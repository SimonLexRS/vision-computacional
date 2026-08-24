"""Genera el dataset de detección YOLO combinado (Sintético + SteelDefectX + NEU-DET).

Combina:
1. 15.000 imágenes sintéticas (Tatheer Abbas, CC BY 4.0) con cajas derivadas de metadata.csv + OpenCV.
2. 480 imágenes reales de defectos industriales de SteelDefectX (Zhao et al., 2024).
3. ~1.770 imágenes reales de NEU-DET (Song & Yan, 2013, Northeastern University) con cajas
   reales anotadas en formato Pascal VOC (XML), para maximizar la generalización frente a
   texturas, grano, reflejos y ruido real.

Estructura de salida:
    yolo_dataset/{train,val,test}/images/<archivo>
    yolo_dataset/{train,val,test}/labels/<archivo>.txt
    yolo_dataset/defects.yaml
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT / "industrial_defect_dataset"
STEEL_DIR = ROOT / "external_datasets" / "steel_defect_x"
STEEL_IMG_DIR = STEEL_DIR / "images"
STEEL_META = STEEL_DIR / "selected_metadata.json"
NEU_IMG_DIR = ROOT / "external_datasets" / "neu_det" / "images"
NEU_ANN_DIR = ROOT / "external_datasets" / "neu_det" / "annotations"
OUT_DIR = ROOT / "yolo_dataset"

DETECT_CLASSES = ["crack", "hole", "rust", "scratch"]
CLASS_ID = {name: i for i, name in enumerate(DETECT_CLASSES)}
IMG_SIZE = 256

STEEL_CLASS_MAP = {
  "Crazing": "crack", "Crease": "crack", "Waist folding": "crack",
  "Bright scratch": "scratch", "Dark scratches": "scratch", "Finishing roll printing": "scratch",
  "Secondary rust skin": "rust", "White rust": "rust", "Pitted surface": "rust",
  "Rolled in scale": "rust", "Red iron sheet": "rust",
  "Oxide scale of plate system": "rust", "Oxide scale of temperature system": "rust",
  "Punching": "hole", "Inclusion": "hole", "Slag inclusion": "hole",
  "Patches": "hole", "Crescent gap": "hole", "Rolled pit": "hole",
}

# Mapeo de las 6 clases de NEU-DET (nombres en los XML VOC) a las 4 del proyecto.
NEU_CLASS_MAP = {
  "crazing": "crack",
  "scratches": "scratch",
  "rolled-in_scale": "rust",
  "pitted_surface": "rust",
  "patches": "hole",
  "inclusion": "hole",
}


def segment_defects(gray: np.ndarray) -> np.ndarray:
  background = cv2.medianBlur(gray, 31)
  diff = cv2.absdiff(gray, background)
  t_otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  _, mask = cv2.threshold(diff, max(t_otsu, 14), 255, cv2.THRESH_BINARY)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  return mask


def boxes_for_synthetic_image(gray: np.ndarray, centers, areas):
  mask = segment_defects(gray)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = [c for c in contours if cv2.contourArea(c) >= 8]
  centroids = []
  for c in contours:
    m = cv2.moments(c)
    centroids.append((m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else None)

  boxes = []
  for (cx, cy), area in zip(centers, areas):
    r_search = max(24.0, 2.0 * float(np.sqrt(area)))
    parts = []
    for c, cent in zip(contours, centroids):
      if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0:
        parts.append(c)
      elif cent is not None and np.hypot(cent[0] - cx, cent[1] - cy) <= r_search:
        parts.append(c)

    if parts:
      pts = np.vstack(parts)
      x, y, w, h = cv2.boundingRect(pts)
      pad = 3
      x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    else:
      side = 1.5 * float(np.sqrt(area))
      x, y, w, h = cx - side / 2, cy - side / 2, side, side

    x1 = float(np.clip(x, 0, IMG_SIZE - 1))
    y1 = float(np.clip(y, 0, IMG_SIZE - 1))
    x2 = float(np.clip(x + w, 1, IMG_SIZE))
    y2 = float(np.clip(y + h, 1, IMG_SIZE))
    box = (x1, y1, x2 - x1, y2 - y1)
    if all(not _same_box(box, b) for b in boxes):
      boxes.append(box)

  return boxes


def _same_box(a, b, tol=2.0) -> bool:
  return all(abs(x - y) <= tol for x, y in zip(a, b))


def to_yolo(box, w_img=IMG_SIZE, h_img=IMG_SIZE):
  x, y, w, h = box
  return ((x + w / 2) / w_img, (y + h / 2) / h_img, w / w_img, h / h_img)


def extract_real_boxes(img_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
  h, w = img_bgr.shape[:2]
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  med = cv2.medianBlur(blur, 21)
  diff = cv2.absdiff(blur, med)
  
  _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
  
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  valid_contours = [c for c in contours if cv2.contourArea(c) >= 30]
  
  boxes = []
  for c in valid_contours:
    x, y, bw, bh = cv2.boundingRect(c)
    if bw >= w * 0.95 and bh >= h * 0.95:
      continue
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)
    boxes.append((float(x), float(y), float(bw), float(bh)))

  if not boxes:
    side_w, side_h = w * 0.5, h * 0.5
    boxes.append(((w - side_w) / 2, (h - side_h) / 2, side_w, side_h))

  return boxes[:4]


def parse_voc_xml(xml_path: Path) -> list[tuple[str, float, float, float, float]]:
  """Lee un XML Pascal VOC y devuelve [(clase, xmin, ymin, xmax, ymax), ...]."""
  tree = ET.parse(xml_path)
  root = tree.getroot()
  objects = []
  for obj in root.findall("object"):
    name = obj.findtext("name", default="").strip()
    box = obj.find("bndbox")
    if box is None:
      continue
    xmin = float(box.findtext("xmin", default="0"))
    ymin = float(box.findtext("ymin", default="0"))
    xmax = float(box.findtext("xmax", default="0"))
    ymax = float(box.findtext("ymax", default="0"))
    if xmax > xmin and ymax > ymin:
      objects.append((name, xmin, ymin, xmax, ymax))
  return objects


def link_or_copy(src: Path, dst: Path) -> None:
  try:
    if dst.exists():
      dst.unlink()
    os.link(src, dst)
  except OSError:
    shutil.copy2(src, dst)


def main():
  print("=== Preparando Dataset Combinado para YOLOv8 (Sintético + Real) ===")
  
  for split in ("train", "val", "test"):
    (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

  meta = pd.read_csv(ROOT / "metadata.csv").set_index("filename")
  print(f"metadata.csv cargado: {len(meta)} registros.")

  # 1. Dataset Sintético Base
  synth_counts = {"train": 0, "val": 0, "test": 0}
  for split in ("train", "val", "test"):
    img_out = OUT_DIR / split / "images"
    lbl_out = OUT_DIR / split / "labels"

    for class_dir in sorted((SYNTH_DIR / split).iterdir()):
      cls = class_dir.name
      for png in sorted(class_dir.glob("*.png")):
        link_or_copy(png, img_out / png.name)
        synth_counts[split] += 1
        label_path = (lbl_out / png.name).with_suffix(".txt")

        if cls not in CLASS_ID:
          label_path.write_text("")
          continue

        if png.name not in meta.index:
          label_path.write_text("")
          continue

        row = meta.loc[png.name]
        centers = ast.literal_eval(row["defect_positions"])
        areas = ast.literal_eval(row["defect_sizes_px"])
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        boxes = boxes_for_synthetic_image(gray, centers, areas)

        lines = [
          f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
          for (cx, cy, w, h) in (to_yolo(b) for b in boxes)
        ]
        label_path.write_text("".join(lines))

  print(f"Sintético procesado: {synth_counts}")

  # 2. Dataset Real SteelDefectX
  real_counts = {"train": 0, "val": 0, "test": 0}
  if STEEL_META.exists() and STEEL_IMG_DIR.exists():
    with open(STEEL_META, "r", encoding="utf-8") as f:
      real_records = json.load(f)

    print(f"\nProcesando {len(real_records)} imágenes reales de SteelDefectX...")
    np.random.seed(42)
    np.random.shuffle(real_records)

    for i, r in enumerate(real_records):
      img_name = r["image_name"]
      c_orig = r.get("class_name")
      if c_orig not in STEEL_CLASS_MAP:
        continue

      c_target = STEEL_CLASS_MAP[c_orig]
      cid = CLASS_ID[c_target]
      src_img = STEEL_IMG_DIR / img_name
      if not src_img.exists():
        continue

      ratio = i / len(real_records)
      split = "train" if ratio < 0.8 else ("val" if ratio < 0.9 else "test")

      img_bgr = cv2.imread(str(src_img))
      if img_bgr is None:
        continue

      h, w = img_bgr.shape[:2]
      boxes = extract_real_boxes(img_bgr)
      
      img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
      out_img_name = f"real_{img_name.replace('.jpg', '.png')}"
      dst_img = OUT_DIR / split / "images" / out_img_name
      dst_txt = OUT_DIR / split / "labels" / (Path(out_img_name).stem + ".txt")

      cv2.imwrite(str(dst_img), img_resized)

      lines = [
        f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        for (cx, cy, bw, bh) in (to_yolo(b, w, h) for b in boxes)
      ]
      dst_txt.write_text("".join(lines))
      real_counts[split] += 1

    print(f"Imágenes reales procesadas: {real_counts}")

  # 3. Dataset Real NEU-DET (cajas reales anotadas en VOC XML)
  neu_counts = {"train": 0, "val": 0, "test": 0}
  if NEU_ANN_DIR.exists() and NEU_IMG_DIR.exists():
    neu_xmls = sorted(NEU_ANN_DIR.glob("*.xml"))
    print(f"\nProcesando {len(neu_xmls)} imágenes reales de NEU-DET...")
    np.random.seed(42)
    np.random.shuffle(neu_xmls)

    for i, xml_path in enumerate(neu_xmls):
      objects = parse_voc_xml(xml_path)
      # Solo objetos de clases mapeadas al proyecto.
      mapped = [(NEU_CLASS_MAP[n], x1, y1, x2, y2)
                for n, x1, y1, x2, y2 in objects if n in NEU_CLASS_MAP]
      if not mapped:
        continue

      src_img = NEU_IMG_DIR / (xml_path.stem + ".jpg")
      if not src_img.exists():
        continue

      ratio = i / len(neu_xmls)
      split = "train" if ratio < 0.8 else ("val" if ratio < 0.9 else "test")

      img_bgr = cv2.imread(str(src_img))
      if img_bgr is None:
        continue
      h, w = img_bgr.shape[:2]

      img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
      out_img_name = f"neu_{xml_path.stem}.png"
      dst_img = OUT_DIR / split / "images" / out_img_name
      dst_txt = OUT_DIR / split / "labels" / (Path(out_img_name).stem + ".txt")

      cv2.imwrite(str(dst_img), img_resized)

      # Cajas reales del XML -> YOLO normalizado (el resize preserva las fracciones).
      lines = []
      for cls, x1, y1, x2, y2 in mapped:
        cx, cy, bw, bh = to_yolo((x1, y1, x2 - x1, y2 - y1), w, h)
        lines.append(f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
      dst_txt.write_text("".join(lines))
      neu_counts[split] += 1

    print(f"Imágenes NEU-DET procesadas: {neu_counts}")
  else:
    print("\nAviso: no se encontró external_datasets/neu_det — "
          "corre python src/download_datasets.py para incluir NEU-DET.")

  # 4. Guardar YAML de configuración para YOLOv8
  yaml_content = f"""train: train/images
val: val/images
test: test/images

names:
  0: crack
  1: hole
  2: rust
  3: scratch
"""
  (OUT_DIR / "defects.yaml").write_text(yaml_content, encoding="utf-8")
  print(f"\nConfiguración guardada en: {OUT_DIR / 'defects.yaml'}")
  print(f"Dataset combinado listo para entrenamiento GPU.")


if __name__ == "__main__":
  main()
