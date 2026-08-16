"""Genera el dataset de detección (formato YOLO) a partir de metadata.csv.

El dataset de clasificación no trae bounding boxes, pero `metadata.csv`
guarda por imagen los centros (`defect_positions`) y áreas aproximadas
(`defect_sizes_px`) de cada defecto sintético. Como los defectos tienen
alto contraste sobre el fondo metálico, se refinan las cajas con
segmentación clásica (OpenCV):

    fondo = mediana grande → |img - fondo| → umbral (Otsu con mínimo)
    → morfología → contornos → contornos asociados a cada centro → bbox

Si la segmentación no encuentra nada cerca de un centro, se usa como
fallback una caja cuadrada de lado 1.5×√area centrada en el punto.

Salida (hard links, sin duplicar los 390 MB de PNGs):

    yolo_dataset/{train,val,test}/images/<archivo>.png
    yolo_dataset/{train,val,test}/labels/<archivo>.txt   (vacío = background)
    yolo_dataset/defects.yaml

Validación visual: `outputs/yolo_labels_sample.png` (4×4, una fila por
clase de defecto) — revisar antes de entrenar.

Uso:
    python src/prepare_yolo_dataset.py
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "industrial_defect_dataset"
OUT_DIR = ROOT / "yolo_dataset"
SAMPLE_PATH = ROOT / "outputs" / "yolo_labels_sample.png"

# Clases del detector. "normal" no genera etiquetas: son imágenes de
# background (archivo .txt vacío), que reducen falsos positivos.
DETECT_CLASSES = ["crack", "hole", "rust", "scratch"]
CLASS_ID = {name: i for i, name in enumerate(DETECT_CLASSES)}

# Colores BGR por clase para la visualización de muestra.
CLASS_COLORS = {
    "crack": (92, 92, 255),    # rojo
    "hole": (67, 159, 255),    # naranja
    "rust": (47, 123, 201),    # marrón
    "scratch": (255, 120, 182)  # violeta
}

IMG_SIZE = 256  # el dataset es 256×256 uniforme (verificado en metadata)


def segment_defects(gray: np.ndarray) -> np.ndarray:
    """Mapa binario de regiones de defecto por diferencia con el fondo.

    El fondo metálico es suave/texturizado y los defectos (líneas
    oscuras, blobs, parches claros) desaparecen con una mediana grande,
    así que |img - fondo| resalta el defecto completo.
    """
    background = cv2.medianBlur(gray, 31)
    diff = cv2.absdiff(gray, background)
    # Otsu con umbral mínimo: en zonas muy uniformes Otsu baja demasiado
    # y deja pasar ruido de textura.
    t_otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(diff, max(t_otsu, 14), 255, cv2.THRESH_BINARY)
    # Cierre: une fragmentos del mismo defecto (p. ej. scratch con puntos
    # claros separados). Apertura: elimina píxeles sueltos de ruido.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def boxes_for_image(gray: np.ndarray, centers, areas):
    """Devuelve (boxes_xywh, n_segmentados, n_fallback) para una imagen.

    Cada centro de metadata agrupa los contornos cercanos y su unión da
    la caja. Sin contornos cercanos → caja cuadrada desde el área.
    """
    mask = segment_defects(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtrar ruido diminuto de una vez.
    contours = [c for c in contours if cv2.contourArea(c) >= 8]
    centroids = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 0:
            centroids.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        else:
            centroids.append(None)

    boxes = []
    n_seg = n_fb = 0
    for (cx, cy), area in zip(centers, areas):
        # Radio generoso: defectos alargados (scratch/crack) se extienden
        # mucho más allá de √(área).
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
            pad = 3  # margen para cubrir el defecto completo
            x, y = x - pad, y - pad
            w, h = w + 2 * pad, h + 2 * pad
            n_seg += 1
        else:
            side = 1.5 * float(np.sqrt(area))
            x, y, w, h = cx - side / 2, cy - side / 2, side, side
            n_fb += 1

        # Recorte al marco de la imagen.
        x1 = float(np.clip(x, 0, IMG_SIZE - 1))
        y1 = float(np.clip(y, 0, IMG_SIZE - 1))
        x2 = float(np.clip(x + w, 1, IMG_SIZE))
        y2 = float(np.clip(y + h, 1, IMG_SIZE))
        box = (x1, y1, x2 - x1, y2 - y1)
        # Dedup: dos centros pueden caer en el mismo contorno.
        if all(not _same_box(box, b) for b in boxes):
            boxes.append(box)

    return boxes, n_seg, n_fb


def _same_box(a, b, tol=2.0) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def to_yolo(box):
    """(x, y, w, h) en px → línea YOLO normalizada (cx cy w h)."""
    x, y, w, h = box
    return ((x + w / 2) / IMG_SIZE, (y + h / 2) / IMG_SIZE,
            w / IMG_SIZE, h / IMG_SIZE)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard link (mismo volumen, 0 bytes extra); copia como fallback."""
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    meta = pd.read_csv(ROOT / "metadata.csv")
    meta = meta.set_index("filename")
    print(f"metadata.csv: {len(meta)} filas")

    stats = {"seg": 0, "fb": 0, "boxes": 0, "images": 0, "background": 0}
    missing_meta = []
    sample_tiles = {c: [] for c in DETECT_CLASSES}  # para la validación visual

    for split in ("train", "val", "test"):
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for class_dir in sorted((DATASET_DIR / split).iterdir()):
            cls = class_dir.name
            for png in sorted(class_dir.glob("*.png")):
                link_or_copy(png, img_out / png.name)
                stats["images"] += 1

                label_path = (lbl_out / png.name).with_suffix(".txt")
                if cls not in CLASS_ID:
                    # normal (u otra no detectable) → background.
                    label_path.write_text("")
                    stats["background"] += 1
                    continue

                if png.name not in meta.index:
                    missing_meta.append(png.name)
                    label_path.write_text("")
                    continue

                row = meta.loc[png.name]
                centers = ast.literal_eval(row["defect_positions"])
                areas = ast.literal_eval(row["defect_sizes_px"])
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                boxes, n_seg, n_fb = boxes_for_image(gray, centers, areas)
                stats["seg"] += n_seg
                stats["fb"] += n_fb
                stats["boxes"] += len(boxes)

                lines = [
                    f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for (cx, cy, w, h) in (to_yolo(b) for b in boxes)
                ]
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

                # Candidatos para la muestra visual: variedad de conteos.
                if cls in sample_tiles and len(sample_tiles[cls]) < 4 and split == "test":
                    if len(sample_tiles[cls]) == 0 or len(boxes) != sample_tiles[cls][-1][1]:
                        sample_tiles[cls].append((png, len(boxes)))

        print(f"{split}: {stats['images']} imágenes acumuladas")

    # --- defects.yaml ---
    yaml_path = OUT_DIR / "defects.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(DETECT_CLASSES))
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n{names}\n"
    )

    # --- Muestra visual 4×4 (una fila por clase) ---
    thumb = 192
    sheet = np.full((thumb * 4, thumb * 4, 3), 24, np.uint8)
    for r, cls in enumerate(DETECT_CLASSES):
        for c, (png, _) in enumerate(sample_tiles[cls][:4]):
            img = cv2.imread(str(png))
            row = meta.loc[png.name]
            centers = ast.literal_eval(row["defect_positions"])
            areas = ast.literal_eval(row["defect_sizes_px"])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes, _, _ = boxes_for_image(gray, centers, areas)
            for (x, y, w, h) in boxes:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                              CLASS_COLORS[cls], 2)
            cv2.putText(img, f"{cls} ({len(boxes)})", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls], 1)
            sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = cv2.resize(
                img, (thumb, thumb), interpolation=cv2.INTER_AREA)
    SAMPLE_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(SAMPLE_PATH), sheet)

    # --- Reporte ---
    total = stats["seg"] + stats["fb"]
    print("\n=== Reporte ===")
    print(f"Imágenes: {stats['images']} (background sin defectos: {stats['background']})")
    print(f"Defectos etiquetados: {total} → cajas únicas: {stats['boxes']}")
    if total:
        print(f"  por segmentación: {stats['seg']} ({100 * stats['seg'] / total:.1f}%)")
        print(f"  por fallback √area: {stats['fb']} ({100 * stats['fb'] / total:.1f}%)")
    if missing_meta:
        print(f"AVISO: {len(missing_meta)} imágenes sin fila en metadata.csv "
              f"(etiquetadas como background), p. ej. {missing_meta[:5]}")
    print(f"YAML: {yaml_path}")
    print(f"Muestra visual: {SAMPLE_PATH} — REVISAR antes de entrenar.")


if __name__ == "__main__":
    main()
