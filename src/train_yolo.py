"""Entrena el detector YOLOv8n en GPU (NVIDIA RTX 5060 Ti) sobre el dataset combinado.

Reproduce la Sección 15 del notebook `entrenar_modelos.ipynb` con exactamente
los mismos hiperparámetros, para entrenar sin abrir Jupyter.

Dataset (3 fuentes, construido por prepare_combined_dataset.py):
    - 15.000 imágenes sintéticas (Tatheer Abbas, CC BY 4.0).
    - ~480 imágenes reales de SteelDefectX (Zhao et al., 2024).
    - ~1.770 imágenes reales de NEU-DET (Song & Yan, 2013) con cajas VOC reales.

Pipeline:
    1. python src/download_datasets.py        (descarga datasets reales)
    2. python src/prepare_combined_dataset.py  (genera yolo_dataset/)
    3. python src/train_yolo.py                (este script)
    4. python src/export_detector_onnx.py      (exporta a la demo web)
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "yolo_dataset" / "defects.yaml"
PROJECT = ROOT / "outputs" / "yolo"


def main() -> None:
  if not DATA_YAML.exists():
    raise SystemExit(
      "No existe yolo_dataset/defects.yaml — corre primero "
      "python src/prepare_combined_dataset.py"
    )

  model = YOLO("yolov8n.pt")  # Pretrained COCO
  
  # Entrenamiento optimizado para NVIDIA GeForce RTX 5060 Ti
  model.train(
    data=str(DATA_YAML),
    imgsz=256,
    epochs=60,
    patience=15,          # Early stopping
    batch=64,
    device=0,             # GPU 0 (NVIDIA RTX 5060 Ti)
    workers=0,            # Estable en Windows sin deadlock
    seed=42,
    optimizer="SGD",      # SGD con momentum (óptimo y libre de bugs MuSGD)
    amp=True,             # Mixed precision automático para máxima velocidad GPU
    # Data Augmentation avanzada para generalización en cámara y web
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    flipud=0.5,
    mosaic=0.5,
    mixup=0.1,
    close_mosaic=10,    # Desactiva mosaico las últimas 10 épocas (igual que el notebook)
    project=str(PROJECT),
    name="train",
    exist_ok=True,
  )

  # Evaluación en el split test físico
  metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=256,
    batch=64,
    project=str(PROJECT),
    name="test_val",
    exist_ok=True,
  )

  class_names = [metrics.names[i] for i in sorted(metrics.names)]
  # Mismo esquema de resultados que la Sección 15 del notebook.
  results = {
    "model": "yolov8n",
    "imgsz": 256,
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "per_class": {
      name: {
        "AP50": float(metrics.box.ap50[i]),
        "AP50-95": float(metrics.box.maps[i]),
      }
      for i, name in enumerate(class_names)
    },
  }
  out = PROJECT / "results.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

  # Resumen por clase en CSV (igual que yolo_test_summary.csv del notebook).
  rows = sorted(
    ({"clase": name, **vals} for name, vals in results["per_class"].items()),
    key=lambda r: r["AP50"],
    reverse=True,
  )
  csv_path = PROJECT / "yolo_test_summary.csv"
  with open(csv_path, "w", encoding="utf-8") as f:
    f.write("clase,AP50,AP50-95\n")
    for r in rows:
      f.write(f"{r['clase']},{r['AP50']},{r['AP50-95']}\n")

  print("\n" + "=" * 50)
  print("RESULTADOS FINALES DE DETECCIÓN (SPLIT TEST):")
  print(json.dumps(results, indent=2, ensure_ascii=False))
  print(f"\nPesos entrenados: {PROJECT / 'train' / 'weights' / 'best.pt'}")
  print(f"Métricas guardadas en: {out} y {csv_path}")


if __name__ == "__main__":
  main()
