"""Entrena el detector YOLOv8n sobre el dataset combinado (3 fuentes).

Reproduce la Sección 15 del notebook `entrenar_modelos.ipynb` con exactamente
los mismos hiperparámetros, para entrenar sin abrir Jupyter. Todos los valores
por defecto son los usados en el resultado reportado.

Dataset (construido por prepare_combined_dataset.py):
    - 15.000 imágenes sintéticas (Tatheer Abbas, CC BY 4.0).
    - ~480 imágenes reales de SteelDefectX (Zhao et al., 2024).
    - ~1.770 imágenes reales de NEU-DET (Song & Yan, 2013) con cajas VOC reales.

Pipeline:
    1. python src/download_datasets.py         (descarga datasets reales)
    2. python src/prepare_combined_dataset.py  (genera yolo_dataset/)
    3. python src/train_yolo.py                (este script)
    4. python src/export_detector_onnx.py      (exporta a la demo web)

Ejemplos:
    python src/train_yolo.py
    python src/train_yolo.py --epochs 100 --batch 32
    python src/train_yolo.py --device cpu --epochs 5      # prueba sin GPU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "yolo_dataset" / "defects.yaml"
PROJECT = ROOT / "outputs" / "yolo"


def parse_args():
  parser = argparse.ArgumentParser(
    description="Entrena YOLOv8n sobre el dataset combinado "
                "(los defaults reproducen el resultado reportado).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--datos", default=str(DATA_YAML),
                      help="Ruta al defects.yaml del dataset YOLO")
  parser.add_argument("--modelo-base", default="yolov8n.pt",
                      help="Checkpoint de partida (pretrained COCO)")
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--imgsz", type=int, default=256)
  parser.add_argument("--patience", type=int, default=15,
                      help="Épocas sin mejora de mAP50 antes de detener")
  parser.add_argument("--device", default="0",
                      help="'0' para la primera GPU NVIDIA, 'cpu' para CPU")
  parser.add_argument("--workers", type=int, default=0,
                      help="0 evita el deadlock del DataLoader en Windows")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--optimizer", default="SGD",
                      help="SGD es estable; 'auto' elige MuSGD, que crashea en CUDA 13")
  parser.add_argument("--project", default=str(PROJECT),
                      help="Carpeta raíz donde Ultralytics guarda las corridas")
  parser.add_argument("--name", default="train",
                      help="Nombre de la corrida dentro de --project")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  data_yaml = Path(args.datos)
  if not data_yaml.exists():
    raise SystemExit(
      f"No existe {data_yaml} — corre primero "
      "python src/prepare_combined_dataset.py"
    )

  print(f"Dataset: {data_yaml}")
  print(f"Dispositivo: {args.device} | épocas: {args.epochs} | batch: {args.batch}")

  model = YOLO(args.modelo_base)  # pretrained COCO

  model.train(
    data=str(data_yaml),
    imgsz=args.imgsz,
    epochs=args.epochs,
    patience=args.patience,   # early stopping
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    seed=args.seed,
    optimizer=args.optimizer,
    amp=True,                 # mixed precision automático
    # Data augmentation avanzada para generalización en cámara y web
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
    close_mosaic=10,          # desactiva mosaico las últimas 10 épocas
    project=args.project,
    name=args.name,
    exist_ok=True,
  )

  # Evaluación en el split test físico
  metrics = model.val(
    data=str(data_yaml),
    split="test",
    imgsz=args.imgsz,
    batch=args.batch,
    project=args.project,
    name="test_val",
    exist_ok=True,
  )

  class_names = [metrics.names[i] for i in sorted(metrics.names)]
  results = {
    "model": "yolov8n",
    "imgsz": args.imgsz,
    "epochs": args.epochs,
    "seed": args.seed,
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

  project_dir = Path(args.project)
  out = project_dir / "results.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

  rows = sorted(
    ({"clase": name, **vals} for name, vals in results["per_class"].items()),
    key=lambda r: r["AP50"],
    reverse=True,
  )
  csv_path = project_dir / "yolo_test_summary.csv"
  with open(csv_path, "w", encoding="utf-8") as f:
    f.write("clase,AP50,AP50-95\n")
    for r in rows:
      f.write(f"{r['clase']},{r['AP50']},{r['AP50-95']}\n")

  print("\n" + "=" * 50)
  print("RESULTADOS FINALES DE DETECCIÓN (SPLIT TEST):")
  print(json.dumps(results, indent=2, ensure_ascii=False))
  print(f"\nPesos entrenados: {project_dir / args.name / 'weights' / 'best.pt'}")
  print(f"Métricas guardadas en: {out} y {csv_path}")


if __name__ == "__main__":
  main()
