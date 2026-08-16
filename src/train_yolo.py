"""Entrena el detector YOLOv8n sobre el dataset derivado de metadata.

Pipeline:
    1. python src/prepare_yolo_dataset.py   (genera yolo_dataset/)
    2. python src/train_yolo.py             (este script)

- imgsz=256: tamaño nativo del dataset (sin letterbox destructivo).
- `normal` está como background (sin etiquetas) para reducir FPs.
- Evalúa en el split test físico y guarda outputs/yolo/results.json.

Las etiquetas se derivan de metadata.csv (posiciones/tamaños
aproximados refinados por segmentación): métricas con esa reserva.
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
            "python src/prepare_yolo_dataset.py"
        )

    model = YOLO("yolov8n.pt")  # pretrained COCO (descarga automática)
    model.train(
        data=str(DATA_YAML),
        imgsz=256,
        epochs=100,
        patience=20,      # early stopping
        batch=64,
        device=0,
        # workers=0: con workers>0 el DataLoader se deadlock en Windows
        # a mitad de epoch (GPU al 1%, log congelado).
        workers=0,
        seed=42,
        # optimizer=auto elige MuSGD (nuevo en ultralytics 8.4), que
        # crashea con torch 2.13/cu130 en esta GPU. SGD es la receta
        # clásica y estable de YOLOv8.
        optimizer="SGD",
        project=str(PROJECT),
        name="train",
    )

    # Evaluación en el split test físico del dataset.
    metrics = model.val(
        data=str(DATA_YAML), split="test", imgsz=256, batch=64,
        project=str(PROJECT), name="test_val",
    )

    class_names = [metrics.names[i] for i in sorted(metrics.names)]
    results = {
        "model": "yolov8n",
        "imgsz": 256,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": {
            name: {
                "ap50": float(metrics.box.ap50[i]),
                "ap50_95": float(metrics.box.maps[i]),
            }
            for i, name in enumerate(class_names)
        },
    }
    out = PROJECT / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nPesos: {PROJECT / 'train' / 'weights' / 'best.pt'}")
    print(f"Métricas: {out}")


if __name__ == "__main__":
    main()
