"""Entrena el detector YOLOv8n en GPU (NVIDIA RTX 5060 Ti) sobre el dataset combinado.

Reproduce la Seccion 15 del notebook `entrenar_modelos.ipynb` con exactamente
los mismos hiperparametros, para entrenar sin abrir Jupyter.

Dataset (3 fuentes, construido por prepare_combined_dataset.py):
    - 15.000 imagenes sinteticas (Tatheer Abbas, CC BY 4.0).
    - ~480 imagenes reales de SteelDefectX (Zhao et al., 2024).
    - ~1.770 imagenes reales de NEU-DET (Song & Yan, 2013) con cajas VOC reales.

Pipeline:
    1. python src/download_datasets.py        (descarga todos los datasets)
    2. python src/prepare_combined_dataset.py  (genera yolo_dataset/)
    3. python src/train_yolo.py                (este script)
    4. python src/export_detector_onnx.py      (exporta a la demo web)

Argumentos:
    --datos     Ruta al archivo YAML del dataset YOLO (default: yolo_dataset/defects.yaml)
    --salida    Carpeta de salida para pesos y metricas (default: outputs/yolo)
    --epochs    Numero maximo de epocas (default: 60)
    --device    Dispositivo: 0 para GPU, cpu para CPU (default: 0)
    --batch     Batch size (default: 64)
    --patience  Paciencia para early stopping (default: 15)
    --seed      Semilla aleatoria (default: 42)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "yolo_dataset" / "defects.yaml"
DEFAULT_PROJECT = ROOT / "outputs" / "yolo"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el detector YOLOv8n sobre el dataset combinado de defectos."
    )
    parser.add_argument(
        "--datos", type=str, default=str(DEFAULT_DATA),
        help="Ruta al archivo YAML del dataset YOLO (default: yolo_dataset/defects.yaml).",
    )
    parser.add_argument(
        "--salida", type=str, default=str(DEFAULT_PROJECT),
        help="Carpeta de salida para pesos y metricas (default: outputs/yolo).",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Numero maximo de epocas (default: 60).",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Dispositivo: 0 para GPU, cpu para CPU (default: 0).",
    )
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Paciencia para early stopping (default: 15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria (default: 42).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=256,
        help="Tamano de imagen (default: 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.datos)
    if not data_yaml.exists():
        raise SystemExit(
            f"No existe {data_yaml} — corre primero "
            "python src/prepare_combined_dataset.py"
        )

    project = Path(args.salida)
    project.mkdir(parents=True, exist_ok=True)

    # Cargar modelo preentrenado COCO
    model = YOLO("yolov8n.pt")

    # Entrenamiento optimizado para NVIDIA GeForce RTX 5060 Ti
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,       # Early stopping
        batch=args.batch,
        device=args.device,           # GPU 0 o cpu
        workers=0,                    # Estable en Windows sin deadlock
        seed=args.seed,
        optimizer="SGD",              # SGD con momentum (optimo y libre de bugs MuSGD)
        amp=True,                     # Mixed precision automatico para maxima velocidad GPU
        # Data Augmentation avanzada para generalizacion en camara y web
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
        close_mosaic=10,              # Desactiva mosaico las ultimas 10 epocas
        project=str(project),
        name="train",
        exist_ok=True,
    )

    # Evaluacion en el split test fisico
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name="test_val",
        exist_ok=True,
    )

    class_names = [metrics.names[i] for i in sorted(metrics.names)]
    # Mismo esquema de resultados que la Seccion 15 del notebook.
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
    out = project / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Resumen por clase en CSV (igual que yolo_test_summary.csv del notebook).
    rows = sorted(
        ({"clase": name, **vals} for name, vals in results["per_class"].items()),
        key=lambda r: r["AP50"],
        reverse=True,
    )
    csv_path = project / "yolo_test_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("clase,AP50,AP50-95\n")
        for r in rows:
            f.write(f"{r['clase']},{r['AP50']},{r['AP50-95']}\n")

    print("\n" + "=" * 50)
    print("RESULTADOS FINALES DE DETECCION (SPLIT TEST):")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nPesos entrenados: {project / 'train' / 'weights' / 'best.pt'}")
    print(f"Metricas guardadas en: {out} y {csv_path}")


if __name__ == "__main__":
    main()
