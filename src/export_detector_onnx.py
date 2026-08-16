"""Exporta el detector YOLOv8n entrenado a ONNX para la demo web.

Análogo a `export_onnx.py` (clasificador), pero para el detector:
exporta el checkpoint con Ultralytics (`format=onnx, imgsz=256, opset=17,
simplify=True`), copia el resultado a `web/detector.onnx` y verifica la
paridad PyTorch vs ONNX Runtime con `verify_detector_onnx.py`.

Uso:
    python src/export_detector_onnx.py \
        --checkpoint outputs/yolo/train-3/weights/best.pt \
        --salida web/detector.onnx
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
  parser = argparse.ArgumentParser(
    description="Exporta el detector YOLOv8n (.pt) a ONNX y verifica la paridad."
  )
  parser.add_argument(
    "--checkpoint",
    default=str(ROOT / "outputs" / "yolo" / "train" / "weights" / "best.pt"),
    help="Ruta al checkpoint .pt de Ultralytics (por defecto: outputs/yolo/train, el desplegado en la web)",
  )
  parser.add_argument(
    "--salida",
    default=str(ROOT / "web" / "detector.onnx"),
    help="Ruta de salida del archivo .onnx para la demo web",
  )
  parser.add_argument("--img-size", type=int, default=256)
  parser.add_argument("--opset", type=int, default=17)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint = Path(args.checkpoint)
  if not checkpoint.exists():
    raise SystemExit(
      f"No existe {checkpoint} — entrena primero con python src/train_yolo.py"
    )

  salida = Path(args.salida)
  salida.parent.mkdir(parents=True, exist_ok=True)

  print(f"Exportando {checkpoint} a ONNX (imgsz={args.img_size}, opset={args.opset})...")
  model = YOLO(str(checkpoint))
  exported = Path(
    model.export(
      format="onnx",
      imgsz=args.img_size,
      opset=args.opset,
      simplify=True,
    )
  )

  shutil.copy2(exported, salida)
  size_mb = salida.stat().st_size / 1e6
  print(f"ONNX guardado en: {salida} ({size_mb:.1f} MB)")

  # Verificación de paridad PyTorch vs ONNX Runtime (mismo decode/NMS que la web).
  print("\nVerificando paridad contra el checkpoint de origen...")
  result = subprocess.run(
    [
      sys.executable,
      str(ROOT / "src" / "verify_detector_onnx.py"),
      "--pt", str(checkpoint),
      "--onnx", str(salida),
    ],
    cwd=str(ROOT),
  )
  if result.returncode != 0:
    raise SystemExit("La exportación ONNX del detector no coincide con PyTorch.")

  print("\nExportación lista: la demo web usa el mismo modelo verificado.")


if __name__ == "__main__":
  main()
