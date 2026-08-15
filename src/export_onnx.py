"""Exporta un checkpoint del proyecto a ONNX para la demo web.

Exporta el modelo y verifica que las predicciones del grafo ONNX
(onnxruntime) coincidan con las de PyTorch sobre imágenes reales del
conjunto de test.

Ejemplo:
    python src/export_onnx.py \
        --checkpoint outputs/checkpoints/mobilenet_v3_small_best.pt \
        --salida web/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import load_checkpoint, make_eval_transform, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exporta un checkpoint .pt a ONNX y verifica la paridad."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al checkpoint .pt guardado por train.py")
    parser.add_argument("--salida", default="web/model.onnx",
                        help="Ruta de salida del archivo .onnx")
    parser.add_argument("--datos", default="industrial_defect_dataset",
                        help="Carpeta del dataset (para verificar con imágenes de test/)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = select_device("cpu")  # la exportación se hace en CPU

    model, checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    print(f"Exportando {checkpoint['model_name']} (clases: {class_names})")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    torch.onnx.export(
        model,
        dummy,
        salida,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        # Un solo archivo autocontenido (sin model.onnx.data externo),
        # más simple de servir en la demo web estática.
        external_data=False,
    )

    size_mb = salida.stat().st_size / 1e6
    print(f"ONNX guardado en: {salida} ({size_mb:.1f} MB)")

    # --- Verificación de paridad PyTorch vs ONNX Runtime ---
    import onnxruntime as ort

    session = ort.InferenceSession(str(salida), providers=["CPUExecutionProvider"])
    transform = make_eval_transform(args.img_size)

    test_dir = Path(args.datos) / "test"
    muestras = []
    for cls in class_names:
        imagenes = sorted((test_dir / cls).glob("*.png"))
        if imagenes:
            muestras.append((cls, imagenes[0]))

    if not muestras:
        print("No se encontraron imágenes de test/; se omite la verificación.")
        return

    print("\nVerificación de paridad (PyTorch vs ONNX Runtime):")
    todas_ok = True
    with torch.no_grad():
        for real, ruta in muestras:
            tensor = transform(Image.open(ruta)).unsqueeze(0)

            logits_pt = model(tensor).numpy()[0]
            logits_onnx = session.run(
                ["logits"], {"input": tensor.numpy()}
            )[0][0]

            pred_pt = class_names[int(logits_pt.argmax())]
            pred_onnx = class_names[int(logits_onnx.argmax())]
            diff = float(np.abs(logits_pt - logits_onnx).max())

            ok = pred_pt == pred_onnx and diff < 1e-3
            todas_ok = todas_ok and ok
            print(
                f"  {ruta.name}: real={real} torch={pred_pt} onnx={pred_onnx} "
                f"max_diff={diff:.2e} {'OK' if ok else 'FALLA'}"
            )

    if not todas_ok:
        raise SystemExit("La exportación ONNX no coincide con PyTorch.")

    print("\nParidad verificada: el modelo ONNX predice igual que PyTorch.")


if __name__ == "__main__":
    main()
