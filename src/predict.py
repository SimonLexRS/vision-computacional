"""Corre inferencia con un modelo guardado sobre una imagen o carpeta.

Funciona en CPU (default), sin necesidad de GPU. Acepta un archivo de
imagen o una carpeta (procesa todas las imágenes dentro).

Ejemplos:
    python src/predict.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
        --imagen industrial_defect_dataset/test/crack/crack_00003.png

    python src/predict.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
        --imagen industrial_defect_dataset/test/rust/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from common import load_checkpoint, make_eval_transform, select_device, set_seed

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de defectos en superficies metálicas (CPU por defecto)."
    )
    parser.add_argument("--modelo", required=True,
                        help="Ruta al checkpoint .pt guardado por train.py")
    parser.add_argument("--imagen", required=True,
                        help="Imagen o carpeta con imágenes a clasificar")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="cpu",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise SystemExit(f"No existe: {path}")


@torch.no_grad()
def predict_image(model, image_path, transform, device, class_names):
    # El context manager asegura cerrar el archivo al procesar carpetas.
    with Image.open(image_path) as img:
        # El modelo ya convierte a RGB de 3 canales (Grayscale→3 en transform).
        tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    return class_names[pred_idx], probs


def main():
    args = parse_args()

    set_seed(args.seed)
    device = select_device(args.device)

    model, checkpoint = load_checkpoint(args.modelo, device)
    class_names = checkpoint["class_names"]
    transform = make_eval_transform(args.img_size)

    images = collect_images(Path(args.imagen))
    if not images:
        raise SystemExit("No se encontraron imágenes en la ruta indicada.")

    print(f"Modelo: {checkpoint['model_name']} | Dispositivo: {device}")
    print(f"Imágenes a clasificar: {len(images)}\n")

    for image_path in images:
        pred_class, probs = predict_image(
            model, image_path, transform, device, class_names
        )
        probs_str = "  ".join(
            f"{name}={prob:.3f}" for name, prob in zip(class_names, probs)
        )
        print(f"{image_path.name}: {pred_class}  [{probs_str}]")


if __name__ == "__main__":
    main()
