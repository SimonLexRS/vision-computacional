"""Evalúa un modelo guardado sobre el split de test (o val) del dataset.

Determinista: sin aleatoriedad en la carga de datos ni en el orden de
los batches, por lo que sobre el mismo .pt y el mismo split siempre
reporta las mismas métricas.

Ejemplo:
    python src/evaluate.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
        --datos industrial_defect_dataset/ --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from common import evaluate, load_checkpoint, make_eval_transform, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa un checkpoint .pt sobre test/ o val/."
    )
    parser.add_argument("--modelo", required=True,
                        help="Ruta al checkpoint .pt guardado por train.py")
    parser.add_argument("--datos", default="industrial_defect_dataset",
                        help="Carpeta del dataset con subcarpetas train/, val/ y test/")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Partición a evaluar (default: test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = select_device(args.device)

    model, checkpoint = load_checkpoint(args.modelo, device)
    class_names = checkpoint["class_names"]

    split_dir = Path(args.datos) / args.split
    if not split_dir.exists():
        raise SystemExit(f"No existe la partición: {split_dir}")

    ds = datasets.ImageFolder(
        split_dir, transform=make_eval_transform(args.img_size)
    )
    if ds.classes != class_names:
        raise SystemExit(
            f"Las clases de {split_dir} ({ds.classes}) no coinciden "
            f"con las del checkpoint ({class_names})."
        )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, device)

    print(f"Modelo:     {checkpoint['model_name']} ({args.modelo})")
    print(f"Split:      {args.split} ({len(ds)} imágenes)")
    print(f"Dispositivo: {device}")
    print()
    print(f"Loss:        {metrics['loss']:.6f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    print()
    print("Matriz de confusión:")
    print(confusion_matrix(metrics["y_true"], metrics["y_pred"]))
    print()
    print(classification_report(
        metrics["y_true"], metrics["y_pred"],
        target_names=class_names, digits=4, zero_division=0,
    ))


if __name__ == "__main__":
    main()
