"""Entrena los modelos del proyecto desde la línea de comandos.

Reproduce el entrenamiento del notebook `entrenar_modelos.ipynb` sin
abrir Jupyter. Con los argumentos por defecto reproduce exactamente el
resultado reportado en el informe.

Ejemplo:
    python src/train.py --datos industrial_defect_dataset/ --salida outputs/checkpoints/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from common import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    evaluate,
    make_eval_transform,
    make_train_transform,
    select_device,
    set_seed,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return running_loss / len(loader.dataset), correct / total


def train_model(model_name, train_loader, val_loader, class_names, args, device):
    # Reiniciar la semilla antes de cada modelo.
    
    set_seed(args.seed)

    # El DataLoader tiene su propio Generator: set_seed() reinicia el RNG global
    # de torch, pero no ese objeto. Sin esta línea, el orden de barajado del
    # modelo N depende de cuántas épocas corrieron los N-1 anteriores, y entrenar
    # un modelo suelto no reproduce su fila de la tabla de resultados.

    if train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)

    model = build_model(model_name, num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Planificador opcional. Con "none" el LR queda constante, que es la
    # configuración con la que se obtuvieron los resultados reportados.
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    best_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    checkpoint_path = Path(args.salida) / f"{model_name}_best.pt"

    print(f"\nEntrenando {model_name} en {device}")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])

        print(
            f"Época {epoch:02d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        # Guardar el mejor modelo según F1 macro de validación.
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": best_state,
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "metrics": {
                        "accuracy": val_metrics["accuracy"],
                        "f1_macro": val_metrics["f1_macro"],
                        "f1_weighted": val_metrics["f1_weighted"],
                    },
                    "config": {
                        "label_smoothing": args.label_smoothing,
                        "scheduler": args.scheduler,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                    }
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: el F1 macro dejó de mejorar.")
                break
        if scheduler is not None:
            scheduler.step()

    training_time = time.time() - start_time

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device)

    return {
        "model": model_name,
        "params": count_parameters(model),
        "epochs_run": len(history["epoch"]),
        "seconds": training_time,
        "val_accuracy": final_metrics["accuracy"],
        "val_f1_macro": final_metrics["f1_macro"],
        "val_f1_weighted": final_metrics["f1_weighted"],
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena los modelos de clasificación de defectos "
                    "(los defaults reproducen el resultado del informe)."
    )
    parser.add_argument("--datos", default="industrial_defect_dataset",
                        help="Carpeta del dataset con subcarpetas train/ y val/")
    parser.add_argument("--salida", default="outputs/checkpoints",
                        help="Carpeta donde se guardan los pesos .pt")
    parser.add_argument("--modelos", nargs="+", default=MODEL_NAMES,
                        choices=MODEL_NAMES,
                        help="Modelos a entrenar (default: los cuatro)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Máximo de épocas (early stopping puede cortar antes)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Suavizado de etiquetas (0.0 = desactivado, el valor reportado)")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                        help="Planificador de learning rate (none = el valor reportado)")

    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas sin mejora de F1 macro antes de detener")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Semilla: {args.seed} | Dispositivo: {device}")

    datos = Path(args.datos)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(
        datos / "train", transform=make_train_transform(args.img_size)
    )
    val_ds = datasets.ImageFolder(
        datos / "val", transform=make_eval_transform(args.img_size)
    )

    if train_ds.classes != val_ds.classes:
        raise SystemExit(
            f"Las clases de train y val no coinciden: "
            f"{train_ds.classes} vs {val_ds.classes}"
        )
    class_names = train_ds.classes

    print(f"Clases: {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        generator=generator, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )

    results = []
    for model_name in args.modelos:
        results.append(
            train_model(model_name, train_loader, val_loader, class_names, args, device)
        )

    # Guardar resumen del experimento junto a los checkpoints.
    summary = {
        "args": vars(args),
        "device": str(device),
        "class_names": class_names,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "results": [{k: v for k, v in r.items() if k != "history"} for r in results],
    }
    summary_path = salida / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEntrenamiento completo. Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
