"""Extrae predicciones incorrectas del modelo de clasificacion y del detector YOLO.

Genera una figura con 10-15 predicciones incorrectas, cada una con una
hipotesis de por que fallo, para incluir en el informe (Seccion Resultados).

El informe (rubrica, Seccion 6) exige: "10-15 predicciones correctas y
10-15 incorrectas, con una hipotesis de por que fallo cada una."

Uso:
    python src/analyze_errors.py
    python src/analyze_errors.py --modelo outputs/checkpoints/cnn_baseline_best.pt
    python src/analyze_errors.py --salida informe/figures/predicciones_incorrectas.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_checkpoint, make_eval_transform, set_seed

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLF_MODEL = ROOT / "outputs" / "checkpoints" / "cnn_baseline_best.pt"
DEFAULT_DATA = ROOT / "industrial_defect_dataset"
DEFAULT_YOLO_MODEL = ROOT / "outputs" / "yolo" / "train" / "weights" / "best.pt"
DEFAULT_YOLO_DATA = ROOT / "yolo_dataset" / "test"
DEFAULT_OUT = ROOT / "informe" / "figures" / "predicciones_incorrectas.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae predicciones incorrectas (clasificacion + deteccion) para el informe."
    )
    parser.add_argument("--modelo", type=str, default=str(DEFAULT_CLF_MODEL),
                        help="Checkpoint .pt del modelo de clasificacion (default: CNN baseline).")
    parser.add_argument("--datos", type=str, default=str(DEFAULT_DATA),
                        help="Carpeta del dataset de clasificacion con subcarpetas train/, val/ y test/.")
    parser.add_argument("--yolo-modelo", type=str, default=str(DEFAULT_YOLO_MODEL),
                        help="Checkpoint .pt del detector YOLOv8n.")
    parser.add_argument("--yolo-datos", type=str, default=str(DEFAULT_YOLO_DATA),
                        help="Carpeta con imagenes de test del dataset YOLO.")
    parser.add_argument("--salida", type=str, default=str(DEFAULT_OUT),
                        help="Ruta de salida para la figura PNG.")
    parser.add_argument("--max-errores", type=int, default=15,
                        help="Numero maximo de predicciones incorrectas a mostrar (default: 15).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo: cpu o cuda (default: cpu).")
    return parser.parse_args()


# ===================== CLASIFICACION =====================

@torch.no_grad()
def find_clf_errors(model, dataset, device, max_errors=5):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    errors = []
    class_names = dataset.classes

    for i, (img, label) in enumerate(loader):
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        pred = output.argmax(dim=1)

        if pred != label:
            probs = torch.softmax(output, dim=1)
            conf = probs[0, pred].item()
            true_conf = probs[0, label].item()
            errors.append({
                "tipo": "clasificacion",
                "path": dataset.samples[i][0],
                "true_label": class_names[label.item()],
                "pred_label": class_names[pred.item()],
                "confidence": conf,
                "true_confidence": true_conf,
                "img_tensor": img.cpu().squeeze(0),
            })
            if len(errors) >= max_errors:
                break
    return errors


def clf_hypothesis(error):
    true_cls = error["true_label"]
    pred_cls = error["pred_label"]
    conf = error["confidence"]
    true_conf = error["true_confidence"]

    hypotheses = {
        ("crack", "normal"): "Grieta de bajo contraste: el modelo no detecta el defecto y clasifica como normal.",
        ("crack", "rust"): "Grieta con bordes oxidados: el modelo atiende al oxido y clasifica como rust.",
        ("rust", "normal"): "Oxido leve o temprano: el modelo no distingue la variacion de tono del oxido inicial.",
        ("rust", "scratch"): "Mancha de oxido alargada que el modelo interpreta como rayon.",
        ("scratch", "crack"): "Rayon profundo que el modelo interpreta como fisura estructural.",
        ("hole", "rust"): "Perforacion pequena con halo oxidado que el modelo clasifica como rust.",
        ("normal", "scratch"): "Textura superficial normal con marcas leves confundidas con rayones.",
    }
    key = (true_cls, pred_cls)
    if key in hypotheses:
        return hypotheses[key]
    return f"Confusion {true_cls} -> {pred_cls}: confianzas cercanas (pred={conf:.2f}, real={true_conf:.2f})."


# ===================== DETECCION YOLO =====================

def find_yolo_errors(yolo_model_path, test_dir, device, max_errors=13):
    from ultralytics import YOLO

    model = YOLO(str(yolo_model_path))
    test_dir = Path(test_dir)
    # Las imagenes pueden estar en test_dir/ o test_dir/images/
    images_dir = test_dir / "images" if (test_dir / "images").exists() else test_dir
    # Los labels pueden estar en test_dir/labels o test_dir/../labels
    label_dir = test_dir / "labels" if (test_dir / "labels").exists() else test_dir.parent / "labels"

    errors = []
    img_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in img_files:
        if len(errors) >= max_errors:
            break

        label_path = label_dir / (img_path.stem + ".txt")

        gt_classes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_classes.append(int(parts[0]))

        results = model(img_path, verbose=False, device=device, conf=0.5)
        pred_classes = []
        pred_confs = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                pred_classes.append(int(box.cls.item()))
                pred_confs.append(float(box.conf.item()))

        class_names = results[0].names if results else {}
        error_type = None
        hypothesis = None

        if len(gt_classes) > 0 and len(pred_classes) == 0:
            gt_cls = class_names.get(gt_classes[0], "?")
            error_type = "falso_negativo"
            hypotheses_fn = {
                "scratch": "Falso negativo (scratch): el rayon es una linea fina y alargada que el detector no localiza. Bajo contraste y caja sensible a desviaciones de pocos pixeles.",
                "rust": "Falso negativo (rust): la mancha de oxido tiene textura variable que el detector no distingue del fondo.",
                "crack": "Falso negativo (crack): la grieta es muy fina o de bajo contraste para 256x256. La caja anotada puede ser muy estrecha.",
                "hole": "Falso negativo (hole): la perforacion es pequena o esta en el borde, fuera del campo efectivo del detector.",
            }
            hypothesis = hypotheses_fn.get(gt_cls, f"Falso negativo: el detector no encuentra el '{gt_cls}' anotado.")

        elif len(gt_classes) > 0 and len(pred_classes) > 0:
            gt_set = set(gt_classes)
            pred_set = set(pred_classes)
            if not gt_set.intersection(pred_set):
                gt_cls = class_names.get(gt_classes[0], "?")
                pred_cls = class_names.get(pred_classes[0], "?")
                error_type = "clase_equivocada"
                hypotheses_ce = {
                    ("scratch", "rust"): "Clase equivocada (scratch->rust): la textura lineal del rayon se confunde con la variacion de tono del oxido.",
                    ("rust", "scratch"): "Clase equivocada (rust->scratch): la mancha de oxido alargada se confunde con un rayon lineal.",
                    ("crack", "scratch"): "Clase equivocada (crack->scratch): la linea de la grieta se confunde con un rayon superficial.",
                    ("scratch", "crack"): "Clase equivocada (scratch->crack): el rayon profundo se confunde con una fisura estructural.",
                    ("hole", "rust"): "Clase equivocada (hole->rust): la oquedad con halo oxidado se clasifica como rust.",
                    ("rust", "hole"): "Clase equivocada (rust->hole): la picadura profunda de oxido se confunde con hole.",
                }
                hypothesis = hypotheses_ce.get((gt_cls, pred_cls),
                    f"Clase equivocada: {gt_cls} -> {pred_cls}. Solapamiento visual entre clases.")
            else:
                continue
        elif len(gt_classes) == 0 and len(pred_classes) > 0:
            pred_cls = class_names.get(pred_classes[0], "?")
            error_type = "falso_positivo"
            hypothesis = f"Falso positivo: el detector encuentra un '{pred_cls}' donde no hay defecto. Posible ruido de textura o reflejo."
        else:
            continue

        if error_type:
            img = Image.open(img_path).convert("RGB")
            errors.append({
                "tipo": "deteccion",
                "error_type": error_type,
                "path": str(img_path),
                "img_pil": img,
                "hypothesis": hypothesis,
                "pred_classes": pred_classes,
                "pred_confs": pred_confs,
                "gt_classes": gt_classes,
                "class_names": class_names,
            })

    return errors


# ===================== VISUALIZACION =====================

def plot_all_errors(clf_errors, yolo_errors, output_path, clf_model_name):
    total = len(clf_errors) + len(yolo_errors)
    if total == 0:
        print("No se encontraron errores.")
        return

    cols = 5
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    idx = 0
    for err in clf_errors:
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        img = err["img_tensor"] * std + mean
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.set_title(
            f"[Clasif] Real: {err['true_label']}\nPred: {err['pred_label']} ({err['confidence']:.2f})",
            fontsize=9, color="red", fontweight="bold",
        )
        ax.set_xlabel(clf_hypothesis(err), fontsize=6.5, wrap=True, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1

    for err in yolo_errors:
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        img = np.array(err["img_pil"])
        ax.imshow(img, cmap="gray")
        etype_label = {
            "falso_negativo": "Falso negativo",
            "falso_positivo": "Falso positivo",
            "clase_equivocada": "Clase equivocada",
        }.get(err["error_type"], err["error_type"])
        pred_str = ", ".join(
            f"{err['class_names'].get(c, '?')}({err['pred_confs'][i]:.2f})"
            for i, c in enumerate(err["pred_classes"])
        ) if err["pred_classes"] else "(sin deteccion)"
        gt_str = ", ".join(
            err["class_names"].get(c, "?") for c in err["gt_classes"]
        ) if err["gt_classes"] else "(sin GT)"
        ax.set_title(
            f"[Det] {etype_label}\nGT: {gt_str} | Pred: {pred_str}",
            fontsize=8, color="darkred", fontweight="bold",
        )
        ax.set_xlabel(err["hypothesis"], fontsize=6.5, wrap=True, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1

    for i in range(total, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)

    plt.suptitle(
        f"Predicciones incorrectas en test ({total} errores: "
        f"clasificacion={len(clf_errors)}, deteccion={len(yolo_errors)})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en: {output_path}")


def main():
    args = parse_args()
    set_seed(42)
    device = args.device

    all_errors_json = []

    # 1. Errores de clasificacion
    clf_errors = []
    if Path(args.modelo).exists():
        print(f"\n=== Errores de Clasificacion ===")
        print(f"Cargando modelo: {args.modelo}")
        model, checkpoint = load_checkpoint(args.modelo, torch.device(device))
        model_name = checkpoint.get("model_name", "modelo")
        print(f"Modelo: {model_name}")

        test_dir = Path(args.datos) / "test"
        if test_dir.exists():
            transform = make_eval_transform(256)
            dataset = datasets.ImageFolder(str(test_dir), transform=transform)
            print(f"Dataset de test: {len(dataset)} imagenes, clases: {dataset.classes}")
            max_clf = min(args.max_errores, 5)
            clf_errors = find_clf_errors(model, dataset, torch.device(device), max_errors=max_clf)
            print(f"Encontradas {len(clf_errors)} predicciones incorrectas de clasificacion.")
            for err in clf_errors:
                h = clf_hypothesis(err)
                print(f"  {err['true_label']:8s} -> {err['pred_label']:8s} (conf={err['confidence']:.2f}) | {h}")
                all_errors_json.append({
                    "tipo": "clasificacion", "path": err["path"],
                    "true_label": err["true_label"], "pred_label": err["pred_label"],
                    "confidence": err["confidence"], "hypothesis": h,
                })
        else:
            print(f"No existe {test_dir}")
    else:
        print(f"No existe el modelo de clasificacion: {args.modelo}")

    # 2. Errores de deteccion YOLO
    yolo_errors = []
    yolo_model = Path(args.yolo_modelo)
    yolo_test = Path(args.yolo_datos)
    if yolo_model.exists() and yolo_test.exists():
        print(f"\n=== Errores de Deteccion YOLO ===")
        print(f"Cargando detector: {yolo_model}")
        max_yolo = args.max_errores - len(clf_errors)
        yolo_errors = find_yolo_errors(yolo_model, yolo_test, device, max_errors=max_yolo)
        print(f"Encontradas {len(yolo_errors)} predicciones incorrectas de deteccion.")
        for err in yolo_errors:
            print(f"  [{err['error_type']}] {Path(err['path']).name} | {err['hypothesis']}")
            all_errors_json.append({
                "tipo": "deteccion", "error_type": err["error_type"],
                "path": err["path"], "hypothesis": err["hypothesis"],
            })
    else:
        print(f"\nNo se encontro el modelo YOLO o el dataset de test YOLO.")

    total = len(clf_errors) + len(yolo_errors)
    print(f"\nTotal de errores encontrados: {total}")

    if total == 0:
        print("No se encontraron errores. No se genera la figura.")
        return

    output_path = Path(args.salida)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf_name = checkpoint.get("model_name", "modelo") if Path(args.modelo).exists() else ""
    plot_all_errors(clf_errors, yolo_errors, output_path, clf_name)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_errors_json, indent=2, ensure_ascii=False))
    print(f"Metadatos guardados en: {json_path}")


if __name__ == "__main__":
    main()
