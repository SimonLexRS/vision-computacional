"""Calibra el umbral del gate de dominio de la demo web.

El gate (MobileNetV3-Small, model.onnx) se entrenó solo con metal
sintético, así que en escenas arbitrarias softmax reparte ~100% entre
las 5 clases igualmente (sobre-confianza fuera de dominio). Para elegir
GATE_THRESHOLD con datos, este script corre el gate (mismo preproceso
que la web: recorte central cuadrado 256², grayscale→3ch, norm.
ImageNet) sobre:

- IN-DOMAIN: muestra del split test del dataset (todas las clases).
- FUERA-DE-DOMINIO (proxy): fotos naturales de sklearn.data
  (china, flower, coffee, astronaut, chelsea, brick/grass/gravel…),
  ruido aleatorio, gradientes suaves y "texto" sintético.

Imprime ambas distribuciones (max-softmax y energía de logits) y
recomienda el umbral que mejor las separa.

Uso:
    python src/calibrate_gate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "web" / "model.onnx"
IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image) -> np.ndarray:
    """Idéntico a la web: crop central cuadrado, gray→3ch, ImageNet, CHW."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    chw = np.stack([gray, gray, gray])  # [3, H, W]
    return ((chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None])[None]


def scores(session, img):
    """Devuelve (max_softmax, energía) para una imagen."""
    logits = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0][0]
    exps = np.exp(logits - logits.max())
    max_softmax = float((exps / exps.sum()).max())
    energy = float(np.log(np.exp(logits).sum()))  # log-sum-exp
    return max_softmax, energy


def ood_samples():
    """Genera/carga imágenes fuera de dominio (proxy de escenas reales)."""
    samples = {}
    # Fotos naturales y texturas de skimage.data (paquete local, sin red).
    try:
        from skimage import data as skdata
        for name in ("astronaut", "chelsea", "coffee", "brick", "grass", "gravel"):
            samples[f"skimage:{name}"] = Image.fromarray(getattr(skdata, name)())
    except ImportError:
        print("AVISO: skimage no disponible; OOD solo con patrones sintéticos.")
    rng = np.random.default_rng(42)
    # Ruido uniforme y gaussiano.
    samples["ruido-uniforme"] = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    g = np.clip(rng.normal(128, 40, (256, 256, 3)), 0, 255).astype(np.uint8)
    samples["ruido-gaussiano"] = Image.fromarray(g)
    # Gradiente suave (escena desenfocada).
    grad = np.linspace(40, 220, 256, dtype=np.uint8)
    grad2d = np.tile(grad, (256, 1))  # [H, W]
    samples["gradiente"] = Image.fromarray(np.stack([grad2d] * 3, axis=2))
    # Patrón tipo texto/líneas sobre fondo claro.
    txt = Image.new("RGB", (256, 256), (235, 235, 235))
    d = ImageDraw.Draw(txt)
    for i in range(12):
        y = 12 + i * 20
        d.rectangle([12, y, 12 + rng.integers(80, 230), y + 8], fill=(40, 40, 40))
    samples["texto"] = txt
    # Escena oscura casi uniforme (cámara tapada).
    samples["oscura"] = Image.fromarray(np.full((256, 256, 3), 25, np.uint8))
    return samples


def det_preprocess(img: Image.Image) -> np.ndarray:
    """Igual que la web: crop central 256², RGB /255, CHW."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def det_count(session, img, conf=0.4):
    """Número de cajas del detector por encima de `conf` (sin NMS)."""
    out = session.run(None, {session.get_inputs()[0].name: det_preprocess(img)})[0]
    scores = out[0, 4:, :]  # [nc, N]
    return int((scores.max(axis=0) >= conf).sum())


def main() -> None:
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    det_session = None
    det_path = ROOT / "web" / "detector.onnx"
    if det_path.exists():
        det_session = ort.InferenceSession(str(det_path), providers=["CPUExecutionProvider"])

    # --- In-domain: muestra del split test ---
    test_dir = ROOT / "industrial_defect_dataset" / "test"
    per_class = 12
    in_scores = []
    for class_dir in sorted(test_dir.iterdir()):
        pngs = sorted(class_dir.glob("*.png"))[:per_class]
        for png in pngs:
            with Image.open(png) as img:
                in_scores.append(scores(session, img))

    # --- Fuera de dominio ---
    ood = {}
    for name, img in ood_samples().items():
        gate = scores(session, img)
        ndet = det_count(det_session, img) if det_session else -1
        ood[name] = (*gate, ndet)

    in_ms = np.array([s[0] for s in in_scores])
    in_en = np.array([s[1] for s in in_scores])
    print(f"IN-DOMAIN (n={len(in_scores)}): max-softmax min={in_ms.min():.4f} "
          f"p5={np.percentile(in_ms, 5):.4f} med={np.median(in_ms):.4f} | "
          f"energía p5={np.percentile(in_en, 5):.3f}")
    print("\nFUERA-DE-DOMINIO:")
    for name, (ms, en, nd) in sorted(ood.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:24s} max-softmax={ms:.4f} energía={en:.3f} cajas_detector={nd}")
    ood_ms = np.array([s[0] for s in ood.values()])
    ood_en = np.array([s[1] for s in ood.values()])

    # Umbral candidato: punto medio entre máximo OOD y p5 in-domain.
    thr_ms = (ood_ms.max() + np.percentile(in_ms, 5)) / 2
    thr_en = (ood_en.max() + np.percentile(in_en, 5)) / 2
    sep_ms = ood_ms.max() < in_ms.min()
    sep_en = ood_en.max() < in_en.min()
    print(f"\nSeparación por max-softmax: {'limpia' if sep_ms else 'SOLAPADA'} "
          f"(OOD máx={ood_ms.max():.4f} vs IN min={in_ms.min():.4f}) → umbral sugerido {thr_ms:.3f}")
    print(f"Separación por energía:    {'limpia' if sep_en else 'SOLAPADA'} "
          f"(OOD máx={ood_en.max():.3f} vs IN min={in_en.min():.3f}) → umbral sugerido {thr_en:.3f}")
    print("\nLlevar el umbral elegido a GATE_THRESHOLD en web/app.js "
          "(con histéresis: entrar ≥ T, salir < T − 0.15).")


if __name__ == "__main__":
    main()
