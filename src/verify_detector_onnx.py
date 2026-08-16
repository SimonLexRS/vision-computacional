"""Verifica la paridad del detector ONNX exportado contra PyTorch.

Corre el .pt de ultralytics y el .onnx (onnxruntime, decode + NMS en
numpy — la misma lógica que lleva la web en JS) sobre imágenes del
split test y compara cajas/clases. Debe ser casi idéntico.

Uso:
    python src/verify_detector_onnx.py \
        --pt outputs/yolo/train/weights/best.pt \
        --onnx web/detector.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 256
CONF = 0.4
IOU_NMS = 0.45
N_IMAGES = 10


def preprocess(img: Image.Image) -> np.ndarray:
    """Igual que la web: resize 256², RGB /255, CHW float32."""
    arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"), dtype=np.float32)
    return (arr / 255.0).transpose(2, 0, 1)[None]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """NMS clásico sobre cajas xyxy. Devuelve índices conservados."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (area_i + area_j - inter + 1e-9) <= iou_thr]
    return keep


def detect_onnx(session: ort.InferenceSession, img: Image.Image):
    """Decode YOLOv8 [1, 4+nc, N] + filtro + NMS. Igual que app.js."""
    out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
    preds = out[0].T  # [N, 4+nc]
    scores = preds[:, 4:]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(preds)), cls]
    m = conf >= CONF
    cx, cy, w, h = preds[m, 0], preds[m, 1], preds[m, 2], preds[m, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    conf, cls = conf[m], cls[m]
    keep = nms(boxes, conf, IOU_NMS) if len(boxes) else []
    return boxes[keep], conf[keep], cls[keep]


def iou(a, b) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", default=str(ROOT / "outputs/yolo/train/weights/best.pt"))
    p.add_argument("--onnx", default=str(ROOT / "web/detector.onnx"))
    args = p.parse_args()

    model = YOLO(args.pt)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    test_imgs = sorted((ROOT / "industrial_defect_dataset/test").rglob("*.png"))
    step = max(1, len(test_imgs) // N_IMAGES)
    sample = test_imgs[::step][:N_IMAGES]

    ious, matched, total_pt = [], 0, 0
    for path in sample:
        img = Image.open(path)
        pt = model.predict(img, imgsz=IMG_SIZE, conf=CONF, iou=IOU_NMS, verbose=False)[0]
        pt_boxes = pt.boxes.xyxy.cpu().numpy() if pt.boxes else np.zeros((0, 4))
        pt_cls = pt.boxes.cls.cpu().numpy().astype(int) if pt.boxes else np.zeros(0, int)
        ox_boxes, _, ox_cls = detect_onnx(session, img)

        total_pt += len(pt_boxes)
        used = set()
        for i, pb in enumerate(pt_boxes):
            best_j, best_iou = -1, 0.0
            for j, ob in enumerate(ox_boxes):
                if j in used or ox_cls[j] != pt_cls[i]:
                    continue
                v = iou(pb, ob)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.5:
                matched += 1
                used.add(best_j)
                ious.append(best_iou)
        print(f"{path.name}: pt={len(pt_boxes)} onnx={len(ox_boxes)}")

    print("\n=== Paridad ONNX vs PyTorch ===")
    print(f"Detecciones PyTorch: {total_pt} | emparejadas (IoU≥0.5, misma clase): {matched}")
    if ious:
        print(f"IoU medio de las emparejadas: {float(np.mean(ious)):.3f}")
    ok = total_pt > 0 and matched / total_pt >= 0.9
    print("RESULTADO:", "OK" if ok else "REVISAR — divergencia alta")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
