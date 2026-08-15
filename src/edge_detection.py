"""Edge detection en tiempo real sobre el stream de la cámara.

Enfoque: **Canny clásico** (escala de grises → desenfoque gaussiano →
Canny). Corre sobrado en una CPU normal (>30 FPS a 640×480), no
requiere descargar modelos y sus umbrales se ajustan en vivo.

Si más adelante se quiere migrar a una red de deep learning (HED,
DexiNed) para bordes "semánticos", solo hay que reemplazar el cuerpo de
`detect_edges(gray)` — p. ej. con `cv2.dnn` y los pesos de HED; el
resto del pipeline (captura, render, teclas, grabación) no cambia.

Uso:
    python src/edge_detection.py                    # webcam 0, overlay
    python src/edge_detection.py --mode map         # mapa blanco/negro
    python src/edge_detection.py --source URL_RTSP  # cámara IP
    python src/edge_detection.py --save salida.mp4  # grabar salida
    python src/edge_detection.py --selftest         # sin cámara (test)

Teclas en vivo:  m = overlay/mapa   +/- = umbral t1   q/ESC = salir
"""

import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_COLOR = (80, 255, 120)  # BGR: verde para los bordes superpuestos


def detect_edges(gray, t1, t2, blur):
    """Devuelve el mapa de bordes (uint8 0/255) de un frame en grises.

    PUNTO DE EXTENSIÓN: para usar HED/DexiNed u otro método, reemplazar
    estas dos líneas manteniendo la firma (grises → mapa 0/255).
    """
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
    return cv2.Canny(smooth, t1, t2)


def render(frame, edges, mode):
    """Construye la imagen a mostrar: SOLO bordes.

    overlay → bordes en color sobre la imagen original.
    map     → mapa de bordes en blanco y negro.
    """
    if mode == "map":
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out = frame.copy()
    out[edges > 0] = OVERLAY_COLOR
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Edge detection en vivo (Canny).")
    p.add_argument("--source", default="0",
                   help="Índice de cámara (0) o URL RTSP/http de cámara IP.")
    p.add_argument("--t1", type=int, default=60, help="Umbral bajo de Canny.")
    p.add_argument("--t2", type=int, default=160, help="Umbral alto de Canny.")
    p.add_argument("--blur", type=int, default=5,
                   help="Kernel del desenfoque gaussiano (impar; 0/1 = sin blur).")
    p.add_argument("--mode", choices=["overlay", "map"], default="overlay",
                   help="overlay = bordes en color sobre la imagen; map = blanco/negro.")
    p.add_argument("--save", default=None, help="Ruta .mp4 para grabar la salida.")
    p.add_argument("--selftest", action="store_true",
                   help="Sin cámara: procesa una imagen de test y guarda outputs/edges_selftest.png.")
    return p.parse_args()


def run_selftest(args):
    """Verificación sin cámara: una imagen del split de test del dataset."""
    sample = next((ROOT / "industrial_defect_dataset" / "test").rglob("*.png"), None)
    if sample is None:
        raise SystemExit("No se encontró ninguna imagen en industrial_defect_dataset/test/")
    frame = cv2.imread(str(sample))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray, args.t1, args.t2, args.blur)
    out_path = ROOT / "outputs" / "edges_selftest.png"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), render(frame, edges, args.mode))
    print(f"Selftest OK: {sample.name} → {out_path} "
          f"({int((edges > 0).sum())} píxeles de borde)")


def main():
    args = parse_args()
    if args.selftest:
        run_selftest(args)
        return

    # --source puede ser índice de cámara o URL de cámara IP.
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {args.source}")

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    t1, mode = args.t1, args.mode
    t_prev = time.perf_counter()
    print("Teclas: m = overlay/mapa | +/- = umbral t1 | q/ESC = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fin del stream o frame inválido.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = detect_edges(gray, t1, args.t2, args.blur)
        view = render(frame, edges, mode)

        now = time.perf_counter()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(view, f"FPS {fps:.0f} | t1={t1} t2={args.t2} | {mode}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY_COLOR, 2)

        cv2.imshow("Edge detection", view)
        if writer:
            writer.write(view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mode = "map" if mode == "overlay" else "overlay"
        elif key in (ord("+"), ord("=")):
            t1 = min(t1 + 10, args.t2 - 10)
        elif key == ord("-"):
            t1 = max(t1 - 10, 10)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
