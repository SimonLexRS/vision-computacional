"""Descarga el dataset sintetico de defectos en superficies metalicas desde Kaggle.

Fuente:
    Tatheer Abbas — "Synthetic Industrial Metal Surface Defects"
    https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects
    Licencia: CC BY 4.0

El dataset contiene 15.000 imagenes PNG 256x256 en escala de grises,
5 clases balanceadas (normal, scratch, crack, rust, hole), con metadatos
en metadata.csv.

Estructura esperada tras la descarga:
    industrial_defect_dataset/
    ├── train/{normal,scratch,crack,rust,hole}/   # 12.000 imagenes
    ├── val/{normal,scratch,crack,rust,hole}/     #  1.500 imagenes
    ├── test/{normal,scratch,crack,rust,hole}/    #  1.500 imagenes
    ├── metadata.csv
    └── config.json

Requisitos:
    - Tener instalado kagglehub (pip install kagglehub) o kaggle CLI.
    - Autenticacion de Kaggle configurada (kaggle.json con API token).
      Ver: https://github.com/Kaggle/kagglehub#authentication

Uso:
    python src/download_synthetic_dataset.py
    python src/download_synthetic_dataset.py --forzar  # re-descarga aunque exista
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = ROOT / "industrial_defect_dataset"
CLASSES = ["normal", "scratch", "crack", "rust", "hole"]
SPLITS = ["train", "val", "test"]

KAGGLE_DATASET = "tatheerabbas/synthetic-industrial-metal-surface-defects"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga el dataset sintetico de Kaggle (15.000 imagenes)."
    )
    parser.add_argument(
        "--forzar", action="store_true",
        help="Re-descarga aunque el dataset ya exista.",
    )
    return parser.parse_args()


def dataset_ya_descargado() -> bool:
    """Verifica si el dataset ya esta completo en industrial_defect_dataset/."""
    if not DEST_DIR.exists():
        return False
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = DEST_DIR / split / cls
            if not cls_dir.exists() or len(list(cls_dir.glob("*.png"))) == 0:
                return False
    # Verificar metadata.csv
    if not (DEST_DIR.parent / "metadata.csv").exists() and not (DEST_DIR / "metadata.csv").exists():
        return False
    return True


def descargar_con_kagglehub() -> Path:
    """Descarga el dataset usando kagglehub (metodo recomendado)."""
    import kagglehub
    print(f"Descargando dataset desde Kaggle: {KAGGLE_DATASET} ...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset descargado en: {path}")
    return Path(path)


def descargar_con_kaggle_cli() -> Path:
    """Descarga el dataset usando la CLI de Kaggle (fallback)."""
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="kaggle_synthetic_"))
    print(f"Descargando dataset via Kaggle CLI a {tmp} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(tmp), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error de Kaggle CLI: {result.stderr}")
        raise RuntimeError(
            "No se pudo descargar con kaggle CLI. Verifica que tengas "
            "kaggle instalado (pip install kaggle) y configurado "
            "(~/.kaggle/kaggle.json). Ver: "
            "https://github.com/Kaggle/kagglehub#authentication"
        )
    print(f"Dataset descargado y descomprimido en: {tmp}")
    return tmp


def organizar_dataset(src_dir: Path) -> None:
    """Organiza las imagenes descargadas en la estructura train/val/test del proyecto.

    Kaggle publica el dataset con imagenes en subcarpetas por clase bajo
    images/train/ y images/val/. Este script:
    1. Copia las imagenes a industrial_defect_dataset/train/ y .../val/.
    2. Divide val/ en val/ (50%) y test/ (50%) de forma estratificada
       con semilla 42, para crear el split de test que usa el proyecto.
    3. Copia metadata.csv y config.json a la raiz del proyecto.
    """
    import random

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    # Buscar la carpeta de imagenes (puede ser images/ o directamente la raiz)
    images_dir = src_dir / "images"
    if not images_dir.exists():
        # A veces el dataset se descomprime con la estructura directamente
        images_dir = src_dir

    # 1. Copiar train/ (12.000 imagenes)
    kaggle_train = images_dir / "train"
    if kaggle_train.exists():
        for cls in CLASSES:
            src_cls = kaggle_train / cls
            dst_cls = DEST_DIR / "train" / cls
            if src_cls.exists():
                dst_cls.mkdir(parents=True, exist_ok=True)
                for img in src_cls.glob("*.png"):
                    shutil.copy2(img, dst_cls / img.name)
                print(f"  train/{cls}: {len(list(dst_cls.glob('*.png')))} imagenes")

    # 2. Copiar val/ y dividirlo en val/ + test/ estratificado
    kaggle_val = images_dir / "val"
    if kaggle_val.exists():
        for cls in CLASSES:
            src_cls = kaggle_val / cls
            if not src_cls.exists():
                continue

            all_imgs = sorted(src_cls.glob("*.png"))
            random.shuffle(all_imgs)
            mid = len(all_imgs) // 2

            val_imgs = all_imgs[:mid]
            test_imgs = all_imgs[mid:]

            dst_val = DEST_DIR / "val" / cls
            dst_test = DEST_DIR / "test" / cls
            dst_val.mkdir(parents=True, exist_ok=True)
            dst_test.mkdir(parents=True, exist_ok=True)

            for img in val_imgs:
                shutil.copy2(img, dst_val / img.name)
            for img in test_imgs:
                shutil.copy2(img, dst_test / img.name)

            print(f"  val/{cls}: {len(val_imgs)} | test/{cls}: {len(test_imgs)}")

    # 3. Copiar metadata.csv y config.json a la raiz del proyecto
    for fname in ["metadata.csv", "config.json"]:
        src_file = src_dir / fname
        if not src_file.exists():
            src_file = images_dir / fname
        if src_file.exists():
            dst_file = ROOT / fname
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  Copiado {fname} -> {dst_file}")

    # Conteo final
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            n = len(list((DEST_DIR / split / cls).glob("*.png")))
            total += n
    print(f"\nDataset sintetico listo: {total} imagenes en {DEST_DIR}")


def main():
    args = parse_args()

    if dataset_ya_descargado() and not args.forzar:
        print("El dataset sintetico ya esta descargado en industrial_defect_dataset/.")
        print("Usa --forzar para re-descargar.")
        return

    print("=== Descarga del Dataset Sintetico (Kaggle, CC BY 4.0) ===")
    print(f"Fuente: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print()

    # Intentar primero con kagglehub, luego con kaggle CLI
    src_dir = None
    try:
        src_dir = descargar_con_kagglehub()
    except ImportError:
        print("kagglehub no instalado, intentando con kaggle CLI...")
        try:
            src_dir = descargar_con_kaggle_cli()
        except FileNotFoundError:
            print("\nERROR: No se pudo descargar el dataset de Kaggle.")
            print("Necesitas configurar acceso a Kaggle:")
            print("  1. pip install kagglehub  (o pip install kaggle)")
            print("  2. Crear un API token en https://www.kaggle.com/settings")
            print("  3. Guardar ~/.kaggle/kaggle.json (Linux/Mac) o")
            print("     C:\\Users\\<usuario>\\.kaggle\\kaggle.json (Windows)")
            print()
            print("Alternativa: descargar manualmente desde")
            print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
            print("y descomprimir en industrial_defect_dataset/")
            raise SystemExit(1)
    except Exception as e:
        print(f"kagglehub fallo: {e}")
        print("Intentando con kaggle CLI...")
        try:
            src_dir = descargar_con_kaggle_cli()
        except Exception:
            print("\nERROR: No se pudo descargar el dataset de Kaggle.")
            print("Verifica tu configuracion de Kaggle (kaggle.json).")
            raise SystemExit(1)

    organizar_dataset(src_dir)
    print("\nDataset sintetico descargado y organizado correctamente.")


if __name__ == "__main__":
    main()
