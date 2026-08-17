"""Descarga y prepara datasets reales de defectos en superficies metálicas.

Fuentes y Créditos Académicos:
1. SteelDefectX Dataset (Zhao et al., 2024):
   - Repositorio Hugging Face: Zhaosxian/SteelDefectX
   - Imágenes reales de defectos industriales en tiras de acero laminado.
   - Categorías: Crazing, Bright/Dark Scratches, Secondary Rust Skin, Pitted Surface,
                 Punching, Inclusion, Patches, Crescent Gap, etc.

2. NEU-DET Steel Surface Defect Database (Song & Yan, 2013, Northeastern University):
   - Mirror en GitHub: siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
     (carpetas IMAGES/ y ANNOTATIONS/ del repo; 1.770 pares imagen/XML de los
     1.800 originales — el dataset oficial tiene archivos faltantes/corruptos).
   - Imágenes reales en escala de grises (200x200), 6 clases balanceadas,
     con cajas reales anotadas en formato Pascal VOC (XML).
   - Categorías: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

3. Synthetic Industrial Metal Surface Defects:
   - Autor: Tatheer Abbas (Kaggle, CC BY 4.0).
   - 15.000 imágenes base en escala de grises (256x256), 5 clases balanceadas.

Nota: los scripts de external_datasets/neu_det/ heredados de un repo de
Faster R-CNN (build_scratch_records.py, predict.py, tfannotation.py,
xml_to_csv.py) NO forman parte de este pipeline; solo se usan las
subcarpetas images/ y annotations/ que descarga este script.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "external_datasets"
STEEL_DIR = EXTERNAL_DIR / "steel_defect_x"
IMAGES_DIR = STEEL_DIR / "images"

NEU_DIR = EXTERNAL_DIR / "neu_det"
NEU_IMAGES_DIR = NEU_DIR / "images"
NEU_ANN_DIR = NEU_DIR / "annotations"
NEU_REPO_ZIP = (
  "https://codeload.github.com/siddhartamukherjee/"
  "NEU-DET-Steel-Surface-Defect-Detection/zip/refs/heads/master"
)

# Mapeo exhaustivo de clases de SteelDefectX a las 4 clases de defectos del proyecto
STEEL_CLASS_MAP = {
  # Crack (Grietas)
  "Crazing": "crack",
  "Crease": "crack",
  "Waist folding": "crack",

  # Scratch (Rayones)
  "Bright scratch": "scratch",
  "Dark scratches": "scratch",
  "Finishing roll printing": "scratch",

  # Rust (Óxido y corrosión)
  "Secondary rust skin": "rust",
  "White rust": "rust",
  "Pitted surface": "rust",
  "Rolled in scale": "rust",
  "Red iron sheet": "rust",
  "Oxide scale of plate system": "rust",
  "Oxide scale of temperature system": "rust",

  # Hole (Perforaciones e inclusiones)
  "Punching": "hole",
  "Inclusion": "hole",
  "Slag inclusion": "hole",
  "Patches": "hole",
  "Crescent gap": "hole",
  "Rolled pit": "hole",
}

# Mapeo de las 6 clases de NEU-DET (nombres tal como vienen en los XML VOC)
# a las 4 clases de defectos del proyecto (consistente con STEEL_CLASS_MAP).
NEU_CLASS_MAP = {
  "crazing": "crack",
  "scratches": "scratch",
  "rolled-in_scale": "rust",
  "pitted_surface": "rust",
  "patches": "hole",
  "inclusion": "hole",
}


def download_single_image(img_name: str) -> bool:
  try:
    hf_path = f"train/{img_name}"
    local_target = IMAGES_DIR / img_name
    if local_target.exists() and local_target.stat().st_size > 0:
      return True

    downloaded = hf_hub_download(
      repo_id="Zhaosxian/SteelDefectX",
      filename=hf_path,
      repo_type="dataset",
    )
    shutil.copy2(downloaded, local_target)
    return True
  except Exception as e:
    print(f"Aviso descargando {img_name}: {e}")
    return False


def fetch_steel_defect_x(max_per_class: int = 150) -> list[dict]:
  """Descarga muestras balanceadas de SteelDefectX para cada categoría."""
  STEEL_DIR.mkdir(parents=True, exist_ok=True)
  IMAGES_DIR.mkdir(parents=True, exist_ok=True)

  print("Descargando metadatos de SteelDefectX...")
  meta_path = hf_hub_download(
    repo_id="Zhaosxian/SteelDefectX",
    filename="train-text.json",
    repo_type="dataset",
  )

  with open(meta_path, "r", encoding="utf-8") as f:
    records = json.load(f)

  print(f"Total registros en metadatos: {len(records)}")

  # Filtrar y balancear por clase mapeada
  by_target_class: dict[str, list[dict]] = {"crack": [], "scratch": [], "rust": [], "hole": []}
  for r in records:
    c = r.get("class_name")
    if c in STEEL_CLASS_MAP:
      mapped = STEEL_CLASS_MAP[c]
      by_target_class[mapped].append(r)

  selected_records = []
  for target_cls, recs in by_target_class.items():
    chosen = recs[:max_per_class]
    selected_records.extend(chosen)
    print(f"  Clase '{target_cls}': {len(chosen)} imágenes seleccionadas (de {len(recs)} disponibles)")

  print(f"\nDescargando {len(selected_records)} imágenes reales en paralelo...")
  img_names = [r["image_name"] for r in selected_records]

  with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(download_single_image, img_names))

  successful = sum(1 for r in results if r)
  print(f"Descarga finalizada: {successful}/{len(selected_records)} imágenes guardadas en {IMAGES_DIR}")

  # Guardar catálogo de registros seleccionados
  out_meta = STEEL_DIR / "selected_metadata.json"
  with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(selected_records, f, indent=2, ensure_ascii=False)

  return selected_records


def fetch_neu_det() -> None:
  """Descarga NEU-DET (imágenes + anotaciones VOC XML) desde el mirror de GitHub.

  Baja el zip del repo y extrae únicamente IMAGES/ y ANNOTATIONS/ a
  external_datasets/neu_det/{images,annotations}/. Es idempotente: si ya
  existen los archivos, no vuelve a descargar.
  """
  NEU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
  NEU_ANN_DIR.mkdir(parents=True, exist_ok=True)

  n_img = len(list(NEU_IMAGES_DIR.glob("*.jpg")))
  n_ann = len(list(NEU_ANN_DIR.glob("*.xml")))
  if n_img > 0 and n_img == n_ann:
    print(f"NEU-DET ya descargado ({n_img} pares imagen/XML); se omite la descarga.")
    return

  zip_path = NEU_DIR / "_neu_det_repo.zip"
  print("Descargando NEU-DET (mirror GitHub, ~30 MB)...")
  urllib.request.urlretrieve(NEU_REPO_ZIP, zip_path)

  prefix_img = "NEU-DET-Steel-Surface-Defect-Detection-master/IMAGES/"
  prefix_ann = "NEU-DET-Steel-Surface-Defect-Detection-master/ANNOTATIONS/"
  n_img = n_ann = 0
  with zipfile.ZipFile(zip_path) as zf:
    for member in zf.namelist():
      if member.startswith(prefix_img) and member.endswith(".jpg"):
        target = NEU_IMAGES_DIR / Path(member).name
      elif member.startswith(prefix_ann) and member.endswith(".xml"):
        target = NEU_ANN_DIR / Path(member).name
      else:
        continue
      with zf.open(member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
      if target.suffix == ".jpg":
        n_img += 1
      else:
        n_ann += 1

  zip_path.unlink()
  print(f"NEU-DET listo: {n_img} imágenes y {n_ann} anotaciones XML en {NEU_DIR}")

  # Conteo por clase (los nombres de archivo son <clase>_<n>.jpg).
  counts: dict[str, int] = {}
  for jpg in sorted(NEU_IMAGES_DIR.glob("*.jpg")):
    cls = jpg.stem.rsplit("_", 1)[0]
    counts[cls] = counts.get(cls, 0) + 1
  for cls, n in sorted(counts.items()):
    mapped = NEU_CLASS_MAP.get(cls, "(sin mapeo)")
    print(f"  Clase '{cls}': {n} imágenes -> '{mapped}'")


def fetch_synthetic_dataset():
  """Descarga el dataset sintetico de Kaggle (15.000 imagenes, CC BY 4.0).

  Delega en src/download_synthetic_dataset.py. Si el dataset ya existe,
  no hace nada (idempotente).
  """
  from download_synthetic_dataset import dataset_ya_descargado, organizar_dataset
  from download_synthetic_dataset import descargar_con_kagglehub, descargar_con_kaggle_cli

  if dataset_ya_descargado():
    print("Dataset sintetico ya descargado en industrial_defect_dataset/; se omite.")
    return

  print("\n--- Descargando dataset sintetico (Kaggle, CC BY 4.0) ---")
  src_dir = None
  try:
    src_dir = descargar_con_kagglehub()
  except ImportError:
    try:
      src_dir = descargar_con_kaggle_cli()
    except FileNotFoundError:
      print("\n  AVISO: No se pudo descargar el dataset sintetico de Kaggle.")
      print("  Necesitas configurar acceso a Kaggle (kaggle.json).")
      print("  Ver instrucciones en README.md o ejecuta:")
      print("    python src/download_synthetic_dataset.py")
      print("  Alternativa: descarga manual desde")
      print("    https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects")
      return
  except Exception as e:
    print(f"\n  AVISO: Error descargando dataset sintetico: {e}")
    print("  Puedes descargarlo manualmente desde Kaggle (ver README).")
    return

  if src_dir is not None:
    organizar_dataset(src_dir)
    print("Dataset sintetico listo.")


def main():
  import argparse

  parser = argparse.ArgumentParser(
    description="Descarga TODOS los datasets del proyecto (sintetico + reales)."
  )
  parser.add_argument(
    "--solo-reales", action="store_true",
    help="Descarga solo los datasets reales (SteelDefectX + NEU-DET), no el sintetico.",
  )
  parser.add_argument(
    "--solo-sintetico", action="store_true",
    help="Descarga solo el dataset sintetico de Kaggle.",
  )
  args = parser.parse_args()

  print("=" * 60)
  print("  DESCARGA DE DATASETS DEL PROYECTO")
  print("  Vision Computacional — Defectos en superficies metalicas")
  print("=" * 60)

  if not args.solo_reales:
    fetch_synthetic_dataset()

  if not args.solo_sintetico:
    print("\n=== Descarga de Datasets Reales (SteelDefectX + NEU-DET) ===")
    records = fetch_steel_defect_x(max_per_class=120)
    print(f"\nSteelDefectX listo: {len(records)} imagenes de defectos reales integradas.")
    print()
    fetch_neu_det()

  print("\n" + "=" * 60)
  print("  TODOS LOS DATASETS ESTAN LISTOS.")
  print("  Siguiente paso: python src/prepare_combined_dataset.py")
  print("=" * 60)


if __name__ == "__main__":
  main()
