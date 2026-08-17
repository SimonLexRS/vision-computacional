"""Genera (o restaura) la partición de test a partir de val/.

El dataset de Kaggle trae solo train/ y val/. El proyecto usa una división
80/10/10 estratificada: se mueven 300 imágenes por clase de val/ a test/.

Dos modos:

1) --manifiesto splits/test_files.txt   (RECOMENDADO)
   Reproduce EXACTAMENTE el split usado en el experimento reportado.
   El manifiesto se genera una sola vez con --exportar-manifiesto.

2) --generar
   Crea el split desde cero con `random.Random(semilla).sample(...)` sobre la
   lista ORDENADA de archivos. Es reproducible entre máquinas, pero solo
   coincidirá con el experimento original si ese split se hizo con este mismo
   procedimiento.

Ejemplos:
    python src/preparar_datos.py --datos industrial_defect_dataset --exportar-manifiesto splits/test_files.txt
    python src/preparar_datos.py --datos industrial_defect_dataset --manifiesto splits/test_files.txt
    python src/preparar_datos.py --datos industrial_defect_dataset --generar --por-clase 300 --semilla 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}


def listar_clases(directorio: Path) -> list[str]:
    return sorted(p.name for p in directorio.iterdir() if p.is_dir())


def listar_imagenes(directorio: Path) -> list[Path]:
    return sorted(
        p for p in directorio.iterdir() if p.suffix.lower() in EXTENSIONES
    )


def exportar_manifiesto(datos: Path, salida: Path) -> None:
    test_dir = datos / "test"
    if not test_dir.exists():
        raise SystemExit(
            "No existe test/. Genera el split primero con --generar, "
            "o ejecuta esto en la máquina donde ya tienes la partición."
        )

    lineas = []
    for clase in listar_clases(test_dir):
        for img in listar_imagenes(test_dir / clase):
            lineas.append(f"{clase}/{img.name}")

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Manifiesto con {len(lineas)} archivos escrito en {salida}")


def aplicar_manifiesto(datos: Path, manifiesto: Path, mover: bool = True) -> None:
    val_dir, test_dir = datos / "val", datos / "test"
    entradas = [
        l.strip() for l in manifiesto.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    movidos, ya_estaban, faltantes = 0, 0, []
    for entrada in entradas:
        clase, nombre = entrada.split("/", 1)
        destino = test_dir / clase / nombre
        origen = val_dir / clase / nombre

        destino.parent.mkdir(parents=True, exist_ok=True)

        if destino.exists():
            ya_estaban += 1
            continue
        if not origen.exists():
            faltantes.append(entrada)
            continue

        (shutil.move if mover else shutil.copy2)(str(origen), str(destino))
        movidos += 1

    print(f"Movidos: {movidos} | ya estaban en test/: {ya_estaban} | faltantes: {len(faltantes)}")
    if faltantes:
        print("ADVERTENCIA: no se encontraron estos archivos en val/:")
        for f in faltantes[:10]:
            print("  ", f)
        raise SystemExit("El split no se pudo reproducir por completo.")


def generar_split(datos: Path, por_clase: int, semilla: int, mover: bool = True) -> None:
    val_dir, test_dir = datos / "val", datos / "test"
    rng = random.Random(semilla)

    for clase in listar_clases(val_dir):
        imagenes = listar_imagenes(val_dir / clase)  # lista ORDENADA: determinista
        if len(imagenes) < por_clase:
            raise SystemExit(
                f"La clase {clase} tiene {len(imagenes)} imágenes en val/ y se piden "
                f"{por_clase} para test. ¿Ya generaste el split antes?"
            )

        seleccion = rng.sample(imagenes, por_clase)
        (test_dir / clase).mkdir(parents=True, exist_ok=True)

        for img in seleccion:
            (shutil.move if mover else shutil.copy2)(
                str(img), str(test_dir / clase / img.name)
            )

        print(f"{clase:10s} -> {por_clase} imágenes movidas a test/")


def resumen(datos: Path) -> None:
    print("\nDistribución final:")
    for particion in ("train", "val", "test"):
        d = datos / particion
        if not d.exists():
            continue
        total = 0
        detalle = []
        for clase in listar_clases(d):
            n = len(listar_imagenes(d / clase))
            total += n
            detalle.append(f"{clase}={n}")
        print(f"  {particion:6s} total={total:6d}  ({', '.join(detalle)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, required=True,
                    help="Carpeta raíz del dataset (con train/ y val/).")
    ap.add_argument("--manifiesto", type=Path,
                    help="Reproduce el split leyendo una lista de archivos.")
    ap.add_argument("--exportar-manifiesto", type=Path,
                    help="Escribe el manifiesto a partir del test/ ya existente.")
    ap.add_argument("--generar", action="store_true",
                    help="Genera el split desde val/ con la semilla dada.")
    ap.add_argument("--por-clase", type=int, default=300)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--copiar", action="store_true",
                    help="Copia en vez de mover (deja val/ intacto).")
    args = ap.parse_args()

    if not args.datos.exists():
        raise SystemExit(f"No existe la carpeta {args.datos}")

    if args.exportar_manifiesto:
        exportar_manifiesto(args.datos, args.exportar_manifiesto)
    elif args.manifiesto:
        aplicar_manifiesto(args.datos, args.manifiesto, mover=not args.copiar)
    elif args.generar:
        generar_split(args.datos, args.por_clase, args.semilla, mover=not args.copiar)
    else:
        raise SystemExit(
            "Elige un modo: --exportar-manifiesto, --manifiesto o --generar."
        )

    resumen(args.datos)


if __name__ == "__main__":
    main()
