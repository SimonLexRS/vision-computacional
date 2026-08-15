# Visión computacional: defectos en superficies metálicas

Proyecto de clasificación multi-clase de defectos industriales en metal: se compara una **CNN entrenada desde cero** (baseline) contra tres modelos con **transfer learning** (PyTorch + CUDA).

**Dataset:** [Synthetic Industrial Metal Surface Defects](https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects) (Tatheer Abbas, CC BY 4.0) — 15.000 imágenes PNG 256×256 en escala de grises, 5 clases balanceadas (`normal`, `scratch`, `crack`, `rust`, `hole`).

Documentación del planteamiento: [`DOCUMENTO_PROYECTO.md`](DOCUMENTO_PROYECTO.md).

## División del dataset (train / val / test)

División **80/10/10 estratificada** (misma proporción por clase, sin solape entre particiones). El conjunto de test se generó moviendo 300 imágenes por clase desde `val/` a `test/` con semilla fija (`SEED = 42`), por lo que la división es reproducible:

| Partición | Imágenes | Por clase | Uso |
|-----------|----------|-----------|-----|
| train | 12.000 | 2.400 | Entrenamiento (con data augmentation) |
| validation | 1.500 | 300 | Early stopping y selección del mejor checkpoint |
| test | 1.500 | 300 | Evaluación final, una sola vez |

## Modelos

| Modelo | Rol | Parámetros |
|--------|-----|------------|
| CNN básica (desde cero) | Baseline | 391K |
| ResNet-18 | Transfer learning | 11.2M |
| EfficientNet-B0 | Transfer learning — trade-off accuracy/coste | 4.0M |
| MobileNetV3-Small | Transfer learning — ligero / edge | 1.5M |

Entrenamiento con pesos ImageNet (la CNN se entrena desde cero), early stopping sobre F1 macro de validación, semilla fija (42) para reproducibilidad.

## Demo web (cámara en vivo)

Demo interactiva desplegada en GitHub Pages: **https://simonlexrs.github.io/vision-computacional/**

- Activa la cámara y enfoca una superficie metálica: la clasificación corre **en vivo, 100% en el navegador** (MobileNetV3-Small exportado a ONNX + ONNX Runtime Web). Ninguna imagen sale del dispositivo.
- Sin cámara disponible, también acepta **subir una imagen**.
- El modelo servido (`web/model.onnx`, 6.4 MB) se genera con `src/export_onnx.py` y su paridad con PyTorch queda verificada en la exportación.

## Cómo ejecutar

```bash
pip install -r requirements.txt
```

### Entrenamiento (línea de comandos, sin Jupyter)

```bash
# Reproduce el resultado del informe (los defaults son los hiperparámetros reportados)
python src/train.py --datos industrial_defect_dataset/ --salida outputs/checkpoints/

# Entrenar solo un modelo, p. ej.:
python src/train.py --modelos mobilenet_v3_small --epochs 20
```

### Evaluación sobre test

```bash
python src/evaluate.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
    --datos industrial_defect_dataset/ --split test
```

### Inferencia

```bash
# CPU por defecto; acepta una imagen o una carpeta
python src/predict.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
    --imagen industrial_defect_dataset/test/crack/
```

O con el notebook sin GPU (carga los pesos con `map_location="cpu"`):

```bash
jupyter notebook notebooks/inferencia_cpu.ipynb
```

### Exportar el modelo para la demo web

```bash
python src/export_onnx.py --checkpoint outputs/checkpoints/mobilenet_v3_small_best.pt \
    --salida web/model.onnx
```

> **Pesos entrenados:** los `.pt` no están versionados (carpeta `outputs/` en `.gitignore`).
> Para obtenerlos sin reentrenar: <!-- TODO(equipo): subir mobilenet_v3_small_best.pt a Drive/HF y pegar la URL -->
> `gdown <URL_DEL_MODELO> -O outputs/checkpoints/mobilenet_v3_small_best.pt`.
> Alternativamente, `python src/train.py` los regenera con los mismos resultados (semilla 42 fijada).

Requisitos: Python 3.12+, PyTorch (CUDA opcional — `train.py` y el notebook usan GPU si está disponible; `evaluate.py`, `predict.py` e `inferencia_cpu.ipynb` corren en CPU). El notebook de entrenamiento completo también puede ejecutarse desde Jupyter: `jupyter notebook entrenar_modelos.ipynb` (`DEVICE_MODE = "cuda"` por defecto; puede cambiarse a `"auto"` o `"cpu"`).

Estructura relevante:

```
├── industrial_defect_dataset/   # train/ + val/ + test/
├── entrenar_modelos.ipynb       # pipeline completo: datos, entrenamiento, evaluación e interpretación
├── notebooks/
│   └── inferencia_cpu.ipynb     # inferencia CPU-only con el modelo entrenado
├── src/                         # scripts CLI (argparse, semillas fijadas)
│   ├── train.py                 # entrena y guarda checkpoints
│   ├── evaluate.py              # métricas sobre test/ o val/
│   ├── predict.py               # inferencia sobre imagen o carpeta (CPU)
│   ├── export_onnx.py           # exporta a ONNX para la demo web
│   └── common.py                # arquitecturas, transforms y utilidades compartidas
├── web/                         # demo web estática (GitHub Pages + ONNX Runtime Web)
├── docs/figures/                # gráficos del experimento (curvas, matrices de confusión)
├── DOCUMENTO_PROYECTO.md
└── requirements.txt             # versiones fijadas
```

## Resultados

Métricas del último entrenamiento (`outputs/experiment_config.json` y `outputs/test_summary.csv`, no versionados). La selección del mejor checkpoint se hizo sobre **validación**; el conjunto de **test** se usó una sola vez, al final.

| Modelo | Epochs | Tiempo (min) | Val F1 macro | Test accuracy | Test F1 macro |
|--------|--------|--------------|--------------|---------------|----------------|
| ResNet-18 | 7 | 7.9 | 1.0000 | 1.0000 | 1.0000 |
| EfficientNet-B0 | 6 | 7.8 | 1.0000 | 1.0000 | 1.0000 |
| MobileNetV3-Small | 6 | 5.9 | 1.0000 | 1.0000 | 1.0000 |
| CNN básica | 17 | 18.8 | 0.9993 | 0.9987 | 0.9987 |

**Lectura de resultados:** no hay brecha val → test (buena generalización, sin sobreajuste a la partición de validación). Los tres modelos con transfer learning alcanzan el 100% en test; la CNN baseline comete los únicos 2 errores del experimento (un `crack` y un `rust` clasificados como `normal`). MobileNetV3-Small iguala a ResNet-18 con ~7× menos parámetros y es el candidato recomendado para despliegue en hardware limitado.

> Nota de honestidad: que 3 de 4 modelos lleguen al 100% indica que el dataset sintético es relativamente fácil frente al caso real (defectos grandes, alto contraste, sin ruido de sensor). Estas métricas son una cota superior bajo condiciones ideales, no una estimación de producción. El análisis completo está en las celdas de interpretación del notebook (secciones 8.1, 9, 10, 11, 12 y 14).

### Imágenes procesadas (batch de entrenamiento)

![Muestras de entrenamiento](docs/figures/sample_batch.png)

### Curvas de entrenamiento (train vs val)

| CNN básica | ResNet-18 |
|---|---|
| ![CNN loss](docs/figures/cnn_baseline_loss.png) | ![ResNet-18 loss](docs/figures/resnet18_loss.png) |
| ![CNN accuracy](docs/figures/cnn_baseline_accuracy.png) | ![ResNet-18 accuracy](docs/figures/resnet18_accuracy.png) |

| EfficientNet-B0 | MobileNetV3-Small |
|---|---|
| ![EfficientNet loss](docs/figures/efficientnet_b0_loss.png) | ![MobileNet loss](docs/figures/mobilenet_v3_small_loss.png) |
| ![EfficientNet accuracy](docs/figures/efficientnet_b0_accuracy.png) | ![MobileNet accuracy](docs/figures/mobilenet_v3_small_accuracy.png) |

### Matrices de confusión (sobre test)

| CNN básica | ResNet-18 |
|---|---|
| ![CNN confusion](docs/figures/cnn_baseline_confusion_matrix.png) | ![ResNet-18 confusion](docs/figures/resnet18_confusion_matrix.png) |

| EfficientNet-B0 | MobileNetV3-Small |
|---|---|
| ![EfficientNet confusion](docs/figures/efficientnet_b0_confusion_matrix.png) | ![MobileNet confusion](docs/figures/mobilenet_v3_small_confusion_matrix.png) |

## Licencia

Dataset bajo **CC BY 4.0** (atribución a Tatheer Abbas). Código del proyecto para uso académico del módulo.
