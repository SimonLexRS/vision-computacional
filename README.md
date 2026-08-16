<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/brand/ucb-logo-dark.png">
    <img src="docs/figures/brand/ucb-logo-vertical.png" alt="Universidad Católica Boliviana" width="230">
  </picture>
</p>

# Visión computacional: detección y clasificación de defectos en superficies metálicas

**Proyecto Final — Módulo Visión Computacional** · Maestría en Ciencia de Datos e Inteligencia Artificial Aplicada · **Universidad Católica Boliviana**

> **Problema:** dada una imagen (o el video en vivo de una cámara) de una superficie metálica, decidir automáticamente si la pieza está en condiciones normales o presenta defectos —grietas, perforaciones, óxido o rayones— y **localizar cada defecto** con su caja y tipo, para apoyar la decisión de aceptar o reprobar la pieza en una línea de inspección de calidad.

## Equipo

| Integrante | Rol |
|:---|:---|
| **Simon Alex Rodriguez Saavedra** | Líder / Integrador |
| **Jennifer Suarez Gutierrez** | Entrenamiento de la red neuronal CNN baseline |
| **Daniel Ribera Añez** | Contribuidor |
| **Daniela Alejandra Caro Arrazola** | Contribuidora |

**Informe técnico (artículo científico, Entregable A):** [`informe/main.pdf`](informe/main.pdf) — fuente LaTeX en [`informe/main.tex`](informe/main.tex).

Proyecto integral de detección de objetos y clasificación multi-clase de defectos industriales en metal: combina un **Gate clasificador de dominio** (MobileNetV3-Small) y un **detector YOLOv8n en tiempo real** entrenados con aceleración por hardware (**GPU NVIDIA GeForce RTX 5060 Ti**) sobre datasets combinados sintéticos y reales.

---

## Datasets, Fuentes y Créditos Académicos

Para garantizar máxima precisión y capacidad de generalización frente a texturas reales, grano de sensor, iluminación variable y variaciones microscópicas de defectos, este proyecto integra y da crédito a las siguientes bases de datos etiquetadas:

### 1. Synthetic Industrial Metal Surface Defects
- **Autor:** Tatheer Abbas (Kaggle).
- **Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Enlace:** [Kaggle Dataset](https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects).
- **Descripción:** 15.000 imágenes PNG $256 \times 256$ en escala de grises, 5 clases perfectamente balanceadas (`normal`, `scratch`, `crack`, `rust`, `hole`), con metadatos asociados (`metadata.csv`) sobre centros, áreas de defecto y parámetros de textura/iluminación.

### 2. SteelDefectX (A Real-World Benchmark Dataset for Steel Surface Defect Detection)
- **Autores:** Zhao et al. (2024).
- **Repositorio:** [Hugging Face Zhaosxian/SteelDefectX](https://huggingface.co/datasets/Zhaosxian/SteelDefectX).
- **Descripción:** 5.454 imágenes reales capturadas en líneas de producción industrial de acero laminado en caliente y frío. Aporta muestras con reflectancia metálica real, ruido de rodillo y defectos auténticos categorizados y mapeados al proyecto:
  - `Crazing`, `Crease`, `Waist folding` $\rightarrow$ **`crack`** (grietas y fisuras).
  - `Bright scratch`, `Dark scratches`, `Finishing roll printing` $\rightarrow$ **`scratch`** (rayones superficiales).
  - `Secondary rust skin`, `White rust`, `Pitted surface`, `Rolled in scale`, `Red iron sheet`, `Oxide scale` $\rightarrow$ **`rust`** (óxido, corrosión y picaduras).
  - `Punching`, `Inclusion`, `Slag inclusion`, `Patches`, `Crescent gap`, `Rolled pit` $\rightarrow$ **`hole`** (perforaciones, oquedades e inclusiones).

### 3. NEU Surface Defect Database (NEU-DET / NEU-CLS)
- **Autores:** Kechen Song y Yunhui Yan (Northeastern University, Shenyang, China).
- **Cita Académica:**
  > Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection*. **Applied Surface Science**, 285, 858-864. DOI: [10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002).
- **Descripción:** Benchmark académico de referencia para detección de defectos en bandas de acero laminadas en caliente (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).
- **Integración:** tercera fuente del dataset de detección YOLO — 1.770 imágenes reales $200 \times 200$ con **cajas reales anotadas** en formato Pascal VOC (XML), obtenidas del [mirror en GitHub](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection) (el dataset oficial tiene archivos faltantes/corruptos; de los 1.800 originales se recuperan 1.770 pares imagen/anotación). Mapeo de clases: `crazing` $\rightarrow$ **`crack`**, `scratches` $\rightarrow$ **`scratch`**, `rolled-in_scale` / `pitted_surface` $\rightarrow$ **`rust`**, `patches` / `inclusion` $\rightarrow$ **`hole`**.

---

## Entrenamiento en GPU (NVIDIA GeForce RTX 5060 Ti)

Todos los modelos se entrenan con aceleración de hardware dedicada:
- **GPU:** NVIDIA GeForce RTX 5060 Ti (CUDA 13.0 / PyTorch 2.13.0).
- **Precisión:** Automatic Mixed Precision (AMP `float16`) para máxima velocidad y eficiencia de memoria.
- **Optimizador:** SGD con momentum ($0.937$) y weight decay ($0.0005$).
- **Data Augmentation Avanzada:**
  - Variación de iluminación y tono: HSV jitter (`hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.4`).
  - Transformaciones geométricas: Rotaciones (`degrees=10.0`), traslación (`translate=0.1`), escalado (`scale=0.2`), flips horizontales y verticales.
  - Multi-escala y mezcla: Mosaico multi-imagen (`mosaic=0.5`) y `mixup=0.1` para detección robusta de defectos pequeños.

---

## 📊 División del Dataset Combinado (Train / Val / Test)

División **80/10/10 estratificada** con semilla fija (`SEED = 42`) para garantizar reproducibilidad total:

| Partición | Imágenes Sintéticas | Reales (SteelDefectX) | Reales (NEU-DET) | Total Imágenes | Uso |
|:---|:---:|:---:|:---:|:---:|:---|
| **train** | 12.000 | 384 | 1.416 | **13.800** | Entrenamiento con Data Augmentation en GPU |
| **val** | 1.500 | 48 | 177 | **1.725** | Early stopping y selección de mejor checkpoint |
| **test** | 1.500 | 48 | 177 | **1.725** | Evaluación final no vista |
| **Total** | **15.000** | **480** | **1.770** | **17.250** | |

> **Nota de versionado:** el `web/detector.onnx` desplegado y las métricas de detección reportadas abajo corresponden al entrenamiento sobre este dataset de **3 fuentes** (17.250 imágenes; el split de test incluye 177 imágenes reales de NEU-DET con cajas anotadas). Una corrida histórica de 100 épocas sobre la versión anterior de 2 fuentes (`outputs/yolo/train-3`, 15.480 imágenes) alcanzó `mAP50 ≈ 0.80` en su propio split de test.

---

## Arquitectura de Modelos

| Modelo | Tipo | Rol | Parámetros / Tamaño |
|:---|:---|:---|:---:|
| **MobileNetV3-Small** | Clasificador de Dominio (Gate) | Discrimina superficie metálica vs entorno fuera de dominio | 1.5M (~6.4 MB ONNX) |
| **YOLOv8n** | Detector de Objetos (Bounding Boxes) | Localiza y clasifica defectos individuales (`crack`, `hole`, `rust`, `scratch`) | 3.2M (~12.2 MB ONNX) |
| **ResNet-18** | Benchmark Clasificación | Comparativa de Transfer Learning | 11.2M |
| **EfficientNet-B0** | Benchmark Clasificación | Comparativa de Transfer Learning | 4.0M |
| **CNN Básica** | Baseline | Red convolucional entrenada desde cero | 391K |

---

## Demo Web en Tiempo Real (Cámara e Inspección)

Demo interactiva desplegada en GitHub Pages: **https://simonlexrs.github.io/vision-computacional/**

- **Inferencia en el Cliente:** Corre 100% en el navegador utilizando **ONNX Runtime Web** con WebAssembly multi-hilo a **20–30+ FPS**. Ninguna imagen o video sale del dispositivo.
- **Pipeline en Dos Etapas:**
  1. **Gate de Dominio (`model.onnx`):** Filtra fondos no metálicos y escenas fuera de dominio (OOD).
  2. **Detector YOLOv8n (`detector.onnx`):** Si la superficie es válida, detecta y acota cada defecto individualmente. Si no hay defectos, reporta claramente **`Normal — sin defectos` (Verde)**.
- **Lupa / Zoom de Inspección (1×, 1.5×, 2×, 3×):** Proyecta sub-regiones ROI en alta densidad al tensor YOLO ($256 \times 256$) para detectar defectos milimétricos y micro-fisuras.
- **Modo Maximizado Inmersivo (HUD):** Permite ver la cámara a pantalla completa (`100vw × 100dvh`) con controles flotantes de zoom, bordes Sobel por clase y métricas de FPS en tiempo real.
- **Filtros Anti-Ruido Calibrados:** Protección contra falsos positivos nocturnos en sensor ISO y calibración por clase (umbral general al 35%, óxido al 50%).

### 📸 Capturas del testing (modelo desplegado)

Pruebas de extremo a extremo sobre la app desplegada, subiendo imágenes del split de **test** (no visto en entrenamiento) y una escena fuera de dominio:

| Detección de grieta | Superficie normal |
|:---:|:---:|
| ![Testing: grieta detectada](docs/figures/demo/demo_crack.png) | ![Testing: superficie normal](docs/figures/demo/demo_normal.png) |
| **Gate OOD (escena no metálica)** | **Acerca del modelo / créditos** |
| ![Testing: fuera de dominio](docs/figures/demo/demo_ood.png) | ![Acerca del modelo](docs/figures/demo/demo_about.png) |

El chip superior muestra clase y confianza; el panel lateral, el conteo por clase y las probabilidades del gate. Más capturas (rayón, óxido, perforación) en [`docs/figures/demo/`](docs/figures/demo/).

---

## Pipeline de Ejecución y Entrenamiento

### 1. Ingesta y Descarga de Datasets
```bash
# Descarga datasets reales de defectos (SteelDefectX + NEU-DET)
python src/download_datasets.py

# Construye el dataset combinado yolo_dataset/ con cajas refinadas
python src/prepare_combined_dataset.py
```

### 2. Entrenamiento en GPU (RTX 5060 Ti)

El entrenamiento de **todos** los modelos —clasificadores (CNN, ResNet-18, EfficientNet-B0, MobileNetV3-Small) **y el detector YOLOv8n**— está integrado en el notebook [`entrenar_modelos.ipynb`](entrenar_modelos.ipynb) (Sección 15 para YOLO). Para reproducirlo sin abrir Jupyter, los scripts equivalentes por línea de comandos son:

```bash
# Entrenar YOLOv8n en GPU CUDA (device=0)
python src/train_yolo.py

# Reentrenar clasificador MobileNetV3 en GPU
python src/train.py --datos industrial_defect_dataset/ --salida outputs/checkpoints/
```

> Ambas vías (notebook y script) usan exactamente los mismos hiperparámetros: `imgsz=256`, SGD momentum `0.937`, weight decay `0.0005`, AMP, seed `42` y la misma data augmentation (HSV jitter, rotación, traslación, escalado, flips, mosaico y mixup). El entrenamiento de YOLO usa **early stopping** (`patience`): si el mAP50 de validación deja de mejorar durante varias épocas, el entrenamiento se detiene antes de llegar al máximo de épocas.

### 3. Exportación y Verificación ONNX
```bash
# Exportar detector a ONNX (exporta, copia a web/ y verifica paridad)
python src/export_detector_onnx.py --checkpoint outputs/yolo/train/weights/best.pt --salida web/detector.onnx

# Exportar clasificador Gate a ONNX
python src/export_onnx.py --checkpoint outputs/checkpoints/mobilenet_v3_small_best.pt --salida web/model.onnx

# Verificar paridad numérica ONNX vs PyTorch (por separado)
python src/verify_detector_onnx.py --pt outputs/yolo/train/weights/best.pt --onnx web/detector.onnx
```

---

## Resultados y Métricas de Evaluación

Evaluación sobre el split de **Test** final ($1.725$ imágenes, dataset de 3 fuentes — ver nota de versionado arriba):

### Detector YOLOv8n (Dataset Combinado)

Resultado del entrenamiento reproducible (notebook Sección 15 / `python src/train_yolo.py`, 60 épocas, `imgsz=256`, SGD, seed `42`) evaluado sobre el split de test:

- **mAP50:** `0.739`
- **mAP50-95:** `0.488`
- **Precisión:** `0.754`
- **Recall:** `0.663`
- **AP50 por Clase:**
  - `crack`: **0.791**
  - `hole`: **0.846**
  - `rust`: **0.677**
  - `scratch`: **0.643**

> **Este es el checkpoint desplegado en la demo web** (`web/detector.onnx`, paridad ONNX↔PyTorch verificada con IoU medio 1.000). Nota de comparación: la corrida histórica de 100 épocas sobre el dataset de 2 fuentes (`outputs/yolo/train-3`) reportó `mAP50 ≈ 0.80`, pero sobre un split de test distinto (sin las 177 imágenes reales de NEU-DET, que elevan la dificultad del benchmark al usar cajas anotadas reales). El entrenamiento usa **early stopping** (`patience=15`); en la corrida reportada la métrica siguió mejorando y se completaron las 60 épocas (1.15 h en la RTX 5060 Ti).

### Modelos de Clasificación (Transfer Learning)
- **MobileNetV3-Small:** Test F1-Macro = `1.0000` (100% de exactitud, 1.5M parámetros).
- **ResNet-18:** Test F1-Macro = `1.0000` (11.2M parámetros).
- **EfficientNet-B0:** Test F1-Macro = `1.0000` (4.0M parámetros).
- **CNN Básica (Desde Cero):** Test F1-Macro = `0.9987`.

---

## ⚠️ Limitaciones conocidas

- **Cajas de entrenamiento heurísticas:** en la fuente sintética y en SteelDefectX las bounding boxes se derivan de metadata + segmentación OpenCV (no de anotación humana); solo NEU-DET aporta cajas reales. Las métricas de detección son por tanto **exploratorias**, no un benchmark estándar.
- **Dominio mayoritariamente sintético:** la clasificación perfecta (F1 = 1.0) es una **cota superior bajo condiciones ideales**; existe riesgo de *sim-to-real gap* en piezas reales.
- **Cambio de benchmark:** al integrar NEU-DET el split de test se volvió más exigente (177 imágenes reales con cajas anotadas), por lo que el mAP50 del detector (0.739) no es comparable directamente con corridas anteriores sobre el test de 2 fuentes.
- **Reproducibilidad:** semillas fijadas (`SEED=42`) en todos los scripts; el entrenamiento en GPU puede variar mínimamente por operaciones no determinísticas de CUDA/AMP.
- Detalle completo en el informe: [`informe/main.pdf`](informe/main.pdf).

---

## 📄 Licencia y Atribución

- **Datasets:**
  - [Synthetic Industrial Metal Surface Defects](https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects) bajo licencia **CC BY 4.0** (Tatheer Abbas).
  - [SteelDefectX](https://huggingface.co/datasets/Zhaosxian/SteelDefectX) (Zhao et al., 2024).
  - [NEU-DET Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) (Song & Yan, Northeastern University).
- **Código y Modelos:** Proyecto desarrollado por el equipo del Proyecto Final (ver [👥 Equipo](#-equipo)) para el módulo de **Visión Computacional**, Maestría en Ciencia de Datos e Inteligencia Artificial Aplicada, **Universidad Católica Boliviana**.
