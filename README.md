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
- **Limitaciones conocidas:** datos generados por procedimiento — defectos de alto contraste, iluminación controlada y sin ruido de sensor. No sustituye validación sobre piezas reales.

### 2. SteelDefectX (A Real-World Benchmark Dataset for Steel Surface Defect Detection)
- **Autores:** Zhao et al. (2024).
- **Repositorio:** [Hugging Face Zhaosxian/SteelDefectX](https://huggingface.co/datasets/Zhaosxian/SteelDefectX).
- **Descripción:** 5.454 imágenes reales capturadas en líneas de producción industrial de acero laminado en caliente y frío. Aporta muestras con reflectancia metálica real, ruido de rodillo y defectos auténticos categorizados y mapeados al proyecto:
  - `Crazing`, `Crease`, `Waist folding` $\rightarrow$ **`crack`** (grietas y fisuras).
  - `Bright scratch`, `Dark scratches`, `Finishing roll printing` $\rightarrow$ **`scratch`** (rayones superficiales).
  - `Secondary rust skin`, `White rust`, `Pitted surface`, `Rolled in scale`, `Red iron sheet`, `Oxide scale` $\rightarrow$ **`rust`** (óxido, corrosión y picaduras).
  - `Punching`, `Inclusion`, `Slag inclusion`, `Patches`, `Crescent gap`, `Rolled pit` $\rightarrow$ **`hole`** (perforaciones, oquedades e inclusiones).
- **Limitación:** no trae cajas anotadas; las bounding boxes se derivan por segmentación OpenCV.

### 3. NEU Surface Defect Database (NEU-DET / NEU-CLS)
- **Autores:** Kechen Song y Yunhui Yan (Northeastern University, Shenyang, China).
- **Cita Académica:**
  > Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection*. **Applied Surface Science**, 285, 858-864. DOI: [10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002).
- **Descripción:** Benchmark académico de referencia para detección de defectos en bandas de acero laminadas en caliente (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).
- **Integración:** tercera fuente del dataset de detección YOLO — 1.770 imágenes reales $200 \times 200$ con **cajas reales anotadas** en formato Pascal VOC (XML), obtenidas del [mirror en GitHub](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection) (el dataset oficial tiene archivos faltantes/corruptos; de los 1.800 originales se recuperan 1.770 pares imagen/anotación). Mapeo de clases: `crazing` $\rightarrow$ **`crack`**, `scratches` $\rightarrow$ **`scratch`**, `rolled-in_scale` / `pitted_surface` $\rightarrow$ **`rust`**, `patches` / `inclusion` $\rightarrow$ **`hole`**.
- **Nota sobre el mapeo:** la reducción de las 6 clases de NEU a las 4 del proyecto es una decisión de este trabajo, no del dataset original. `patches → hole` es la equivalencia más discutible y se documenta como tal.

---

## División de los Datasets

El proyecto usa **dos conjuntos distintos**, uno por tarea. Es importante no confundirlos: las métricas de clasificación provienen únicamente del dataset sintético, y las de detección del dataset combinado.

### A. Clasificación — solo dataset sintético (15.000 imágenes)

División **80/10/10 estratificada** (misma proporción por clase, sin solape). El dataset original solo define `train`/`val`; el conjunto de test se construyó moviendo 300 imágenes por clase desde `val/` con semilla fija (`SEED = 42`).

| Partición | Imágenes | Por clase | Uso |
|:---|:---:|:---:|:---|
| **train** | 12.000 | 2.400 | Entrenamiento con data augmentation |
| **val** | 1.500 | 300 | Early stopping y selección del mejor checkpoint |
| **test** | 1.500 | 300 | Evaluación final, una sola vez |

La lista exacta de archivos que componen `test/` está versionada en [`splits/test_files.txt`](splits/test_files.txt), de modo que la partición es reproducible archivo por archivo:

```bash
python src/preparar_datos.py --datos industrial_defect_dataset --manifiesto splits/test_files.txt
```

> La columna `split` de `metadata.csv` corresponde al split original de Kaggle, por lo que las imágenes movidas a `test/` siguen marcadas ahí como `val`. La fuente de verdad de la partición es `splits/test_files.txt`.

### B. Detección — dataset combinado de 3 fuentes (17.250 imágenes)

| Partición | Sintéticas | SteelDefectX | NEU-DET | Total | Uso |
|:---|:---:|:---:|:---:|:---:|:---|
| **train** | 12.000 | 384 | 1.416 | **13.800** | Entrenamiento con augmentation en GPU |
| **val** | 1.500 | 48 | 177 | **1.725** | Early stopping y selección de checkpoint |
| **test** | 1.500 | 48 | 177 | **1.725** | Evaluación final no vista |
| **Total** | **15.000** | **480** | **1.770** | **17.250** | |

> **Sobre la estratificación:** para la fuente sintética el split es estratificado por clase (hereda las carpetas de A). Para SteelDefectX y NEU-DET es un split **aleatorio con semilla fija 42** (`np.random.shuffle` + corte por índice en `prepare_combined_dataset.py`), no estratificado: el balance por clase resulta aproximado, no exacto.

> **Nota de versionado:** el `web/detector.onnx` desplegado y las métricas de detección reportadas abajo corresponden al entrenamiento sobre este dataset de **3 fuentes**. Una corrida histórica de 100 épocas sobre la versión anterior de 2 fuentes (`outputs/yolo/train-3`, 15.480 imágenes) alcanzó `mAP50 ≈ 0.80` en su propio split de test.

---

## Arquitectura de Modelos

| Modelo | Tipo | Rol | Parámetros / Tamaño |
|:---|:---|:---|:---:|
| **MobileNetV3-Small** | Clasificador de Dominio (Gate) | Discrimina superficie metálica vs entorno fuera de dominio | 1.5M (~6.4 MB ONNX) |
| **YOLOv8n** | Detector de Objetos (Bounding Boxes) | Localiza y clasifica defectos individuales (`crack`, `hole`, `rust`, `scratch`) | 3.2M (~12.2 MB ONNX) |
| **ResNet-18** | Benchmark Clasificación | Comparativa de Transfer Learning | 11.2M |
| **EfficientNet-B0** | Benchmark Clasificación | Comparativa de Transfer Learning | 4.0M |
| **CNN Básica** | Baseline | Red convolucional entrenada desde cero | 391K |

### Entrenamiento en GPU (NVIDIA GeForce RTX 5060 Ti)

- **GPU:** NVIDIA GeForce RTX 5060 Ti (CUDA 13.0 / PyTorch 2.13.0).
- **Precisión:** Automatic Mixed Precision (AMP `float16`) en el detector.
- **Optimizadores:** AdamW (`lr=3e-4`, `weight_decay=1e-4`) en los clasificadores; SGD con momentum ($0.937$) y weight decay ($0.0005$) en YOLOv8n.
- **Data augmentation del detector:** HSV jitter (`hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.4`), rotaciones (`degrees=10.0`), traslación (`translate=0.1`), escalado (`scale=0.2`), flips, mosaico (`mosaic=0.5`) y `mixup=0.1`.
- **Data augmentation de los clasificadores:** flips horizontal y vertical, rotación ±15°, jitter de brillo y contraste ±0.15.

---

## Demo Web en Tiempo Real (Cámara e Inspección)

Demo interactiva desplegada en GitHub Pages: **https://simonlexrs.github.io/vision-computacional/**

- **Inferencia en el cliente:** corre 100% en el navegador con **ONNX Runtime Web** y WebAssembly multi-hilo a **20–30+ FPS**. Ninguna imagen o video sale del dispositivo.
- **Pipeline en dos etapas:**
  1. **Gate de dominio (`model.onnx`):** filtra fondos no metálicos y escenas fuera de dominio (OOD).
  2. **Detector YOLOv8n (`detector.onnx`):** si la superficie es válida, detecta y acota cada defecto individualmente. Sin defectos, reporta **`Normal — sin defectos` (verde)**.
- **Lupa / zoom de inspección (1×, 1.5×, 2×, 3×):** proyecta sub-regiones ROI en alta densidad al tensor YOLO ($256 \times 256$) para detectar defectos milimétricos y micro-fisuras.
- **Modo maximizado inmersivo (HUD):** cámara a pantalla completa (`100vw × 100dvh`) con controles flotantes de zoom, bordes Sobel por clase y métricas de FPS en tiempo real.
- **Filtros anti-ruido calibrados:** protección contra falsos positivos nocturnos en sensor ISO y calibración por clase (umbral general al 35%, óxido al 50%).

El umbral del gate se elige con datos, no a ojo: `python src/calibrate_gate.py` compara la distribución de max-softmax y energía de logits entre imágenes in-domain y fuera de dominio, y sugiere el punto de corte que mejor las separa (se aplica en `web/app.js` con histéresis: entrar ≥ T, salir < T − 0.15).

### Capturas del testing (modelo desplegado)

Pruebas de extremo a extremo sobre la app desplegada, subiendo imágenes del split de **test** (no visto en entrenamiento) y una escena fuera de dominio:

| Detección de grieta | Superficie normal |
|:---:|:---:|
| ![Testing: grieta detectada](docs/figures/demo/demo_crack.png) | ![Testing: superficie normal](docs/figures/demo/demo_normal.png) |
| **Gate OOD (escena no metálica)** | **Acerca del modelo / créditos** |
| ![Testing: fuera de dominio](docs/figures/demo/demo_ood.png) | ![Acerca del modelo](docs/figures/demo/demo_about.png) |

El chip superior muestra clase y confianza; el panel lateral, el conteo por clase y las probabilidades del gate. Más capturas (rayón, óxido, perforación) en [`docs/figures/demo/`](docs/figures/demo/).

---

## Instalación

```bash
git clone https://github.com/SimonLexRS/vision-computacional.git
cd vision-computacional

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requiere **Python 3.12**. El entrenamiento aprovecha GPU NVIDIA con CUDA si está disponible; **la evaluación y la inferencia funcionan en CPU sin ningún cambio**.

### Datos y pesos entrenados

Ni el dataset ni los checkpoints se versionan en el repositorio (`industrial_defect_dataset/`, `external_datasets/`, `yolo_dataset/` y `outputs/` están en `.gitignore`).

```bash
# Dataset sintético: descargar de Kaggle y descomprimir en industrial_defect_dataset/
# Datasets reales (SteelDefectX + NEU-DET): descarga automática
python src/download_datasets.py

# Pesos entrenados del clasificador: se regeneran con
python src/train.py --datos industrial_defect_dataset/ --salida outputs/checkpoints/
```

Los modelos ONNX desplegados (`web/model.onnx`, `web/detector.onnx`) **sí** están versionados, así que la demo web funciona directamente tras el clone.

---

## Pipeline de Ejecución

### 1. Preparación de datos

```bash
# Reconstruir la partición de test del dataset sintético (clasificación)
python src/preparar_datos.py --datos industrial_defect_dataset --manifiesto splits/test_files.txt

# Descargar datasets reales (SteelDefectX + NEU-DET)
python src/download_datasets.py

# Construir el dataset combinado de detección con cajas refinadas
python src/prepare_combined_dataset.py
```

### 2. Entrenamiento

El entrenamiento de **todos** los modelos —clasificadores y detector— está integrado en el notebook [`entrenar_modelos.ipynb`](entrenar_modelos.ipynb) (Sección 15 para YOLO). Para reproducirlo sin abrir Jupyter:

```bash
# Clasificadores (los defaults reproducen exactamente la tabla de resultados)
python src/train.py --datos industrial_defect_dataset/ --salida outputs/checkpoints/

# Un solo modelo (reproduce su fila de la tabla: el Generator del DataLoader se
# reinicia por modelo, así que el resultado no depende del orden de entrenamiento)
python src/train.py --modelos mobilenet_v3_small

# Detector YOLOv8n
python src/train_yolo.py --datos yolo_dataset/defects.yaml --device 0
```

> Ambas vías (notebook y script) usan los mismos hiperparámetros y la misma semilla (42). El entrenamiento de YOLO usa early stopping (`patience=15`) sobre el mAP50 de validación.

### 3. Evaluación e inferencia (funciona en CPU)

```bash
# Métricas sobre el split de test: accuracy, F1 macro, matriz de confusión y reporte
# por clase. Determinista: mismo .pt + mismo split = mismas métricas.
python src/evaluate.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
    --datos industrial_defect_dataset/ --split test

# Inferencia sobre una imagen suelta o una carpeta completa (CPU por defecto)
python src/predict.py --modelo outputs/checkpoints/mobilenet_v3_small_best.pt \
    --imagen industrial_defect_dataset/test/crack/

# Notebook equivalente sin GPU: carga los pesos con map_location="cpu", predice
# sobre imágenes de ejemplo, visualiza el resultado y mide la latencia en CPU.
jupyter notebook notebooks/inferencia_cpu.ipynb
```

### 4. Exportación y verificación ONNX

```bash
# Clasificador Gate → web/model.onnx (exporta y verifica paridad numérica)
python src/export_onnx.py --checkpoint outputs/checkpoints/mobilenet_v3_small_best.pt \
    --salida web/model.onnx

# Detector → web/detector.onnx (exporta, copia a web/ y verifica paridad)
python src/export_detector_onnx.py --checkpoint outputs/yolo/train/weights/best.pt \
    --salida web/detector.onnx

# Verificación de paridad por separado
python src/verify_detector_onnx.py --pt outputs/yolo/train/weights/best.pt \
    --onnx web/detector.onnx

# Calibración del umbral del gate de dominio
python src/calibrate_gate.py
```

### 5. Utilidad adicional: edge detection en vivo

Script independiente (`src/edge_detection.py`) que muestra solo los bordes del stream de la cámara con **Canny clásico** (grises → desenfoque gaussiano → Canny), a >30 FPS en CPU.

```bash
python src/edge_detection.py                # webcam 0, bordes en verde
python src/edge_detection.py --mode map     # mapa de bordes blanco/negro
python src/edge_detection.py --selftest     # sin cámara: prueba con una imagen de test
```

Teclas en vivo: `m` alterna overlay/mapa, `+`/`-` ajustan el umbral `t1`, `q`/ESC sale.

---

## Reproducibilidad

- **Semilla 42** fijada en `common.set_seed()`: `random`, `numpy`, `torch`, `torch.cuda`, con `cudnn.deterministic = True` y `cudnn.benchmark = False`. Se reinicia al inicio de cada modelo **junto con el `Generator` del DataLoader**, de modo que cada modelo recibe el mismo orden de batches se entrene solo o dentro de la lista completa.
- **`evaluate.py` no tiene aleatoriedad** (`shuffle=False`, sin augmentation): el mismo `.pt` sobre el mismo split da siempre las mismas métricas.
- **Los defaults de `train.py` son los hiperparámetros exactos** de la tabla de resultados. `--label-smoothing` (default `0.0`) y `--scheduler` (default `none`) existen como opciones, pero los resultados reportados usan los valores por defecto.
- **Cada checkpoint guarda su configuración** (`label_smoothing`, `scheduler`, `lr`, `weight_decay`, `batch_size`, `seed`) junto a las métricas de validación y los nombres de clase.
- **Particiones versionadas:** `splits/test_files.txt` fija el split de test archivo por archivo; los splits de las fuentes reales se generan con `np.random.seed(42)`.
- **Limitación:** algunas operaciones de cuDNN y AMP no son determinísticas bit a bit. En la misma máquina y versión de PyTorch el resultado es estable; entre GPUs o versiones distintas puede haber diferencias en los últimos decimales. **No se promete igualdad bit a bit.**

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt                  # versiones fijadas
├── informe/                          # Entregable A: main.tex, figuras y main.pdf
├── entrenar_modelos.ipynb            # pipeline completo (clasificadores + YOLO, Sección 15)
├── notebooks/
│   └── inferencia_cpu.ipynb          # inferencia CPU-only con el modelo entrenado
├── src/
│   ├── common.py                     # arquitecturas, transforms, semillas, checkpoints
│   ├── preparar_datos.py             # reconstruye la partición de test (clasificación)
│   ├── download_datasets.py          # descarga SteelDefectX + NEU-DET
│   ├── prepare_combined_dataset.py   # dataset YOLO de 3 fuentes con cajas refinadas
│   ├── train.py                      # entrena los 4 clasificadores
│   ├── evaluate.py                   # métricas sobre test/ o val/
│   ├── predict.py                    # inferencia sobre imagen o carpeta (CPU)
│   ├── train_yolo.py                 # entrena el detector YOLOv8n
│   ├── export_onnx.py                # clasificador → ONNX + paridad
│   ├── export_detector_onnx.py       # detector → ONNX + paridad
│   ├── verify_detector_onnx.py       # paridad ONNX vs PyTorch del detector
│   ├── calibrate_gate.py             # calibra el umbral del gate de dominio
│   └── edge_detection.py             # edge detection en vivo (Canny, OpenCV)
├── web/                              # demo estática (GitHub Pages + ONNX Runtime Web)
├── splits/test_files.txt             # partición de test versionada
├── docs/figures/                     # curvas, matrices de confusión y capturas de la demo
├── industrial_defect_dataset/        # dataset sintético (no versionado)
├── external_datasets/                # SteelDefectX + NEU-DET (no versionado)
├── yolo_dataset/                     # dataset combinado de detección (no versionado)
├── outputs/                          # checkpoints y métricas (no versionado)
└── .gitignore
```

---

## Resultados y Métricas de Evaluación

### Clasificación — dataset sintético (test: 1.500 imágenes)

La selección del mejor checkpoint se hizo sobre **validación**; el conjunto de **test** se usó una sola vez, al final.

| Modelo | Parámetros | Test Accuracy | Test F1-Macro |
|:---|:---:|:---:|:---:|
| **MobileNetV3-Small** | 1.5M | 1.0000 | **1.0000** |
| **ResNet-18** | 11.2M | 1.0000 | **1.0000** |
| **EfficientNet-B0** | 4.0M | 1.0000 | **1.0000** |
| **CNN Básica (desde cero)** | 391K | 0.9987 | 0.9987 |

No hay brecha validación → test, lo que descarta sobreajuste a la partición de validación. La CNN baseline comete los únicos 2 errores del experimento (un `crack` y un `rust` clasificados como `normal`). **MobileNetV3-Small es el modelo desplegado:** iguala a ResNet-18 con ~7× menos parámetros y produce el ONNX más liviano para la demo web.

### Detección — dataset combinado (test: 1.725 imágenes, incluye 177 reales de NEU-DET)

Entrenamiento reproducible (notebook Sección 15 / `python src/train_yolo.py`, 60 épocas, `imgsz=256`, SGD, seed 42):

| Métrica | Valor |
|:---|:---:|
| mAP50 | **0.739** |
| mAP50-95 | 0.488 |
| Precisión | 0.754 |
| Recall | 0.663 |

**AP50 por clase:** `hole` 0.846 · `crack` 0.791 · `rust` 0.677 · `scratch` 0.643.

> Este es el checkpoint desplegado en la demo web (`web/detector.onnx`, paridad ONNX↔PyTorch verificada). La corrida histórica de 100 épocas sobre el dataset de 2 fuentes reportó `mAP50 ≈ 0.80`, pero **sobre un split de test distinto** (sin las 177 imágenes reales de NEU-DET, que elevan la dificultad al usar cajas anotadas reales), por lo que ambas cifras no son directamente comparables.

---

## Limitaciones conocidas

- **Cajas de entrenamiento heurísticas:** en la fuente sintética y en SteelDefectX las bounding boxes se derivan de metadata + segmentación OpenCV, no de anotación humana; solo NEU-DET aporta cajas reales. Las métricas de detección son **exploratorias**, no un benchmark estándar.
- **Clasificación entrenada solo con datos sintéticos:** el F1 = 1.0 es una **cota superior bajo condiciones ideales**. Que tres arquitecturas distintas saturen la métrica indica que la dificultad del dataset sintético está por debajo del problema industrial real (*sim-to-real gap*).
- **El benchmark de detección cambió** al integrar NEU-DET, así que el mAP50 actual no es comparable con corridas anteriores.
- **Distribución de clases irreal:** en una línea de producción la clase `normal` domina y los defectos son eventos raros; el balance perfecto del dataset sintético no representa esa asimetría.
- **Una sola etiqueta por imagen** en la tarea de clasificación: una pieza real puede presentar óxido y rayadura simultáneamente.
- **Determinismo:** semillas fijadas en todos los scripts, pero las operaciones no determinísticas de CUDA/AMP pueden producir variaciones mínimas entre entornos.

Detalle completo en el informe: [`informe/main.pdf`](informe/main.pdf).

---

## Licencia y Atribución

- **Datasets:**
  - [Synthetic Industrial Metal Surface Defects](https://www.kaggle.com/datasets/tatheerabbas/synthetic-industrial-metal-surface-defects) bajo licencia **CC BY 4.0** (Tatheer Abbas).
  - [SteelDefectX](https://huggingface.co/datasets/Zhaosxian/SteelDefectX) (Zhao et al., 2024).
  - [NEU-DET Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) (Song & Yan, Northeastern University).
- **Código y Modelos:** Proyecto desarrollado por el equipo del Proyecto Final (ver [Equipo](#equipo)) para el módulo de **Visión Computacional**, Maestría en Ciencia de Datos e Inteligencia Artificial Aplicada, **Universidad Católica Boliviana**.
