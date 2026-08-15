/**
 * Demo web: clasificación de defectos en superficies metálicas.
 *
 * MobileNetV3-Small exportado a ONNX, ejecutado 100% en el navegador
 * con ONNX Runtime Web (WASM). El preprocesamiento replica el del
 * entrenamiento: resize 256×256, grayscale → 3 canales, normalización
 * ImageNet.
 */

"use strict";

const IMG_SIZE = 256;
const CLASS_NAMES = ["crack", "hole", "normal", "rust", "scratch"];
const CLASS_LABELS = {
  crack: "Grieta",
  hole: "Perforación",
  normal: "Normal",
  rust: "Óxido",
  scratch: "Rayón",
};
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const INFERENCE_INTERVAL_MS = 200;
// En modo multi se infiere un batch de N regiones por tick: más trabajo
// por inferencia, así que el intervalo es mayor.
const MULTI_INFERENCE_INTERVAL_MS = 350;
const CLASS_COLORS = {
  crack: "#ff5c5c",
  hole: "#ff9f43",
  normal: "#34c77b",
  rust: "#c97b2f",
  scratch: "#b678ff",
};

// --- Elementos de la UI ---
const video = document.getElementById("video");
const videoWrapper = document.getElementById("video-wrapper");
const roiGuide = document.getElementById("roi-guide");
const overlay = document.getElementById("overlay");
const placeholder = document.getElementById("placeholder");
const chip = document.getElementById("prediction-chip");
const chipClass = document.getElementById("chip-class");
const chipConf = document.getElementById("chip-conf");
const btnCamera = document.getElementById("btn-camera");
const btnMulti = document.getElementById("btn-multi");
const btnFullscreen = document.getElementById("btn-fullscreen");
const cameraPanel = document.getElementById("camera-panel");
const fileInput = document.getElementById("file-input");
const badge = document.getElementById("engine-badge");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const latencyEl = document.getElementById("latency");
const barsEl = document.getElementById("bars");
const preprocessCanvas = document.getElementById("preprocess");
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });

let session = null;
let cameraStream = null;
let inferenceTimer = null;
let inferenceInFlight = false;
let frameCount = 0;
let fpsWindowStart = performance.now();
// Suavizado exponencial de probabilidades para estabilizar la
// predicción en vivo (evita parpadeo entre clases frame a frame).
let smoothProbs = null;
const SMOOTHING = 0.4;
// Modo multi: una EMA por celda de la cuadrícula.
let multiMode = false;
let smoothGrid = null;

// Ajusta el marco de "zona de análisis" al cuadrado central del
// wrapper (es exactamente la región que recorta el preprocesamiento).
function layoutRoi() {
  const side = Math.min(videoWrapper.clientWidth, videoWrapper.clientHeight) * 0.85;
  roiGuide.style.width = `${side}px`;
  roiGuide.style.height = `${side}px`;
}

window.addEventListener("resize", layoutRoi);

// --- Barras de probabilidad ---
const barFills = {};
const barValues = {};

function buildBars() {
  for (const name of CLASS_NAMES) {
    const row = document.createElement("li");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${CLASS_LABELS[name]}</span>
      <div class="bar-track"><div class="bar-fill" style="background: var(--${name})"></div></div>
      <span class="bar-value">0%</span>
    `;
    barsEl.appendChild(row);
    barFills[name] = row.querySelector(".bar-fill");
    barValues[name] = row.querySelector(".bar-value");
  }
}

function updateBars(probs) {
  CLASS_NAMES.forEach((name, i) => {
    const pct = probs[i] * 100;
    barFills[name].style.width = `${pct}%`;
    barValues[name].textContent = `${pct.toFixed(1)}%`;
  });
}

// --- Carga del modelo ONNX ---
async function loadModel() {
  try {
    // ONNX Runtime Web auto-hospedado en vendor/ort/ (sin CDN).
    // Se usa document.baseURI: el loader jsep se importa como módulo ES y
    // las rutas relativas se resolverían contra la carpeta del bundle.
    ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });
    badge.textContent = "modelo listo";
    badge.classList.add("ready");
  } catch (err) {
    console.error(err);
    badge.textContent = "error al cargar el modelo";
    badge.classList.add("error");
    statusEl.textContent = "No se pudo cargar model.onnx: " + err.message;
  }
}

// --- Preprocesamiento (idéntico al entrenamiento) ---
// Dibuja una región cuadrada del source en el canvas 256×256 y
// devuelve sus píxeles RGBA.
function drawRegionToCanvas(source, sx, sy, side) {
  preprocessCtx.drawImage(source, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE);
  return preprocessCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
}

// Convierte píxeles RGBA a tensor normalizado (grayscale → 3 canales,
// normalización ImageNet, layout CHW) escribiendo en `target` desde
// `offset` (permite armar batches sin copias extra).
function pixelsToTensor(pixels, target, offset) {
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    // Grayscale (luminosidad), replicado a 3 canales.
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    target[offset + i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    target[offset + plane + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    target[offset + 2 * plane + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
}

function preprocess(source, srcWidth, srcHeight) {
  // Recorte central cuadrado, como imagen completa encuadrada.
  const side = Math.min(srcWidth, srcHeight);
  const sx = (srcWidth - side) / 2;
  const sy = (srcHeight - side) / 2;

  const pixels = drawRegionToCanvas(source, sx, sy, side);
  const tensor = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  pixelsToTensor(pixels, tensor, 0);

  return new ort.Tensor("float32", tensor, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// --- Inferencia ---
async function runInference(source, srcWidth, srcHeight) {
  // Guard anti-solapamiento: si la inferencia anterior no terminó,
  // se omite este tick (evita ejecuciones concurrentes acumuladas).
  if (!session || inferenceInFlight) return;
  inferenceInFlight = true;

  try {
    const input = preprocess(source, srcWidth, srcHeight);

    const t0 = performance.now();
    const output = await session.run({ input });
    const latency = performance.now() - t0;

    const probs = softmax(Array.from(output.logits.data));
    smoothProbs = smoothProbs
      ? probs.map((p, i) => SMOOTHING * p + (1 - SMOOTHING) * smoothProbs[i])
      : probs;
    const bestIdx = smoothProbs.indexOf(Math.max(...smoothProbs));
    const bestName = CLASS_NAMES[bestIdx];

    chip.hidden = false;
    chipClass.textContent = CLASS_LABELS[bestName];
    chipClass.style.color = `var(--${bestName})`;
    chipConf.textContent = `${(smoothProbs[bestIdx] * 100).toFixed(1)}%`;

    updateBars(smoothProbs);
    latencyEl.textContent = `Inferencia: ${latency.toFixed(0)} ms`;

    tickFps();
  } catch (err) {
    console.error("Error en la inferencia:", err);
    statusEl.textContent = "Error en la inferencia: " + err.message;
  } finally {
    inferenceInFlight = false;
  }
}

// Contador de FPS compartido por ambos modos de inferencia.
function tickFps() {
  frameCount++;
  const now = performance.now();
  if (now - fpsWindowStart >= 1000) {
    fpsEl.textContent = `FPS: ${((frameCount * 1000) / (now - fpsWindowStart)).toFixed(1)}`;
    frameCount = 0;
    fpsWindowStart = now;
  }
}

// --- Detección múltiple: cuadrícula de regiones clasificadas en batch ---
// 3×3 en escritorio; 2×2 cuando el panel es angosto (móvil).
function gridDims() {
  const small = videoWrapper.clientWidth < 480;
  return small ? { rows: 2, cols: 2 } : { rows: 3, cols: 3 };
}

async function runMultiInference() {
  if (!session || inferenceInFlight || !cameraStream) return;
  inferenceInFlight = true;

  try {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const { rows, cols } = gridDims();
    const n = rows * cols;
    const plane = IMG_SIZE * IMG_SIZE;
    const batch = new Float32Array(n * 3 * plane);
    const cells = [];

    let k = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Cuadrado centrado dentro de cada celda de la cuadrícula.
        const cw = vw / cols;
        const ch = vh / rows;
        const side = Math.min(cw, ch);
        const sx = c * cw + (cw - side) / 2;
        const sy = r * ch + (ch - side) / 2;
        pixelsToTensor(drawRegionToCanvas(video, sx, sy, side), batch, k * 3 * plane);
        cells.push({ sx, sy, side });
        k++;
      }
    }

    const input = new ort.Tensor("float32", batch, [n, 3, IMG_SIZE, IMG_SIZE]);
    const t0 = performance.now();
    const output = await session.run({ input });
    const latency = performance.now() - t0;

    const logits = output.logits.data;
    if (!smoothGrid || smoothGrid.length !== n) {
      smoothGrid = Array.from({ length: n }, () => null);
    }
    const results = [];
    for (let i = 0; i < n; i++) {
      const probs = softmax(Array.from(logits.slice(i * CLASS_NAMES.length, (i + 1) * CLASS_NAMES.length)));
      smoothGrid[i] = smoothGrid[i]
        ? probs.map((p, j) => SMOOTHING * p + (1 - SMOOTHING) * smoothGrid[i][j])
        : probs;
      results.push(smoothGrid[i]);
    }

    drawDetections(cells, results, vw, vh);

    // Barras: promedio de las regiones. Chip: la detección más confiada.
    const avg = CLASS_NAMES.map((_, j) => results.reduce((a, p) => a + p[j], 0) / n);
    updateBars(avg);

    let bestCell = 0;
    let bestConf = 0;
    results.forEach((p, i) => {
      const m = Math.max(...p);
      if (m > bestConf) {
        bestConf = m;
        bestCell = i;
      }
    });
    const bestIdx = results[bestCell].indexOf(bestConf);
    const bestName = CLASS_NAMES[bestIdx];
    chip.hidden = false;
    chipClass.textContent = CLASS_LABELS[bestName];
    chipClass.style.color = `var(--${bestName})`;
    chipConf.textContent = `${(bestConf * 100).toFixed(1)}%`;

    latencyEl.textContent = `Inferencia: ${latency.toFixed(0)} ms (${n} regiones)`;
    tickFps();
  } catch (err) {
    console.error("Error en la inferencia múltiple:", err);
    statusEl.textContent = "Error en la inferencia: " + err.message;
  } finally {
    inferenceInFlight = false;
  }
}

// Dibuja las cajas etiquetadas sobre el overlay (coordenadas de video
// escaladas al tamaño del wrapper).
function drawDetections(cells, results, vw, vh) {
  const w = videoWrapper.clientWidth;
  const h = videoWrapper.clientHeight;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const scaleX = w / vw;
  const scaleY = h / vh;
  const fontSize = Math.max(11, Math.round(w / 70));
  ctx.font = `600 ${fontSize}px "Segoe UI", system-ui, sans-serif`;

  cells.forEach((cell, i) => {
    const probs = results[i];
    const bestIdx = probs.indexOf(Math.max(...probs));
    const name = CLASS_NAMES[bestIdx];
    const color = CLASS_COLORS[name];
    const x = cell.sx * scaleX;
    const y = cell.sy * scaleY;
    const s = cell.side * scaleX;

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, s, s);

    const label = `${CLASS_LABELS[name]} ${(probs[bestIdx] * 100).toFixed(0)}%`;
    const labelH = fontSize + 8;
    ctx.fillStyle = "rgba(10, 14, 24, 0.82)";
    ctx.fillRect(x, y, ctx.measureText(label).width + 12, labelH);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 6, y + labelH - 7);
  });
}

function setMultiMode(on) {
  multiMode = on && !!cameraStream;
  btnMulti.disabled = !cameraStream;
  btnMulti.classList.toggle("active", multiMode);
  btnMulti.textContent = multiMode ? "Detección simple" : "Detección múltiple";
  smoothGrid = null;
  if (!cameraStream) return;

  clearInterval(inferenceTimer);
  if (multiMode) {
    roiGuide.hidden = true;
    overlay.hidden = false; // el overlay pasa a ser la capa de cajas
    inferenceTimer = setInterval(() => {
      if (video.readyState >= 2) runMultiInference();
    }, MULTI_INFERENCE_INTERVAL_MS);
  } else {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    overlay.hidden = true;
    roiGuide.hidden = false;
    layoutRoi();
    smoothProbs = null;
    inferenceTimer = setInterval(() => {
      if (video.readyState >= 2) {
        runInference(video, video.videoWidth, video.videoHeight);
      }
    }, INFERENCE_INTERVAL_MS);
  }
}

// --- Cámara ---
async function startCamera() {
  btnCamera.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      "No se pudo acceder a la cámara (" + err.name + "). Puedes usar 'Subir imagen'.";
    btnCamera.disabled = false;
    return;
  }

  stopUploadedImage();
  video.srcObject = cameraStream;
  video.hidden = false;
  overlay.hidden = true;
  placeholder.style.display = "none";
  await video.play();

  // El wrapper adopta la proporción real del stream: con object-fit
  // cover y proporción coincidente nada queda recortado ni con franjas.
  if (video.videoWidth && video.videoHeight) {
    videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    // Variable CSS usada por el layout de pantalla completa.
    videoWrapper.style.setProperty("--video-ar", video.videoWidth / video.videoHeight);
  }
  smoothProbs = null;
  roiGuide.hidden = false;
  layoutRoi();

  statusEl.textContent = "Cámara activa — analizando en vivo.";
  btnCamera.textContent = "Detener cámara";
  btnCamera.disabled = false;
  btnMulti.disabled = false;

  clearInterval(inferenceTimer);
  inferenceTimer = setInterval(() => {
    if (video.readyState >= 2) {
      runInference(video, video.videoWidth, video.videoHeight);
    }
  }, INFERENCE_INTERVAL_MS);

  // Gancho de prueba automatizada: ?multi=1 activa el modo multi.
  if (new URLSearchParams(location.search).has("multi")) {
    setMultiMode(true);
  }
}

function stopCamera() {
  clearInterval(inferenceTimer);
  inferenceTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  // Reset del modo multi (requiere cámara activa).
  multiMode = false;
  smoothGrid = null;
  btnMulti.disabled = true;
  btnMulti.classList.remove("active");
  btnMulti.textContent = "Detección múltiple";
  video.srcObject = null;
  placeholder.style.display = "flex";
  chip.hidden = true;
  roiGuide.hidden = true;
  smoothProbs = null;
  statusEl.textContent = "Cámara detenida.";
  btnCamera.textContent = "Activar cámara";
  fpsEl.textContent = "FPS: —";
}

// --- Subir imagen (fallback sin cámara) ---
function stopUploadedImage() {
  overlay.hidden = true;
}

function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // Mostrar la imagen en el área del video.
    if (cameraStream) stopCamera();
    video.hidden = true;

    // El overlay se dibuja a la resolución de la imagen (tope 1024 px)
    // y el wrapper adopta su proporción: object-fit contain evita cortes.
    const scale = Math.min(1, 1024 / Math.max(img.width, img.height));
    overlay.width = Math.round(img.width * scale);
    overlay.height = Math.round(img.height * scale);
    const ctx = overlay.getContext("2d");
    ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
    overlay.hidden = false;
    placeholder.style.display = "none";
    videoWrapper.style.aspectRatio = `${overlay.width} / ${overlay.height}`;
    videoWrapper.style.setProperty("--video-ar", overlay.width / overlay.height);
    smoothProbs = null;
    roiGuide.hidden = false;
    layoutRoi();

    runInference(img, img.width, img.height);
    statusEl.textContent = "Imagen analizada: " + file.name;
    URL.revokeObjectURL(url);
  };
  img.src = url;
  fileInput.value = "";
}

// --- Init ---
buildBars();
loadModel();
btnCamera.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else startCamera();
});
fileInput.addEventListener("change", handleUpload);
btnMulti.addEventListener("click", () => setMultiMode(!multiMode));

// --- Pantalla completa ---
if (!document.fullscreenEnabled) {
  btnFullscreen.hidden = true; // p. ej. iOS Safari en iPhone
}
btnFullscreen.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await cameraPanel.requestFullscreen();
  } catch (err) {
    console.error("No se pudo cambiar pantalla completa:", err);
  }
});
document.addEventListener("fullscreenchange", () => {
  btnFullscreen.textContent = document.fullscreenElement
    ? "Salir de pantalla completa"
    : "Pantalla completa";
  layoutRoi();
});

// Gancho de prueba automatizada: ?autocam=1 activa la cámara sin clic
// (se usa con --use-fake-device-for-media-stream en tests headless).
if (new URLSearchParams(location.search).has("autocam")) {
  startCamera();
}
