/**
 * Demo web: clasificación de defectos en superficies metálicas.
 *
 * MobileNetV3-Small exportado a ONNX, ejecutado 100% en el navegador
 * con ONNX Runtime Web (WASM). El preprocesamiento replica el del
 * entrenamiento: resize 256×256, grayscale → 3 canales, normalización
 * ImageNet.
 *
 * El reconocimiento se muestra como un borde de color alrededor de
 * toda la imagen (un color por clase) con transición suave; si la
 * confianza es baja se muestra "Indefinido" en gris, porque el modelo
 * se entrenó con superficies metálicas y las escenas arbitrarias son
 * fuera de su dominio.
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
const INFERENCE_INTERVAL_MS = 150;
// Por debajo de esta confianza suavizada la escena se considera fuera
// de dominio y se reporta "Indefinido" en lugar de una clase al azar.
const CONF_THRESHOLD = 0.45;

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
const btnEdges = document.getElementById("btn-edges");
const btnFullscreen = document.getElementById("btn-fullscreen");
const cameraPanel = document.getElementById("camera-panel");
const viewsEl = document.getElementById("views");
const edgesPanel = document.getElementById("edges-panel");
const edgesView = document.getElementById("edges-view");
const edgesCtx = edgesView.getContext("2d");
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
const SMOOTHING = 0.3;

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
// normalización ImageNet, layout CHW).
function pixelsToTensor(pixels, target) {
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    // Grayscale (luminosidad), replicado a 3 canales.
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    target[i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    target[plane + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    target[2 * plane + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
}

function preprocess(source, srcWidth, srcHeight) {
  // Recorte central cuadrado, como imagen completa encuadrada.
  const side = Math.min(srcWidth, srcHeight);
  const sx = (srcWidth - side) / 2;
  const sy = (srcHeight - side) / 2;

  const pixels = drawRegionToCanvas(source, sx, sy, side);
  const tensor = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  pixelsToTensor(pixels, tensor);

  return new ort.Tensor("float32", tensor, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// Publica la predicción en la UI: chip, borde de color alrededor de
// toda la imagen y barras.
function showPrediction(probs) {
  const bestIdx = probs.indexOf(Math.max(...probs));
  const bestName = CLASS_NAMES[bestIdx];
  const bestConf = probs[bestIdx];
  const confident = bestConf >= CONF_THRESHOLD;

  // Borde completo de la imagen en el color de la clase (gris si la
  // confianza es baja: escena fuera del dominio del modelo).
  videoWrapper.style.setProperty(
    "--det-color",
    confident ? `var(--${bestName})` : "var(--muted)"
  );
  videoWrapper.classList.add("detecting");

  chip.hidden = false;
  chipClass.textContent = confident ? CLASS_LABELS[bestName] : "Indefinido";
  chipClass.style.color = confident ? `var(--${bestName})` : "var(--muted)";
  chipConf.textContent = `${(bestConf * 100).toFixed(1)}%`;

  updateBars(probs);
}

// --- Detección de bordes en vivo (Sobel sobre canvas, vista aparte) ---
// Equivalente práctico de Canny para visualización en vivo: el
// downscale del drawImage ya actúa como suavizado, luego magnitud de
// Sobel + umbral. Barato en CPU (~384 px de ancho, loop por rAF).
const EDGES_WIDTH = 384;
const EDGE_THRESHOLD = 80; // magnitud Sobel mínima (0-1020); subir = menos ruido
const edgeSrc = document.createElement("canvas");
const edgeSrcCtx = edgeSrc.getContext("2d", { willReadFrequently: true });
let edgesOn = false;
let edgesRaf = 0;
let edgeGray = null; // buffers reutilizados (sin allocations por frame)
let edgeOut = null;

function computeEdges() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  const aw = EDGES_WIDTH;
  const ah = Math.max(1, Math.round((EDGES_WIDTH * vh) / vw));
  if (edgeSrc.width !== aw || edgeSrc.height !== ah) {
    edgeSrc.width = aw;
    edgeSrc.height = ah;
    edgesView.width = aw;
    edgesView.height = ah;
    edgeGray = new Float32Array(aw * ah);
    edgeOut = edgesCtx.createImageData(aw, ah);
  }

  edgeSrcCtx.drawImage(video, 0, 0, aw, ah);
  const sd = edgeSrcCtx.getImageData(0, 0, aw, ah).data;
  const od = edgeOut.data;
  const g = edgeGray;

  for (let i = 0; i < aw * ah; i++) {
    g[i] = 0.299 * sd[i * 4] + 0.587 * sd[i * 4 + 1] + 0.114 * sd[i * 4 + 2];
  }

  for (let y = 1; y < ah - 1; y++) {
    for (let x = 1; x < aw - 1; x++) {
      const i = y * aw + x;
      const gx =
        -g[i - aw - 1] - 2 * g[i - 1] - g[i + aw - 1] +
        g[i - aw + 1] + 2 * g[i + 1] + g[i + aw + 1];
      const gy =
        -g[i - aw - 1] - 2 * g[i - aw] - g[i - aw + 1] +
        g[i + aw - 1] + 2 * g[i + aw] + g[i + aw + 1];
      const v = Math.abs(gx) + Math.abs(gy) > EDGE_THRESHOLD ? 255 : 0;
      const j = i * 4;
      od[j] = v;
      od[j + 1] = v;
      od[j + 2] = v;
      od[j + 3] = 255;
    }
  }
  edgesCtx.putImageData(edgeOut, 0, 0);
}

function edgesLoop() {
  if (!edgesOn) return;
  if (video.readyState >= 2) computeEdges();
  edgesRaf = requestAnimationFrame(edgesLoop);
}

function setEdges(on) {
  edgesOn = on && !!cameraStream;
  btnEdges.disabled = !cameraStream;
  btnEdges.classList.toggle("active", edgesOn);
  btnEdges.textContent = edgesOn ? "Ocultar bordes" : "Detección de bordes";
  edgesPanel.hidden = !edgesOn;
  viewsEl.classList.toggle("edges-on", edgesOn);
  if (edgesOn) {
    edgesLoop();
  } else {
    cancelAnimationFrame(edgesRaf);
  }
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

    showPrediction(smoothProbs);
    latencyEl.textContent = `Inferencia: ${latency.toFixed(0)} ms`;

    frameCount++;
    const now = performance.now();
    if (now - fpsWindowStart >= 1000) {
      fpsEl.textContent = `FPS: ${((frameCount * 1000) / (now - fpsWindowStart)).toFixed(1)}`;
      frameCount = 0;
      fpsWindowStart = now;
    }
  } catch (err) {
    console.error("Error en la inferencia:", err);
    statusEl.textContent = "Error en la inferencia: " + err.message;
  } finally {
    inferenceInFlight = false;
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
  btnEdges.disabled = false;

  clearInterval(inferenceTimer);
  inferenceTimer = setInterval(() => {
    if (video.readyState >= 2) {
      runInference(video, video.videoWidth, video.videoHeight);
    }
  }, INFERENCE_INTERVAL_MS);

  // Gancho de prueba automatizada: ?edges=1 activa la vista de bordes.
  if (new URLSearchParams(location.search).has("edges")) {
    setEdges(true);
  }
}

function stopCamera() {
  clearInterval(inferenceTimer);
  inferenceTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  // Reset de la vista de bordes (requiere cámara activa).
  edgesOn = false;
  cancelAnimationFrame(edgesRaf);
  edgesPanel.hidden = true;
  viewsEl.classList.remove("edges-on");
  btnEdges.disabled = true;
  btnEdges.classList.remove("active");
  btnEdges.textContent = "Detección de bordes";
  video.srcObject = null;
  placeholder.style.display = "flex";
  chip.hidden = true;
  roiGuide.hidden = true;
  smoothProbs = null;
  // Quitar el borde de reconocimiento.
  videoWrapper.classList.remove("detecting");
  videoWrapper.style.removeProperty("--det-color");
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
    // y el wrapper adopta su proporción: object-fit cover sin cortes.
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
btnEdges.addEventListener("click", () => setEdges(!edgesOn));

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
