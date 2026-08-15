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

// --- Elementos de la UI ---
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const placeholder = document.getElementById("placeholder");
const chip = document.getElementById("prediction-chip");
const chipClass = document.getElementById("chip-class");
const chipConf = document.getElementById("chip-conf");
const btnCamera = document.getElementById("btn-camera");
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
let frameCount = 0;
let fpsWindowStart = performance.now();

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
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
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
function preprocess(source, srcWidth, srcHeight) {
  // Recorte central cuadrado, como imagen completa encuadrada.
  const side = Math.min(srcWidth, srcHeight);
  const sx = (srcWidth - side) / 2;
  const sy = (srcHeight - side) / 2;

  preprocessCtx.drawImage(
    source, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE
  );

  const { data } = preprocessCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const tensor = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    // Grayscale (luminosidad), replicado a 3 canales.
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    // Normalización ImageNet, layout CHW.
    tensor[i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    tensor[IMG_SIZE * IMG_SIZE + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    tensor[2 * IMG_SIZE * IMG_SIZE + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

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
  if (!session) return;

  const input = preprocess(source, srcWidth, srcHeight);

  const t0 = performance.now();
  const output = await session.run({ input });
  const latency = performance.now() - t0;

  const probs = softmax(Array.from(output.logits.data));
  const bestIdx = probs.indexOf(Math.max(...probs));
  const bestName = CLASS_NAMES[bestIdx];

  chip.hidden = false;
  chipClass.textContent = CLASS_LABELS[bestName];
  chipClass.style.color = `var(--${bestName})`;
  chipConf.textContent = `${(probs[bestIdx] * 100).toFixed(1)}%`;

  updateBars(probs);
  latencyEl.textContent = `Inferencia: ${latency.toFixed(0)} ms`;

  frameCount++;
  const now = performance.now();
  if (now - fpsWindowStart >= 1000) {
    fpsEl.textContent = `FPS: ${((frameCount * 1000) / (now - fpsWindowStart)).toFixed(1)}`;
    frameCount = 0;
    fpsWindowStart = now;
  }
}

// --- Cámara ---
async function startCamera() {
  btnCamera.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
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

  statusEl.textContent = "Cámara activa — analizando en vivo.";
  btnCamera.textContent = "Detener cámara";
  btnCamera.disabled = false;

  clearInterval(inferenceTimer);
  inferenceTimer = setInterval(() => {
    if (video.readyState >= 2) {
      runInference(video, video.videoWidth, video.videoHeight);
    }
  }, INFERENCE_INTERVAL_MS);
}

function stopCamera() {
  clearInterval(inferenceTimer);
  inferenceTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  video.srcObject = null;
  placeholder.style.display = "flex";
  chip.hidden = true;
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

    const ctx = overlay.getContext("2d");
    // Encuadre tipo "cover".
    const scale = Math.max(overlay.width / img.width, overlay.height / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.drawImage(img, (overlay.width - w) / 2, (overlay.height - h) / 2, w, h);
    overlay.hidden = false;
    placeholder.style.display = "none";

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
