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
// Modo seguimiento: tracks activos con EMA por clase (estado en la
// sección de seguimiento, más abajo).
let multiMode = false;

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

// --- Seguimiento de objetos: regiones dinámicas + CNN + tracker ---
// El modelo es un clasificador, no un detector: las regiones candidatas
// se proponen por saliencia visual (gradiente local: textura/bordes) y
// movimiento (diferencia contra el frame anterior), cada una se
// clasifica con la CNN en un solo batch, y un tracker las sigue frame a
// frame con suavizado exponencial, como haría el ojo humano.
const ANALYSIS_WIDTH = 192;
const analysisCanvas = document.createElement("canvas");
const analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });
let prevGray = null; // frame anterior en baja resolución (movimiento)
let tracks = [];
let nextTrackId = 1;
const TRACK_ALPHA = 0.4; // suavizado de la caja que sigue al objeto
const TRACK_MAX_MISSED = 3; // ticks sin match antes de soltar el track

// Detecta regiones salientes/en movimiento. Devuelve cajas cuadradas
// { sx, sy, side } en coordenadas del video.
function detectRegions(vw, vh) {
  const aw = ANALYSIS_WIDTH;
  const ah = Math.max(1, Math.round((ANALYSIS_WIDTH * vh) / vw));
  if (analysisCanvas.width !== aw || analysisCanvas.height !== ah) {
    analysisCanvas.width = aw;
    analysisCanvas.height = ah;
    prevGray = null;
  }
  analysisCtx.drawImage(video, 0, 0, aw, ah);
  const { data } = analysisCtx.getImageData(0, 0, aw, ah);

  // Escala de grises.
  const gray = new Float32Array(aw * ah);
  for (let i = 0; i < aw * ah; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }

  // Energía = gradiente local + movimiento contra el frame anterior.
  const energy = new Float32Array(aw * ah);
  for (let y = 1; y < ah - 1; y++) {
    for (let x = 1; x < aw - 1; x++) {
      const i = y * aw + x;
      let e = Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + aw] - gray[i - aw]);
      if (prevGray) e += 2 * Math.abs(gray[i] - prevGray[i]);
      energy[i] = e;
    }
  }
  prevGray = gray;

  // Suavizado 5×5 del mapa de energía: los bordes finos (anillos) se
  // convierten en regiones gruesas detectables por área.
  const blurred = new Float32Array(aw * ah);
  for (let y = 0; y < ah; y++) {
    for (let x = 0; x < aw; x++) {
      let s = 0;
      let cnt = 0;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= aw || ny < 0 || ny >= ah) continue;
          s += energy[ny * aw + nx];
          cnt++;
        }
      }
      blurred[y * aw + x] = s / cnt;
    }
  }

  // Umbral adaptativo: media + 1 desviación estándar.
  let sum = 0;
  let sum2 = 0;
  for (let i = 0; i < blurred.length; i++) {
    sum += blurred[i];
    sum2 += blurred[i] * blurred[i];
  }
  const mean = sum / blurred.length;
  const std = Math.sqrt(Math.max(0, sum2 / blurred.length - mean * mean));
  const threshold = mean + std;

  // Máscara binaria + dilatación 3×3 para cerrar huecos.
  const raw = new Uint8Array(aw * ah);
  let any = false;
  for (let i = 0; i < blurred.length; i++) {
    if (blurred[i] > threshold) {
      raw[i] = 1;
      any = true;
    }
  }
  if (!any) return [];
  const mask = new Uint8Array(aw * ah);
  for (let y = 0; y < ah; y++) {
    for (let x = 0; x < aw; x++) {
      if (!raw[y * aw + x]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < aw && ny >= 0 && ny < ah) mask[ny * aw + nx] = 1;
        }
      }
    }
  }

  // Componentes conexas (BFS 4-conectado) → caja por región.
  const labels = new Int32Array(aw * ah).fill(-1);
  const queue = new Int32Array(aw * ah);
  const boxes = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start] >= 0) continue;
    const id = boxes.length;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    labels[start] = id;
    let minX = aw;
    let maxX = 0;
    let minY = ah;
    let maxY = 0;
    let count = 0;
    while (head < tail) {
      const p = queue[head++];
      const px = p % aw;
      const py = (p / aw) | 0;
      count++;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
      if (px > 0 && mask[p - 1] && labels[p - 1] < 0) { labels[p - 1] = id; queue[tail++] = p - 1; }
      if (px < aw - 1 && mask[p + 1] && labels[p + 1] < 0) { labels[p + 1] = id; queue[tail++] = p + 1; }
      if (py > 0 && mask[p - aw] && labels[p - aw] < 0) { labels[p - aw] = id; queue[tail++] = p - aw; }
      if (py < ah - 1 && mask[p + aw] && labels[p + aw] < 0) { labels[p + aw] = id; queue[tail++] = p + aw; }
    }
    boxes.push({ minX, maxX, minY, maxY, area: count });
  }

  // Filtrar por área mínima (~1.5% del frame), expandir 30% y hacer
  // cuadradas; en pantallas angostas se siguen menos objetos.
  const minArea = aw * ah * 0.015;
  const maxBoxes = videoWrapper.clientWidth < 480 ? 3 : 5;
  const scaleX = vw / aw;
  const scaleY = vh / ah;
  return boxes
    .filter((b) => b.area >= minArea)
    .sort((a, b) => b.area - a.area)
    .slice(0, maxBoxes)
    .map((b) => {
      const w = (b.maxX - b.minX + 1) * scaleX * 1.3;
      const h = (b.maxY - b.minY + 1) * scaleY * 1.3;
      const cx = ((b.minX + b.maxX + 1) / 2) * scaleX;
      const cy = ((b.minY + b.maxY + 1) / 2) * scaleY;
      const side = Math.min(Math.max(w, h), Math.min(vw, vh));
      const sx = Math.min(Math.max(cx - side / 2, 0), vw - side);
      const sy = Math.min(Math.max(cy - side / 2, 0), vh - side);
      return { sx, sy, side };
    });
}

function boxIou(a, b) {
  const x1 = Math.max(a.sx, b.sx);
  const y1 = Math.max(a.sy, b.sy);
  const x2 = Math.min(a.sx + a.side, b.sx + b.side);
  const y2 = Math.min(a.sy + a.side, b.sy + b.side);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.side * a.side + b.side * b.side - inter;
  return union > 0 ? inter / union : 0;
}

// Asocia detecciones a tracks existentes (IoU o cercanía de centroides)
// y suaviza la caja: la etiqueta sigue al objeto sin saltos bruscos.
function updateTracks(detections, vw, vh) {
  const maxDist = Math.hypot(vw, vh) * 0.2;
  const used = new Set();

  for (const track of tracks) {
    let best = -1;
    let bestIou = 0.2; // umbral mínimo de solapamiento
    detections.forEach((det, i) => {
      if (used.has(i)) return;
      const ov = boxIou(track.box, det);
      if (ov > bestIou) {
        bestIou = ov;
        best = i;
      }
    });
    if (best < 0) {
      // Sin solapamiento: match por distancia de centroides.
      const tcx = track.box.sx + track.box.side / 2;
      const tcy = track.box.sy + track.box.side / 2;
      let bestD = maxDist;
      detections.forEach((det, i) => {
        if (used.has(i)) return;
        const d = Math.hypot(det.sx + det.side / 2 - tcx, det.sy + det.side / 2 - tcy);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
    }
    if (best >= 0) {
      used.add(best);
      const det = detections[best];
      track.box = {
        sx: TRACK_ALPHA * det.sx + (1 - TRACK_ALPHA) * track.box.sx,
        sy: TRACK_ALPHA * det.sy + (1 - TRACK_ALPHA) * track.box.sy,
        side: TRACK_ALPHA * det.side + (1 - TRACK_ALPHA) * track.box.side,
      };
      track.missed = 0;
      track.detIndex = best;
    } else {
      track.missed++;
      track.detIndex = -1;
    }
  }

  // Detecciones sin match → tracks nuevos.
  detections.forEach((det, i) => {
    if (used.has(i)) return;
    tracks.push({ id: nextTrackId++, box: { ...det }, probs: null, missed: 0, detIndex: i });
  });

  tracks = tracks.filter((t) => t.missed <= TRACK_MAX_MISSED);
}

async function runTrackingInference() {
  if (!session || inferenceInFlight || !cameraStream) return;
  inferenceInFlight = true;

  try {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const detections = detectRegions(vw, vh);
    updateTracks(detections, vw, vh);

    // Clasificar las regiones detectadas en un solo batch.
    let latency = 0;
    if (detections.length > 0) {
      const n = detections.length;
      const plane = IMG_SIZE * IMG_SIZE;
      const batch = new Float32Array(n * 3 * plane);
      detections.forEach((det, i) => {
        pixelsToTensor(drawRegionToCanvas(video, det.sx, det.sy, det.side), batch, i * 3 * plane);
      });

      const input = new ort.Tensor("float32", batch, [n, 3, IMG_SIZE, IMG_SIZE]);
      const t0 = performance.now();
      const output = await session.run({ input });
      latency = performance.now() - t0;

      const logits = output.logits.data;
      for (const track of tracks) {
        if (track.detIndex < 0) continue;
        const off = track.detIndex * CLASS_NAMES.length;
        const probs = softmax(Array.from(logits.slice(off, off + CLASS_NAMES.length)));
        track.probs = track.probs
          ? probs.map((p, j) => SMOOTHING * p + (1 - SMOOTHING) * track.probs[j])
          : probs;
      }
    }

    drawTracks();

    // Barras: promedio de los tracks activos. Chip: el más confiado.
    const withProbs = tracks.filter((t) => t.probs);
    if (withProbs.length > 0) {
      const avg = CLASS_NAMES.map((_, j) => withProbs.reduce((a, t) => a + t.probs[j], 0) / withProbs.length);
      updateBars(avg);
      let best = withProbs[0];
      withProbs.forEach((t) => {
        if (Math.max(...t.probs) > Math.max(...best.probs)) best = t;
      });
      const bestIdx = best.probs.indexOf(Math.max(...best.probs));
      const bestName = CLASS_NAMES[bestIdx];
      chip.hidden = false;
      chipClass.textContent = CLASS_LABELS[bestName];
      chipClass.style.color = `var(--${bestName})`;
      chipConf.textContent = `${(best.probs[bestIdx] * 100).toFixed(1)}%`;
    } else {
      chip.hidden = true;
    }

    latencyEl.textContent = detections.length
      ? `Inferencia: ${latency.toFixed(0)} ms (${tracks.length} objeto${tracks.length === 1 ? "" : "s"})`
      : "Buscando objetos…";
    tickFps();
  } catch (err) {
    console.error("Error en el seguimiento:", err);
    statusEl.textContent = "Error en la inferencia: " + err.message;
  } finally {
    inferenceInFlight = false;
  }
}

// Dibuja las cajas de los tracks sobre el overlay (coordenadas de
// video escaladas al tamaño del wrapper).
function drawTracks() {
  const w = videoWrapper.clientWidth;
  const h = videoWrapper.clientHeight;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const scaleX = w / vw;
  const scaleY = h / vh;
  const fontSize = Math.max(11, Math.round(w / 70));
  ctx.font = `600 ${fontSize}px "Segoe UI", system-ui, sans-serif`;

  for (const track of tracks) {
    if (!track.probs) continue;
    const bestIdx = track.probs.indexOf(Math.max(...track.probs));
    const name = CLASS_NAMES[bestIdx];
    const color = CLASS_COLORS[name];
    const x = track.box.sx * scaleX;
    const y = track.box.sy * scaleY;
    const s = track.box.side * scaleX;

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, s, s);

    const label = `${CLASS_LABELS[name]} ${(track.probs[bestIdx] * 100).toFixed(0)}%`;
    const labelH = fontSize + 8;
    ctx.fillStyle = "rgba(10, 14, 24, 0.82)";
    ctx.fillRect(x, y, ctx.measureText(label).width + 12, labelH);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 6, y + labelH - 7);
  }
}

function setMultiMode(on) {
  multiMode = on && !!cameraStream;
  btnMulti.disabled = !cameraStream;
  btnMulti.classList.toggle("active", multiMode);
  btnMulti.textContent = multiMode ? "Detección simple" : "Seguimiento de objetos";
  tracks = [];
  prevGray = null;
  if (!cameraStream) return;

  clearInterval(inferenceTimer);
  if (multiMode) {
    roiGuide.hidden = true;
    overlay.hidden = false; // el overlay pasa a ser la capa de cajas
    inferenceTimer = setInterval(() => {
      if (video.readyState >= 2) runTrackingInference();
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
  // Reset del modo seguimiento (requiere cámara activa).
  multiMode = false;
  tracks = [];
  prevGray = null;
  btnMulti.disabled = true;
  btnMulti.classList.remove("active");
  btnMulti.textContent = "Seguimiento de objetos";
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
