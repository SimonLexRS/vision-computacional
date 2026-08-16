/**
 * Demo web: detección individual de defectos en superficies metálicas.
 *
 * Pipeline en dos etapas, 100% en el navegador (ONNX Runtime Web, WASM):
 *
 *   1. GATE DE DOMINIO — MobileNetV3-Small (model.onnx), clasificador de
 *      5 clases (crack, hole, normal, rust, scratch). Si su confianza
 *      suavizada < GATE_THRESHOLD la escena es fuera de dominio: se
 *      reporta "Indefinido" y NO se muestra ninguna detección ni borde.
 *      Así solo se muestran las clases para las que se entrenó el sistema.
 *
 *   2. DETECTOR — YOLOv8n (detector.onnx), 4 clases de defecto
 *      (crack, hole, rust, scratch). Solo corre cuando el gate está en
 *      dominio. Localiza y clasifica cada defecto individualmente;
 *      decode + NMS se hacen aquí en JS. En dominio y sin detecciones
 *      → la superficie es "Normal".
 *
 * La detección de bordes (Sobel) queda subordinada al detector: solo se
 * pintan bordes DENTRO de las cajas detectadas, en el color de su clase.
 *
 * Preprocesamiento por modelo (mismo recorte central cuadrado 256×256):
 *   gate     → grayscale replicado a 3 canales + normalización ImageNet.
 *   detector → RGB /255 (como el entrenamiento de ultralytics).
 */

"use strict";

const IMG_SIZE = 256;
// Orden alfabético de ImageFolder (igual que en el entrenamiento del gate).
const CLS_NAMES = ["crack", "hole", "normal", "rust", "scratch"];
// Clases del detector (defects.yaml). "normal" no existe: es background.
const DET_NAMES = ["crack", "hole", "rust", "scratch"];
const CLASS_LABELS = {
  crack: "Grieta",
  hole: "Perforación",
  normal: "Normal",
  rust: "Óxido",
  scratch: "Rayón",
};
// RGB por clase para dibujo en canvas (mismos colores que tokens.css).
const CLASS_RGB = {
  crack: [255, 92, 92],
  hole: [255, 159, 67],
  normal: [52, 199, 123],
  rust: [201, 123, 47],
  scratch: [182, 120, 255],
};
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const TICK_MS = 200;
// Por debajo de esta confianza suavizada la escena se considera fuera
// de dominio y se reporta "Indefinido" en lugar de una clase al azar.
const GATE_THRESHOLD = 0.45;
const DET_THRESHOLD = 0.4; // score mínimo por caja del detector
const NMS_IOU = 0.45;

// --- Elementos de la UI ---
const video = document.getElementById("video");
const videoWrapper = document.getElementById("video-wrapper");
const roiGuide = document.getElementById("roi-guide");
const overlay = document.getElementById("overlay");
const detectOverlay = document.getElementById("detect-overlay");
const detectCtx = detectOverlay.getContext("2d");
const placeholder = document.getElementById("placeholder");
const chip = document.getElementById("prediction-chip");
const chipDot = document.getElementById("chip-dot");
const chipClass = document.getElementById("chip-class");
const chipConf = document.getElementById("chip-conf");
const detCount = document.getElementById("det-count");
const btnCamera = document.getElementById("btn-camera");
const btnEdges = document.getElementById("btn-edges");
const btnFullscreen = document.getElementById("btn-fullscreen");
const cameraPanel = document.getElementById("camera-panel");
const edgesOverlay = document.getElementById("edges-overlay");
const edgesCtx = edgesOverlay.getContext("2d");
const fileInput = document.getElementById("file-input");
const badgeCls = document.getElementById("badge-cls");
const badgeDet = document.getElementById("badge-det");
const badgeCam = document.getElementById("badge-cam");
const statusEl = document.getElementById("status");
const sysState = document.getElementById("sys-state");
const sysStateText = document.getElementById("sys-state-text");
const detList = document.getElementById("det-list");
const countsEl = document.getElementById("counts");
const fpsEl = document.getElementById("fps");
const latencyClsEl = document.getElementById("latency-cls");
const latencyDetEl = document.getElementById("latency-det");
const barsEl = document.getElementById("bars");
const preprocessCanvas = document.getElementById("preprocess");
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });

let clsSession = null;
let detSession = null;
let clsInput = "input";
let clsOutput = "logits";
let detInput = "images";
let detOutput = "output0";
let cameraStream = null;
let tickTimer = null;
let inferenceInFlight = false;
let frameCount = 0;
let fpsWindowStart = performance.now();
// Suavizado exponencial de probabilidades del gate para estabilizar la
// predicción en vivo (evita parpadeo entre clases frame a frame).
let smoothProbs = null;
const SMOOTHING = 0.3;
// Detecciones activas (coords del frame fuente), con seguimiento
// temporal ligero contra parpadeo.
let activeDets = [];
let prevDets = [];
let edgesOn = false;
// Última imagen estática analizada (para re-correr el pipeline al
// activar "Bordes por detección" sin cámara).
let lastStaticSource = null;

// Ajusta el marco de "zona de análisis" al cuadrado central del
// wrapper (es exactamente la región que recorta el preprocesamiento).
function layoutRoi() {
  const side = Math.min(videoWrapper.clientWidth, videoWrapper.clientHeight) * 0.85;
  roiGuide.style.width = `${side}px`;
  roiGuide.style.height = `${side}px`;
}

window.addEventListener("resize", layoutRoi);

// --- Pills de estado ---
function setPill(el, state) {
  // state: "loading" | "ready" | "error" | "idle"
  el.classList.remove("is-loading", "is-ready", "is-error");
  if (state === "loading") el.classList.add("is-loading");
  if (state === "ready") el.classList.add("is-ready");
  if (state === "error") el.classList.add("is-error");
}

// --- Barras de probabilidad del gate ---
const barFills = {};
const barValues = {};

function buildBars() {
  for (const name of CLS_NAMES) {
    const row = document.createElement("li");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${CLASS_LABELS[name]}</span>
      <div class="bar-track"><div class="bar-fill" style="background: var(--color-${name})"></div></div>
      <span class="bar-value">0%</span>
    `;
    barsEl.appendChild(row);
    barFills[name] = row.querySelector(".bar-fill");
    barValues[name] = row.querySelector(".bar-value");
  }
}

function updateBars(probs) {
  CLS_NAMES.forEach((name, i) => {
    const pct = probs[i] * 100;
    barFills[name].style.width = `${pct}%`;
    barValues[name].textContent = `${pct.toFixed(1)}%`;
  });
}

// --- Conteo por clase (detector) ---
const countNums = {};

function buildCounts() {
  for (const name of DET_NAMES) {
    const cell = document.createElement("div");
    cell.className = "count-cell";
    cell.innerHTML = `
      <span class="count-num" style="color: var(--color-${name})">0</span>
      <span class="count-label">${CLASS_LABELS[name]}</span>
    `;
    countsEl.appendChild(cell);
    countNums[name] = cell.querySelector(".count-num");
  }
}

function updateCounts(dets) {
  const tally = Object.fromEntries(DET_NAMES.map((n) => [n, 0]));
  for (const d of dets) tally[d.name]++;
  for (const name of DET_NAMES) countNums[name].textContent = tally[name];
}

// --- Lista de detecciones activas ---
function updateDetList(dets) {
  detList.innerHTML = "";
  if (!dets.length) {
    detList.innerHTML = '<li class="det-empty">— sin detecciones —</li>';
    return;
  }
  for (const d of dets.slice(0, 8)) {
    const row = document.createElement("li");
    row.className = "det-row";
    row.innerHTML = `
      <span class="pill-dot" style="background: var(--color-${d.name})"></span>
      <span class="det-name">${CLASS_LABELS[d.name]}</span>
      <span class="det-conf">${(d.score * 100).toFixed(1)}%</span>
    `;
    detList.appendChild(row);
  }
}

// --- Carga de los modelos ONNX ---
// `modelsReady` se resuelve cuando ambos intentos terminaron (éxito o
// error): la imagen estática espera aquí antes de analizarse.
let modelsReady;

async function loadModels() {
  // ONNX Runtime Web auto-hospedado en vendor/ort/ (sin CDN).
  // Se usa document.baseURI: el loader jsep se importa como módulo ES y
  // las rutas relativas se resolverían contra la carpeta del bundle.
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;

  const jobs = [
    ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] })
      .then((s) => {
        clsSession = s;
        clsInput = s.inputNames[0];
        clsOutput = s.outputNames[0];
        setPill(badgeCls, "ready");
      })
      .catch((err) => {
        console.error("gate:", err);
        setPill(badgeCls, "error");
        statusEl.textContent = "No se pudo cargar model.onnx: " + err.message;
      }),
    ort.InferenceSession.create("detector.onnx", { executionProviders: ["wasm"] })
      .then((s) => {
        detSession = s;
        detInput = s.inputNames[0];
        detOutput = s.outputNames[0];
        setPill(badgeDet, "ready");
      })
      .catch((err) => {
        console.error("detector:", err);
        setPill(badgeDet, "error");
        statusEl.textContent = "No se pudo cargar detector.onnx: " + err.message;
      }),
  ];
  await Promise.all(jobs);
}

// runTick se omite silenciosamente si el gate aún no cargó; para
// fuentes estáticas (sin ticks periódicos) hay que esperar a los modelos.
function runTickWhenReady(source, w, h) {
  if (clsSession) {
    runTick(source, w, h);
  } else {
    modelsReady.then(() => runTick(source, w, h));
  }
}

// --- Preprocesamiento ---
// Un solo draw del recorte central cuadrado a 256×256 alimenta los dos
// tensores (cada modelo normaliza distinto).
function drawRegionToCanvas(source, sx, sy, side) {
  preprocessCtx.drawImage(source, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE);
  return preprocessCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
}

// Gate: grayscale (luminosidad) replicado a 3 canales, norm. ImageNet, CHW.
function pixelsToClsTensor(pixels, target) {
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    target[i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    target[plane + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    target[2 * plane + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
}

// Detector: RGB /255, CHW (preproceso de ultralytics).
function pixelsToDetTensor(pixels, target) {
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    target[i] = pixels[i * 4] / 255;
    target[plane + i] = pixels[i * 4 + 1] / 255;
    target[2 * plane + i] = pixels[i * 4 + 2] / 255;
  }
}

function preprocess(source, srcWidth, srcHeight) {
  // Recorte central cuadrado, como imagen completa encuadrada.
  const side = Math.min(srcWidth, srcHeight);
  const sx = (srcWidth - side) / 2;
  const sy = (srcHeight - side) / 2;

  const pixels = drawRegionToCanvas(source, sx, sy, side);
  const clsData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const detData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  pixelsToClsTensor(pixels, clsData);
  pixelsToDetTensor(pixels, detData);

  return {
    clsTensor: new ort.Tensor("float32", clsData, [1, 3, IMG_SIZE, IMG_SIZE]),
    detTensor: new ort.Tensor("float32", detData, [1, 3, IMG_SIZE, IMG_SIZE]),
    crop: { sx, sy, side },
  };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// --- Decode del detector (YOLOv8 [1, 4+nc, N]) + NMS en JS ---
function decodeDetections(tensor) {
  const [, channels, n] = tensor.dims; // [1, 8, N]
  const d = tensor.data;
  const dets = [];
  for (let i = 0; i < n; i++) {
    let bestCls = 0;
    let bestScore = 0;
    for (let c = 0; c < channels - 4; c++) {
      const s = d[(4 + c) * n + i];
      if (s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }
    if (bestScore < DET_THRESHOLD) continue;
    const cx = d[i];
    const cy = d[n + i];
    const w = d[2 * n + i];
    const h = d[3 * n + i];
    dets.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      score: bestScore,
      cls: bestCls,
    });
  }
  return dets;
}

function iou(a, b) {
  const xx1 = Math.max(a.x1, b.x1);
  const yy1 = Math.max(a.y1, b.y1);
  const xx2 = Math.min(a.x2, b.x2);
  const yy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(dets) {
  const order = [...dets].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of order) {
    if (keep.every((k) => k.cls !== d.cls || iou(k, d) <= NMS_IOU)) keep.push(d);
  }
  return keep;
}

// De coordenadas del tensor 256×256 a coordenadas del frame fuente.
function toSourceCoords(d, crop) {
  const scale = crop.side / IMG_SIZE;
  return {
    x1: crop.sx + d.x1 * scale,
    y1: crop.sy + d.y1 * scale,
    x2: crop.sx + d.x2 * scale,
    y2: crop.sy + d.y2 * scale,
    score: d.score,
    cls: d.cls,
    name: DET_NAMES[d.cls],
  };
}

// Seguimiento temporal ligero: empareja por clase + IoU y suaviza
// coords/score; las no emparejadas sobreviven un tick con decay.
function track(dets) {
  const used = new Set();
  const out = dets.map((d) => {
    let best = -1;
    let bestIou = 0.3; // umbral de emparejamiento
    prevDets.forEach((p, j) => {
      if (used.has(j) || p.name !== d.name) return;
      const v = iou(d, p);
      if (v > bestIou) {
        bestIou = v;
        best = j;
      }
    });
    if (best >= 0) {
      used.add(best);
      const p = prevDets[best];
      const mix = (a, b) => 0.5 * a + 0.5 * b;
      return {
        x1: mix(d.x1, p.x1), y1: mix(d.y1, p.y1),
        x2: mix(d.x2, p.x2), y2: mix(d.y2, p.y2),
        score: mix(d.score, p.score), cls: d.cls, name: d.name, misses: 0,
      };
    }
    return { ...d, misses: 0 };
  });
  // Retener una vez las detecciones que el detector "parpadeó".
  prevDets.forEach((p, j) => {
    if (used.has(j) || p.misses > 0) return;
    const decayed = { ...p, score: p.score * 0.6, misses: 1 };
    if (decayed.score >= 0.25) out.push(decayed);
  });
  prevDets = out;
  return out;
}

function resetTracker() {
  prevDets = [];
}

// --- Publicación en la UI ---
function showResults(probs, gateConfident, dets) {
  const bestIdx = probs.indexOf(Math.max(...probs));
  const gateName = CLS_NAMES[bestIdx];
  const gateConf = probs[bestIdx];

  let detColor = "var(--color-oos)";
  let label = "Indefinido";
  let confText = `${(gateConf * 100).toFixed(1)}%`;

  if (gateConfident && dets.length) {
    const top = dets[0]; // vienen ordenados por score tras NMS+track
    detColor = `var(--color-${top.name})`;
    label = CLASS_LABELS[top.name];
    confText = `${(top.score * 100).toFixed(1)}%`;
  } else if (gateConfident) {
    detColor = "var(--color-normal)";
    label = "Normal — sin defectos";
    confText = `${(gateConf * 100).toFixed(1)}%`;
  }

  // Marco del wrapper en el color de estado (gris si fuera de dominio).
  videoWrapper.style.setProperty("--det-color", detColor);
  videoWrapper.classList.add("detecting");

  chip.hidden = false;
  chipClass.textContent = label;
  chipClass.style.color = detColor;
  chipConf.textContent = confText;

  // Contador de defectos.
  const showCount = gateConfident && dets.length > 0;
  detCount.hidden = !showCount;
  if (showCount) {
    detCount.textContent = `${dets.length} defecto${dets.length > 1 ? "s" : ""}`;
  }

  // Estado del sistema.
  sysState.classList.remove("is-idle", "is-in", "is-out");
  if (gateConfident) {
    sysState.classList.add("is-in");
    sysStateText.textContent = "En dominio — inspeccionando";
  } else {
    sysState.classList.add("is-out");
    sysStateText.textContent = "Fuera de dominio";
  }

  updateBars(probs);
  updateCounts(gateConfident ? dets : []);
  updateDetList(gateConfident ? dets : []);
}

// --- Dibujo de cajas (esquinas tipo target, estilo HMI) ---
function drawDetections(srcWidth, srcHeight, dets) {
  if (!dets.length) {
    detectCtx.clearRect(0, 0, detectOverlay.width, detectOverlay.height);
    detectOverlay.hidden = true;
    return;
  }
  if (detectOverlay.width !== srcWidth || detectOverlay.height !== srcHeight) {
    detectOverlay.width = srcWidth;
    detectOverlay.height = srcHeight;
  }
  detectOverlay.hidden = false;
  const ctx = detectCtx;
  ctx.clearRect(0, 0, srcWidth, srcHeight);

  const fontPx = Math.max(12, Math.round(srcHeight / 26));
  ctx.font = `600 ${fontPx}px ui-monospace, Consolas, monospace`;
  ctx.lineWidth = Math.max(2, Math.round(srcHeight / 300));

  for (const d of dets) {
    const [r, g, b] = CLASS_RGB[d.name];
    const color = `rgb(${r} ${g} ${b})`;
    const w = d.x2 - d.x1;
    const h = d.y2 - d.y1;
    const arm = Math.max(10, Math.min(w, h) * 0.3); // largo de esquina

    ctx.strokeStyle = color;
    ctx.beginPath(); // 4 esquinas tipo target
    ctx.moveTo(d.x1, d.y1 + arm); ctx.lineTo(d.x1, d.y1); ctx.lineTo(d.x1 + arm, d.y1);
    ctx.moveTo(d.x2 - arm, d.y1); ctx.lineTo(d.x2, d.y1); ctx.lineTo(d.x2, d.y1 + arm);
    ctx.moveTo(d.x2, d.y2 - arm); ctx.lineTo(d.x2, d.y2); ctx.lineTo(d.x2 - arm, d.y2);
    ctx.moveTo(d.x1 + arm, d.y2); ctx.lineTo(d.x1, d.y2); ctx.lineTo(d.x1, d.y2 - arm);
    ctx.stroke();

    // Etiqueta "Clase 92%" sobre la esquina superior izquierda.
    const text = `${CLASS_LABELS[d.name]} ${(d.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width;
    const pad = fontPx * 0.35;
    const tx = d.x1;
    const ty = d.y1 - fontPx - pad * 2 >= 0 ? d.y1 - fontPx - pad * 2 : d.y1;
    ctx.fillStyle = "rgba(8, 11, 16, 0.85)";
    ctx.fillRect(tx, ty, tw + pad * 2, fontPx + pad * 2);
    ctx.fillStyle = color;
    ctx.fillText(text, tx + pad, ty + pad + fontPx * 0.78);
  }
}

// --- Detección de bordes subordinada al detector ---
// Sobel sobre el frame completo, pero SOLO se pintan los píxeles de
// borde que caen dentro de una caja detectada, en el color de su clase.
// Sin detecciones (o fuera de dominio) → overlay limpio: el sistema no
// "detecta todo", solo muestra las clases entrenadas.
const EDGES_WIDTH = 384;
const EDGE_THRESHOLD = 80; // magnitud Sobel mínima (0-1020)
const edgeSrc = document.createElement("canvas");
const edgeSrcCtx = edgeSrc.getContext("2d", { willReadFrequently: true });
let edgeGray = null; // buffers reutilizados (sin allocations por tick)
let edgeMag = null;
let edgeOut = null;

function updateEdges(source, srcWidth, srcHeight, gateConfident, dets) {
  if (!edgesOn || !gateConfident || !dets.length) {
    edgesCtx.clearRect(0, 0, edgesOverlay.width, edgesOverlay.height);
    edgesOverlay.hidden = true;
    return;
  }

  const aw = EDGES_WIDTH;
  const ah = Math.max(1, Math.round((EDGES_WIDTH * srcHeight) / srcWidth));
  if (edgeSrc.width !== aw || edgeSrc.height !== ah) {
    edgeSrc.width = aw;
    edgeSrc.height = ah;
    edgesOverlay.width = aw;
    edgesOverlay.height = ah;
    edgeGray = new Float32Array(aw * ah);
    edgeMag = new Float32Array(aw * ah);
    edgeOut = edgesCtx.createImageData(aw, ah);
  }

  edgeSrcCtx.drawImage(source, 0, 0, aw, ah);
  const sd = edgeSrcCtx.getImageData(0, 0, aw, ah).data;
  const g = edgeGray;
  const mag = edgeMag;
  const od = edgeOut.data;

  for (let i = 0; i < aw * ah; i++) {
    g[i] = 0.299 * sd[i * 4] + 0.587 * sd[i * 4 + 1] + 0.114 * sd[i * 4 + 2];
  }
  mag.fill(0);
  for (let y = 1; y < ah - 1; y++) {
    for (let x = 1; x < aw - 1; x++) {
      const i = y * aw + x;
      const gx =
        -g[i - aw - 1] - 2 * g[i - 1] - g[i + aw - 1] +
        g[i - aw + 1] + 2 * g[i + 1] + g[i + aw + 1];
      const gy =
        -g[i - aw - 1] - 2 * g[i - aw] - g[i - aw + 1] +
        g[i + aw - 1] + 2 * g[i + aw] + g[i + aw + 1];
      mag[i] = Math.abs(gx) + Math.abs(gy);
    }
  }

  // Pintar solo dentro de cada caja, con el color de su clase.
  const sx = aw / srcWidth;
  const sy = ah / srcHeight;
  od.fill(0);
  for (const d of dets) {
    const [r, gg, bb] = CLASS_RGB[d.name];
    const bx1 = Math.max(1, Math.floor(d.x1 * sx));
    const by1 = Math.max(1, Math.floor(d.y1 * sy));
    const bx2 = Math.min(aw - 1, Math.ceil(d.x2 * sx));
    const by2 = Math.min(ah - 1, Math.ceil(d.y2 * sy));
    for (let y = by1; y < by2; y++) {
      for (let x = bx1; x < bx2; x++) {
        const i = y * aw + x;
        if (mag[i] > EDGE_THRESHOLD) {
          const j = i * 4;
          od[j] = r;
          od[j + 1] = gg;
          od[j + 2] = bb;
          od[j + 3] = 255;
        }
      }
    }
  }
  edgesOverlay.hidden = false;
  edgesCtx.putImageData(edgeOut, 0, 0);
}

function hasSource() {
  // Hay algo que analizar: stream de cámara o imagen subida visible.
  return !!cameraStream || !overlay.hidden;
}

function setEdges(on) {
  edgesOn = on && hasSource();
  btnEdges.disabled = !hasSource();
  btnEdges.classList.toggle("is-on", edgesOn);
  btnEdges.textContent = edgesOn ? "Ocultar bordes" : "Bordes por detección";
  if (!edgesOn) {
    edgesCtx.clearRect(0, 0, edgesOverlay.width, edgesOverlay.height);
    edgesOverlay.hidden = true;
  } else if (!cameraStream && lastStaticSource) {
    // Imagen estática: no hay ticks periódicos, re-analizar una vez
    // para pintar (o limpiar) los bordes de inmediato.
    runTickWhenReady(lastStaticSource.img, lastStaticSource.w, lastStaticSource.h);
  }
}

// --- Tick de inferencia (gate → detector → UI) ---
async function runTick(source, srcWidth, srcHeight) {
  // Guard anti-solapamiento: si el tick anterior no terminó, se omite
  // (evita ejecuciones concurrentes acumuladas).
  if (!clsSession || inferenceInFlight) return;
  inferenceInFlight = true;

  try {
    const { clsTensor, detTensor, crop } = preprocess(source, srcWidth, srcHeight);

    // 1) Gate de dominio.
    const t0 = performance.now();
    const clsOut = await clsSession.run({ [clsInput]: clsTensor });
    const clsMs = performance.now() - t0;

    const probs = softmax(Array.from(clsOut[clsOutput].data));
    smoothProbs = smoothProbs
      ? probs.map((p, i) => SMOOTHING * p + (1 - SMOOTHING) * smoothProbs[i])
      : probs;
    const gateConfident = Math.max(...smoothProbs) >= GATE_THRESHOLD;

    // 2) Detector (solo en dominio).
    let dets = [];
    let detMs = null;
    if (gateConfident && detSession) {
      const t1 = performance.now();
      const detOut = await detSession.run({ [detInput]: detTensor });
      detMs = performance.now() - t1;
      dets = nms(decodeDetections(detOut[detOutput]))
        .sort((a, b) => b.score - a.score)
        .map((d) => toSourceCoords(d, crop));
      dets = track(dets);
    } else if (!gateConfident) {
      resetTracker();
    }
    const shown = gateConfident ? dets : [];
    activeDets = shown;

    showResults(smoothProbs, gateConfident, shown);
    drawDetections(srcWidth, srcHeight, shown);
    updateEdges(source, srcWidth, srcHeight, gateConfident, shown);

    latencyClsEl.textContent = `${clsMs.toFixed(0)}`;
    if (detMs !== null) latencyDetEl.textContent = `${detMs.toFixed(0)}`;
    else latencyDetEl.textContent = "—";

    frameCount++;
    const now = performance.now();
    if (now - fpsWindowStart >= 1000) {
      fpsEl.textContent = ((frameCount * 1000) / (now - fpsWindowStart)).toFixed(1);
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
  lastStaticSource = null;
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
  resetTracker();
  roiGuide.hidden = false;
  layoutRoi();

  statusEl.textContent = "Cámara activa — inspeccionando en vivo.";
  btnCamera.textContent = "Detener cámara";
  btnCamera.disabled = false;
  btnEdges.disabled = false;
  setPill(badgeCam, "ready");
  sysState.classList.remove("is-idle");
  sysState.classList.add("is-out");
  sysStateText.textContent = "Fuera de dominio";

  clearInterval(tickTimer);
  tickTimer = setInterval(() => {
    if (video.readyState >= 2) {
      runTick(video, video.videoWidth, video.videoHeight);
    }
  }, TICK_MS);

  // Gancho de prueba automatizada: ?edges=1 activa la vista de bordes.
  if (new URLSearchParams(location.search).has("edges")) {
    setEdges(true);
  }
}

function stopCamera() {
  clearInterval(tickTimer);
  tickTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  // Reset de la vista de bordes (requiere cámara activa).
  edgesOn = false;
  edgesOverlay.hidden = true;
  edgesCtx.clearRect(0, 0, edgesOverlay.width, edgesOverlay.height);
  btnEdges.disabled = true;
  btnEdges.classList.remove("is-on");
  btnEdges.textContent = "Bordes por detección";
  detectOverlay.hidden = true;
  detectCtx.clearRect(0, 0, detectOverlay.width, detectOverlay.height);
  video.srcObject = null;
  placeholder.style.display = "flex";
  chip.hidden = true;
  detCount.hidden = true;
  roiGuide.hidden = true;
  smoothProbs = null;
  resetTracker();
  activeDets = [];
  // Quitar el marco de reconocimiento.
  videoWrapper.classList.remove("detecting");
  videoWrapper.style.removeProperty("--det-color");
  statusEl.textContent = "Cámara detenida.";
  btnCamera.textContent = "Activar cámara";
  setPill(badgeCam, "idle");
  sysState.classList.remove("is-in", "is-out");
  sysState.classList.add("is-idle");
  sysStateText.textContent = "Esperando entrada";
  fpsEl.textContent = "—";
  latencyClsEl.textContent = "—";
  latencyDetEl.textContent = "—";
}

// --- Subir imagen (fallback sin cámara) ---
function stopUploadedImage() {
  overlay.hidden = true;
}

// Muestra y analiza una imagen estática (misma ruta para upload y para
// el gancho de prueba ?imgtest=).
function analyzeImage(img, name) {
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
  resetTracker();
  roiGuide.hidden = false;
  layoutRoi();
  lastStaticSource = { img, w: img.width, h: img.height };
  btnEdges.disabled = false;

  runTickWhenReady(img, img.width, img.height);
  statusEl.textContent = "Imagen analizada: " + name;
}

function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    analyzeImage(img, file.name);
    URL.revokeObjectURL(url);
  };
  img.src = url;
  fileInput.value = "";
}

// --- Init ---
buildBars();
buildCounts();
modelsReady = loadModels();
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
const urlParams = new URLSearchParams(location.search);
if (urlParams.has("autocam")) {
  startCamera();
}

// Gancho de prueba: ?imgtest=<ruta> analiza una imagen servida por el
// propio host (p. ej. capturas del split test), sin cámara ni diálogo.
// Se puede combinar con &edges=1 para activar la vista de bordes.
const imgtest = urlParams.get("imgtest");
if (imgtest) {
  const img = new Image();
  img.onload = () => {
    analyzeImage(img, imgtest);
    if (urlParams.has("edges")) setEdges(true);
  };
  img.src = imgtest;
}
