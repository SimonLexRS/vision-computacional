/**
 * Demo web: detección individual de defectos en superficies metálicas.
 *
 * Pipeline en dos etapas, 100% en el navegador (ONNX Runtime Web, WASM):
 *
 *   1. GATE DE DOMINIO — MobileNetV3-Small (model.onnx), clasificador de
 *      5 clases (crack, hole, normal, rust, scratch).
 *
 *   2. DETECTOR — YOLOv8n (detector.onnx), 4 clases de defecto
 *      (crack, hole, rust, scratch).
 *
 *   3. LUPA / ZOOM DE ALTA DEFINICIÓN:
 *      - Permite inspeccionar defectos pequeños/milimétricos (1x, 1.5x, 2x, 3x)
 *        mediante zoom de cámara y sub-región de alta densidad de píxeles.
 *
 *   4. MODO MAXIMIZADO INMERSIVO:
 *      - Vista en pantalla completa con controles HUD integrados.
 *
 *   5. DETECCIÓN ROBUSTA DE SUPERFICIES NORMALES Y FILTROS ANTI-RUIDO.
 */

"use strict";

const IMG_SIZE = 256;
const CLS_NAMES = ["crack", "hole", "normal", "rust", "scratch"];
const DET_NAMES = ["crack", "hole", "rust", "scratch"];
const CLASS_LABELS = {
  crack: "Grieta",
  hole: "Perforación",
  normal: "Normal",
  rust: "Óxido",
  scratch: "Rayón",
};
const CLASS_RGB = {
  crack: [255, 92, 92],
  hole: [255, 159, 67],
  normal: [52, 199, 123],
  rust: [201, 123, 47],
  scratch: [182, 120, 255],
};
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// --- Presets y Umbrales Calibrados ---
const PRESETS = {
  balanced: {
    detThr: 0.35,
    rustThr: 0.50,
    minLight: 20,
    maxSat: 0.42,
    gateHi: 0.85,
    name: "Equilibrado",
  },
  zoom_macro: {
    detThr: 0.30,
    rustThr: 0.45,
    minLight: 15,
    maxSat: 0.45,
    gateHi: 0.80,
    name: "Macro / Defectos Pequeños",
  },
  night: {
    detThr: 0.45,
    rustThr: 0.65,
    minLight: 15,
    maxSat: 0.35,
    gateHi: 0.90,
    name: "Nocturno / Poca Luz",
  },
  high_prec: {
    detThr: 0.55,
    rustThr: 0.70,
    minLight: 25,
    maxSat: 0.30,
    gateHi: 0.94,
    name: "Alta Precisión",
  },
  sensitive: {
    detThr: 0.25,
    rustThr: 0.35,
    minLight: 15,
    maxSat: 0.50,
    gateHi: 0.75,
    name: "Ultra Sensible",
  },
};

let DET_THRESHOLD = 0.35;  // Score mínimo general para cajas YOLO (permite detectar defectos pequeños)
let RUST_THRESHOLD = 0.50; // Score específico para óxido (filtra ruido oscuro de sensor)
let MIN_LIGHT = 20;        // Luminancia mínima para descartar escenas oscuras (0 - 70)
let MAX_SAT = 0.42;        // Saturación máxima (permite tonos cálidos en metal, rechaza vegetación)
let GATE_HI = 0.85;        // Umbral para entrar a dominio
let GATE_LO = 0.72;        // Umbral para salir de dominio
let ROI_SCALE = 0.95;      // Escala del área de análisis
let currentZoom = 1.0;     // Nivel de zoom activo (1.0, 1.5, 2.0, 3.0)
const NMS_IOU = 0.45;
const MAX_BOXES = 8;

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

// Elementos de calibración y umbrales
const sliderDetThr = document.getElementById("slider-det-thr");
const valDetThr = document.getElementById("val-det-thr");
const sliderRustThr = document.getElementById("slider-rust-thr");
const valRustThr = document.getElementById("val-rust-thr");
const sliderMinLight = document.getElementById("slider-min-light");
const valMinLight = document.getElementById("val-min-light");
const sliderMaxSat = document.getElementById("slider-max-sat");
const valMaxSat = document.getElementById("val-max-sat");
const sliderGateHi = document.getElementById("slider-gate-hi");
const valGateHi = document.getElementById("val-gate-hi");
const selectRoiSize = document.getElementById("select-roi-size");
const btnResetThresholds = document.getElementById("btn-reset-thresholds");
const presetPills = document.querySelectorAll(".preset-pill");

const fillLight = document.getElementById("fill-light");
const valCurLight = document.getElementById("val-cur-light");
const fillSat = document.getElementById("fill-sat");
const valCurSat = document.getElementById("val-cur-sat");

// Elementos de HUD y Zoom
const zoomToolbar = document.getElementById("zoom-toolbar");
const zoomPills = document.querySelectorAll(".zoom-pill");
const btnHudClose = document.getElementById("btn-hud-close");
const hudBottomBar = document.getElementById("hud-bottom-bar");
const btnHudEdges = document.getElementById("btn-hud-edges");
const btnHudCamera = document.getElementById("btn-hud-camera");
const hudMetrics = document.getElementById("hud-metrics");

let clsSession = null;
let detSession = null;
let clsInput = "input";
let clsOutput = "logits";
let detInput = "images";
let detOutput = "output0";
let cameraStream = null;
let isStreaming = false;
let videoFrameCallbackHandle = null;
let animFrameHandle = null;
let inferenceInFlight = false;
let isMaximized = false;
let frameCount = 0;
let fpsWindowStart = performance.now();
let lastFpsVal = "—";

let smoothProbs = null;
const SMOOTHING = 0.35;
let activeDets = [];
let prevDets = [];
let edgesOn = false;
let inDomain = false;
let lastStaticSource = null;

// --- Ajuste de ROI y Visor ---
function layoutRoi() {
  const side = Math.min(videoWrapper.clientWidth, videoWrapper.clientHeight) * ROI_SCALE;
  roiGuide.style.width = `${Math.round(side)}px`;
  roiGuide.style.height = `${Math.round(side)}px`;
}

window.addEventListener("resize", layoutRoi);

function syncVideoLayout() {
  if (!video.videoWidth || !video.videoHeight) return;
  if (!isMaximized) {
    videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  videoWrapper.style.setProperty("--video-ar", video.videoWidth / video.videoHeight);
  layoutRoi();
}
video.addEventListener("resize", syncVideoLayout);
window.addEventListener("orientationchange", () => setTimeout(syncVideoLayout, 250));

// --- Control de Zoom (Lupa para defectos pequeños) ---
function setZoom(level) {
  currentZoom = parseFloat(level) || 1.0;
  zoomPills.forEach((p) => {
    p.classList.toggle("active", parseFloat(p.dataset.zoom) === currentZoom);
  });

  // Intentar aplicar zoom óptico/hardware si la cámara lo soporta
  if (cameraStream) {
    const track = cameraStream.getVideoTracks()[0];
    if (track) {
      const caps = track.getCapabilities ? track.getCapabilities() : {};
      if (caps.zoom) {
        const minZ = caps.zoom.min || 1;
        const maxZ = caps.zoom.max || 3;
        const targetZ = Math.min(maxZ, Math.max(minZ, currentZoom));
        track.applyConstraints({ advanced: [{ zoom: targetZ }] }).catch(() => {});
      }
    }
  }

  // Si hay imagen estática, reanalizar con el nuevo zoom
  if (!cameraStream && lastStaticSource) {
    runTickWhenReady(lastStaticSource.img, lastStaticSource.w, lastStaticSource.h);
  }
}

function initZoomEvents() {
  zoomPills.forEach((p) => {
    p.addEventListener("click", () => setZoom(p.dataset.zoom));
  });
}

// --- Modo Maximizado / Pantalla Completa Inmersivo ---
function toggleMaximizedMode(force) {
  isMaximized = typeof force === "boolean" ? force : !isMaximized;
  cameraPanel.classList.toggle("is-maximized", isMaximized);

  if (isMaximized) {
    btnFullscreen.textContent = "⛶ Salir de maximizado";
    btnFullscreen.classList.add("is-on");
    btnHudClose.hidden = false;
    hudBottomBar.hidden = false;
    zoomToolbar.hidden = !hasSource();
    syncVideoLayout();
  } else {
    btnFullscreen.textContent = "⛶ Modo Maximizado";
    btnFullscreen.classList.remove("is-on");
    btnHudClose.hidden = true;
    hudBottomBar.hidden = true;
    syncVideoLayout();
  }
}

// --- Persistencia y Sincronización de Umbrales ---
function loadSavedSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("inspector_settings") || "{}");
    if (saved.detThr !== undefined) DET_THRESHOLD = saved.detThr;
    if (saved.rustThr !== undefined) RUST_THRESHOLD = saved.rustThr;
    if (saved.minLight !== undefined) MIN_LIGHT = saved.minLight;
    if (saved.maxSat !== undefined) MAX_SAT = saved.maxSat;
    if (saved.gateHi !== undefined) {
      GATE_HI = saved.gateHi;
      GATE_LO = Math.max(0.60, GATE_HI - 0.12);
    }
    if (saved.roiScale !== undefined) ROI_SCALE = saved.roiScale;
  } catch (e) {
    console.warn("No se pudieron cargar los ajustes guardados:", e);
  }
  syncControlsUI();
}

function saveSettings() {
  try {
    const settings = {
      detThr: DET_THRESHOLD,
      rustThr: RUST_THRESHOLD,
      minLight: MIN_LIGHT,
      maxSat: MAX_SAT,
      gateHi: GATE_HI,
      roiScale: ROI_SCALE,
    };
    localStorage.setItem("inspector_settings", JSON.stringify(settings));
  } catch (e) {
    console.warn("No se pudieron guardar los ajustes:", e);
  }
}

function syncControlsUI() {
  if (sliderDetThr) {
    sliderDetThr.value = Math.round(DET_THRESHOLD * 100);
    valDetThr.textContent = `${Math.round(DET_THRESHOLD * 100)}%`;
  }
  if (sliderRustThr) {
    sliderRustThr.value = Math.round(RUST_THRESHOLD * 100);
    valRustThr.textContent = `${Math.round(RUST_THRESHOLD * 100)}%`;
  }
  if (sliderMinLight) {
    sliderMinLight.value = Math.round(MIN_LIGHT);
    valMinLight.textContent = `${Math.round(MIN_LIGHT)}`;
  }
  if (sliderMaxSat) {
    sliderMaxSat.value = Math.round(MAX_SAT * 100);
    valMaxSat.textContent = `${Math.round(MAX_SAT * 100)}%`;
  }
  if (sliderGateHi) {
    sliderGateHi.value = Math.round(GATE_HI * 100);
    valGateHi.textContent = `${Math.round(GATE_HI * 100)}%`;
  }
  if (selectRoiSize) {
    selectRoiSize.value = ROI_SCALE.toFixed(2);
  }
  layoutRoi();
}

function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;
  DET_THRESHOLD = p.detThr;
  RUST_THRESHOLD = p.rustThr;
  MIN_LIGHT = p.minLight;
  MAX_SAT = p.maxSat;
  GATE_HI = p.gateHi;
  GATE_LO = Math.max(0.60, GATE_HI - 0.12);

  presetPills.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === presetKey);
  });

  syncControlsUI();
  saveSettings();

  if (!cameraStream && lastStaticSource) {
    runTickWhenReady(lastStaticSource.img, lastStaticSource.w, lastStaticSource.h);
  }
}

function initThresholdEvents() {
  if (sliderDetThr) {
    sliderDetThr.addEventListener("input", (e) => {
      DET_THRESHOLD = Number(e.target.value) / 100;
      valDetThr.textContent = `${e.target.value}%`;
      presetPills.forEach((p) => p.classList.remove("active"));
      saveSettings();
    });
  }
  if (sliderRustThr) {
    sliderRustThr.addEventListener("input", (e) => {
      RUST_THRESHOLD = Number(e.target.value) / 100;
      valRustThr.textContent = `${e.target.value}%`;
      presetPills.forEach((p) => p.classList.remove("active"));
      saveSettings();
    });
  }
  if (sliderMinLight) {
    sliderMinLight.addEventListener("input", (e) => {
      MIN_LIGHT = Number(e.target.value);
      valMinLight.textContent = `${e.target.value}`;
      presetPills.forEach((p) => p.classList.remove("active"));
      saveSettings();
    });
  }
  if (sliderMaxSat) {
    sliderMaxSat.addEventListener("input", (e) => {
      MAX_SAT = Number(e.target.value) / 100;
      valMaxSat.textContent = `${e.target.value}%`;
      presetPills.forEach((p) => p.classList.remove("active"));
      saveSettings();
    });
  }
  if (sliderGateHi) {
    sliderGateHi.addEventListener("input", (e) => {
      GATE_HI = Number(e.target.value) / 100;
      GATE_LO = Math.max(0.60, GATE_HI - 0.12);
      valGateHi.textContent = `${e.target.value}%`;
      presetPills.forEach((p) => p.classList.remove("active"));
      saveSettings();
    });
  }
  if (selectRoiSize) {
    selectRoiSize.addEventListener("change", (e) => {
      ROI_SCALE = parseFloat(e.target.value);
      layoutRoi();
      saveSettings();
    });
  }
  if (btnResetThresholds) {
    btnResetThresholds.addEventListener("click", () => applyPreset("balanced"));
  }
  presetPills.forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });
}

// --- Pills de estado ---
function setPill(el, state) {
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
    const pct = (probs ? probs[i] : 0) * 100;
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
  for (const d of dets.slice(0, MAX_BOXES)) {
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
let modelsReady;

async function loadModels() {
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
  const threads = Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.wasm.numThreads = threads;

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

function runTickWhenReady(source, w, h) {
  if (clsSession) {
    runTick(source, w, h);
  } else {
    modelsReady.then(() => runTick(source, w, h));
  }
}

// --- Preprocesamiento con Zoom de Alta Definición ---
function drawRegionToCanvas(source, sx, sy, side) {
  preprocessCtx.drawImage(source, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE);
  return preprocessCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
}

function analyzeSceneQuality(pixels) {
  const n = IMG_SIZE * IMG_SIZE;
  let sumY = 0;
  let sumSat = 0;

  for (let i = 0; i < n; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];

    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    sumY += y;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const sat = max > 0 ? delta / max : 0;
    sumSat += sat;
  }

  const meanY = sumY / n;
  const meanSat = sumSat / n;

  if (fillLight && valCurLight) {
    const lightPct = Math.min(100, Math.max(0, (meanY / 255) * 100));
    fillLight.style.width = `${lightPct.toFixed(0)}%`;
    valCurLight.textContent = `${meanY.toFixed(0)} / 255`;
    if (meanY < MIN_LIGHT) {
      fillLight.style.background = "var(--color-crack)";
    } else {
      fillLight.style.background = "var(--color-accent)";
    }
  }

  if (fillSat && valCurSat) {
    const satPct = Math.min(100, Math.max(0, meanSat * 100));
    fillSat.style.width = `${satPct.toFixed(0)}%`;
    valCurSat.textContent = `${satPct.toFixed(1)}%`;
    if (meanSat > MAX_SAT) {
      fillSat.style.background = "var(--color-hole)";
    } else {
      fillSat.style.background = "var(--color-normal)";
    }
  }

  const isTooDark = meanY < MIN_LIGHT;
  const isTooSaturated = meanSat > MAX_SAT;

  return { meanY, meanSat, isTooDark, isTooSaturated };
}

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

function pixelsToDetTensor(pixels, target) {
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    target[i] = pixels[i * 4] / 255;
    target[plane + i] = pixels[i * 4 + 1] / 255;
    target[2 * plane + i] = pixels[i * 4 + 2] / 255;
  }
}

function preprocess(source, srcWidth, srcHeight) {
  const baseSide = Math.min(srcWidth, srcHeight);
  // Con zoom activo, se recorta una sub-región más pequeña y se alimenta a 256x256,
  // multiplicando la resolución efectiva para detectar grietas o rayones delgados
  const zoomedSide = baseSide / currentZoom;
  const sx = (srcWidth - zoomedSide) / 2;
  const sy = (srcHeight - zoomedSide) / 2;

  const pixels = drawRegionToCanvas(source, sx, sy, zoomedSide);
  const clsData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const detData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  pixelsToClsTensor(pixels, clsData);
  pixelsToDetTensor(pixels, detData);

  return {
    clsTensor: new ort.Tensor("float32", clsData, [1, 3, IMG_SIZE, IMG_SIZE]),
    detTensor: new ort.Tensor("float32", detData, [1, 3, IMG_SIZE, IMG_SIZE]),
    crop: { sx, sy, side: zoomedSide },
    pixels,
  };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// --- Decode del detector (YOLOv8 [1, 4+nc, N]) ---
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

    const requiredThreshold = DET_NAMES[bestCls] === "rust" ? RUST_THRESHOLD : DET_THRESHOLD;
    if (bestScore < requiredThreshold) continue;

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
    if (keep.every((k) => iou(k, d) <= NMS_IOU)) keep.push(d);
  }
  return keep;
}

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

function track(dets) {
  const used = new Set();
  const out = dets.map((d) => {
    let best = -1;
    let bestIou = 0.3;
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
        x1: mix(d.x1, p.x1),
        y1: mix(d.y1, p.y1),
        x2: mix(d.x2, p.x2),
        y2: mix(d.y2, p.y2),
        score: mix(d.score, p.score),
        cls: d.cls,
        name: d.name,
        misses: 0,
      };
    }
    return { ...d, misses: 0 };
  });

  prevDets.forEach((p, j) => {
    if (used.has(j) || p.misses > 0) return;
    const decayed = { ...p, score: p.score * 0.6, misses: 1 };
    const thr = decayed.name === "rust" ? RUST_THRESHOLD * 0.5 : DET_THRESHOLD * 0.5;
    if (decayed.score >= thr) out.push(decayed);
  });
  prevDets = out;
  return out;
}

function resetTracker() {
  prevDets = [];
}

// --- Publicación en la UI y Detección de Superficies Normales ---
function showResults(probs, dets, oodReason) {
  const bestIdx = probs ? probs.indexOf(Math.max(...probs)) : 0;
  const gateTop = probs ? CLS_NAMES[bestIdx] : "normal";
  const gateConf = probs ? probs[bestIdx] : 0;

  let color = "var(--color-oos)";
  let label = oodReason || "Indefinido";
  let confText = probs ? `${(gateConf * 100).toFixed(1)}%` : "—";

  if (inDomain) {
    if (dets.length > 0) {
      const top = dets[0];
      color = `var(--color-${top.name})`;
      label = CLASS_LABELS[top.name];
      confText = `${(top.score * 100).toFixed(1)}%`;
    } else {
      // Si está en dominio y YOLO no detecta defectos, la superficie es Normal
      color = "var(--color-normal)";
      label = "Normal — sin defectos";
      confText = gateTop === "normal" ? `${(gateConf * 100).toFixed(1)}%` : "100%";
    }
  }

  chip.hidden = false;
  chipDot.style.background = color;
  chipClass.textContent = label;
  chipClass.style.color = color;
  chipConf.textContent = confText;

  const showCount = inDomain && dets.length > 0;
  detCount.hidden = !showCount;
  if (showCount) {
    detCount.textContent = `${dets.length} defecto${dets.length > 1 ? "s" : ""}`;
  }

  roiGuide.hidden = !hasSource() || dets.length > 0;

  sysState.classList.remove("is-idle", "is-in", "is-out");
  if (!inDomain) {
    sysState.classList.add("is-out");
    sysStateText.textContent = oodReason || "Fuera de dominio";
  } else if (dets.length > 0) {
    sysState.classList.add("is-in");
    sysStateText.textContent = "En dominio — inspeccionando";
  } else {
    sysState.classList.add("is-in");
    sysStateText.textContent = "En dominio — sin defectos";
  }

  updateBars(probs);
  updateCounts(inDomain ? dets : []);
  updateDetList(inDomain ? dets : []);
}

// --- Dibujo de cajas con estilo HMI ---
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
    const arm = Math.max(10, Math.min(w, h) * 0.3);

    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(d.x1, d.y1 + arm); ctx.lineTo(d.x1, d.y1); ctx.lineTo(d.x1 + arm, d.y1);
    ctx.moveTo(d.x2 - arm, d.y1); ctx.lineTo(d.x2, d.y1); ctx.lineTo(d.x2, d.y1 + arm);
    ctx.moveTo(d.x2, d.y2 - arm); ctx.lineTo(d.x2, d.y2); ctx.lineTo(d.x2 - arm, d.y2);
    ctx.moveTo(d.x1 + arm, d.y2); ctx.lineTo(d.x1, d.y2); ctx.lineTo(d.x1, d.y2 - arm);
    ctx.stroke();

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

// --- Detección de bordes (Sobel) ---
const EDGES_WIDTH = 384;
const EDGE_THRESHOLD = 80;
const edgeSrc = document.createElement("canvas");
const edgeSrcCtx = edgeSrc.getContext("2d", { willReadFrequently: true });
let edgeGray = null;
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
  return !!cameraStream || !overlay.hidden;
}

function setEdges(on) {
  edgesOn = on && hasSource();
  btnEdges.disabled = !hasSource();
  btnEdges.classList.toggle("is-on", edgesOn);
  btnEdges.textContent = edgesOn ? "Ocultar bordes" : "Bordes por detección";
  if (btnHudEdges) btnHudEdges.classList.toggle("is-on", edgesOn);

  if (!edgesOn) {
    edgesCtx.clearRect(0, 0, edgesOverlay.width, edgesOverlay.height);
    edgesOverlay.hidden = true;
  } else if (!cameraStream && lastStaticSource) {
    runTickWhenReady(lastStaticSource.img, lastStaticSource.w, lastStaticSource.h);
  }
}

// --- Tick de inferencia ---
async function runTick(source, srcWidth, srcHeight) {
  if (!clsSession || inferenceInFlight) return;
  inferenceInFlight = true;

  try {
    const { clsTensor, detTensor, crop, pixels } = preprocess(source, srcWidth, srcHeight);

    // 0) Validación de calidad de escena
    const quality = analyzeSceneQuality(pixels);

    if (quality.isTooDark) {
      inDomain = false;
      resetTracker();
      activeDets = [];
      showResults(null, [], "Poca luz (Anti-ruido nocturno activo)");
      drawDetections(srcWidth, srcHeight, []);
      updateEdges(source, srcWidth, srcHeight, false, []);
      latencyClsEl.textContent = "—";
      latencyDetEl.textContent = "—";
      return;
    }

    if (quality.isTooSaturated) {
      inDomain = false;
      resetTracker();
      activeDets = [];
      showResults(null, [], "Fuera de dominio (Color no metálico)");
      drawDetections(srcWidth, srcHeight, []);
      updateEdges(source, srcWidth, srcHeight, false, []);
      latencyClsEl.textContent = "—";
      latencyDetEl.textContent = "—";
      return;
    }

    // 1) Gate de dominio
    const t0 = performance.now();
    const clsOut = await clsSession.run({ [clsInput]: clsTensor });
    const clsMs = performance.now() - t0;

    const probs = softmax(Array.from(clsOut[clsOutput].data));
    smoothProbs = smoothProbs
      ? probs.map((p, i) => SMOOTHING * p + (1 - SMOOTHING) * smoothProbs[i])
      : probs;

    const gateMax = Math.max(...smoothProbs);
    const gateTop = CLS_NAMES[smoothProbs.indexOf(gateMax)];
    if (!inDomain && (gateMax >= GATE_HI || gateTop === "normal")) inDomain = true;
    else if (inDomain && gateMax < GATE_LO && gateTop !== "normal") inDomain = false;

    // 2) Detector YOLOv8n
    let dets = [];
    let detMs = null;
    if (inDomain && detSession) {
      const t1 = performance.now();
      const detOut = await detSession.run({ [detInput]: detTensor });
      detMs = performance.now() - t1;

      dets = nms(decodeDetections(detOut[detOutput]))
        .sort((a, b) => b.score - a.score)
        .map((d) => toSourceCoords(d, crop));

      // Si el gate detecta un defecto con alta certeza y el detector coincide, se filtra por clase;
      // si no hay cajas o el gate sugiere normal, se preserva el estado
      if (gateTop !== "normal" && gateMax >= GATE_HI) {
        dets = dets.filter((d) => d.name === gateTop);
      }
      dets = track(dets).slice(0, MAX_BOXES);
    } else if (!inDomain) {
      resetTracker();
    }
    const shown = inDomain ? dets : [];
    activeDets = shown;

    showResults(smoothProbs, shown, inDomain ? null : "Fuera de dominio");
    drawDetections(srcWidth, srcHeight, shown);
    updateEdges(source, srcWidth, srcHeight, inDomain, shown);

    latencyClsEl.textContent = `${clsMs.toFixed(0)}`;
    if (detMs !== null) latencyDetEl.textContent = `${detMs.toFixed(0)}`;
    else latencyDetEl.textContent = "—";

    frameCount++;
    const now = performance.now();
    if (now - fpsWindowStart >= 1000) {
      lastFpsVal = ((frameCount * 1000) / (now - fpsWindowStart)).toFixed(1);
      fpsEl.textContent = lastFpsVal;
      if (hudMetrics) hudMetrics.textContent = `FPS: ${lastFpsVal} · ${clsMs.toFixed(0)}+${detMs ? detMs.toFixed(0) : 0}ms`;
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

// --- Bucle continuo de alta velocidad (20-30+ FPS) ---
function scheduleNextFrame() {
  if (!isStreaming || !cameraStream) return;

  if ("requestVideoFrameCallback" in video) {
    videoFrameCallbackHandle = video.requestVideoFrameCallback(async () => {
      if (isStreaming && video.readyState >= 2) {
        await runTick(video, video.videoWidth, video.videoHeight);
      }
      scheduleNextFrame();
    });
  } else {
    animFrameHandle = requestAnimationFrame(async () => {
      if (isStreaming && video.readyState >= 2) {
        await runTick(video, video.videoWidth, video.videoHeight);
      }
      scheduleNextFrame();
    });
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
  zoomToolbar.hidden = false;
  await video.play();

  syncVideoLayout();
  smoothProbs = null;
  inDomain = false;
  resetTracker();
  roiGuide.hidden = false;
  layoutRoi();

  statusEl.textContent = "Cámara activa — inspección en vivo de alta velocidad.";
  btnCamera.textContent = "Detener cámara";
  btnCamera.disabled = false;
  btnEdges.disabled = false;
  if (btnHudCamera) btnHudCamera.textContent = "Detener";
  setPill(badgeCam, "ready");
  sysState.classList.remove("is-idle");
  sysState.classList.add("is-out");
  sysStateText.textContent = "Fuera de dominio";

  isStreaming = true;
  frameCount = 0;
  fpsWindowStart = performance.now();
  scheduleNextFrame();

  if (new URLSearchParams(location.search).has("edges")) {
    setEdges(true);
  }
}

function stopCamera() {
  isStreaming = false;
  if (videoFrameCallbackHandle && "cancelVideoFrameCallback" in video) {
    video.cancelVideoFrameCallback(videoFrameCallbackHandle);
    videoFrameCallbackHandle = null;
  }
  if (animFrameHandle) {
    cancelAnimationFrame(animFrameHandle);
    animFrameHandle = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
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
  zoomToolbar.hidden = true;
  smoothProbs = null;
  inDomain = false;
  resetTracker();
  activeDets = [];
  statusEl.textContent = "Cámara detenida.";
  btnCamera.textContent = "Activar cámara";
  if (btnHudCamera) btnHudCamera.textContent = "Cámara";
  setPill(badgeCam, "idle");
  sysState.classList.remove("is-in", "is-out");
  sysState.classList.add("is-idle");
  sysStateText.textContent = "Esperando entrada";
  fpsEl.textContent = "—";
  latencyClsEl.textContent = "—";
  latencyDetEl.textContent = "—";
}

// --- Subir imagen (fallback) ---
function stopUploadedImage() {
  overlay.hidden = true;
}

function analyzeImage(img, name) {
  if (cameraStream) stopCamera();
  video.hidden = true;

  const scale = Math.min(1, 1024 / Math.max(img.width, img.height));
  overlay.width = Math.round(img.width * scale);
  overlay.height = Math.round(img.height * scale);
  const ctx = overlay.getContext("2d");
  ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
  overlay.hidden = false;
  placeholder.style.display = "none";
  zoomToolbar.hidden = false;
  if (!isMaximized) {
    videoWrapper.style.aspectRatio = `${overlay.width} / ${overlay.height}`;
  }
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
loadSavedSettings();
initThresholdEvents();
initZoomEvents();
modelsReady = loadModels();

btnCamera.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else startCamera();
});
fileInput.addEventListener("change", handleUpload);
btnEdges.addEventListener("click", () => setEdges(!edgesOn));

btnFullscreen.addEventListener("click", () => toggleMaximizedMode());
if (btnHudClose) btnHudClose.addEventListener("click", () => toggleMaximizedMode(false));
if (btnHudCamera) {
  btnHudCamera.addEventListener("click", () => {
    if (cameraStream) stopCamera();
    else startCamera();
  });
}
if (btnHudEdges) btnHudEdges.addEventListener("click", () => setEdges(!edgesOn));

// Ganchos de prueba
const urlParams = new URLSearchParams(location.search);
if (urlParams.has("autocam")) {
  startCamera();
}

const imgtest = urlParams.get("imgtest");
if (imgtest) {
  const img = new Image();
  img.onload = () => {
    analyzeImage(img, imgtest);
    if (urlParams.has("edges")) setEdges(true);
  };
  img.src = imgtest;
}
