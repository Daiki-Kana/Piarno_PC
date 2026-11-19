/**
 * Maracas Web App (MicroPose WebGPU + MediaPipe フォールバック)
 *
 * パー（開いた手）で手を振る動作を「正規化2D速度ピークからの急減速・折り返し（加速度）」で検知し、
 * Web Audio API のフィルタ付きノイズ合成で低遅延のマラカス音を鳴らす専用アプリ。
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { createHandpose } from "@svenflow/micro-handpose";

// ==========================================================================
// DOM要素
// ==========================================================================
const video = document.getElementById("webcam");
const canvas = document.getElementById("output-canvas");
const canvasCtx = canvas.getContext("2d");
const startModal = document.getElementById("start-modal");
const startPlayBtn = document.getElementById("start-play-btn");
const startPlayText = document.getElementById("start-play-text");

const telemetryPill = document.getElementById("pc-telemetry-pill");
const telemetryCamFps = document.getElementById("telemetry-cam-fps");
const telemetryAppFps = document.getElementById("telemetry-app-fps");
const telemetryRes = document.getElementById("telemetry-res");
const telemetryMs = document.getElementById("telemetry-ms");
const telemetryMode = document.getElementById("telemetry-mode");
const telemetryEngine = document.getElementById("telemetry-engine");

const cameraSelectBtn = document.getElementById("camera-select-btn");
const cameraSelectMenu = document.getElementById("camera-select-menu");
const cameraListEl = document.getElementById("camera-list");
const mirrorToggleBtn = document.getElementById("mirror-toggle-btn");
const engineToggleBtn = document.getElementById("engine-toggle-btn");
const trackingLineToggleBtn = document.getElementById("tracking-line-toggle-btn");

const cameraErrorOverlay = document.getElementById("camera-error-overlay");
const cameraErrorTitle = document.getElementById("camera-error-title");
const cameraErrorMessage = document.getElementById("camera-error-message");
const cameraErrorReloadBtn = document.getElementById("camera-error-reload-btn");
const cameraErrorCloseBtn = document.getElementById("camera-error-close-btn");

// ==========================================================================
// 設定・永続化状態
// ==========================================================================
const PC_CAMERA_STORAGE_KEY = "maracas_pc_camera_id";
const PC_MIRROR_STORAGE_KEY = "maracas_pc_mirror";
const PC_ENGINE_PREF_STORAGE_KEY = "maracas_pc_engine_pref";
const PC_TRACKING_LINE_STORAGE_KEY = "maracas_pc_tracking_line";

const MAX_APP_FPS = 60;
const MIN_APP_DISPATCH_INTERVAL_MS = 10.0;

let currentCameraId = (() => {
  try {
    return localStorage.getItem(PC_CAMERA_STORAGE_KEY) || "";
  } catch {
    return "";
  }
})();

let isMirrorEnabled = (() => {
  try {
    const saved = localStorage.getItem(PC_MIRROR_STORAGE_KEY);
    return saved === null ? true : saved === "true";
  } catch {
    return true;
  }
})();

let preferredEngine = (() => {
  try {
    const saved = localStorage.getItem(PC_ENGINE_PREF_STORAGE_KEY);
    return saved === "MICRO" ? "MICRO" : "MEDIAPIPE";
  } catch {
    return "MEDIAPIPE";
  }
})();

let showTrackingLines = (() => {
  try {
    const saved = localStorage.getItem(PC_TRACKING_LINE_STORAGE_KEY);
    return saved === null ? true : saved === "true";
  } catch {
    return true;
  }
})();

// ==========================================================================
// Web Audio API マラカス音響エンジン
// ==========================================================================
// WebAudioFont 実機マラカスサンプル ＆ Web Audio フォールバック音響エンジン
// ==========================================================================
const MARACAS_PRESET_URL =
  "https://surikov.github.io/webaudiofontdata/sound/12870_0_FluidR3_GM_sf2_file.js";
const MARACAS_PRESET_VAR = "_drum_70_0_FluidR3_GM_sf2_file";

let audioCtx = null;
let noiseBuffer = null;
let maracasStrokeIndex = 0;
let wafPlayer = null;
let maracasPreset = null;
let maracasClosedFilter = null;
let isPresetLoading = false;

/**
 * WebAudioFont の Maracas (GM Drum Note 70) プリセットを非同期ロード
 */
function loadMaracasPreset(ctx) {
  if (maracasPreset || isPresetLoading || !wafPlayer) return;
  isPresetLoading = true;
  try {
    wafPlayer.loader.startLoad(ctx, MARACAS_PRESET_URL, MARACAS_PRESET_VAR);
    wafPlayer.loader.waitLoad(() => {
      const loaded = window[MARACAS_PRESET_VAR];
      if (loaded) {
        maracasPreset = loaded;
        console.log("[AUDIO] WebAudioFont Maracas サンプルのロードが完了しました");
      }
      isPresetLoading = false;
    });
  } catch (err) {
    console.warn("[AUDIO] WebAudioFont ロード失敗、フォールバック合成を使用します:", err);
    isPresetLoading = false;
  }
}

/**
 * AudioContext をユーザー操作コンテキスト内で初期化・再開
 */
function ensureAudioContext() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) {
      audioCtx = new Ctx({ latencyHint: "interactive" });
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  if (audioCtx && !wafPlayer && typeof window.WebAudioFontPlayer === "function") {
    wafPlayer = new window.WebAudioFontPlayer();
    maracasClosedFilter = audioCtx.createBiquadFilter();
    maracasClosedFilter.type = "lowpass";
    maracasClosedFilter.frequency.setValueAtTime(2600, audioCtx.currentTime);
    maracasClosedFilter.connect(audioCtx.destination);
    loadMaracasPreset(audioCtx);
  }
  if (audioCtx && !noiseBuffer) {
    // サンプルロード前用の0.25秒ホワイトノイズバッファを事前生成
    const length = Math.floor(audioCtx.sampleRate * 0.25);
    noiseBuffer = audioCtx.createBuffer(1, length, audioCtx.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      data[i] = Math.random() * 2 - 1;
    }
  }
  return audioCtx;
}

/**
 * 粒の擦れ・シャリ感を補う高域シェイカーレイヤーを再生
 */
function playShakerGrainLayer(ctx, clamped, isOpen, delaySec = 0, gainScale = 0.28) {
  if (!noiseBuffer) return;

  const startAt = ctx.currentTime + delaySec;
  const src = ctx.createBufferSource();
  src.buffer = noiseBuffer;

  const hpFilter = ctx.createBiquadFilter();
  hpFilter.type = "highpass";
  hpFilter.frequency.setValueAtTime(
    isOpen ? 4600 + clamped * 1000 : 1400 + clamped * 500,
    startAt
  );

  const bpFilter = ctx.createBiquadFilter();
  bpFilter.type = "bandpass";
  bpFilter.frequency.setValueAtTime(
    isOpen ? 6800 + clamped * 1400 : 2500 + clamped * 700,
    startAt
  );
  bpFilter.Q.setValueAtTime(isOpen ? 1.1 : 1.4, startAt);

  const gainNode = ctx.createGain();
  const peakGain = gainScale * (0.45 + clamped * 0.55);
  const decaySec = (isOpen ? 0.065 : 0.085) + clamped * 0.04;

  // 10msのスウェルで「シュワッ」と粒が擦れる立ち上がりを作る
  gainNode.gain.setValueAtTime(0.001, startAt);
  gainNode.gain.linearRampToValueAtTime(peakGain, startAt + 0.01);
  gainNode.gain.exponentialRampToValueAtTime(0.0001, startAt + decaySec);

  src.connect(hpFilter);
  hpFilter.connect(bpFilter);
  bpFilter.connect(gainNode);
  gainNode.connect(ctx.destination);

  src.start(startAt, Math.random() * 0.05, decaySec + 0.02);
}

/**
 * ロード前・オフライン時のフォールバック合成マラカス音
 */
function playMaracasSynthFallback(ctx, clamped, isOpen) {
  playShakerGrainLayer(ctx, clamped, isOpen, 0, 0.75);
  playShakerGrainLayer(ctx, clamped, isOpen, 0.024, 0.42);
}

/**
 * マラカス音を発音する（WebAudioFont実機サンプル＋粒跳ね返り＋高域シャカ感）
 * @param {number} intensity 振りの強さ (0.25〜1.0)
 * @param {boolean} isOpen パー（true: 高い音） / グー（false: 低めの音）
 */
function playMaracasSound(intensity = 0.7, isOpen = true) {
  const ctx = ensureAudioContext();
  if (!ctx) return;

  const clamped = Math.min(1.0, Math.max(0.2, intensity));

  if (maracasPreset && wafPlayer) {
    try {
      const now = ctx.currentTime;
      const basePitch = isOpen ? 75 : 64;
      const targetNode = isOpen || !maracasClosedFilter ? ctx.destination : maracasClosedFilter;
      const volume = Math.min(1.0, 0.45 + clamped * 0.55);

      // 1打目（主音）＋ 24ms後の2打目（粒の跳ね返り）でシャカシャカ感を演出
      wafPlayer.queueWaveTable(ctx, targetNode, maracasPreset, now, basePitch, 0.35, volume);
      wafPlayer.queueWaveTable(ctx, targetNode, maracasPreset, now + 0.024, basePitch + 2, 0.25, volume * 0.45);
      // 粒の擦れる高域シャリ感を薄くブレンド
      playShakerGrainLayer(ctx, clamped, isOpen, 0, isOpen ? 0.34 : 0.24);
      return;
    } catch (err) {
      console.warn("[AUDIO] WebAudioFont 再生エラー、フォールバック使用:", err);
    }
  }

  playMaracasSynthFallback(ctx, clamped, isOpen);
}

// ==========================================================================
// パー/グー判定 ＆ マラカス振り（縦：下方向への振り始め立ち上がり即時発音）検知
// ==========================================================================
const SHAKE_TRIGGER_SPEED_TH = 2.6;     // 即時発音とみなす下方向正規化速度（palmSize / 秒）
const SHAKE_RESET_SPEED_TH = 0.8;       // 再トリガー許可のための速度低下閾値（palmSize / 秒）
const SHAKE_MIN_STROKE_DIST = 0.12;     // 誤爆防止の最小下移動距離（palmSize比）
const SHAKE_MIN_PALM_SCALE_RATIO = 0.97;// カメラから遠ざかる後ろ引きを弾くサイズ維持率
const SHAKE_COOLDOWN_MS = 200;          // 連続誤爆を防ぐ無検知時間（ms）

const createMaracasHandState = () => ({
  prevCenter: null,
  prevTimeMs: null,
  prevPalmSize: null,
  downStrokeDist: 0,
  canTrigger: true,
  lastHitTimeMs: 0,
  lostFrames: 0
});

const maracasSlots = [createMaracasHandState(), createMaracasHandState()];
const shakeRipples = [];

/**
 * 指定した手スロットのマラカス検知状態をリセット
 */
function resetMaracasHandState(handState) {
  handState.prevCenter = null;
  handState.prevTimeMs = null;
  handState.prevPalmSize = null;
  handState.downStrokeDist = 0;
  handState.canTrigger = true;
}

function resetAllMaracasStates() {
  for (const slot of maracasSlots) {
    resetMaracasHandState(slot);
    slot.lostFrames = 0;
  }
}

/**
 * 手が開いた状態（パー）かどうかを判定
 * 4本の指（人差し指〜小指）のうち3本以上が伸びていればパーとみなす
 * @param {Array<{x: number, y: number}>} lm 正規化ランドマーク配列
 * @returns {boolean}
 */
function isOpenPalm(lm) {
  if (!lm || lm.length < 21) return false;
  const wrist = lm[0];
  const fingerPairs = [
    [8, 6],   // 人差し指
    [12, 10], // 中指
    [16, 14], // 薬指
    [20, 18]  // 小指
  ];
  let extendedCount = 0;
  for (const [tipIdx, pipIdx] of fingerPairs) {
    const dTip = Math.hypot(lm[tipIdx].x - wrist.x, lm[tipIdx].y - wrist.y);
    const dPip = Math.hypot(lm[pipIdx].x - wrist.x, lm[pipIdx].y - wrist.y);
    if (dTip > dPip * 1.12) {
      extendedCount++;
    }
  }
  return extendedCount >= 3;
}

/**
 * 1フレーム分の手のひら動きからマラカス発音を判定する
 * @param {Array<{x: number, y: number}>} lm 正規化ランドマーク (0..1)
 * @param {Array<{x: number, y: number}>} pxLm ピクセル座標ランドマーク
 * @param {number} nowMs
 * @returns {{ isOpen: boolean, shook: boolean, centerPx: {x: number, y: number} }}
 */
function evaluateMaracasShake(handState, lm, pxLm, nowMs) {
  const isOpen = isOpenPalm(lm);
  const centerNorm = {
    x: (lm[0].x + lm[9].x) * 0.5,
    y: (lm[0].y + lm[9].y) * 0.5
  };
  const centerPx = {
    x: (pxLm[0].x + pxLm[9].x) * 0.5,
    y: (pxLm[0].y + pxLm[9].y) * 0.5
  };

  // 手の基準長（手首〜中指付け根）でカメラ距離によるスケール差を正規化
  const palmSize = Math.max(0.04, Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y));

  if (handState.prevCenter === null || handState.prevTimeMs === null) {
    handState.prevCenter = centerNorm;
    handState.prevTimeMs = nowMs;
    handState.prevPalmSize = palmSize;
    return { isOpen, shook: false, centerPx };
  }

  const dt = Math.max((nowMs - handState.prevTimeMs) / 1000, 1 / 240);
  const deltaDownNorm = (centerNorm.y - handState.prevCenter.y) / palmSize;
  // Y軸下方向（上から下への振り下ろし）の正規化速度を算出
  const downSpeed = deltaDownNorm / dt;

  // 下向きに連続移動した距離を累積し、上向きに戻ったらリセット
  if (deltaDownNorm > 0) {
    handState.downStrokeDist += deltaDownNorm;
  } else {
    handState.downStrokeDist = 0;
  }

  // カメラから遠ざかる後ろ引き（手の見かけサイズ縮小）をブロック
  const isNotRetreating =
    handState.prevPalmSize === null ||
    palmSize >= handState.prevPalmSize * SHAKE_MIN_PALM_SCALE_RATIO;

  handState.prevCenter = centerNorm;
  handState.prevTimeMs = nowMs;
  handState.prevPalmSize = palmSize;

  if (!handState.canTrigger && downSpeed < SHAKE_RESET_SPEED_TH) {
    handState.canTrigger = true;
  }

  let shook = false;
  const isInCooldown = nowMs - handState.lastHitTimeMs < SHAKE_COOLDOWN_MS;
  // 振り始めの加速立ち上がり（サイズ維持＋最小ストローク長達成時）で即座に発音
  if (
    handState.canTrigger &&
    !isInCooldown &&
    isNotRetreating &&
    handState.downStrokeDist >= SHAKE_MIN_STROKE_DIST &&
    downSpeed >= SHAKE_TRIGGER_SPEED_TH
  ) {
    const intensity = Math.min(1.0, Math.max(0.35, (downSpeed - 2.0) / 5.5));
    playMaracasSound(intensity, isOpen);
    handState.canTrigger = false;
    handState.downStrokeDist = 0;
    handState.lastHitTimeMs = nowMs;
    shook = true;

    // 波紋ビジュアルエフェクトを追加
    shakeRipples.push({
      x: centerPx.x,
      y: centerPx.y,
      radius: 14,
      maxRadius: 56 + intensity * 44,
      alpha: 0.9
    });
  }

  return { isOpen, shook, centerPx };
}

// ==========================================================================
// 描画（骨格ライン ＋ シェイクリップル）
// ==========================================================================
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

const LOST_RESET_FRAMES = 4;
const HAND_MATCH_MAX_DIST = 0.35;

/**
 * 検出結果の判定・描画を行う（位置連続性マッチングによる両手同時演奏対応）
 * @param {object} results
 */
function processDetectionResults(results) {
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  const nowMs = performance.now();
  const validHands = (results?.landmarks || []).filter((lm) => lm && lm.length >= 21).slice(0, 2);
  const width = canvas.width;
  const height = canvas.height;

  // 各検出手の中心座標を算出
  const handCenters = validHands.map((lm) => ({
    x: (lm[0].x + lm[9].x) * 0.5,
    y: (lm[0].y + lm[9].y) * 0.5
  }));

  // 前フレームの座標に近いスロットへ優先割り当て（Left/Rightラベル揺れによる状態混同を防止）
  const assignedSlotForHand = new Array(validHands.length).fill(-1);
  const usedSlots = new Set();

  const pairs = [];
  for (let hIdx = 0; hIdx < validHands.length; hIdx++) {
    for (let sIdx = 0; sIdx < maracasSlots.length; sIdx++) {
      const prev = maracasSlots[sIdx].prevCenter;
      if (!prev) continue;
      const dist = Math.hypot(handCenters[hIdx].x - prev.x, handCenters[hIdx].y - prev.y);
      if (dist <= HAND_MATCH_MAX_DIST) {
        pairs.push({ hIdx, sIdx, dist });
      }
    }
  }
  pairs.sort((a, b) => a.dist - b.dist);

  for (const { hIdx, sIdx } of pairs) {
    if (assignedSlotForHand[hIdx] !== -1 || usedSlots.has(sIdx)) continue;
    assignedSlotForHand[hIdx] = sIdx;
    usedSlots.add(sIdx);
  }

  // 未割り当ての手を空きスロットへ割り当て
  for (let hIdx = 0; hIdx < validHands.length; hIdx++) {
    if (assignedSlotForHand[hIdx] !== -1) continue;
    for (let sIdx = 0; sIdx < maracasSlots.length; sIdx++) {
      if (!usedSlots.has(sIdx)) {
        resetMaracasHandState(maracasSlots[sIdx]);
        assignedSlotForHand[hIdx] = sIdx;
        usedSlots.add(sIdx);
        break;
      }
    }
  }

  let anyShook = false;
  let anyOpen = false;

  for (let hIdx = 0; hIdx < validHands.length; hIdx++) {
    const sIdx = assignedSlotForHand[hIdx];
    if (sIdx === -1) continue;

    const rawLm = validHands[hIdx];
    const handState = maracasSlots[sIdx];
    handState.lostFrames = 0;

    const pxLm = rawLm.map((pt) => ({
      x: pt.x * width,
      y: pt.y * height
    }));

    const { isOpen, shook, centerPx } = evaluateMaracasShake(handState, rawLm, pxLm, nowMs);
    if (shook) anyShook = true;
    if (isOpen) anyOpen = true;

    if (showTrackingLines) {
      drawHandSkeleton(pxLm, centerPx, isOpen);
    }
  }

  // 未割り当てスロットのロスト処理
  for (let sIdx = 0; sIdx < maracasSlots.length; sIdx++) {
    if (!usedSlots.has(sIdx)) {
      const handState = maracasSlots[sIdx];
      handState.lostFrames++;
      if (handState.lostFrames >= LOST_RESET_FRAMES) {
        resetMaracasHandState(handState);
      }
    }
  }

  if (telemetryMode) {
    if (usedSlots.size === 0) {
      telemetryMode.textContent = "IDLE";
    } else {
      telemetryMode.textContent = anyShook ? "SHAKE" : anyOpen ? "OPEN" : "CLOSED";
    }
  }

  renderRipples();
}

/**
 * 手の骨格と手のひら中心マーカーを描画
 * @param {Array<{x: number, y: number}>} pts
 * @param {{x: number, y: number}} centerPx
 * @param {boolean} isOpen
 */
function drawHandSkeleton(pts, centerPx, isOpen) {
  canvasCtx.save();
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";
  canvasCtx.strokeStyle = isOpen ? "rgba(255, 255, 255, 0.85)" : "rgba(255, 255, 255, 0.28)";
  canvasCtx.lineWidth = isOpen ? 2.5 : 1.5;

  canvasCtx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    canvasCtx.moveTo(pts[a].x, pts[a].y);
    canvasCtx.lineTo(pts[b].x, pts[b].y);
  }
  canvasCtx.stroke();

  canvasCtx.fillStyle = isOpen ? "#ffffff" : "rgba(255, 255, 255, 0.35)";
  for (const pt of pts) {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x, pt.y, 3.0, 0, Math.PI * 2);
    canvasCtx.fill();
  }

  // 手のひら中心マーカー
  canvasCtx.beginPath();
  canvasCtx.arc(centerPx.x, centerPx.y, 6.0, 0, Math.PI * 2);
  canvasCtx.strokeStyle = isOpen ? "#ffffff" : "rgba(255, 255, 255, 0.35)";
  canvasCtx.lineWidth = 2.0;
  canvasCtx.stroke();

  canvasCtx.restore();
}

/**
 * シェイク発火時のモノクロ波紋アニメーションを描画
 */
function renderRipples() {
  if (shakeRipples.length === 0) return;

  canvasCtx.save();
  for (let i = shakeRipples.length - 1; i >= 0; i--) {
    const r = shakeRipples[i];
    canvasCtx.beginPath();
    canvasCtx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
    canvasCtx.strokeStyle = `rgba(255, 255, 255, ${r.alpha.toFixed(2)})`;
    canvasCtx.lineWidth = 2.5;
    canvasCtx.stroke();

    r.radius += (r.maxRadius - r.radius) * 0.25;
    r.alpha -= 0.08;
    if (r.alpha <= 0.02) {
      shakeRipples.splice(i, 1);
    }
  }
  canvasCtx.restore();
}

// ==========================================================================
// トラッキングエンジン（MicroPose WebGPU ＋ MediaPipe フォールバック）
// ==========================================================================
let handLandmarker = null;
let microHandpose = null;
let microInputCanvas = null;
let microInputCtx = null;
let activeTrackingEngine = "NONE";
let trackingInitPromise = null;

let currentStream = null;
let isPredicting = false;
let isInferringFrame = false;
let pendingVideoFrameTime = null;
let lastVideoTimestamp = -1;
let lastVideoTime = -1;
let lastAppProcessedTime = 0;

let appFrameCount = 0;
let lastAppFpsTime = performance.now();
let camFrameCount = 0;
let lastPresentedFrames = -1;
let lastCamFpsTime = performance.now();

/**
 * MicroPose 用の中間入力 Canvas を確保・同期する（毎フレームの createImageBitmap 回避）
 * モデル内部入力(224x224)に合わせて長辺最大384pxに収め、モバイルのGPU転送負荷を削減
 */
function ensureMicroInputCanvas(vw, vh) {
  const maxDim = 384;
  let targetW = vw || 640;
  let targetH = vh || 360;
  if (Math.max(targetW, targetH) > maxDim) {
    const scale = maxDim / Math.max(targetW, targetH);
    targetW = Math.round(targetW * scale);
    targetH = Math.round(targetH * scale);
  }
  if (!microInputCanvas) {
    microInputCanvas = document.createElement("canvas");
    microInputCtx = microInputCanvas.getContext("2d", { alpha: false });
  }
  if (microInputCanvas.width !== targetW || microInputCanvas.height !== targetH) {
    microInputCanvas.width = targetW;
    microInputCanvas.height = targetH;
  }
  return microInputCanvas;
}

let mediaPipeDelegateLabel = "MP-GPU";
let isSwitchingEngine = false;

/**
 * MicroPose (WebGPU) エンジンの遅延初期化
 * @returns {Promise<boolean>} 利用可能なら true
 */
async function ensureMicroEngine() {
  if (microHandpose) return true;
  if (typeof navigator === "undefined" || !navigator.gpu) return false;

  try {
    const adapter = await navigator.gpu.requestAdapter();
    const arch = adapter?.info?.architecture || "";
    const isSoftware =
      !adapter ||
      adapter.isFallbackAdapter === true ||
      arch.toLowerCase().includes("swiftshader");

    if (isSoftware) {
      console.warn("[TRACKING ENGINE] WebGPU 実GPUアダプタが利用不可:", adapter?.info);
      return false;
    }

    microHandpose = await createHandpose({
      maxHands: 2,
      scoreThreshold: 0.45,
      palmScoreThreshold: 0.45
    });
    console.log("[TRACKING ENGINE] MicroPose (WebGPU) 準備完了", adapter.info);
    return true;
  } catch (err) {
    console.warn("[TRACKING ENGINE] MicroPose 初期化失敗:", err);
    microHandpose = null;
    return false;
  }
}

/**
 * MediaPipe (WebGL GPU / CPU) エンジンの遅延初期化
 */
async function ensureMediaPipeEngine() {
  if (handLandmarker) return true;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const modelPaths = [
    "./models/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  ];

  for (const modelPath of modelPaths) {
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelPath, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.3
      });
      mediaPipeDelegateLabel = "MP-GPU";
      console.log("[TRACKING ENGINE] MediaPipe (GPU) 準備完了");
      return true;
    } catch {
      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelPath, delegate: "CPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.3
        });
        mediaPipeDelegateLabel = "MP-CPU";
        console.log("[TRACKING ENGINE] MediaPipe (CPU) 準備完了");
        return true;
      } catch {
        // 次のモデルパスを試行
      }
    }
  }
  return false;
}

/**
 * トラッキングエンジンの初期化・切り替え
 * @param {"MICRO"|"MEDIAPIPE"} [targetPref]
 */
async function switchTrackingEngine(targetPref = preferredEngine) {
  if (isSwitchingEngine) return;
  isSwitchingEngine = true;
  if (engineToggleBtn) engineToggleBtn.disabled = true;

  try {
    if (targetPref === "MICRO") {
      const ok = await ensureMicroEngine();
      if (ok) {
        activeTrackingEngine = "MICRO_HANDPOSE";
        preferredEngine = "MICRO";
        if (microHandpose && typeof microHandpose.reset === "function") {
          microHandpose.reset();
        }
        syncEngineToggleUI();
        resetAllMaracasStates();
        return;
      }
    }

    const mpOk = await ensureMediaPipeEngine();
    if (!mpOk) {
      throw new Error("ハンドトラッキングモデルの初期化に失敗しました。");
    }
    activeTrackingEngine = "MEDIAPIPE";
    preferredEngine = "MEDIAPIPE";
    syncEngineToggleUI();
    resetAllMaracasStates();
  } finally {
    isSwitchingEngine = false;
    if (engineToggleBtn) engineToggleBtn.disabled = false;
  }
}

async function initTrackingEngine() {
  await switchTrackingEngine(preferredEngine);
}

// ==========================================================================
// Webカメラ制御（外付け60fpsカメラ自動選択）
// ==========================================================================
function getCompatibleUserMedia(constraints) {
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }
  const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
  if (legacy) {
    return new Promise((resolve, reject) => legacy.call(navigator, constraints, resolve, reject));
  }
  return Promise.reject(new Error("WebカメラAPIがサポートされていません。"));
}

async function refreshCameraList() {
  if (!cameraListEl || !navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === "videoinput");
    cameraListEl.innerHTML = "";
    videoInputs.forEach((device, index) => {
      const btn = document.createElement("button");
      btn.className = "camera-menu-item";
      if (device.deviceId && device.deviceId === currentCameraId) {
        btn.classList.add("active");
      }
      btn.textContent = device.label || `Camera ${index + 1}`;
      btn.title = btn.textContent;
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (cameraSelectMenu) cameraSelectMenu.classList.add("hidden");
        if (device.deviceId && device.deviceId !== currentCameraId) {
          await startCamera(device.deviceId, true);
        }
      });
      cameraListEl.appendChild(btn);
    });
  } catch (err) {
    console.warn("カメラ一覧取得スキップ:", err);
  }
}

async function startCamera(deviceId = null, isManualSelect = false) {
  isPredicting = false;
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }

  const isBuiltInCameraLabel = (label = "") =>
    /integrated|built-in|facetime|internal/i.test(label);

  let preferredDeviceId = deviceId;
  if (!isManualSelect && navigator.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === "videoinput" && d.deviceId);
      const savedDev = preferredDeviceId ? videoInputs.find((d) => d.deviceId === preferredDeviceId) : null;
      const externalCam = videoInputs.find((d) => d.label && !isBuiltInCameraLabel(d.label));
      if (externalCam && (!savedDev || isBuiltInCameraLabel(savedDev.label))) {
        preferredDeviceId = externalCam.deviceId;
        console.log(`[CAM AUTO] 外付けカメラを優先選択: ${externalCam.label}`);
      }
    } catch {}
  }

  const buildConstraints = (targetDeviceId) => {
    const base = targetDeviceId ? { deviceId: { exact: targetDeviceId } } : {};
    return [
      { video: { ...base, width: { exact: 1024 }, height: { exact: 576 }, frameRate: { ideal: 60, max: 60, min: 50 } }, audio: false },
      { video: { ...base, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, max: 60, min: 50 } }, audio: false },
      { video: { ...base, width: { ideal: 1024 }, height: { ideal: 576 }, frameRate: { ideal: 60, max: 60, min: 30 } }, audio: false },
      { video: { ...base, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, max: 60, min: 30 } }, audio: false },
      { video: { ...base, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { ...base, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
      { video: targetDeviceId ? { deviceId: { exact: targetDeviceId } } : true, audio: false }
    ];
  };

  let stream = null;
  let lastError = null;
  const tryList = preferredDeviceId
    ? [...buildConstraints(preferredDeviceId), ...buildConstraints(null)]
    : buildConstraints(null);

  for (const constraints of tryList) {
    try {
      stream = await getCompatibleUserMedia(constraints);
      if (stream) break;
    } catch (e) {
      lastError = e;
      if (e.name === "NotAllowedError" || e.name === "PermissionDeniedError" || e.name === "SecurityError") {
        throw e;
      }
    }
  }

  if (!stream) {
    throw lastError || new Error("Webカメラの起動に失敗しました。");
  }

  const initialTrack = stream.getVideoTracks()[0];
  if (!isManualSelect && initialTrack && isBuiltInCameraLabel(initialTrack.label) && navigator.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const externalCam = devices.find(
        (d) => d.kind === "videoinput" && d.deviceId && d.label && !isBuiltInCameraLabel(d.label)
      );
      if (externalCam) {
        stream.getTracks().forEach((t) => t.stop());
        return startCamera(externalCam.deviceId, true);
      }
    } catch {}
  }

  currentStream = stream;
  video.srcObject = currentStream;

  const track = stream.getVideoTracks()[0];
  if (track) {
    try {
      if (typeof track.getCapabilities === "function" && typeof track.applyConstraints === "function") {
        const caps = track.getCapabilities();
        const targetFps = caps.frameRate?.max ? Math.min(MAX_APP_FPS, caps.frameRate.max) : MAX_APP_FPS;
        await track.applyConstraints({ frameRate: { ideal: targetFps, max: MAX_APP_FPS } });
      }
    } catch {}

    try {
      if (typeof track.getSettings === "function") {
        const settings = track.getSettings();
        if (settings.deviceId) {
          currentCameraId = settings.deviceId;
          try {
            localStorage.setItem(PC_CAMERA_STORAGE_KEY, currentCameraId);
          } catch {}
        }
        if (telemetryRes && settings.width && settings.height) {
          telemetryRes.textContent = `${settings.width}×${settings.height}`;
        }
      }
    } catch {}
  }

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = async () => {
      try {
        await video.play();
        resolve();
      } catch (err) {
        reject(err);
      }
    };
    video.onerror = (err) => reject(err);
  });

  updateCanvasResolution();
  await refreshCameraList();

  appFrameCount = 0;
  camFrameCount = 0;
  lastPresentedFrames = -1;
  lastAppProcessedTime = 0;
  isInferringFrame = false;
  pendingVideoFrameTime = null;
  lastAppFpsTime = performance.now();
  lastCamFpsTime = performance.now();
  resetAllMaracasStates();
  if (microHandpose && typeof microHandpose.reset === "function") {
    microHandpose.reset();
  }

  isPredicting = true;
  scheduleCameraFpsMonitor();
  schedulePredictLoop();
}

function updateCanvasResolution() {
  if (video.videoWidth && video.videoHeight) {
    const aspect = video.videoWidth / video.videoHeight;
    document.documentElement.style.setProperty("--video-aspect", aspect.toFixed(4));
    // モバイルの2D描画負荷を抑えるため、表示Canvasの内部解像度は長辺最大640pxに制限
    const maxDrawDim = 640;
    let targetW = video.videoWidth;
    let targetH = video.videoHeight;
    if (Math.max(targetW, targetH) > maxDrawDim) {
      const scale = maxDrawDim / Math.max(targetW, targetH);
      targetW = Math.round(targetW * scale);
      targetH = Math.round(targetH * scale);
    }
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
  }
}

// ==========================================================================
// 推論ループ ＆ FPS計測
// ==========================================================================
function shouldRunAppFrame(nowMs) {
  if (nowMs - lastAppProcessedTime < MIN_APP_DISPATCH_INTERVAL_MS) {
    return false;
  }
  lastAppProcessedTime = nowMs;
  return true;
}

async function processVideoFrame(frameTime) {
  if (!isPredicting || isInferringFrame) return;
  isInferringFrame = true;

  try {
    if (video.videoWidth && canvas.width === 0) {
      updateCanvasResolution();
    }

    const now = typeof frameTime === "number" ? frameTime : performance.now();
    const timestampMs = Math.max(now, lastVideoTimestamp + 1.0);
    lastVideoTimestamp = timestampMs;

    const startTime = performance.now();
    let results = { landmarks: [] };

    if (activeTrackingEngine === "MICRO_HANDPOSE" && microHandpose) {
      const inputCanvas = ensureMicroInputCanvas(video.videoWidth, video.videoHeight);
      microInputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
      const hands = await microHandpose.detect(inputCanvas);
      if (!isPredicting) return;
      results = {
        landmarks: hands.map((h) => h.landmarks),
        handednesses: hands.map((h) => [
          { categoryName: h.handedness === "right" ? "Right" : "Left" }
        ])
      };
    } else if (handLandmarker) {
      results = handLandmarker.detectForVideo(video, timestampMs);
    }

    const calcDuration = performance.now() - startTime;
    if (telemetryMs) {
      telemetryMs.textContent = `${calcDuration.toFixed(1)} ms`;
    }

    processDetectionResults(results);
    updateAppFps();
  } catch (err) {
    console.warn("[INFERENCE] フレーム推論エラー:", err);
  } finally {
    isInferringFrame = false;
  }
}

/**
 * カメラ供給FPSの独立計測（推論ループと切り離して実供給FPSを正確に計測）
 */
function scheduleCameraFpsMonitor() {
  if (!isPredicting) return;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback((_, metadata) => {
      if (!isPredicting) return;
      updateCameraFps(performance.now(), metadata?.presentedFrames);
      scheduleCameraFpsMonitor();
    });
  }
}

/**
 * 本家デモ同様の requestAnimationFrame 直列推論ループ
 * （await 完了後に次フレームを予約することでフレームペーシングの乱れを防止）
 */
function schedulePredictLoop() {
  if (!isPredicting) return;

  requestAnimationFrame(async (now) => {
    if (!isPredicting) return;
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype) && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        updateCameraFps(performance.now());
      }
      if (shouldRunAppFrame(performance.now())) {
        await processVideoFrame(now);
      }
    }
    schedulePredictLoop();
  });
}

function updateCameraFps(nowMs, presentedFrames) {
  if (typeof presentedFrames === "number") {
    if (lastPresentedFrames < 0 || presentedFrames < lastPresentedFrames) {
      camFrameCount += 1;
    } else {
      camFrameCount += presentedFrames - lastPresentedFrames;
    }
    lastPresentedFrames = presentedFrames;
  } else {
    camFrameCount += 1;
  }

  const elapsed = nowMs - lastCamFpsTime;
  if (elapsed >= 500) {
    const camFps = Math.round((camFrameCount * 1000) / elapsed);
    if (telemetryCamFps) {
      telemetryCamFps.textContent = `CAM ${camFps}`;
    }
    camFrameCount = 0;
    lastCamFpsTime = nowMs;
  }
}

function updateAppFps() {
  appFrameCount++;
  const now = performance.now();
  const elapsed = now - lastAppFpsTime;
  if (elapsed >= 500) {
    const fps = Math.round((appFrameCount * 1000) / elapsed);
    if (telemetryAppFps) {
      telemetryAppFps.textContent = `APP ${fps} FPS`;
    }
    appFrameCount = 0;
    lastAppFpsTime = now;
  }
}

// ==========================================================================
// UIコントロール・起動フロー
// ==========================================================================
function syncMirrorUI() {
  if (video) video.classList.toggle("mirrored", isMirrorEnabled);
  if (canvas) canvas.classList.toggle("mirrored", isMirrorEnabled);
  if (mirrorToggleBtn) {
    mirrorToggleBtn.classList.toggle("off", !isMirrorEnabled);
    mirrorToggleBtn.setAttribute("aria-pressed", String(isMirrorEnabled));
    mirrorToggleBtn.title = isMirrorEnabled ? "Mirror: ON" : "Mirror: OFF";
  }
}

function syncEngineToggleUI() {
  const isMicro = activeTrackingEngine === "MICRO_HANDPOSE";
  const label = isMicro ? "MICRO" : activeTrackingEngine === "MEDIAPIPE" ? mediaPipeDelegateLabel : preferredEngine;
  if (telemetryEngine) {
    telemetryEngine.textContent = label;
  }
  if (engineToggleBtn) {
    engineToggleBtn.classList.toggle("off", !isMicro);
    engineToggleBtn.setAttribute("aria-pressed", String(isMicro));
    engineToggleBtn.title = `Engine: ${label}`;
  }
}

function syncTrackingLineToggleUI() {
  if (!trackingLineToggleBtn) return;
  trackingLineToggleBtn.classList.toggle("off", !showTrackingLines);
  trackingLineToggleBtn.setAttribute("aria-pressed", String(showTrackingLines));
  trackingLineToggleBtn.title = showTrackingLines ? "Tracking Lines: ON" : "Tracking Lines: OFF";
}

function showCameraErrorOverlay(message, title = "Camera Unavailable") {
  if (cameraErrorTitle) cameraErrorTitle.textContent = title;
  if (cameraErrorMessage) cameraErrorMessage.textContent = message;
  if (cameraErrorOverlay) cameraErrorOverlay.classList.remove("hidden");
}

async function handleStartPlay() {
  if (!startPlayBtn) return;
  startPlayBtn.disabled = true;
  if (startPlayText) startPlayText.textContent = "STARTING...";

  ensureAudioContext();

  try {
    if (trackingInitPromise) {
      await trackingInitPromise;
    } else {
      await initTrackingEngine();
    }
    await startCamera(currentCameraId || null, false);
    if (startModal) startModal.classList.add("hidden");
  } catch (error) {
    console.error("起動エラー:", error);
    startPlayBtn.disabled = false;
    if (startPlayText) startPlayText.textContent = "START";
    showCameraErrorOverlay(error?.message || "Camera access was denied or no webcam was found.");
  }
}

function initApp() {
  // デバッグテレメトリは ?debug=1 指定時のみ表示（通常は非表示）
  const isDebugMode = new URLSearchParams(window.location.search).get("debug") === "1";
  if (telemetryPill) {
    telemetryPill.classList.toggle("hidden", !isDebugMode);
  }

  syncMirrorUI();
  syncEngineToggleUI();
  syncTrackingLineToggleUI();

  trackingInitPromise = initTrackingEngine().catch((err) => {
    console.warn("モデル先行ロードエラー:", err);
  });

  if (startPlayBtn) {
    startPlayBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleStartPlay();
    });
  }

  if (cameraSelectBtn && cameraSelectMenu) {
    cameraSelectBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await refreshCameraList();
      cameraSelectMenu.classList.toggle("hidden");
    });
    document.addEventListener("click", (e) => {
      if (!cameraSelectMenu.classList.contains("hidden") && !cameraSelectMenu.contains(e.target)) {
        cameraSelectMenu.classList.add("hidden");
      }
    });
  }

  if (mirrorToggleBtn) {
    mirrorToggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      isMirrorEnabled = !isMirrorEnabled;
      try {
        localStorage.setItem(PC_MIRROR_STORAGE_KEY, String(isMirrorEnabled));
      } catch {}
      syncMirrorUI();
    });
  }

  if (engineToggleBtn) {
    engineToggleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const nextPref = activeTrackingEngine === "MICRO_HANDPOSE" ? "MEDIAPIPE" : "MICRO";
      try {
        localStorage.setItem(PC_ENGINE_PREF_STORAGE_KEY, nextPref);
      } catch {}
      await switchTrackingEngine(nextPref);
    });
  }

  if (trackingLineToggleBtn) {
    trackingLineToggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showTrackingLines = !showTrackingLines;
      try {
        localStorage.setItem(PC_TRACKING_LINE_STORAGE_KEY, String(showTrackingLines));
      } catch {}
      syncTrackingLineToggleUI();
    });
  }

  if (cameraErrorCloseBtn && cameraErrorOverlay) {
    cameraErrorCloseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      cameraErrorOverlay.classList.add("hidden");
    });
  }

  if (cameraErrorReloadBtn) {
    cameraErrorReloadBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      window.location.reload();
    });
  }

  video.addEventListener("resize", updateCanvasResolution);
  window.addEventListener("resize", updateCanvasResolution);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
