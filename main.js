/**
 * MediaPipe Hand Landmarker 最小検証環境 (インカメラ・鏡像完全一致)
 * 
 * 机面スレスレの水平アングルから手を認識させる前提の最小実装。
 * 遅延検証のため、平滑化フィルター（移動平均や1 Euro Filter等）は一切挟まず、
 * 検出された生（Raw）のランドマーク座標をダイレクトにCanvasへ描画します。
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { createHandpose } from "@svenflow/micro-handpose";
import * as Tone from "tone";
import { BlanchDetector } from "./blanchDetector.js";

// DOM要素
const video = document.getElementById("webcam");
const canvas = document.getElementById("output-canvas");
const canvasCtx = canvas.getContext("2d");
const notesCanvas = document.getElementById("notes-canvas");
const notesCtx = notesCanvas ? notesCanvas.getContext("2d") : null;
const statusText = document.getElementById("status-text");
const fpsCounter = document.getElementById("fps-counter");
const inferenceTime = document.getElementById("inference-time");
const handsCount = document.getElementById("hands-count");
const camInfo = document.getElementById("cam-info");
const errorBox = document.getElementById("error-box");
const cameraErrorOverlay = document.getElementById("camera-error-overlay");
const cameraErrorTitle = document.getElementById("camera-error-title");
const cameraErrorMessage = document.getElementById("camera-error-message");
const cameraErrorReloadBtn = document.getElementById("camera-error-reload-btn");
const cameraErrorCloseBtn = document.getElementById("camera-error-close-btn");
const startModal = document.getElementById("start-modal");
const startPlayBtn = document.getElementById("start-play-btn");
const startPlayText = document.getElementById("start-play-text");
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
const oneEuroToggleBtn = document.getElementById("one-euro-toggle-btn");
const tipXVal = document.getElementById("tip-x-val");
const tipYVal = document.getElementById("tip-y-val");
const relYVal = document.getElementById("rel-y-val");
const thVal = document.getElementById("th-val");
const velYVal = document.getElementById("vel-y-val");
const vThVal = document.getElementById("v-th-val");
const stateVal = document.getElementById("state-val");
const tapCountVal = document.getElementById("tap-count-val");
const csvStatus = document.getElementById("csv-status");
const csvFileInput = document.getElementById("csv-file-input");
const testSoundBtn = document.getElementById("test-sound-btn");
const targetFingerVal = document.getElementById("target-finger-val");
const targetTapProgress = document.getElementById("target-tap-progress");
const songProgressBadge = document.getElementById("song-progress-badge");
const debugPanel = document.getElementById("debug-panel");

// デバッグHUDの表示状態フラグ（非表示時は毎フレームのDOM書き込み・文字列演算をスキップ）
let isDebugPanelVisible = false;
function checkDebugPanelVisibility() {
  if (!debugPanel) {
    isDebugPanelVisible = false;
    return;
  }
  isDebugPanelVisible = window.getComputedStyle(debugPanel).display !== "none";
}
checkDebugPanelVisibility();
window.addEventListener("resize", checkDebugPanelVisibility);

// トラッキングライン（骨格線・指先マーカー）の表示フラグ（デフォルト非表示・localStorageで状態保存）
export const TRACKING_LINES_STORAGE_KEY = "piarno_show_tracking_lines";
export let showTrackingLines = localStorage.getItem(TRACKING_LINES_STORAGE_KEY) === "true";

// ==========================================================================
// BlanchTouch 実験用設定・判定モード管理
// ==========================================================================
export const DETECTION_MODES = {
  LEGACY: "legacy",
  BLANCH_ONLY: "blanch_only",
  HYBRID: "hybrid"
};

// Hybridモードの結合ルール（後からOR条件や従来判定フォールバックへ容易に切替可能）
export const HYBRID_COMBINE_RULES = {
  AND: "AND",
  OR: "OR",
  LEGACY_FALLBACK: "LEGACY_FALLBACK"
};
const HYBRID_COMBINE_RULE = HYBRID_COMBINE_RULES.AND;
const HYBRID_RY_PROXIMITY_RATIO = 0.85;

// PC版は5本指・楽曲演奏モードを標準とし、判定方式は位置・速度ベース(LEGACY)をデフォルトにする
export let IS_FREE_SESSION = true;
const FREE_SESSION_SONG_ID = "free_session";
const FREE_SESSION_FINGER_KEYS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"];
const FREE_SESSION_NOTES = {
  THUMB: { noteName: "C5", label: "ド" },
  INDEX: { noteName: "D5", label: "レ" },
  MIDDLE: { noteName: "E5", label: "ミ" },
  RING: { noteName: "F5", label: "ファ" },
  PINKY: { noteName: "G5", label: "ソ" }
};
const FREE_SESSION_FINGER_KEY = "INDEX";
const ALL_HAND_JOINT_INDICES = Array.from({ length: 21 }, (_, i) => i);
const FREE_SESSION_INTER_FINGER_COOLDOWN_MS = 150; // つられ動き誤検知を防ぐ他指ブロック時間（ms）
const FREE_SESSION_LEGATO_VY_MULTIPLIER = 1.3; // 他指が机についている間の新規打鍵速度閾値倍率

const PC_CAMERA_STORAGE_KEY = "piarno_pc_camera_id";
const PC_MIRROR_STORAGE_KEY = "piarno_pc_mirror";
const PC_ONE_EURO_STORAGE_KEY = "piarno_pc_one_euro";

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

let isOneEuroEnabled = (() => {
  try {
    const saved = localStorage.getItem(PC_ONE_EURO_STORAGE_KEY);
    return saved === null ? true : saved === "true";
  } catch {
    return true;
  }
})();

const BLANCH_STORAGE_KEYS = {
  MODE: "piarno_pc_detection_mode",
  THRESHOLD: "piarno_blanch_threshold",
  ROI_SIZE: "piarno_blanch_roi_size",
  SMOOTHING: "piarno_blanch_smoothing"
};

const BLANCH_DEFAULTS = {
  mode: DETECTION_MODES.LEGACY,
  onThreshold: 3.5,
  roiSize: 24,
  smoothingAlpha: 0.35
};

function loadSavedNumber(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const val = parseFloat(raw);
    return Number.isFinite(val) ? val : fallback;
  } catch {
    return fallback;
  }
}

let detectionMode = (() => {
  try {
    const saved = localStorage.getItem(BLANCH_STORAGE_KEYS.MODE);
    return Object.values(DETECTION_MODES).includes(saved) ? saved : BLANCH_DEFAULTS.mode;
  } catch {
    return BLANCH_DEFAULTS.mode;
  }
})();

const createBlanchDetector = () =>
  new BlanchDetector({
    onThreshold: loadSavedNumber(BLANCH_STORAGE_KEYS.THRESHOLD, BLANCH_DEFAULTS.onThreshold),
    roiSize: loadSavedNumber(BLANCH_STORAGE_KEYS.ROI_SIZE, BLANCH_DEFAULTS.roiSize),
    smoothingAlpha: loadSavedNumber(BLANCH_STORAGE_KEYS.SMOOTHING, BLANCH_DEFAULTS.smoothingAlpha)
  });

const blanchDetector = createBlanchDetector();
const freeFingerBlanchDetectors = {
  THUMB: createBlanchDetector(),
  INDEX: createBlanchDetector(),
  MIDDLE: createBlanchDetector(),
  RING: createBlanchDetector(),
  PINKY: createBlanchDetector()
};

/**
 * 全てのBlanchDetectorインスタンスへ設定変更を反映
 * @param {object} params
 */
function updateAllBlanchSettings(params) {
  blanchDetector.updateSettings(params);
  Object.values(freeFingerBlanchDetectors).forEach((det) => det.updateSettings(params));
}

/**
 * 全てのBlanchDetectorインスタンスのベースラインをリセット
 */
function resetAllBlanchBaselines() {
  blanchDetector.resetBaseline();
  Object.values(freeFingerBlanchDetectors).forEach((det) => det.resetBaseline());
}

// デバッグオーバーレイはデフォルト非表示（?debug=1 指定時または右上の照準ボタンで表示）
let isBlanchDebugVisible = (() => {
  const param = new URLSearchParams(window.location.search).get("debug");
  return param === "1";
})();

/**
 * モード別の打鍵トリガー判定（IDLE -> TOUCHED）
 * @returns {boolean}
 */
function evaluateTapTrigger(mode, currentRy, hitRyTh, tipVy, hitVyTh, blanchResult) {
  const legacyHit = currentRy >= hitRyTh && tipVy >= hitVyTh;
  const blanchHit = Boolean(blanchResult && blanchResult.valid && blanchResult.isBlanching);

  if (mode === DETECTION_MODES.LEGACY) {
    return legacyHit;
  }
  if (mode === DETECTION_MODES.BLANCH_ONLY) {
    return blanchHit;
  }

  // Hybridモード（デフォルト: 誤爆防止優先のAND条件）
  if (HYBRID_COMBINE_RULE === HYBRID_COMBINE_RULES.OR) {
    return legacyHit || blanchHit;
  }
  if (HYBRID_COMBINE_RULE === HYBRID_COMBINE_RULES.LEGACY_FALLBACK) {
    return blanchHit ? (currentRy >= hitRyTh * HYBRID_RY_PROXIMITY_RATIO) : legacyHit;
  }
  // AND条件: 指が鍵盤付近にある（従来変位閾値の近傍）かつ ブランチング検出
  const isNearKeybed = currentRy >= hitRyTh * HYBRID_RY_PROXIMITY_RATIO;
  return isNearKeybed && blanchHit;
}

/**
 * モード別のリフト復帰判定（TOUCHED -> IDLE）
 * @returns {boolean}
 */
function evaluateLiftTrigger(mode, currentRy, hitRyTh, liftRyTh, tipVy, blanchResult) {
  const isLiftByPosition = currentRy <= liftRyTh;
  const isLiftByRebound = tipVy <= -100 && currentRy <= (hitRyTh - 0.02);
  const legacyLift = isLiftByPosition || isLiftByRebound;
  const blanchLift = Boolean(blanchResult && !blanchResult.isBlanching);

  if (mode === DETECTION_MODES.LEGACY) {
    return legacyLift;
  }
  if (mode === DETECTION_MODES.BLANCH_ONLY) {
    return blanchLift;
  }

  // Hybridモード: AND時はどちらかが復帰すればIDLEへ戻し連続打鍵に備える
  if (HYBRID_COMBINE_RULE === HYBRID_COMBINE_RULES.OR) {
    return legacyLift && blanchLift;
  }
  return legacyLift || blanchLift;
}

// テスト用CSVデータセット群（ファイル別）
export let testDataDatasets = [];
// 全ファイルのフレームを結合したフラット配列
export let testDataFrames = [];

// Web Audio API サウンドエンジン ＆ Tone.js Salamander Grand Piano
let audioCtx = null;
let tapCount = 0;
let pianoSampler = null;
let isSamplerLoaded = false;

// 周波数（Hz）からノート名への高精度マッピング（自然な中央オクターブ C4〜G4 を右手メロディ、低音 G2〜B3 を自動伴奏に設定）
const FREQ_TO_NOTE_MAP = {
  // 自動低音伴奏域（G2〜B3）
  98.00: "G2",
  110.00: "A2",
  123.47: "B2",
  130.81: "C3",
  146.83: "D3",
  164.81: "E3",
  174.61: "F3",
  185.00: "F#3",
  196.00: "G3",
  220.00: "A3",
  246.94: "B3",
  // 右手メロディ中央域（C4〜G4）
  261.63: "C4",
  293.66: "D4",
  329.63: "E4",
  349.23: "F4",
  392.00: "G4",
  // （互換用高音域 C5〜G5）
  523.25: "C5",
  587.33: "D5",
  659.25: "E5",
  698.46: "F5",
  783.99: "G5"
};

// ノート名から周波数（Hz）への逆引きマッピング（フォールバック合成時にも使用）
const NOTE_TO_FREQ_MAP = Object.entries(FREQ_TO_NOTE_MAP).reduce((acc, [f, n]) => {
  acc[n] = parseFloat(f);
  return acc;
}, {});

/**
 * Tone.js Salamander Grand Piano 音源サンプラーの初期化
 */
function initPianoSampler() {
  try {
    pianoSampler = new Tone.Sampler({
      urls: {
        A0: "A0.mp3",
        C1: "C1.mp3",
        "D#1": "Ds1.mp3",
        "F#1": "Fs1.mp3",
        A1: "A1.mp3",
        C2: "C2.mp3",
        "D#2": "Ds2.mp3",
        "F#2": "Fs2.mp3",
        A2: "A2.mp3",
        C3: "C3.mp3",
        "D#3": "Ds3.mp3",
        "F#3": "Fs3.mp3",
        A3: "A3.mp3",
        C4: "C4.mp3",
        "D#4": "Ds4.mp3",
        "F#4": "Fs4.mp3",
        A4: "A4.mp3",
        C5: "C5.mp3",
        "D#5": "Ds5.mp3",
        "F#5": "Fs5.mp3",
        A5: "A5.mp3",
        C6: "C6.mp3",
        "D#6": "Ds6.mp3",
        "F#6": "Fs6.mp3",
        A6: "A6.mp3",
        C7: "C7.mp3",
        "D#7": "Ds7.mp3",
        "F#7": "Fs7.mp3",
        A7: "A7.mp3",
        C8: "C8.mp3"
      },
      // 同音連打（ミ・ミ、ド・ド等）時に音が不自然にチョップされず、自然なピアノの減衰・オーバーラップが持続するようリリースを拡張
      release: 2.4,
      baseUrl: "https://tonejs.github.io/audio/salamander/",
      onload: () => {
        isSamplerLoaded = true;
        console.log("[AUDIO] Salamander Grand Piano サンプル音源のロードが完了しました");
      }
    }).toDestination();
  } catch (err) {
    console.warn("[AUDIO] Tone.Sampler 初期化エラー:", err);
  }
}

// サンプラーの初期化を実行
initPianoSampler();

// 学習済みCSVから自動導出される閾値モデル（デフォルト値付き）
let hitRyThreshold = 0.70;
let liftRyThreshold = 0.45;
let trainedHitSamples = 0;

// 打鍵ステートマシン管理（IDLE: 待機, TOUCHED: 机面接触中・リフト待ち）
let tapState = "IDLE";
let lastFreeSessionHitTime = 0;
const freeFingerTapStates = {
  THUMB: "IDLE",
  INDEX: "IDLE",
  MIDDLE: "IDLE",
  RING: "IDLE",
  PINKY: "IDLE"
};

/**
 * フリーセッション用5本指の打鍵状態をすべてIDLEにリセット
 */
function resetFreeFingerStates() {
  lastFreeSessionHitTime = 0;
  for (const key of FREE_SESSION_FINGER_KEYS) {
    freeFingerTapStates[key] = "IDLE";
  }
}

// 画面自動消灯防止（Screen Wake Lock API）の管理用センチネル
let wakeLockSentinel = null;

/**
 * 画面スリープ防止（Screen Wake Lock API）の要求
 * iPad等の演奏中に画面が自動消灯しないよう画面ロックを保持
 */
export async function requestWakeLock() {
  if ("wakeLock" in navigator && typeof navigator.wakeLock.request === "function") {
    try {
      if (wakeLockSentinel && !wakeLockSentinel.released) {
        return;
      }
      wakeLockSentinel = await navigator.wakeLock.request("screen");
      wakeLockSentinel.addEventListener("release", () => {
        console.log("[WAKE LOCK] スクリーンロックが解放されました");
        wakeLockSentinel = null;
      });
      console.log("[WAKE LOCK] 画面スリープ防止を有効化しました");
    } catch (err) {
      console.warn("[WAKE LOCK] 画面スリープ防止の取得スキップ（非対応または低電力モード等）:", err);
    }
  }
}

/**
 * スクリーンロックの安全な解放
 */
export async function releaseWakeLock() {
  if (wakeLockSentinel) {
    try {
      await wakeLockSentinel.release();
    } catch (err) {
      console.warn("[WAKE LOCK] 解放エラー:", err);
    }
    wakeLockSentinel = null;
  }
}

/**
 * AudioContextの初期化・再開（ブラウザのAutoplay PolicyおよびiOS Safariのサスペンド・interrupted対応）
 */
export function ensureAudioContext() {
  if (Tone.context.state !== "running") {
    Tone.start().catch((err) => {
      console.warn("[AUDIO] Tone.start エラー:", err);
    });
  }

  const rawCtx = Tone.getContext().rawContext;
  if (rawCtx && (rawCtx.state === "suspended" || rawCtx.state === "interrupted")) {
    rawCtx.resume().catch((err) => {
      console.warn("[AUDIO] AudioContext resume failed:", err);
    });
  }

  // 初回サウンド有効化バナーがあれば非表示にする
  const banner = document.getElementById("audio-start-banner");
  if (banner && !banner.classList.contains("hidden")) {
    banner.classList.add("hidden");
  }

  return rawCtx;
}

/**
 * サンプルロード前やエラー時のオシレーター波形合成フォールバック（Web Audio API 加算合成）
 * @param {number} freq 周波数 (Hz)
 * @param {number} duration 持続時間 (秒)
 */
function playSynthFallback(freq = 523.25, duration = 0.8) {
  const ctx = ensureAudioContext();
  if (!ctx) return;

  try {
    const now = ctx.currentTime;
    const masterGain = ctx.createGain();

    const oscBase = ctx.createOscillator();
    const gainBase = ctx.createGain();
    oscBase.type = "triangle";
    oscBase.frequency.setValueAtTime(freq, now);
    gainBase.gain.setValueAtTime(0.40, now);
    oscBase.connect(gainBase);
    gainBase.connect(masterGain);

    const oscHarmonic = ctx.createOscillator();
    const gainHarmonic = ctx.createGain();
    oscHarmonic.type = "sine";
    oscHarmonic.frequency.setValueAtTime(freq * 2, now);
    gainHarmonic.gain.setValueAtTime(0.20, now);
    oscHarmonic.connect(gainHarmonic);
    gainHarmonic.connect(masterGain);

    masterGain.gain.setValueAtTime(0.001, now);
    masterGain.gain.linearRampToValueAtTime(0.65, now + 0.004);
    // 自然な減衰
    masterGain.gain.exponentialRampToValueAtTime(0.0001, now + Math.max(0.35, duration));

    masterGain.connect(ctx.destination);

    oscBase.start(now);
    oscHarmonic.start(now);
    oscBase.stop(now + duration + 0.05);
    oscHarmonic.stop(now + duration + 0.05);
  } catch (err) {
    console.warn("[AUDIO] playSynthFallback 発音エラー:", err);
  }
}

/**
 * Salamander Grand Piano実機サンプリング音源 ＆ フォールバック合成によるピアノ発音（和音・オプション対応）
 * @param {number|string|Array<string|number>} targetFreqOrNote 周波数 (Hz)、音名 ("C5" 等)、または音名の配列 (["G3", "C4"] 等)
 * @param {boolean} countAsTap 打鍵回数としてカウントするかどうか（自動伴奏等はfalse）
 * @param {object} options 発音オプション { duration: string|number, isBass: boolean }
 */
export function playTapSound(targetFreqOrNote = 523.25, countAsTap = true, options = {}) {
  // 配列が渡された場合は和音（各構成音）として自然に同時発音
  if (Array.isArray(targetFreqOrNote)) {
    targetFreqOrNote.forEach((note) => {
      playTapSound(note, false, options);
    });
    if (countAsTap) {
      tapCount++;
      if (tapCountVal) tapCountVal.textContent = `${tapCount}`;
    }
    return;
  }

  ensureAudioContext();

  const duration = options.duration || (options.isBass ? "1m" : "2n");
  const fallbackSec = duration === "1m" ? 2.4 : duration === "2n" ? 1.2 : 0.6;

  let noteName = null;
  let freq = 523.25;

  if (typeof targetFreqOrNote === "string") {
    noteName = targetFreqOrNote;
    freq = NOTE_TO_FREQ_MAP[noteName] || 523.25;
  } else if (typeof targetFreqOrNote === "number") {
    freq = targetFreqOrNote;
    const roundedFreq = Math.round(freq * 100) / 100;
    noteName = FREQ_TO_NOTE_MAP[roundedFreq] || freq;
  }

  if (isSamplerLoaded && pianoSampler) {
    try {
      pianoSampler.triggerAttackRelease(noteName, duration);
    } catch (e) {
      console.warn("[AUDIO] Sampler 発音エラー、フォールバック合成を使用:", e);
      playSynthFallback(freq, fallbackSec);
    }
  } else {
    // サンプル音源ロード前・未完了時はフォールバック合成で確実に発音
    playSynthFallback(freq, fallbackSec);
  }

  if (countAsTap) {
    tapCount++;
    if (tapCountVal) {
      tapCountVal.textContent = `${tapCount}`;
    }
  }
}

// 状態管理
let handLandmarker = null;
let microHandpose = null;
let microInputCanvas = null;
let microInputCtx = null;
let activeTrackingEngine = "NONE"; // "MICRO_HANDPOSE" | "MEDIAPIPE"
let isInferringFrame = false;
let pendingVideoFrameTime = null;

/**
 * MicroPose 用の中間入力 Canvas を確保・同期する（毎フレームの createImageBitmap オーバーヘッド回避）
 * アスペクト比を維持したまま長辺最大640pxに収めることでGPUテクスチャ転送負荷も最小化
 * @param {number} vw
 * @param {number} vh
 * @returns {HTMLCanvasElement}
 */
function ensureMicroInputCanvas(vw, vh) {
  const maxDim = 640;
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
let currentStream = null;
let isPredicting = false;
// 上限FPS（カメラ・アプリともに60fpsを上限に固定。50/60fpsジッターによる半減を防ぐため10ms未満の過剰到着のみ間引き）
const MAX_APP_FPS = 60;
const MIN_APP_DISPATCH_INTERVAL_MS = 10.0;

let lastVideoTime = -1;
let lastFpsUpdateTime = performance.now();
let frameCount = 0;
let lastAppProcessedTime = 0;
let cameraFrameCount = 0;
let lastPresentedFrames = -1;
let lastCameraFpsUpdateTime = performance.now();

// 人差し指先端（TIP）の最新生ピクセル座標
export const currentTip = {
  x: 0,
  y: 0
};

/**
 * 1 Euro Filter（ワンユーロフィルター）
 * 低速時はジッター（微小な震え）を完全に平滑化し、高速打鍵時は遅延ゼロで追従する標準適応フィルター
 */
class OneEuroFilter {
  constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  reset(x = null, timestamp = null) {
    this.xPrev = x;
    this.dxPrev = 0;
    this.tPrev = timestamp;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, timestamp = performance.now()) {
    if (this.xPrev === null || this.tPrev === null) {
      this.xPrev = x;
      this.dxPrev = 0;
      this.tPrev = timestamp;
      return x;
    }

    const dt = Math.max((timestamp - this.tPrev) / 1000, 0.001);
    this.tPrev = timestamp;

    // 速度（1階差分）の平滑化
    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxHat;

    // 速度に応じた動的カットオフ周波数（素早い動きでは自動で高くなり遅延ゼロ）
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff, dt);

    // 信号の平滑化
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;

    return xHat;
  }
}

/**
 * 2D関節座標用 1 Euro Filter
 */
class PointFilter {
  constructor(minCutoff = 1.2, beta = 0.008) {
    this.xf = new OneEuroFilter(minCutoff, beta);
    this.yf = new OneEuroFilter(minCutoff, beta);
    this.rawYPrev = null;
    this.rawTimePrev = null;
    this.rawVy = 0;
    this.isBypassed = false;
  }

  reset(x = null, y = null, timestamp = null) {
    this.xf.reset(x, timestamp);
    this.yf.reset(y, timestamp);
    this.rawYPrev = y;
    this.rawTimePrev = timestamp;
    this.rawVy = 0;
  }

  /**
   * 過去座標からの引きずりをバイパスし、指定座標で即座にスナップ初期化
   * @param {number} x
   * @param {number} y
   * @param {number} timestamp
   * @param {object} out
   */
  snap(x, y, timestamp, out = null) {
    this.reset(x, y, timestamp);
    if (out) {
      out.x = x;
      out.y = y;
      return out;
    }
    return {
      x,
      y
    };
  }

  filter(x, y, timestamp, out = null, bypass = false) {
    // 再ON時の座標飛びを防ぐため内部フィルタ状態も常に追従させる
    const fx = this.xf.filter(x, timestamp);
    const fy = this.yf.filter(y, timestamp);

    if (this.rawYPrev !== null && this.rawTimePrev !== null) {
      const dt = Math.max((timestamp - this.rawTimePrev) / 1000, 1 / 240);
      this.rawVy = (y - this.rawYPrev) / dt;
    }
    this.rawYPrev = y;
    this.rawTimePrev = timestamp;
    this.isBypassed = bypass;

    const outX = bypass ? x : fx;
    const outY = bypass ? y : fy;
    if (out) {
      out.x = outX;
      out.y = outY;
      return out;
    }
    return {
      x: outX,
      y: outY
    };
  }

  /**
   * Y軸下降速度（ピクセル/秒）を取得（バイパス時は生座標差分から算出）
   * 下向き移動時は正の値（> 0）、上向き移動時は負の値（< 0）
   * @returns {number}
   */
  getVelocityY() {
    return this.isBypassed ? this.rawVy : this.yf.dxPrev;
  }
}

// 日本語指名マッピング（本システムは右手のみでの演奏を前提）
export const FINGER_NAMES = {
  THUMB: "親指",
  INDEX: "人差し指",
  MIDDLE: "中指",
  RING: "薬指",
  PINKY: "小指"
};

// 対象指の定義（右手 親指〜小指までの全指・ハ長調基本ポジション C5〜G5）
export const FINGER_CONFIGS = {
  THUMB: {
    fingerNum: 1,
    key: "THUMB",
    hand: "右手",
    shortName: "親指",
    label: "右手 親指 (1: ド)",
    freq: 523.25, // C5 (ド)
    baseIdx: 2,   // MCP (付け根)
    p1Idx: 1,     // CMC
    p2Idx: 2,     // MCP
    p3Idx: 3,     // IP
    tipIdx: 4,    // TIP
    indices: [1, 2, 3, 4]
  },
  INDEX: {
    fingerNum: 2,
    key: "INDEX",
    hand: "右手",
    shortName: "人差し指",
    label: "右手 人差し指 (2: レ)",
    freq: 587.33, // D5 (レ)
    baseIdx: 5,   // MCP
    p1Idx: 5,     // MCP
    p2Idx: 6,     // PIP
    p3Idx: 7,     // DIP
    tipIdx: 8,    // TIP
    indices: [5, 6, 7, 8]
  },
  MIDDLE: {
    fingerNum: 3,
    key: "MIDDLE",
    hand: "右手",
    shortName: "中指",
    label: "右手 中指 (3: ミ)",
    freq: 659.25, // E5 (ミ)
    baseIdx: 9,   // MCP
    p1Idx: 9,     // MCP
    p2Idx: 10,    // PIP
    p3Idx: 11,    // DIP
    tipIdx: 12,   // TIP
    indices: [9, 10, 11, 12]
  },
  RING: {
    fingerNum: 4,
    key: "RING",
    hand: "右手",
    shortName: "薬指",
    label: "右手 薬指 (4: ファ)",
    freq: 698.46, // F5 (ファ)
    baseIdx: 13,  // MCP
    p1Idx: 13,    // MCP
    p2Idx: 14,    // PIP
    p3Idx: 15,    // DIP
    tipIdx: 16,   // TIP
    indices: [13, 14, 15, 16]
  },
  PINKY: {
    fingerNum: 5,
    key: "PINKY",
    hand: "右手",
    shortName: "小指",
    label: "右手 小指 (5: ソ)",
    freq: 783.99, // G5 (ソ)
    baseIdx: 17,  // MCP
    p1Idx: 17,    // MCP
    p2Idx: 18,    // PIP
    p3Idx: 19,    // DIP
    tipIdx: 20,   // TIP
    indices: [17, 18, 19, 20]
  }
};

// 『よろこびのうた（Ode to Joy）』運指・音名シーケンス定義（右手5音固定 C5〜G5・楽譜通りの左手自動伴奏 G2〜C4・全16小節/62音）
// ※演奏者は右手のみを使用します（左手の操作・打鍵指示は一切ありません。低音・和音はアプリ側の自動伴奏音です）
// 1:親指(C5/ド), 2:人差し指(D5/レ), 3:中指(E5/ミ★第1音), 4:薬指(F5/ファ), 5:小指(G5/ソ)
export const ODE_TO_JOY_SEQUENCE = [
  // ==========================================
  // 第1節（フレーズ1）: ミ ミ ファ ソ ｜ ソ ファ ミ レ ｜ ド ド レ ミ ｜ ミ レ レ ─
  // ==========================================
  // 小節1: ミ ミ ファ ソ - 左手: C3(ド) → [G3, C4](ソ・ド) → E3(ミ) → [G3, C4](ソ・ド)
  { step: 1,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 2,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  { step: 3,  phrase: 1, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: "E3" },
  { step: 4,  phrase: 1, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: ["G3", "C4"] },
  // 小節2: ソ ファ ミ レ - 左手: G2(低音ソ) → [G3, B3](ソ・シ) → D3(レ) → [G3, B3](ソ・シ)
  { step: 5,  phrase: 1, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: "G2" },
  { step: 6,  phrase: 1, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: ["G3", "B3"] },
  { step: 7,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "D3" },
  { step: 8,  phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G3", "B3"] },
  // 小節3: ド ド レ ミ - 左手: C3(ド) → [G3, C4](ソ・ド) → E3(ミ) → [G3, C4](ソ・ド)
  { step: 9,  phrase: 1, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: "C3" },
  { step: 10, phrase: 1, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["G3", "C4"] },
  { step: 11, phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "E3" },
  { step: 12, phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  // 小節4: ミ レ レ ─ - 左手: G2 → [D3, F3] → [G2, D3]
  { step: 13, phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "G2" },
  { step: 14, phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["D3", "F3"] },
  { step: 15, phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G2", "D3"] },

  // ==========================================
  // 第2節（フレーズ2）: ミ ミ ファ ソ ｜ ソ ファ ミ レ ｜ ド ド レ ミ ｜ レ ド ド ─
  // ==========================================
  // 小節5: ミ ミ ファ ソ - 左手: C3 → [G3, C4] → E3 → [G3, C4]
  { step: 16, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 17, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  { step: 18, phrase: 2, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: "E3" },
  { step: 19, phrase: 2, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: ["G3", "C4"] },
  // 小節6: ソ ファ ミ レ - 左手: G2 → [G3, B3] → D3 → [G3, B3]
  { step: 20, phrase: 2, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: "G2" },
  { step: 21, phrase: 2, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: ["G3", "B3"] },
  { step: 22, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "D3" },
  { step: 23, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G3", "B3"] },
  // 小節7: ド ド レ ミ - 左手: C3 → [G3, C4] → E3 → [G3, C4]
  { step: 24, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: "C3" },
  { step: 25, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["G3", "C4"] },
  { step: 26, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "E3" },
  { step: 27, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  // 小節8: レ ド ド ─ - 左手: [C3, G3, C4] → G2 → [C3, E3, C4]
  { step: 28, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["C3", "G3", "C4"] },
  { step: 29, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: "G2" },
  { step: 30, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["C3", "E3", "C4"] },

  // ==========================================
  // 第3節（フレーズ3）: レ レ ミ ド ｜ レ ミ(短) ファ(短) ミ ド ｜ レ ミ(短) ファ(短) ミ レ ｜ ド レ ソ ─
  // ==========================================
  // 小節9: レ レ ミ ド - 左手: [G2, G3] → [F3, G3] → [G2, G3] → [F3, G3]
  { step: 31, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G2", "G3"] },
  { step: 32, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["F3", "G3"] },
  { step: 33, phrase: 3, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G2", "G3"] },
  { step: 34, phrase: 3, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["F3", "G3"] },
  // 小節10: レ ミ(短) ファ(短) ミ ド - 左手: [G2, G3] → 伴奏保持 → [F3, G3] → [G2, G3] → [F3, G3]
  { step: 35, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G2", "G3"] },
  { step: 36, phrase: 3, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 37, phrase: 4, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: ["F3", "G3"] },
  { step: 38, phrase: 3, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G2", "G3"] },
  { step: 39, phrase: 3, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["F3", "G3"] },
  // 小節11: レ ミ(短) ファ(短) ミ レ - 左手: [G2, G3] → 伴奏保持 → [F3, G3] → [G2, G3] → E3
  { step: 40, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G2", "G3"] },
  { step: 41, phrase: 3, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 42, phrase: 4, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: ["F3", "G3"] },
  { step: 43, phrase: 3, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G2", "G3"] },
  { step: 44, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "E3" },
  // 小節12: ド レ ソ ─ - 左手: [C3, E3] → [D3, F#3] → [G2, G3, B3]
  { step: 45, phrase: 3, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["C3", "E3"] },
  { step: 46, phrase: 3, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["D3", "F#3"] },
  { step: 47, phrase: 3, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: ["G2", "G3", "B3"] },

  // ==========================================
  // 第4節（フレーズ4）: ミ ミ ファ ソ ｜ ソ ファ ミ レ ｜ ド ド レ ミ ｜ レ ド ド ─
  // ==========================================
  // 小節13: ミ ミ ファ ソ - 左手: C3 → [G3, C4] → E3 → [G3, C4]
  { step: 48, phrase: 4, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 49, phrase: 4, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  { step: 50, phrase: 4, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: "E3" },
  { step: 51, phrase: 4, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: ["G3", "C4"] },
  // 小節14: ソ ファ ミ レ - 左手: G2 → [G3, B3] → D3 → [G3, B3]
  { step: 52, phrase: 4, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: "G2" },
  { step: 53, phrase: 4, fingerNum: 4, fingerKey: "RING",   note: "ファ", rightNote: "F5", freq: 698.46, autoLeftNote: ["G3", "B3"] },
  { step: 54, phrase: 4, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "D3" },
  { step: 55, phrase: 4, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["G3", "B3"] },
  // 小節15: ド ド レ ミ - 左手: C3 → [G3, C4] → E3 → [G3, C4]
  { step: 56, phrase: 4, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: "C3" },
  { step: 57, phrase: 4, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["G3", "C4"] },
  { step: 58, phrase: 4, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "E3" },
  { step: 59, phrase: 4, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: ["G3", "C4"] },
  // 小節16: レ ド ド ─ - 左手: [C3, G3, C4] → [G2, D3, G3] → [C3, E3, G3, C4]
  { step: 60, phrase: 4, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: ["C3", "G3", "C4"] },
  { step: 61, phrase: 4, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["G2", "D3", "G3"] },
  { step: 62, phrase: 4, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: ["C3", "E3", "G3", "C4"] }
];

// 『メリーさんのひつじ（Mary Had a Little Lamb）』運指・音名シーケンス定義（右手4音 C5, D5, E5, G5・自動伴奏 C3, G3・全8小節/26音）
// ※演奏者は右手のみを使用します（左手の操作・打鍵指示は一切ありません。低音はアプリ側の自動伴奏音です）
// 1:親指(C5/ド), 2:人差し指(D5/レ), 3:中指(E5/ミ★第1音), 5:小指(G5/ソ)
export const MARY_HAD_A_LITTLE_LAMB_SEQUENCE = [
  // ==========================================
  // 前半（フレーズ1）: ミ レ ド レ ｜ ミ ミ ミ ─ ｜ レ レ レ ─ ｜ ミ ソ ソ ─
  // ==========================================
  // 小節1: ミ レ ド レ - 自動伴奏: C3
  { step: 1,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 2,  phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  { step: 3,  phrase: 1, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: null },
  { step: 4,  phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  // 小節2: ミ ミ ミ ─ - 自動伴奏: C3
  { step: 5,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 6,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 7,  phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  // 小節3: レ レ レ ─ - 自動伴奏: G3
  { step: 8,  phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "G3" },
  { step: 9,  phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  { step: 10, phrase: 1, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  // 小節4: ミ ソ ソ ─ - 自動伴奏: C3
  { step: 11, phrase: 1, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 12, phrase: 1, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: null },
  { step: 13, phrase: 1, fingerNum: 5, fingerKey: "PINKY",  note: "ソ", rightNote: "G5", freq: 783.99, autoLeftNote: null },

  // ==========================================
  // 後半（フレーズ2）: ミ レ ド レ ｜ ミ ミ ミ ミ ｜ レ レ ミ レ ｜ ド ─ ─ ─
  // ==========================================
  // 小節5: ミ レ ド レ - 自動伴奏: C3
  { step: 14, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 15, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  { step: 16, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: null },
  { step: 17, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  // 小節6: ミ ミ ミ ミ - 自動伴奏: C3
  { step: 18, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: "C3" },
  { step: 19, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 20, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 21, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  // 小節7: レ レ ミ レ - 自動伴奏: G3
  { step: 22, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: "G3" },
  { step: 23, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  { step: 24, phrase: 2, fingerNum: 3, fingerKey: "MIDDLE", note: "ミ", rightNote: "E5", freq: 659.25, autoLeftNote: null },
  { step: 25, phrase: 2, fingerNum: 2, fingerKey: "INDEX",  note: "レ", rightNote: "D5", freq: 587.33, autoLeftNote: null },
  // 小節8: ド ─ ─ ─ - 自動伴奏: C3
  { step: 26, phrase: 2, fingerNum: 1, fingerKey: "THUMB",  note: "ド", rightNote: "C5", freq: 523.25, autoLeftNote: "C3" }
];

// 演奏曲リスト（自分のペースで1音ずつ進めるステップ演奏）
export const SONGS = {
  ode_to_joy: {
    id: "ode_to_joy",
    title: "よろこびのうた",
    sequence: ODE_TO_JOY_SEQUENCE
  },
  mary_had_a_little_lamb: {
    id: "mary_had_a_little_lamb",
    title: "メリーさんの羊",
    sequence: MARY_HAD_A_LITTLE_LAMB_SEQUENCE
  },
  mary: {
    id: "mary_had_a_little_lamb",
    title: "メリーさんの羊",
    sequence: MARY_HAD_A_LITTLE_LAMB_SEQUENCE
  }
};

export let currentSongId = "ode_to_joy";
export let currentSequence = SONGS.ode_to_joy.sequence;
export let currentSongStep = 0; // 現在の進行ステップ (0 〜 sequence.length - 1)
export let currentFingerKey = IS_FREE_SESSION ? FREE_SESSION_FINGER_KEY : currentSequence[0].fingerKey; // 初期ターゲット指

// 3秒カウントダウン状態管理
export let isCountingDown = false;
let countdownTimerId = null;

/**
 * 3秒カウントダウンの実行（3 → 2 → 1 → START!）
 * @param {Function} callback カウントダウン完了後のコールバック
 */
export function startCountdown(callback) {
  if (countdownTimerId) {
    clearInterval(countdownTimerId);
    countdownTimerId = null;
  }
  isCountingDown = true;
  tapState = "IDLE";
  resetFreeFingerStates();
  updateStateHud("COUNTDOWN", false);

  const overlay = document.getElementById("countdown-overlay");
  const textEl = document.getElementById("countdown-text");
  if (!overlay || !textEl) {
    isCountingDown = false;
    if (callback) callback();
    return;
  }

  overlay.classList.remove("hidden");
  let count = 3;
  textEl.textContent = `${count}`;
  textEl.style.transform = "scale(1.2)";
  setTimeout(() => {
    if (textEl) textEl.style.transform = "scale(1.0)";
  }, 60);

  countdownTimerId = setInterval(() => {
    count--;
    if (count > 0) {
      textEl.textContent = `${count}`;
      textEl.style.transform = "scale(1.2)";
      setTimeout(() => {
        if (textEl) textEl.style.transform = "scale(1.0)";
      }, 60);
    } else if (count === 0) {
      textEl.textContent = "START!";
      textEl.style.transform = "scale(1.3)";
      setTimeout(() => {
        if (textEl) textEl.style.transform = "scale(1.0)";
      }, 60);
    } else {
      clearInterval(countdownTimerId);
      countdownTimerId = null;
      overlay.classList.add("hidden");
      isCountingDown = false;
      updateStateHud("IDLE", false);
      if (callback) callback();
    }
  }, 1000);
}

// 完走クリア演出の状態管理（クリア演出中は打鍵認識を一時停止）
export let isClearing = false;
let clearTimerId = null;

/**
 * 完走時の白黒ミニマルクリア通知のフェード表示（約1.5秒間）
 * @param {Function} onComplete フェードアウト完了後のコールバック
 */
export function showClearNotification(onComplete) {
  const overlay = document.getElementById("clear-overlay");
  if (!overlay) {
    if (onComplete) onComplete();
    return;
  }

  isClearing = true;
  overlay.classList.remove("hidden");
  // 強制リフローまたは微小ディレイでCSS opacity トランジションを確実に開始
  requestAnimationFrame(() => {
    overlay.classList.add("show");
  });

  if (clearTimerId) clearTimeout(clearTimerId);

  // 約1.5秒間シンプルにフェード表示した後、フェードアウトしてループ復帰
  clearTimerId = setTimeout(() => {
    overlay.classList.remove("show");
    setTimeout(() => {
      overlay.classList.add("hidden");
      isClearing = false;
      if (onComplete) onComplete();
    }, 300); // CSSのtransition 0.3s完了後にhidden化
  }, 1500);
}

/**
 * 楽曲の切り替え
 * @param {string} songId
 * @param {boolean} withCountdown カウントダウンを開始するかどうか（起動時は手動制御）
 */
export function selectSong(songId, withCountdown = true) {
  const isFreeMode = songId === FREE_SESSION_SONG_ID;
  const targetSong = isFreeMode ? null : SONGS[songId];
  if (!isFreeMode && !targetSong) return;
  const canonicalId = isFreeMode ? FREE_SESSION_SONG_ID : targetSong.id;

  // クリア演出中であればタイマーとオーバーレイをリセット
  if (clearTimerId) {
    clearTimeout(clearTimerId);
    clearTimerId = null;
  }
  isClearing = false;
  const clearOverlay = document.getElementById("clear-overlay");
  if (clearOverlay) {
    clearOverlay.classList.remove("show");
    clearOverlay.classList.add("hidden");
  }

  IS_FREE_SESSION = isFreeMode;
  currentSongId = canonicalId;
  tapState = "IDLE";
  resetFreeFingerStates();
  resetAllBlanchBaselines();

  if (typeof noteProgressMap !== "undefined") {
    noteProgressMap.clear();
  }

  // 右上ポップアップメニューのアクティブクラス更新
  document.querySelectorAll(".song-menu-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.song === canonicalId);
  });

  // 起動モーダルの選択状態も同期
  document.querySelectorAll(".start-song-item").forEach((btn) => {
    const isSelected = btn.dataset.song === canonicalId;
    btn.classList.toggle("selected", isSelected);
    const check = btn.querySelector(".start-song-check");
    if (check) check.textContent = isSelected ? "●" : "○";
  });

  // メニューを閉じる
  const menu = document.getElementById("song-select-menu");
  if (menu) menu.classList.add("hidden");

  if (isFreeMode) {
    // フリーセッション：5本指自由演奏モード
    renderSongGuideUI();
    if (withCountdown) {
      startCountdown(() => {
        console.log("[MODE SELECT] Free Session 開始");
      });
    }
    return;
  }

  currentSequence = targetSong.sequence;
  currentSongStep = 0;

  // 1音目のターゲット指をセット＆ガイドUI再描画
  setTargetFinger(currentSequence[0].fingerKey);
  renderSongGuideUI();

  if (withCountdown) {
    // カウントダウン開始
    startCountdown(() => {
      console.log(`[SONG] ${SONGS[songId].title} 開始！第1音: ${currentSequence[0].note} (${currentSequence[0].fingerKey})`);
    });
  }
}

// 手の全21ランドマーク専用の適応平滑化フィルター（1 Euro Filter）
// 手首（0）から親指・人差し指・中指・薬指・小指（20）まで全関節を独立して常時平滑化
const landmarkFilters = Array.from({ length: 21 }, () => new PointFilter(1.2, 0.008));

// 全21関節点用の平滑化座標オブジェクトプール（毎フレームの新規オブジェクト生成・破棄によるGCスパイクを完全排除）
const smoothedLandmarksPool = Array.from({ length: 21 }, () => ({ x: 0, y: 0 }));

// 一時的な検出ロスト対策（打鍵時の陰影による瞬断で骨格が点滅・ジャンプするのを防止しつつ、退出時は約0.1秒で即消去）
let lostFrames = 0;
const MAX_LOST_FRAMES = 0; // カクつき防止のため静止補間を無効化し生追従を優先
let hasValidSmoothedLandmarks = false;
let lastRawLandmarks = null;

// 右手トラッキングロック用：追従中の右手手首（Landmark 0）の正規化座標
let lastTrackedWrist = null;
// 平滑化対象指の切り替え検知用（指変更時にスナップ初期化）
let lastFilteredFingerKey = null;

// 手の骨格コネクション定義（全21ランドマーク間の接続ペア [startIdx, endIdx]）
const HAND_CONNECTIONS = [
  // 手のひら
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  // 親指
  [1, 2], [2, 3], [3, 4],
  // 人差し指（背景薄表示）
  [5, 6], [6, 7], [7, 8],
  // 中指
  [9, 10], [10, 11], [11, 12],
  // 薬指
  [13, 14], [14, 15], [15, 16],
  // 小指
  [17, 18], [18, 19], [19, 20]
];

// 指ごとの対象外骨格接続線のキャッシュ（毎フレームのfilter処理・配列アロケーションを完全排除）
const cachedOtherConnections = {};
export let currentOtherConnections = [];

/**
 * ターゲット指変更時の骨格接続線キャッシュ更新
 * @param {string} fingerKey
 */
export function updateCachedConnections(fingerKey) {
  if (cachedOtherConnections[fingerKey]) {
    currentOtherConnections = cachedOtherConnections[fingerKey];
    return;
  }
  const cfg = FINGER_CONFIGS[fingerKey] || FINGER_CONFIGS.THUMB;
  const targetIndices = cfg.indices;
  const connections = HAND_CONNECTIONS.filter(
    ([s, e]) => !(targetIndices.includes(s) && targetIndices.includes(e))
  );
  cachedOtherConnections[fingerKey] = connections;
  currentOtherConnections = connections;
}

// 指ごとの推奨平滑化下降速度閾値（ピクセル/秒）
// 打鍵時は急峻な下降（300〜1600 px/s）が発生し、静止待機時（0〜8 px/s）を完全に遮断
export const FINGER_VY_THRESHOLDS = {
  THUMB: 160,
  INDEX: 220,
  MIDDLE: 220,
  RING: 180,
  PINKY: 180
};

// 指ごとの学習閾値モデル
// 速度ゲート（下降速度 vy）を併用するため、静止待機誤爆を心配せず変位閾値を浅く緩和して軽いタッチを確実に検知
export const fingerThresholdModels = {
  THUMB: { hitRy: 0.35, liftRy: 0.28, samples: 0 },
  MIDDLE: { hitRy: 0.60, liftRy: 0.52, samples: 0 },
  RING: { hitRy: 0.55, liftRy: 0.48, samples: 0 },
  PINKY: { hitRy: 0.50, liftRy: 0.42, samples: 0 },
  INDEX: { hitRy: 0.60, liftRy: 0.52, samples: 0 }
};

// 指ごとのデータセット格納用
export const fingerDatasets = {
  THUMB: [],
  MIDDLE: [],
  RING: [],
  PINKY: [],
  INDEX: []
};

/**
 * ターゲット指の変更と個別閾値の適用
 * ※指が切り替わっても全ランドマークフィルターはリセットせず継続し、座標飛びを防止
 * @param {string} fingerKey
 */
export function setTargetFinger(fingerKey) {
  if (!FINGER_CONFIGS[fingerKey]) return;
  currentFingerKey = fingerKey;

  // 骨格接続線キャッシュを更新
  updateCachedConnections(fingerKey);

  // 指別学習モデルの閾値を適用
  applyFingerThreshold(fingerKey);

  // 指が切り替わったらブランチング基準値をリセット（異なる指の肌色混同防止）
  blanchDetector.resetBaseline();

  console.log(`[FINGER TARGET] ターゲット指: ${FINGER_CONFIGS[fingerKey].label} (TH: ${hitRyThreshold.toFixed(2)})`);
}

// 初回ターゲット指の骨格接続線を初期化
updateCachedConnections(currentFingerKey);

/**
 * 画面右上の右手運指ガイドUIを更新（指定指を純白ハイライト）
 * @param {string} fingerKey "THUMB" | "INDEX" | "MIDDLE" | "RING" | "PINKY"
 * @param {number|string} fingerNum 1〜5
 * @param {string} note "ド", "レ", "ミ" 等
 */
export function updateHandGuideWidget(fingerKey, fingerNum, note) {
  const fingerEls = document.querySelectorAll(".hand-finger");
  fingerEls.forEach((el) => {
    if (el.dataset.finger === fingerKey) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  const label = document.getElementById("hand-guide-label");
  if (label) {
    label.textContent = `右手 ${fingerNum || ""} ${note || ""}`.trim();
  }
}

/**
 * 楽曲進行とターゲット指UIの更新
 */
export function renderSongGuideUI() {
  if (IS_FREE_SESSION) {
    // フリーセッション時は5本指すべてが演奏対象のため、ガイドは待機状態にし曲進捗は非表示にする
    updateHandGuideWidget("", "", "");
    if (targetFingerVal) {
      targetFingerVal.textContent = "右手 5本指 (ド〜ソ)";
    }
    if (targetTapProgress) {
      targetTapProgress.textContent = "FREE";
    }
    if (songProgressBadge) {
      songProgressBadge.textContent = "FREE";
      songProgressBadge.classList.add("hidden");
    }
    return;
  }

  const currentItem = currentSequence[currentSongStep];
  if (!currentItem) return;

  // 画面右上：右手運指ガイドウィジェットのハイライト更新
  updateHandGuideWidget(currentItem.fingerKey, currentItem.fingerNum, currentItem.note);

  if (targetFingerVal) {
    const fingerName = FINGER_NAMES[currentItem.fingerKey] || currentItem.fingerKey;
    targetFingerVal.textContent = `右手 ${currentItem.fingerNum} ${currentItem.note} (${fingerName})`;
    targetFingerVal.classList.add("spike-highlight");
    setTimeout(() => {
      if (targetFingerVal) targetFingerVal.classList.remove("spike-highlight");
    }, 200);
  }
  if (targetTapProgress) {
    targetTapProgress.textContent = `${currentItem.step} / ${currentSequence.length}`;
  }
  // 画面上部：ミニマル進捗バッジ（白黒ミニマル）
  if (songProgressBadge) {
    songProgressBadge.textContent = `${currentItem.step} / ${currentSequence.length}`;
  }
}

/**
 * 現在の音符のハイライト色を取得
 * - 複数弾く指（同一指の連続）：1打目＝黄色、2打目＝緑色、3打目＝青色
 * - 単発（1回のみ弾く指）：水色（シアン）
 * @param {number} stepIndex
 */
export function getTargetFingerColor(stepIndex) {
  const current = IS_FREE_SESSION ? null : currentSequence[stepIndex];
  if (!current) {
    return {
      stroke: "rgba(0, 229, 255, 0.95)",
      halo: "rgba(0, 229, 255, 0.25)",
      accent: "rgba(0, 229, 255, 0.65)",
      glow: "rgba(0, 229, 255, 0.8)",
      fill: "#00e5ff",
      r: 0,
      g: 229,
      b: 255,
      rgb: "0, 229, 255",
      name: "cyan"
    };
  }

  // 連続打鍵グループの先頭を探索
  let start = stepIndex;
  while (start > 0 && currentSequence[start - 1].fingerKey === current.fingerKey) {
    start--;
  }

  // 連続打鍵グループの末尾を探索
  let end = stepIndex;
  while (end < currentSequence.length - 1 && currentSequence[end + 1].fingerKey === current.fingerKey) {
    end++;
  }

  const groupLen = end - start + 1;
  const idxInGroup = stepIndex - start; // 0: 1打目, 1: 2打目, 2: 3打目

  if (groupLen > 1) {
    if (idxInGroup === 0) {
      // 1打目: 黄色 (Yellow)
      return {
        stroke: "rgba(255, 230, 0, 0.95)",
        halo: "rgba(255, 230, 0, 0.25)",
        accent: "rgba(255, 230, 0, 0.65)",
        glow: "rgba(255, 230, 0, 0.85)",
        fill: "#fff700",
        r: 255,
        g: 230,
        b: 0,
        rgb: "255, 230, 0",
        name: "yellow"
      };
    } else if (idxInGroup === 1) {
      // 2打目: 緑色 (Green)
      return {
        stroke: "rgba(0, 255, 136, 0.95)",
        halo: "rgba(0, 255, 136, 0.25)",
        accent: "rgba(0, 255, 136, 0.65)",
        glow: "rgba(0, 255, 136, 0.85)",
        fill: "#00ff88",
        r: 0,
        g: 255,
        b: 136,
        rgb: "0, 255, 136",
        name: "green"
      };
    } else {
      // 3打目: 青色 (Blue)
      return {
        stroke: "rgba(0, 180, 255, 0.95)",
        halo: "rgba(0, 180, 255, 0.25)",
        accent: "rgba(0, 180, 255, 0.65)",
        glow: "rgba(0, 180, 255, 0.85)",
        fill: "#00b4d8",
        r: 0,
        g: 180,
        b: 255,
        rgb: "0, 180, 255",
        name: "blue"
      };
    }
  }

  // 単発: 水色 (Cyan)
  return {
    stroke: "rgba(0, 229, 255, 0.95)",
    halo: "rgba(0, 229, 255, 0.25)",
    accent: "rgba(0, 229, 255, 0.65)",
    glow: "rgba(0, 229, 255, 0.85)",
    fill: "#00e5ff",
    r: 0,
    g: 229,
    b: 255,
    rgb: "0, 229, 255",
    name: "cyan"
  };
}

// ==========================================================================
// 運指誘導ノーツシステム（画面上部から右下の各指先へ向かって降る光る音符）
// ==========================================================================

// 手のイラストSVG内における各指先の中央ピクセル座標（200x200 viewBox基準）
const GUIDE_FINGER_SVG_COORDS = {
  THUMB:  { x: 42,   y: 104 },
  INDEX:  { x: 85.5, y: 36  },
  MIDDLE: { x: 106,  y: 24  },
  RING:   { x: 126.5,y: 34  },
  PINKY:  { x: 146.5,y: 52  }
};

/**
 * 画面右下の手のイラスト内の各指先スクリーン座標（px）を取得
 * @param {string} fingerKey
 * @returns {{ x: number, y: number }}
 */
export function getGuideFingerScreenPos(fingerKey) {
  const widget = document.getElementById("hand-guide-widget");
  if (!widget) {
    return { x: window.innerWidth - 80, y: window.innerHeight - 80 };
  }
  const rect = widget.getBoundingClientRect();
  const scale = rect.width / 200;
  const c = GUIDE_FINGER_SVG_COORDS[fingerKey] || GUIDE_FINGER_SVG_COORDS.MIDDLE;
  return {
    x: rect.left + c.x * scale,
    y: rect.top + c.y * scale
  };
}

// 各ノーツのスムーズな進行度キャッシュ (stepIndex -> 現在の進行度 0.0〜1.0)
export const noteProgressMap = new Map();

// 打鍵HIT時の弾けるパーティクル配列
const noteHitParticles = [];

/**
 * 打鍵HIT時の光のパーティクル＆リングエフェクトを発生
 * @param {number} stepIndex
 */
export function triggerNoteHitEffect(stepIndex) {
  const item = IS_FREE_SESSION ? { fingerKey: FREE_SESSION_FINGER_KEY } : currentSequence[stepIndex];
  if (!item) return;
  const targetPos = getGuideFingerScreenPos(item.fingerKey);
  const color = getTargetFingerColor(stepIndex);

  // 1. 弾けるリング（ショックウェーブ）
  noteHitParticles.push({
    type: "ring",
    x: targetPos.x,
    y: targetPos.y,
    radius: 10,
    maxRadius: 52,
    color: color.stroke || "#00e5ff",
    alpha: 1.0,
    decay: 0.05
  });

  // 2. 放射状の光火花パーティクル
  const count = 12;
  for (let i = 0; i < count; i++) {
    const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.4;
    const speed = 2.5 + Math.random() * 3.5;
    noteHitParticles.push({
      type: "spark",
      x: targetPos.x,
      y: targetPos.y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size: 3 + Math.random() * 2.5,
      color: color.fill || "#00e5ff",
      alpha: 1.0,
      decay: 0.035 + Math.random() * 0.02
    });
  }
}

/**
 * ノーツCanvasの解像度同期
 */
function resizeNotesCanvas() {
  if (!notesCanvas || !notesCtx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (notesCanvas.width !== w * dpr || notesCanvas.height !== h * dpr) {
    notesCanvas.width = w * dpr;
    notesCanvas.height = h * dpr;
    notesCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}
window.addEventListener("resize", resizeNotesCanvas);

let notesCanvasClearedInFreeMode = false;

/**
 * 毎フレームのノーツ描画とアニメーション更新（画面上部から指先への落下）
 */
function updateAndRenderNotes() {
  if (!notesCanvas || !notesCtx) return;

  // フリーセッション中（パーティクル非表示時）は全画面再描画とDOM座標取得をスキップして負荷を削減
  if (IS_FREE_SESSION && noteHitParticles.length === 0) {
    if (!notesCanvasClearedInFreeMode) {
      resizeNotesCanvas();
      notesCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      notesCanvasClearedInFreeMode = true;
    }
    return;
  }
  notesCanvasClearedInFreeMode = false;

  resizeNotesCanvas();

  const w = window.innerWidth;
  const h = window.innerHeight;
  notesCtx.clearRect(0, 0, w, h);

  const now = performance.now();

  // 1. 各指のガイドライン（画面上端から指先への極薄ネオンレーン）
  const fingerKeys = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"];
  fingerKeys.forEach((fKey) => {
    const pos = getGuideFingerScreenPos(fKey);
    const isTarget = fKey === currentFingerKey;
    notesCtx.beginPath();
    notesCtx.moveTo(pos.x, 24);
    notesCtx.lineTo(pos.x, pos.y);
    notesCtx.strokeStyle = isTarget ? "rgba(0, 229, 255, 0.18)" : "rgba(255, 255, 255, 0.04)";
    notesCtx.lineWidth = isTarget ? 2.0 : 1.0;
    notesCtx.stroke();
  });

  // 2. カウントダウン中や完走中・フリーセッション中以外の時、ノーツを描画
  if (!IS_FREE_SESSION && currentSequence && currentSequence.length > 0 && !isClearing) {
    // 現在のステップから最大4音先までを逆順で描画（先頭ノーツを手前に表示）
    const maxAhead = 4;
    for (let i = maxAhead - 1; i >= 0; i--) {
      const stepIdx = currentSongStep + i;
      if (stepIdx >= currentSequence.length) continue;

      const item = currentSequence[stepIdx];
      const targetPos = getGuideFingerScreenPos(item.fingerKey);
      const color = getTargetFingerColor(stepIdx);

      // 目標進行度: 先頭(i=0)は 1.0 (指先に完全着地)、それ以降は 0.72, 0.44, 0.16 と上部に配置
      const targetProg = i === 0 ? 1.0 : Math.max(0.12, 1.0 - i * 0.28);

      // スムーズな補間移動（Lerp）
      let curProg = noteProgressMap.get(stepIdx);
      if (curProg === undefined) {
        curProg = Math.max(0, targetProg - 0.35);
      }
      curProg += (targetProg - curProg) * 0.14;
      noteProgressMap.set(stepIdx, curProg);

      // 画面上のY座標（画面上端 44px から指先 targetPos.y へ）
      const startY = 44;
      const curX = targetPos.x;
      const curY = startY + (targetPos.y - startY) * curProg;

      // 先頭ノーツ（現在叩くべき音）は指先で優しくパルス発光
      const isLead = i === 0;
      const pulse = isLead ? 1.0 + 0.08 * Math.sin(now * 0.007) : 1.0;
      const alpha = Math.min(1.0, curProg * 1.5);

      // ノーツの描画（ネオン発光カプセル）
      notesCtx.save();
      notesCtx.translate(curX, curY);
      notesCtx.scale(pulse, pulse);
      notesCtx.globalAlpha = alpha;

      const capsuleW = 54;
      const capsuleH = 28;
      const radius = 14;

      // 外側ネオングロー
      notesCtx.beginPath();
      notesCtx.roundRect(-capsuleW / 2, -capsuleH / 2, capsuleW, capsuleH, radius);
      notesCtx.fillStyle = color.halo || "rgba(0, 229, 255, 0.3)";
      notesCtx.fill();

      // メインカプセル背景（ダーク半透明）
      notesCtx.beginPath();
      notesCtx.roundRect(-capsuleW / 2, -capsuleH / 2, capsuleW, capsuleH, radius);
      notesCtx.fillStyle = "rgba(10, 10, 15, 0.88)";
      notesCtx.fill();

      // ネオン境界線
      notesCtx.strokeStyle = color.stroke || "#00e5ff";
      notesCtx.lineWidth = isLead ? 2.5 : 1.6;
      notesCtx.stroke();

      // カプセル内テキスト（運指番号＋音名、例: "3 ミ"）
      notesCtx.fillStyle = "#ffffff";
      notesCtx.font = "bold 13px 'Inter', sans-serif";
      notesCtx.textAlign = "center";
      notesCtx.textBaseline = "middle";
      notesCtx.fillText(`${item.fingerNum} ${item.note}`, 0, 1);

      notesCtx.restore();
    }
  }

  // 3. パーティクル・ショックウェーブの描画と更新
  for (let i = noteHitParticles.length - 1; i >= 0; i--) {
    const p = noteHitParticles[i];
    p.alpha -= p.decay;
    if (p.alpha <= 0) {
      noteHitParticles.splice(i, 1);
      continue;
    }

    notesCtx.save();
    notesCtx.globalAlpha = Math.max(0, p.alpha);

    if (p.type === "ring") {
      p.radius += (p.maxRadius - p.radius) * 0.16;
      notesCtx.beginPath();
      notesCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      notesCtx.strokeStyle = p.color;
      notesCtx.lineWidth = 3.0 * p.alpha;
      notesCtx.stroke();
    } else if (p.type === "spark") {
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.94;
      p.vy *= 0.94;
      notesCtx.beginPath();
      notesCtx.arc(p.x, p.y, p.size * p.alpha, 0, Math.PI * 2);
      notesCtx.fillStyle = p.color;
      notesCtx.fill();
    }

    notesCtx.restore();
  }

  requestAnimationFrame(updateAndRenderNotes);
}

// ノーツ描画ループを開始
requestAnimationFrame(updateAndRenderNotes);



/**
 * 指定指の学習済み閾値をアクティブ閾値にセット
 * @param {string} fingerKey
 */
export function applyFingerThreshold(fingerKey) {
  const model = fingerThresholdModels[fingerKey];
  if (model) {
    hitRyThreshold = model.hitRy;
    liftRyThreshold = model.liftRy;
    trainedHitSamples = model.samples;
  }
  if (thVal) {
    thVal.textContent = hitRyThreshold.toFixed(2);
  }
  if (vThVal) {
    const vyTh = FINGER_VY_THRESHOLDS[fingerKey] || 200;
    vThVal.textContent = `${vyTh}`;
  }
}


/**
 * CSVステータス表示の更新
 * @param {string} msg
 */
function updateCsvStatus(msg) {
  if (csvStatus) {
    csvStatus.textContent = msg;
  }
}

/**
 * CSVテキストをパースして相対変位ryおよび打鍵フラグを抽出
 * （timestamp,rx,ry,rz,vx,vy,vz,label 形式、および frame,timestamp,mcp_x... 形式の両方に対応）
 * @param {string} text CSVテキスト
 * @returns {Array<{ timestamp: number, ry: number, isHit: boolean }>}
 */
/**
 * CSVテキストをパースして相対変位ryおよび打鍵フラグを抽出
 * （打鍵ラベルが未付与のtapファイルの場合、変位ピークを自動検出して打鍵点として認識）
 * @param {string} text CSVテキスト
 * @param {string} filename ファイル名
 * @returns {Array<{ timestamp: number, ry: number, isHit: boolean }>}
 */
export function parseCsv(text, filename = "") {
  if (!text || typeof text !== "string") return [];
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const headerLine = lines[0].toLowerCase();
  const headers = headerLine.split(",").map((h) => h.trim());

  const ryIdx = headers.indexOf("ry");
  const labelIdx = headers.indexOf("label") !== -1 ? headers.indexOf("label") : headers.indexOf("is_hit");
  const timeIdx = headers.indexOf("timestamp") !== -1 ? headers.indexOf("timestamp") : (headers.indexOf("timestamp_ms") !== -1 ? headers.indexOf("timestamp_ms") : headers.indexOf("time_sec"));

  const mcpYIdx = headers.indexOf("index_mcp_y") !== -1 ? headers.indexOf("index_mcp_y") : headers.indexOf("mcp_y");
  const tipYIdx = headers.indexOf("index_tip_y") !== -1 ? headers.indexOf("index_tip_y") : headers.indexOf("tip_y");

  const results = [];
  let manualHitCount = 0;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(",").map((c) => c.trim());

    let ry = null;
    let isHit = false;
    let timestamp = 0;

    if (timeIdx !== -1 && cols[timeIdx]) {
      timestamp = parseFloat(cols[timeIdx]) || 0;
    }

    if (labelIdx !== -1 && cols[labelIdx]) {
      isHit = parseInt(cols[labelIdx], 10) === 1;
      if (isHit) manualHitCount++;
    }

    if (ryIdx !== -1 && cols[ryIdx] !== undefined && cols[ryIdx] !== "") {
      ry = parseFloat(cols[ryIdx]);
    } else if (mcpYIdx !== -1 && tipYIdx !== -1 && cols[mcpYIdx] && cols[tipYIdx]) {
      const my = parseFloat(cols[mcpYIdx]);
      const ty = parseFloat(cols[tipYIdx]);
      if (!isNaN(my) && !isNaN(ty)) {
        ry = ty - my;
      }
    }

    if (ry !== null && !isNaN(ry)) {
      results.push({
        timestamp,
        ry,
        isHit
      });
    }
  }

  // もしファイル名に "tap" が含まれており、手動打鍵マークが0件の場合は自動ピーク検出を実行
  if (filename.toLowerCase().includes("tap") && manualHitCount === 0 && results.length > 10) {
    autoDetectTapPeaks(results);
  }

  return results;
}

/**
 * 打鍵データ波形から机面接触ピーク（極大点）を自動検出して isHit を付与
 * @param {Array<{ timestamp: number, ry: number, isHit: boolean }>} frames
 */
function autoDetectTapPeaks(frames) {
  const rys = frames.map((f) => f.ry);
  const len = rys.length;
  if (len < 5) return;

  // 全体の平均値と中央値を算出
  const sortedRys = [...rys].sort((a, b) => a - b);
  const medianRy = sortedRys[Math.floor(len * 0.5)];
  const p75Ry = sortedRys[Math.floor(len * 0.75)];

  let lastHitIdx = -100;
  let autoCount = 0;

  for (let i = 3; i < len - 3; i++) {
    const cur = rys[i];
    // 局所極大（前後より高く、かつ75%タイル以上の山）
    if (
      cur > rys[i - 1] &&
      cur >= rys[i + 1] &&
      cur > rys[i - 2] &&
      cur >= rys[i + 2] &&
      cur > p75Ry &&
      (cur - medianRy) > 0.08 &&
      (i - lastHitIdx) > 10 // クールダウン（最低10フレーム離れている）
    ) {
      frames[i].isHit = true;
      lastHitIdx = i;
      autoCount++;
    }
  }

  console.log(`[CSV PEAK] 打鍵ピーク自動検出: ${autoCount} 箇所の打鍵点を抽出 (Median=${medianRy.toFixed(2)}, P75=${p75Ry.toFixed(2)})`);
}

/**
 * 読み込んだ学習済みCSVデータから各指の打鍵判定閾値を自動導出
 */
function trainModelFromCsv() {
  const fingerKeys = ["THUMB", "MIDDLE", "RING", "PINKY", "INDEX"];
  for (const fKey of fingerKeys) {
    trainModelForFinger(fKey);
  }
  // 現在選択されている指の閾値を反映
  applyFingerThreshold(currentFingerKey);
}

/**
 * 特定指のデータセットから打鍵閾値を導出（負値変位・各指の可動域に完全対応）
 * @param {string} fingerKey
 */
function trainModelForFinger(fingerKey) {
  const datasets = fingerDatasets[fingerKey] || [];
  if (datasets.length === 0) return;

  const frames = datasets.flatMap((d) => d.frames);
  if (frames.length === 0) return;

  // 1. 打鍵データ（isHit = true）の抽出
  const hitRys = frames
    .filter((f) => f.isHit && f.ry !== null && !isNaN(f.ry))
    .map((f) => f.ry)
    .sort((a, b) => a - b);

  // 2. 空中動作データ（純粋な空中浮遊データ neg_air のみを使用）
  // ※ neg_idle (静止待機) や neg_slide (スライド) は机面接触を含むため空中基準から除外
  const airDatasets = datasets.filter((d) => {
    const fn = d.filename.toLowerCase();
    return fn.includes("neg_air");
  });

  let maxAirRy = -Infinity;
  if (airDatasets.length > 0) {
    const airFrames = airDatasets.flatMap((d) => d.frames);
    const validAirRys = airFrames
      .filter((f) => f.ry !== null && !isNaN(f.ry))
      .map((f) => f.ry)
      .sort((a, b) => a - b);
    if (validAirRys.length > 0) {
      // 90パーセンタイルを空中最大変位として採用（異常外れ値をカット）
      maxAirRy = validAirRys[Math.floor(validAirRys.length * 0.90)];
    }
  }

  // 空中データが無い場合は打鍵以外のデータの下位70%を参照
  if (maxAirRy === -Infinity) {
    const negRys = frames
      .filter((f) => !f.isHit && f.ry !== null && !isNaN(f.ry))
      .map((f) => f.ry)
      .sort((a, b) => a - b);
    maxAirRy = negRys.length > 0 ? negRys[Math.floor(negRys.length * 0.70)] : 0.0;
  }

  const samples = hitRys.length;
  let hitTh = 0.0;
  let liftTh = 0.0;

  if (samples > 0) {
    // 打鍵サンプルの中央値・第25%パーセンタイル値
    const medianHitRy = hitRys[Math.floor(samples * 0.50)];
    const p25HitRy = hitRys[Math.floor(samples * 0.25)];

    if (medianHitRy > maxAirRy) {
      // 空中最大と打鍵中央値の中間点に打鍵閾値を設定
      // 薬指（RING）は他指（中指・小指）との連動によるつられ下がり誤検知を防ぐため、やや高め（0.46）に設定
      // 親指（THUMB）は机面への垂直変位が小さいため緩和（0.30）して反応感度を向上
      const hitRatio = fingerKey === "THUMB" ? 0.30 : (fingerKey === "RING" ? 0.46 : 0.45);
      hitTh = maxAirRy + (medianHitRy - maxAirRy) * hitRatio;

      // 速度ゲートにより机面待機誤爆が防止されているため、浅い打鍵でも確実に反応するよう適正範囲内にクランプ
      const minHitTh = fingerKey === "THUMB" ? 0.30 : (fingerKey === "PINKY" ? 0.45 : 0.50);
      hitTh = Math.max(hitTh, minHitTh);

      // リフト閾値：全指で打鍵位置からわずかに指を浮かせるだけで素早くIDLE復帰できるよう、
      // ヒステリシス幅を極小（0.03）に設定
      const hysteresis = 0.03;
      liftTh = hitTh - hysteresis;
    } else {
      // 外れ値等で打鍵中央値が空中を下回る場合の適応フォールバック
      hitTh = p25HitRy;
      liftTh = hitTh - 0.03;
    }
  } else {
    // 打鍵サンプルが0件の場合の安全マージン（親指は垂直変位が小さいため0.10、薬指は誤検知防止で0.20、他指は0.18）
    const safetyMargin = fingerKey === "THUMB" ? 0.10 : (fingerKey === "RING" ? 0.20 : 0.18);
    hitTh = maxAirRy + safetyMargin;
    liftTh = hitTh - 0.03;
  }

  fingerThresholdModels[fingerKey] = {
    hitRy: parseFloat(hitTh.toFixed(2)),
    liftRy: parseFloat(liftTh.toFixed(2)),
    samples
  };

  console.log(
    `[MODEL: ${fingerKey}] 学習完了: 打鍵=${samples}件 (空中Max=${maxAirRy.toFixed(3)}) -> 打鍵TH=${hitTh.toFixed(2)}, リフトTH=${liftTh.toFixed(2)}`
  );
}


/**
 * 複数CSVテキストを一括パースしてデータセット配列に格納
 * @param {Array<{ name: string, content: string }>} files
 */
function ingestCsvFiles(files) {
  for (const file of files) {
    const parsed = parseCsv(file.content, file.name);
    if (parsed.length > 0) {
      const existIdx = testDataDatasets.findIndex((d) => d.filename === file.name);
      const datasetEntry = {
        filename: file.name,
        frames: parsed,
        hitCount: parsed.filter((f) => f.isHit).length
      };

      if (existIdx !== -1) {
        testDataDatasets[existIdx] = datasetEntry;
      } else {
        testDataDatasets.push(datasetEntry);
      }
    }
  }

  // 全フレームのフラット配列を生成
  testDataFrames = testDataDatasets.flatMap((d) => d.frames);

  if (testDataDatasets.length > 0) {
    const totalHits = testDataFrames.filter((f) => f.isHit).length;
    trainModelFromCsv();
    updateCsvStatus(`LOADED (${testDataDatasets.length} files, ${totalHits} hits)`);
    console.log(
      `[CSV] 全読込成功: 計 ${testDataDatasets.length} ファイル, ${testDataFrames.length} フレーム (打鍵マーク: ${totalHits})`
    );
  } else {
    updateCsvStatus("NOT LOADED");
    if (thVal) thVal.textContent = hitRyThreshold.toFixed(2);
  }
}

/**
 * ファイルパスまたはファイル名から指キー（THUMB, MIDDLE, RING, PINKY, INDEX）を判定
 * @param {string} path
 * @returns {string}
 */
export function detectFingerKey(path) {
  const p = path.toLowerCase();
  if (p.includes("thumb")) return "THUMB";
  if (p.includes("middle")) return "MIDDLE";
  if (p.includes("ring")) return "RING";
  if (p.includes("pinky")) return "PINKY";
  if (p.includes("index")) return "INDEX";
  return "THUMB";
}

/**
 * public/dataset/ 配下の各指CSVデータセットを自動非同期フェッチして読み込む
 */
export async function loadTestDataCsv() {
  const loadedFiles = [];

  // 初期化
  Object.keys(fingerDatasets).forEach((k) => (fingerDatasets[k] = []));

  try {
    const manifestRes = await fetch("./dataset/manifest.json");
    if (manifestRes.ok) {
      const manifestData = await manifestRes.json();
      let fileList = [];
      if (Array.isArray(manifestData)) {
        fileList = manifestData;
      } else if (manifestData && typeof manifestData === "object") {
        if (manifestData.fingers) {
          fileList = Object.values(manifestData.fingers).flat();
        } else if (Array.isArray(manifestData.files)) {
          fileList = manifestData.files;
        }
      }

      for (const fname of fileList) {
        if (!fname) continue;
        try {
          const csvRes = await fetch(`./dataset/${fname}`);
          if (csvRes.ok) {
            const text = await csvRes.text();
            const displayName = fname.includes("/") ? fname.split("/").pop() : fname;
            const fingerKey = detectFingerKey(fname);
            const parsed = parseCsv(text, displayName);
            if (parsed.length > 0) {
              const entry = {
                filename: displayName,
                path: fname,
                fingerKey,
                frames: parsed,
                hitCount: parsed.filter((f) => f.isHit).length
              };
              fingerDatasets[fingerKey].push(entry);
              loadedFiles.push({ name: displayName, content: text });
            }
          }
        } catch (err) {
          console.warn(`[CSV] ${fname} の取得スキップ:`, err);
        }
      }
    }
  } catch (err) {
    console.warn("[CSV] manifest.json の読み込みをスキップ:", err);
  }

  if (loadedFiles.length > 0) {
    ingestCsvFiles(loadedFiles);
    trainModelFromCsv();
  } else {
    updateCsvStatus("NO CSV DATA");
    applyFingerThreshold(currentFingerKey);
  }
}


// 先行MediaPipeモデルロード用Promise
let handLandmarkerPromise = null;
let selectedStartSongId = FREE_SESSION_SONG_ID;

/**
 * 初期化処理（PC版）
 */
async function init() {
  prepareApp();
}

/**
 * 起動前準備（モデル読込の先行開始と楽曲選択モーダルのセットアップ）
 */
function prepareApp() {
  resetDebugMetrics();
  updateStatus("READY");
  applyMirrorUI();

  // テスト用CSVの非同期読み込み（未配置でもブロックせず実行）
  loadTestDataCsv();

  // MediaPipe Hand Landmarker の読み込みを先行バックグラウンド実行（スタート押下時の待ち時間を極小化）
  if (!handLandmarkerPromise) {
    handLandmarkerPromise = initHandLandmarker().catch((err) => {
      console.warn("先行モデル読込エラー（スタート時に再試行）:", err);
      handLandmarkerPromise = null;
    });
  }

  // 起動モーダルのイベント初期化
  setupStartModal();
}

/**
 * 起動時楽曲選択モーダルのイベント登録
 */
function setupStartModal() {
  if (!startModal) return;

  const startSongItems = document.querySelectorAll(".start-song-item");
  startSongItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      const songId = item.dataset.song;
      if (!songId || !SONGS[songId]) return;

      selectedStartSongId = songId;

      // 選択状態のUI更新
      startSongItems.forEach((btn) => {
        const isSelected = btn.dataset.song === songId;
        btn.classList.toggle("selected", isSelected);
        const check = btn.querySelector(".start-song-check");
        if (check) check.textContent = isSelected ? "●" : "○";
      });

      // ターゲット指やガイドUIも背景側で事前に反映しておく
      selectSong(songId, false);
    });
  });

  if (startPlayBtn) {
    startPlayBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await handleStartPlay();
    });
  }
}

/**
 * 「START」ボタン押下時の実行処理（音声アンロック・60fps Webカメラ起動・カウントダウン）
 */
async function handleStartPlay() {
  if (!startPlayBtn) return;

  // ボタンをローディング状態にして二重クリックを防止
  startPlayBtn.disabled = true;
  if (startPlayText) startPlayText.textContent = "STARTING...";

  // 1. ユーザー操作の同期スタック内で Web Audio API を即時有効化
  ensureAudioContext();

  try {
    updateStatus("LOADING MODEL...");
    if (handLandmarkerPromise) {
      await handLandmarkerPromise;
    } else {
      await initHandLandmarker();
    }

    // 選択された曲をセット（カウントダウンはカメラ起動後に行うため false）
    selectSong(selectedStartSongId, false);

    // 2. 60fps Webカメラの起動
    updateStatus("STARTING CAMERA...");
    await startCamera(currentCameraId || null);

    // 3. 起動モーダルを非表示
    if (startModal) {
      startModal.classList.add("hidden");
    }

    // 4. カウントダウン開始
    await requestWakeLock();
    startCountdown(() => {
      console.log(`[START] 演奏開始: ${SONGS[selectedStartSongId]?.title || "Free Session"}`);
    });
  } catch (error) {
    console.error("起動エラー:", error);
    startPlayBtn.disabled = false;
    if (startPlayText) startPlayText.textContent = "START";

    const guidanceMsg = getCameraErrorMessage(error);
    showCameraErrorOverlay(guidanceMsg);
    showError(error.message || guidanceMsg);
    updateStatus("ERROR");
  }
}

/**
 * ハンドトラッキングエンジンの初期化
 * WebGPU対応環境では高速な @svenflow/micro-handpose を優先し、非対応時は MediaPipe へフォールバック
 */
async function initHandLandmarker() {
  // 1. MicroPose (WebGPU) の初期化を試行
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const adapterArch = adapter?.info?.architecture || "";
      const isSoftwareWebGpu =
        !adapter ||
        adapter.isFallbackAdapter === true ||
        adapterArch.toLowerCase().includes("swiftshader");

      if (!isSoftwareWebGpu) {
        microHandpose = await createHandpose({
          maxHands: 1,
          scoreThreshold: 0.4,
          palmScoreThreshold: 0.4
        });
        activeTrackingEngine = "MICRO_HANDPOSE";
        if (telemetryEngine) telemetryEngine.textContent = "MICRO";
        console.log("[TRACKING ENGINE] MicroPose (WebGPU) を有効化しました", adapter.info);
        return;
      } else {
        console.warn(
          "[TRACKING ENGINE] WebGPU 実GPUアダプタが利用不可（ソフトウェアエミュレーション等）のため、MediaPipe へフォールバックします",
          adapter?.info
        );
      }
    } catch (microErr) {
      console.warn("[TRACKING ENGINE] MicroPose (WebGPU) 初期化失敗、MediaPipe へフォールバック:", microErr);
      microHandpose = null;
    }
  } else {
    console.warn("[TRACKING ENGINE] WebGPU 非対応のため MediaPipe へフォールバックします");
  }

  // 2. MediaPipe Hand Landmarker フォールバック
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const modelPaths = [
    "./models/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  ];

  let loaded = false;
  let lastErr = null;

  for (const modelPath of modelPaths) {
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.3
      });
      loaded = true;
      activeTrackingEngine = "MEDIAPIPE";
      if (telemetryEngine) telemetryEngine.textContent = "MP-GPU";
      console.log("[TRACKING ENGINE] MediaPipe (GPU) を有効化しました");
      break;
    } catch (gpuErr) {
      console.warn("GPU失敗、CPUへフォールバック:", gpuErr);
      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: modelPath,
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.4,
          minHandPresenceConfidence: 0.4,
          minTrackingConfidence: 0.3
        });
        loaded = true;
        activeTrackingEngine = "MEDIAPIPE";
        if (telemetryEngine) telemetryEngine.textContent = "MP-CPU";
        console.log("[TRACKING ENGINE] MediaPipe (CPU) を有効化しました");
        break;
      } catch (err) {
        lastErr = err;
      }
    }
  }

  if (!loaded) {
    throw new Error(`モデルロード失敗: ${lastErr?.message || "不明なエラー"}`);
  }
}

/**
 * getUserMedia 互換処理
 * @param {MediaStreamConstraints} constraints
 */
function getCompatibleUserMedia(constraints) {
  // セキュアコンテキスト（HTTPS / localhost）の先行検証
  if (!window.isSecureContext) {
    const secErr = new Error(
      "非セキュア環境 (HTTP) のためカメラがブロックされています。HTTPS (https://...) でアクセスしてください。"
    );
    secErr.name = "SecurityError";
    throw secErr;
  }

  if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  const legacyGetUserMedia =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

  if (legacyGetUserMedia) {
    return new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, constraints, resolve, reject);
    });
  }

  const notSupportedErr = new Error("このブラウザはカメラAPIに対応していません。");
  notSupportedErr.name = "NotSupportedError";
  throw notSupportedErr;
}

// PC標準アスペクト比（16:9）で初期表示を設定
document.documentElement.style.setProperty("--video-aspect", (16 / 9).toFixed(4));

/**
 * 鏡像（左右反転）表示の反映
 */
function applyMirrorUI() {
  if (video) video.classList.toggle("mirrored", isMirrorEnabled);
  if (canvas) canvas.classList.toggle("mirrored", isMirrorEnabled);
  if (mirrorToggleBtn) {
    mirrorToggleBtn.classList.toggle("off", !isMirrorEnabled);
    mirrorToggleBtn.setAttribute("aria-pressed", String(isMirrorEnabled));
    mirrorToggleBtn.title = isMirrorEnabled ? "Mirror: ON" : "Mirror: OFF";
  }
}

/**
 * 接続中のWebカメラ一覧を取得し、右上メニューを更新
 */
async function refreshCameraList() {
  if (!cameraListEl || !navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");
    cameraListEl.innerHTML = "";

    videoDevices.forEach((device, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "camera-menu-item";
      const isActive = currentCameraId
        ? device.deviceId === currentCameraId
        : idx === 0;
      if (isActive) btn.classList.add("active");
      btn.textContent = device.label || `Camera ${idx + 1}`;
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

/**
 * PC向け 60fps Webカメラの起動・切り替え
 * @param {string | null} [deviceId=null]
 * @param {boolean} [isManualSelect=false] ユーザーがUIから手動選択したかどうか
 */
async function startCamera(deviceId = null, isManualSelect = false) {
  // 既存ストリームがある場合は停止してから切り替え
  isPredicting = false;
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }

  const isBuiltInCameraLabel = (label = "") =>
    /integrated|built-in|facetime|internal/i.test(label);

  // 自動起動時は保存IDが内蔵30fpsカメラであっても外付け60fpsカメラを優先探索
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
      // 60fps 対応かつ転送・推論が最軽量な 1024x576 @ 60fps を最優先で要求
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
      if (
        e.name === "NotAllowedError" ||
        e.name === "PermissionDeniedError" ||
        e.name === "SecurityError"
      ) {
        throw e;
      }
      console.warn("制約でのカメラ起動失敗、次を試行:", e);
    }
  }

  if (!stream) {
    throw lastError || new Error("Webカメラの取得に失敗しました。接続とアクセス許可を確認してください。");
  }

  // 自動起動時に内蔵30fpsカメラを掴んだ場合、外付けカメラが存在すれば自動切替
  const initialTrack = stream.getVideoTracks()[0];
  if (!isManualSelect && initialTrack && isBuiltInCameraLabel(initialTrack.label) && navigator.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const externalCam = devices.find(
        (d) => d.kind === "videoinput" && d.deviceId && d.label && !isBuiltInCameraLabel(d.label)
      );
      if (externalCam) {
        console.log(`[CAM AUTO] 内蔵カメラから外付けカメラへ自動切替: ${externalCam.label}`);
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
      if (typeof track.getCapabilities === "function") {
        const caps = track.getCapabilities();
        console.log("[CAM CAPABILITIES]", caps);
        const targetFps = caps.frameRate?.max ? Math.min(MAX_APP_FPS, caps.frameRate.max) : MAX_APP_FPS;
        if (typeof track.applyConstraints === "function") {
          await track.applyConstraints({ frameRate: { ideal: targetFps, max: MAX_APP_FPS } });
        }
      }
    } catch (err) {
      console.warn("FPS制約適用スキップ:", err);
    }

    try {
      if (typeof track.getSettings === "function") {
        const settings = track.getSettings();
        if (settings.deviceId) {
          currentCameraId = settings.deviceId;
          try {
            localStorage.setItem(PC_CAMERA_STORAGE_KEY, currentCameraId);
          } catch {}
        }
        const fpsLabel = settings.frameRate ? `${Math.round(settings.frameRate)}fps` : "60fps";
        const resStr = settings.width && settings.height ? `${settings.width}×${settings.height}` : "--×--";
        if (telemetryRes) telemetryRes.textContent = resStr;
        if (camInfo) camInfo.textContent = `${resStr} ${fpsLabel}`;
      }
    } catch (settingErr) {
      console.warn("カメラ設定取得スキップ:", settingErr);
    }
  }

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = async () => {
      try {
        await video.play();
        resolve();
      } catch (playErr) {
        reject(playErr);
      }
    };
    video.onerror = (err) => reject(err);
  });

  updateCanvasResolution();
  await refreshCameraList();

  // カメラ起動・切替時にFPS計測状態とMicroPose ROI状態をリセット
  frameCount = 0;
  cameraFrameCount = 0;
  lastPresentedFrames = -1;
  lastAppProcessedTime = 0;
  isInferringFrame = false;
  pendingVideoFrameTime = null;
  lastFpsUpdateTime = performance.now();
  lastCameraFpsUpdateTime = performance.now();
  if (microHandpose && typeof microHandpose.reset === "function") {
    microHandpose.reset();
  }

  isPredicting = true;
  schedulePredictLoop();
  await requestWakeLock();
  updateStatus("TRACKING");
}

/**
 * Canvas内部解像度およびコンテナアスペクト比をビデオのネイティブピクセル数に同期
 */
function updateCanvasResolution() {
  if (video.videoWidth && video.videoHeight) {
    const aspect = video.videoWidth / video.videoHeight;
    document.documentElement.style.setProperty("--video-aspect", aspect.toFixed(4));

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log(`[CANVAS] ビデオ解像度・アスペクト比を完全同期: ${canvas.width}x${canvas.height} (比率 ${aspect.toFixed(3)}:1)`);
    }
  }
}

let lastVideoTimestamp = -1;

/**
 * 毎フレームの推論処理（単一フレーム・非同期ガード付き）
 * @param {number} frameTime
 */
async function processVideoFrame(frameTime) {
  if (!isPredicting || isInferringFrame) return;
  isInferringFrame = true;

  try {
    // カメラ解像度とCanvas解像度の完全一致を常時保証
    if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
      updateCanvasResolution();
    }

    // 正確なフレーム時間をミリ秒で算出（MediaPipe VIDEOモードが要求する厳密な単調増加を保証）
    const now = typeof frameTime === "number" ? frameTime : performance.now();
    const timestampMs = Math.max(now, lastVideoTimestamp + 1.0);
    lastVideoTimestamp = timestampMs;

    const startTime = performance.now();
    let results = { landmarks: [], handednesses: [] };

    if (activeTrackingEngine === "MICRO_HANDPOSE" && microHandpose) {
      // 本家デモ同様に中間Canvas経由で入力し、毎フレームの createImageBitmap() 負荷を回避
      const inputCanvas = ensureMicroInputCanvas(video.videoWidth, video.videoHeight);
      microInputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
      const hands = await microHandpose.detect(inputCanvas);
      if (!isPredicting) return;
      results = {
        landmarks: hands.map((h) => h.landmarks),
        handednesses: hands.map((h) => [
          {
            categoryName: h.handedness === "right" ? "Right" : "Left",
            score: h.score
          }
        ])
      };
    } else if (handLandmarker) {
      results = handLandmarker.detectForVideo(video, timestampMs);
    }

    const calcDuration = performance.now() - startTime;
    if (telemetryMs) {
      telemetryMs.textContent = `${calcDuration.toFixed(1)} ms`;
    }
    if (isDebugPanelVisible) {
      inferenceTime.textContent = `${calcDuration.toFixed(1)} ms`;
    }

    // 生ランドマーク座標の直接描画・打鍵判定
    drawRawHandLandmarks(results);

    // FPS更新
    updateFps();
  } catch (err) {
    console.warn("[INFERENCE] フレーム推論エラー:", err);
  } finally {
    isInferringFrame = false;
    // 推論中に到着していた最新フレームがあれば、次コールバックを待たずに即座に処理して30fps半減を防止
    if (isPredicting && pendingVideoFrameTime !== null) {
      const nextFrameTime = pendingVideoFrameTime;
      pendingVideoFrameTime = null;
      const nowMs = performance.now();
      if (shouldRunAppFrame(nowMs)) {
        processVideoFrame(nextFrameTime);
      }
    }
  }
}

/**
 * 新規カメラフレームの推論ディスパッチ（推論中は最新フレームを保留）
 * @param {number} frameTime
 */
function dispatchVideoFrame(frameTime) {
  if (isInferringFrame) {
    pendingVideoFrameTime = frameTime;
    return;
  }
  const nowMs = performance.now();
  if (shouldRunAppFrame(nowMs)) {
    processVideoFrame(frameTime);
  }
}

/**
 * アプリ推論ループを上限60fpsに制限するためのフレーム実行判定
 * @param {number} nowMs
 * @returns {boolean}
 */
function shouldRunAppFrame(nowMs) {
  if (nowMs - lastAppProcessedTime < MIN_APP_DISPATCH_INTERVAL_MS) {
    return false;
  }
  lastAppProcessedTime = nowMs;
  return true;
}

/**
 * requestVideoFrameCallback または requestAnimationFrame による推論ループ
 * （カメラの物理フレーム到着に同期しつつ、アプリ推論の上限を60fpsに固定）
 */
function schedulePredictLoop() {
  if (!isPredicting) return;

  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback((now, metadata) => {
      if (!isPredicting) return;
      // 推論完了待ちによる1フレームスキップを防ぐため、処理前に次フレームを予約
      schedulePredictLoop();
      const nowMs = performance.now();
      updateCameraFps(nowMs, metadata?.presentedFrames);
      const frameTime =
        metadata && typeof metadata.presentationTime === "number"
          ? metadata.presentationTime
          : now;
      dispatchVideoFrame(frameTime);
    });
  } else {
    requestAnimationFrame((now) => {
      if (!isPredicting) return;
      schedulePredictLoop();
      if (video.currentTime !== lastVideoTime && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        lastVideoTime = video.currentTime;
        const nowMs = performance.now();
        updateCameraFps(nowMs);
        dispatchVideoFrame(now);
      }
    });
  }
}

// 追従時の最大許容移動距離（正規化座標系：画面幅の25%）
// 1フレーム（約16ms）で右手手首がこれ以上離れることは物理的にあり得ないため、左手への飛び移りを完全遮断
const MAX_WRIST_TRACK_DISTANCE = 0.25;

// ナックル外積の明確な判定閾値（ゼロ近傍の不感帯用）
const CROSS_RIGHT_THRESHOLD = -0.002;
const CROSS_LEFT_THRESHOLD = 0.002;

// 画面左端のセーフティ境界（生カメラ画像で x < 0.02 等の極端な欠損フレームのみ除外）
const MIN_SAFETY_X = 0.02;

/**
 * 手のひらナックルの幾何学的向き（2Dベクトル外積）による右手判定（ヒステリシス対応）
 * 手首(0)→人差し指付け根(5) と 手首(0)→小指付け根(17) の外積を計算
 * 生カメラ画像（CSS scaleX(-1)反転前）において、手の甲が上（机面打鍵姿勢）の場合：
 * - 物理的な右手: 人差し指(5)が右(+X)、小指(17)が左(-X) → cross < 0
 * - 物理的な左手: 人差し指(5)が左(-X)、小指(17)が右(+X) → cross > 0
 * @param {Array<object>} hand 手の全21ランドマーク配列
 * @param {boolean} isCurrentlyTracked 現在追従中の右手候補（位置が近い）かどうか
 * @returns {boolean} 物理的な右手であれば true、左手であれば false
 */
function isAnatomicallyRightHand(hand, isCurrentlyTracked = false) {
  if (!hand || hand.length < 21) return false;

  const p0 = hand[0];   // 手首 (WRIST)
  const p5 = hand[5];   // 人差し指付け根 (INDEX_FINGER_MCP)
  const p17 = hand[17]; // 小指付け根 (PINKY_MCP)

  const vIndexX = p5.x - p0.x;
  const vIndexY = p5.y - p0.y;
  const vPinkyX = p17.x - p0.x;
  const vPinkyY = p17.y - p0.y;

  // 2D外積（Cross Product）
  const cross = vIndexX * vPinkyY - vIndexY * vPinkyX;

  // 明確に右手（人差し指が右、小指が左）
  if (cross < CROSS_RIGHT_THRESHOLD) {
    return true;
  }
  // 明確に左手（人差し指が左、小指が右）
  if (cross > CROSS_LEFT_THRESHOLD) {
    return false;
  }

  // ゼロ近傍の不感帯（ローアングルでの揺らぎ）：
  // 現在追従中の右手（前回の右手位置に近い）であれば、打鍵時の指屈伸によるブレとみなして右手判定を維持
  if (isCurrentlyTracked) {
    return true;
  }

  // 未追従時のゼロ近傍は、負側であれば右手候補として扱う
  return cross < 0;
}

/**
 * 位置連続性（Nearest-Neighbor）とナックル外積ヒステリシスに基づく右手セレクター
 * - 追従中は候補数に関わらず画面幅25%以上の距離ジャンプを無条件遮断（左手飛び移り防止）
 * - ナックル外積のヒステリシスにより、ローアングル打鍵時のチャタリング（判定点滅）を解消
 * - 右手が画面から消えた場合は左手に乗り換えず、右手が戻るまで待機（null返却）
 * @param {object} results MediaPipe HandLandmarkerの検出結果
 * @returns {Array<object> | null} 追従対象の右手の全21ランドマーク配列（未検出時はnull）
 */
function selectRightHandLandmarks(results) {
  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return null;
  }

  const hands = results.landmarks;

  // ========================================================
  // 1. 追従中（lastTrackedWrist が存在する場合）：
  // ========================================================
  if (lastTrackedWrist) {
    // 検出された全手の中から、前回の右手位置に最も近い手を探す
    let closestHand = hands[0];
    let minDistance = Infinity;

    for (let i = 0; i < hands.length; i++) {
      const hand = hands[i];
      const wrist = hand[0];
      const dist = Math.hypot(wrist.x - lastTrackedWrist.x, wrist.y - lastTrackedWrist.y);
      if (dist < minDistance) {
        minDistance = dist;
        closestHand = hand;
      }
    }

    // 【要件1: 距離リミッターの無条件適用】
    // 候補数に関わらず、前回の右手位置から画面幅25%以上離れている手は絶対に採用しない（左手へのワープを100%遮断）
    if (minDistance > MAX_WRIST_TRACK_DISTANCE) {
      return null;
    }

    // 最も近い手が右手であるかナックル幾何判定（追従中ヒステリシス有効）
    // （前回の右手位置の至近にある手であっても、明確に左手形状をしている場合は除外）
    if (!isAnatomicallyRightHand(closestHand, true)) {
      return null;
    }

    // 画面端スレスレの欠損セーフティ
    if (closestHand[0].x < MIN_SAFETY_X) {
      return null;
    }

    return closestHand;
  }

  // ========================================================
  // 2. 未追従状態（初回出現時または完全ロスト後の初回復帰時）：
  // ========================================================
  // ナックル外積により物理的な右手と判定される候補のみを抽出（左手はここで100%除外）
  const rightCandidates = hands.filter((hand) => {
    if (!isAnatomicallyRightHand(hand, false)) return false;
    if (hand[0].x < MIN_SAFETY_X) return false;
    return true;
  });

  if (rightCandidates.length === 0) {
    return null;
  }

  // 右手候補が1つの場合はそれを採用
  if (rightCandidates.length === 1) {
    return rightCandidates[0];
  }

  // 複数候補がある場合は、生カメラ画像で最も左側（x座標が最小＝正面カメラにおける右手側）を選択
  let leftmostHand = rightCandidates[0];
  let minX = Infinity;

  for (let i = 0; i < rightCandidates.length; i++) {
    const hand = rightCandidates[i];
    const wrist = hand[0];
    if (wrist.x < minX) {
      minX = wrist.x;
      leftmostHand = hand;
    }
  }

  return leftmostHand;
}

/**
 * 手の全21ランドマークの適応平滑化描画、打鍵・リフト用特徴量の算出
 * 手首・手のひら・全指先を独立した 1 Euro Filter で常時平滑化し、ジッターと飛びを完全排除
 * @param {object} results
 */
function drawRawHandLandmarks(results) {
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  const fingerConfig = FINGER_CONFIGS[currentFingerKey] || FINGER_CONFIGS.THUMB;
  // フリーセッション時は5本指すべての関節(0..20)を平滑化し、楽曲モード時は手首+対象指の5点のみ平滑化
  const activeJointIndices = IS_FREE_SESSION
    ? ALL_HAND_JOINT_INDICES
    : [0, fingerConfig.p1Idx, fingerConfig.p2Idx, fingerConfig.p3Idx, fingerConfig.tipIdx];
  const width = canvas.width;
  const height = canvas.height;
  const now = performance.now();

  // 位置連続性による右手セレクターを実行（左手は完全除外）
  const selectedRightHand = selectRightHandLandmarks(results);
  const hasHands = selectedRightHand !== null;

  if (!hasHands) {
    lostFrames++;
    // 3フレーム以内（約50ms）の一時的ロストであれば直前の平滑化座標で描画を維持し、画面の点滅・ジャンプを防止
    if (lostFrames <= MAX_LOST_FRAMES && hasValidSmoothedLandmarks) {
      // 直前フレームの平滑化座標でフィルターを現在タイムスタンプ（now）で空回し更新し、復帰時のdt拡大による速度微分スパイクを防止
      for (const i of activeJointIndices) {
        const pt = smoothedLandmarksPool[i];
        landmarkFilters[i].filter(pt.x, pt.y, now, pt);
      }
    } else {
      if (isDebugPanelVisible) {
        handsCount.textContent = "0";
      }
      const hadAnyPressed =
        tapState !== "IDLE" ||
        (IS_FREE_SESSION && FREE_SESSION_FINGER_KEYS.some((k) => freeFingerTapStates[k] !== "IDLE"));
      if (hadAnyPressed) {
        tapState = "IDLE";
        resetFreeFingerStates();
        updateStateHud("IDLE", false);
      }
      // 完全に画角から外れた場合のみフィルターと追従位置をリセット
      landmarkFilters.forEach((f) => f.reset());
      hasValidSmoothedLandmarks = false;
      lastRawLandmarks = null;
      lastTrackedWrist = null; // ★完全ロスト時は右手トラッキング位置もリセット
      lastFilteredFingerKey = null; // ★完全ロスト時は平滑化対象指もリセット
      resetAllBlanchBaselines();
      resetDebugMetrics();
      return;
    }
  } else {
    lostFrames = 0;
  }

  if (isDebugPanelVisible) {
    const totalDetected = results && results.landmarks ? results.landmarks.length : 0;
    handsCount.textContent = hasHands ? `${totalDetected} (右手ロック)` : "1 (補間)";
  }

  // 画面外（完全未検出状態）からの再出現初フレームかどうかを判定
  const isReacquired = hasHands && !hasValidSmoothedLandmarks;

  // メインの手のランドマーク生座標（右手セレクターで選択された右手のみを供給・左手データは完全破棄）
  const landmarks = hasHands ? selectedRightHand : lastRawLandmarks;
  if (!landmarks) return;
  lastRawLandmarks = landmarks;

  // 追従中の右手手首の生正規化座標で lastTrackedWrist を毎フレーム更新
  lastTrackedWrist = { x: landmarks[0].x, y: landmarks[0].y };

  // 演奏・描画に必要な関節点のみ独立平滑化（全21点から5点に絞りフィルタ計算負荷を約76%削減）
  if (hasHands) {
    const isFingerChanged = lastFilteredFingerKey !== currentFingerKey;
    const needSnap = isReacquired || isFingerChanged;

    if (needSnap) {
      // 画面外復帰または指切り替え初フレーム：過去の古い座標からの引きずりをバイパスし、新座標で即座にスナップ初期化
      for (const i of activeJointIndices) {
        const raw = landmarks[i];
        landmarkFilters[i].snap(raw.x * width, raw.y * height, now, smoothedLandmarksPool[i]);
      }
      lastFilteredFingerKey = currentFingerKey;
    } else {
      // 通常トラッキング時：適応平滑化（1 Euro Filter / トグルOFF時は生座標バイパス）
      for (const i of activeJointIndices) {
        const raw = landmarks[i];
        landmarkFilters[i].filter(raw.x * width, raw.y * height, now, smoothedLandmarksPool[i], !isOneEuroEnabled);
      }
    }
    hasValidSmoothedLandmarks = true;
  }
  const smoothedLandmarks = smoothedLandmarksPool;

  // フリーセッション時は5本指すべてを独立して判定・発音・描画する
  if (IS_FREE_SESSION) {
    evaluateFreeSessionFingers(landmarks, smoothedLandmarks, height, isReacquired);
    return;
  }

  // 1. 対象指以外の骨格・関節点描画は非表示（視覚ノイズ排除および描画負荷削減）

  // 2. 選択中指の平滑化ピクセル座標
  const smoothP1 = smoothedLandmarks[fingerConfig.p1Idx];
  const smoothP2 = smoothedLandmarks[fingerConfig.p2Idx];
  const smoothP3 = smoothedLandmarks[fingerConfig.p3Idx];
  const smoothTip = smoothedLandmarks[fingerConfig.tipIdx];

  // 最新平滑化座標の保持
  currentTip.x = smoothTip.x;
  currentTip.y = smoothTip.y;

  // 手首（Landmark 0: Wrist）と対象指の付け根（baseIdx）間の平滑化2Dピクセル距離（手の基準長）
  // ※生のlandmarksやジッターの大きいz深度を完全撤去し、平滑化済み2D座標のみを用いて算出
  const smoothWrist = smoothedLandmarks[0];
  const smoothBaseLm = smoothedLandmarks[fingerConfig.baseIdx];
  const baseDistPx = Math.hypot(smoothBaseLm.x - smoothWrist.x, smoothBaseLm.y - smoothWrist.y);
  // 画面高の5%を下回らないよう安全最小値を設定して分母微小化スパイクを防止
  const safeBaseDistPx = Math.max(baseDistPx, height * 0.05);

  // 対象指の基準点ピクセル座標（smoothP2 または smoothP1）
  const smoothBase = (fingerConfig.baseIdx === fingerConfig.p1Idx) ? smoothP1 : smoothP2;

  // 完全平滑化・2Dピクセル比に基づく安定した相対変位 ry
  const currentRy = (smoothTip.y - smoothBase.y) / safeBaseDistPx;

  // 現在のターゲット音符とハイライト色（単発＝水色、連続＝黄色→緑色→青色）
  const targetColor = getTargetFingerColor(currentSongStep);

  // 対象指先端の平滑化下降速度（ピクセル/秒）を取得（下向き > 0、上向き < 0）
  const tipFilter = landmarkFilters[fingerConfig.tipIdx];
  const tipVy = tipFilter ? tipFilter.getVelocityY() : 0;
  const hitVyThreshold = FINGER_VY_THRESHOLDS[currentFingerKey] || 200;

  // 2.5 BlanchTouch 指先ROI色変化解析（Blanch/Hybridモード時、またはデバッグ表示ON時のみ実行）
  const shouldRunBlanch = detectionMode !== DETECTION_MODES.LEGACY || isBlanchDebugVisible;
  const blanchResult = shouldRunBlanch
    ? blanchDetector.processFrame(
        video,
        landmarks[fingerConfig.tipIdx],
        landmarks[fingerConfig.p3Idx],
        tapState === "IDLE"
      )
    : null;

  if (isBlanchDebugVisible && blanchResult) {
    updateBlanchDebugOverlay(blanchResult);
  }

  // 3. 打鍵認識（Legacy / BlanchOnly / Hybrid モード対応）
  // カウントダウン中（3・2・1）、完走クリア演出中、および画面外復帰初フレームは打鍵認識を安全にスキップ
  if (tapState === "IDLE" && !isCountingDown && !isClearing && !isReacquired) {
    const isTapTriggered = evaluateTapTrigger(
      detectionMode,
      currentRy,
      hitRyThreshold,
      tipVy,
      hitVyThreshold,
      blanchResult
    );
    if (isTapTriggered) {
      tapState = "TOUCHED";

      // 該当指先でノーツHIT弾けエフェクトを発火
      triggerNoteHitEffect(currentSongStep);

      // 右手運指ガイドウィジェットの打鍵フィードバック（一瞬ポップ）
      const handGuideWidget = document.getElementById("hand-guide-widget");
      if (handGuideWidget) {
        handGuideWidget.classList.add("hit-pop");
        setTimeout(() => {
          handGuideWidget.classList.remove("hit-pop");
        }, 160);
      }

      // 現在の音符を取得
      const currentTarget = currentSequence[currentSongStep];

      // 右手正解メロディ音を発音（クリアなアタック）
      playTapSound(currentTarget.rightNote || currentTarget.freq, true, {
        duration: "2n",
        isBass: false
      });

      // 楽譜通りの左手自動伴奏（単音または和音）が設定されている場合は重ねて自動発音
      if (currentTarget.autoLeftNote) {
        playTapSound(currentTarget.autoLeftNote, false, {
          duration: "1m",
          isBass: true
        });
      }
      updateStateHud("TOUCHED", true);

      console.log(
        `[SONG HIT] [${SONGS[currentSongId]?.title || ""}] Step ${currentTarget.step}/${currentSequence.length} [右手打鍵] 運指:${currentTarget.fingerNum} (${currentTarget.note}) 色:${targetColor.name} ry=${currentRy.toFixed(3)} >= TH:${hitRyThreshold.toFixed(2)}, vy=${tipVy.toFixed(0)} >= V_TH:${hitVyThreshold}${currentTarget.autoLeftNote ? ` [自動伴奏: ${currentTarget.autoLeftNote}]` : ""}`
      );

      const isLastStep = currentSongStep === currentSequence.length - 1;

      if (isLastStep) {
        // 完走時：白黒ミニマルクリア通知（約1.5秒間）を表示後、インデックス0（第1音）へ自動ループ復帰
        console.log(`[SONG COMPLETE] 全${currentSequence.length}音を完走！クリア通知を表示します`);
        showClearNotification(() => {
          currentSongStep = 0;
          noteProgressMap.clear();
          const resetTarget = currentSequence[0];
          setTargetFinger(resetTarget.fingerKey);
          renderSongGuideUI();
          console.log(`[SONG LOOP] 第1音 (${resetTarget.note} / ${resetTarget.fingerKey}) へリセット完了`);
        });
      } else {
        // 通常進行：次の音符へステップ進行
        currentSongStep = currentSongStep + 1;
        const nextTarget = currentSequence[currentSongStep];

        // 次のターゲット指へ自動切り替えとガイド更新
        setTargetFinger(nextTarget.fingerKey);
        renderSongGuideUI();
      }
    }
  } else if (tapState === "TOUCHED") {
    // 指のリフト復帰（モード別に判定）
    if (evaluateLiftTrigger(detectionMode, currentRy, hitRyThreshold, liftRyThreshold, tipVy, blanchResult)) {
      tapState = "IDLE";
      updateStateHud("IDLE", false);
    }
  }

  // 4. デバッグHUDのリアルタイム表示更新（平滑化座標、相対変位、下降速度）
  updateDebugMetrics(smoothTip.x, smoothTip.y, currentRy, tipVy);

  // 5. 指定された指の骨格描画（表示フラグ showTrackingLines がONの時のみ描画）
  if (showTrackingLines) {
    drawFingerTrackingOverlay(smoothP1, smoothP2, smoothP3, smoothTip, targetColor);
  }
}

/**
 * フリーセッション用：5本指すべての独立打鍵判定・指別音程発音・描画
 * @param {Array} landmarks 生ランドマーク配列
 * @param {Array} smoothedLandmarks 平滑化済みピクセル座標配列
 * @param {number} height Canvas高さ
 * @param {boolean} isReacquired 再出現初フレームかどうか
 */
function evaluateFreeSessionFingers(landmarks, smoothedLandmarks, height, isReacquired) {
  const smoothWrist = smoothedLandmarks[0];
  const shouldRunBlanch = detectionMode !== DETECTION_MODES.LEGACY || isBlanchDebugVisible;
  const idleColor = getTargetFingerColor(0);
  const pressedColor = {
    stroke: "rgba(255, 255, 255, 0.98)",
    halo: "rgba(255, 255, 255, 0.32)",
    accent: "rgba(255, 255, 255, 0.78)",
    fill: "#ffffff"
  };

  const nowMs = performance.now();
  const hasOtherTouchedFinger = FREE_SESSION_FINGER_KEYS.some(
    (k) => freeFingerTapStates[k] === "TOUCHED"
  );
  const tapCandidates = [];
  const drawTargets = [];
  let debugBlanchResult = null;
  let debugMetricTarget = null;

  for (const fingerKey of FREE_SESSION_FINGER_KEYS) {
    const fCfg = FINGER_CONFIGS[fingerKey];
    const noteInfo = FREE_SESSION_NOTES[fingerKey];
    const model = fingerThresholdModels[fingerKey] || { hitRy: 0.60, liftRy: 0.52 };
    const fingerHitRyTh = model.hitRy;
    const fingerLiftRyTh = model.liftRy;
    const fingerHitVyTh = FINGER_VY_THRESHOLDS[fingerKey] || 200;
    // 他指が押下中のレガート時は速度閾値を厳しくしてつられ動きによる誤検知を防止
    const effectiveHitVyTh = hasOtherTouchedFinger
      ? fingerHitVyTh * FREE_SESSION_LEGATO_VY_MULTIPLIER
      : fingerHitVyTh;

    const smoothP1 = smoothedLandmarks[fCfg.p1Idx];
    const smoothP2 = smoothedLandmarks[fCfg.p2Idx];
    const smoothP3 = smoothedLandmarks[fCfg.p3Idx];
    const smoothTip = smoothedLandmarks[fCfg.tipIdx];
    const smoothBaseLm = smoothedLandmarks[fCfg.baseIdx];

    const baseDistPx = Math.hypot(smoothBaseLm.x - smoothWrist.x, smoothBaseLm.y - smoothWrist.y);
    const safeBaseDistPx = Math.max(baseDistPx, height * 0.05);
    const smoothBase = fCfg.baseIdx === fCfg.p1Idx ? smoothP1 : smoothP2;
    const currentRy = (smoothTip.y - smoothBase.y) / safeBaseDistPx;

    const tipFilter = landmarkFilters[fCfg.tipIdx];
    const tipVy = tipFilter ? tipFilter.getVelocityY() : 0;

    const detector = freeFingerBlanchDetectors[fingerKey];
    const isFingerIdle = freeFingerTapStates[fingerKey] === "IDLE";
    const blanchResult = shouldRunBlanch && detector
      ? detector.processFrame(video, landmarks[fCfg.tipIdx], landmarks[fCfg.p3Idx], isFingerIdle)
      : null;

    if (fingerKey === FREE_SESSION_FINGER_KEY) {
      debugBlanchResult = blanchResult;
      debugMetricTarget = { x: smoothTip.x, y: smoothTip.y, ry: currentRy, vy: tipVy };
    }

    if (isFingerIdle && !isCountingDown && !isClearing && !isReacquired) {
      const isTapTriggered = evaluateTapTrigger(
        detectionMode,
        currentRy,
        fingerHitRyTh,
        tipVy,
        effectiveHitVyTh,
        blanchResult
      );
      if (isTapTriggered) {
        // 1打鍵につき最有力1本のみ採用するためのスコア算出
        const score =
          detectionMode === DETECTION_MODES.BLANCH_ONLY
            ? (blanchResult ? blanchResult.smoothScore : 0)
            : (tipVy / effectiveHitVyTh) + Math.max(0, currentRy - fingerHitRyTh) * 2;
        tapCandidates.push({
          fingerKey,
          fCfg,
          noteInfo,
          currentRy,
          fingerHitRyTh,
          tipVy,
          fingerHitVyTh: effectiveHitVyTh,
          blanchResult,
          score
        });
      }
    } else if (!isFingerIdle) {
      if (evaluateLiftTrigger(detectionMode, currentRy, fingerHitRyTh, fingerLiftRyTh, tipVy, blanchResult)) {
        freeFingerTapStates[fingerKey] = "IDLE";
      }
    }

    drawTargets.push({ fingerKey, smoothP1, smoothP2, smoothP3, smoothTip });
  }

  // 他指が机についていてもクールダウン経過後なら最有力1指を採用してレガート打鍵を許可
  if (
    tapCandidates.length > 0 &&
    nowMs - lastFreeSessionHitTime >= FREE_SESSION_INTER_FINGER_COOLDOWN_MS
  ) {
    let winner = tapCandidates[0];
    for (let i = 1; i < tapCandidates.length; i++) {
      if (tapCandidates[i].score > winner.score) {
        winner = tapCandidates[i];
      }
    }

    freeFingerTapStates[winner.fingerKey] = "TOUCHED";
    lastFreeSessionHitTime = nowMs;

    playTapSound(winner.noteInfo.noteName, true, { duration: "2n", isBass: false });
    updateHandGuideWidget(winner.fingerKey, winner.fCfg.fingerNum, winner.noteInfo.label);

    const widget = document.getElementById("hand-guide-widget");
    if (widget) {
      widget.classList.add("hit-pop");
      setTimeout(() => {
        widget.classList.remove("hit-pop");
      }, 160);
    }

    updateStateHud("TOUCHED", true);
    console.log(
      `[FREE HIT] [${detectionMode}] 指:${winner.fCfg.shortName}(${winner.noteInfo.noteName}/${winner.noteInfo.label}) 候補:${tapCandidates.length}本 ry=${winner.currentRy.toFixed(3)} >= TH:${winner.fingerHitRyTh.toFixed(2)}, vy=${winner.tipVy.toFixed(0)} >= V_TH:${winner.fingerHitVyTh} blanchScore=${winner.blanchResult ? winner.blanchResult.smoothScore.toFixed(2) : "-"}`
    );
  }

  const anyPressed = FREE_SESSION_FINGER_KEYS.some((k) => freeFingerTapStates[k] === "TOUCHED");
  if (!anyPressed) {
    updateStateHud("IDLE", false);
  }

  if (showTrackingLines) {
    for (const item of drawTargets) {
      const drawColor = freeFingerTapStates[item.fingerKey] === "TOUCHED" ? pressedColor : idleColor;
      drawFingerTrackingOverlay(item.smoothP1, item.smoothP2, item.smoothP3, item.smoothTip, drawColor);
    }
  }

  if (isBlanchDebugVisible && debugBlanchResult) {
    updateBlanchDebugOverlay(debugBlanchResult);
  }
  if (debugMetricTarget) {
    updateDebugMetrics(debugMetricTarget.x, debugMetricTarget.y, debugMetricTarget.ry, debugMetricTarget.vy);
  }
}

/**
 * 1本の指の骨格・関節点・指先ターゲットマークを描画
 * @param {object} smoothP1
 * @param {object} smoothP2
 * @param {object} smoothP3
 * @param {object} smoothTip
 * @param {object} targetColor
 */
function drawFingerTrackingOverlay(smoothP1, smoothP2, smoothP3, smoothTip, targetColor) {
  canvasCtx.save();
  canvasCtx.lineCap = "round";
  canvasCtx.lineJoin = "round";

  const drawBonePath = () => {
    canvasCtx.beginPath();
    canvasCtx.moveTo(smoothP1.x, smoothP1.y);
    canvasCtx.lineTo(smoothP2.x, smoothP2.y);
    canvasCtx.lineTo(smoothP3.x, smoothP3.y);
    canvasCtx.lineTo(smoothTip.x, smoothTip.y);
  };

  // 軽量2層ライン描画（描画負荷を抑えつつ視認性を維持）
  canvasCtx.strokeStyle = targetColor.stroke;
  canvasCtx.lineWidth = 4.5;
  drawBonePath();
  canvasCtx.stroke();

  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.9)";
  canvasCtx.lineWidth = 1.8;
  drawBonePath();
  canvasCtx.stroke();

  // 関節点（P1, P2, P3）の軽量ドット描画
  canvasCtx.fillStyle = targetColor.fill;
  [smoothP1, smoothP2, smoothP3].forEach((pt) => {
    canvasCtx.beginPath();
    canvasCtx.arc(pt.x, pt.y, 4.0, 0, 2 * Math.PI);
    canvasCtx.fill();
  });

  // 指先（TIP）のハイライトターゲット描画
  drawTipTargetMark(smoothTip.x, smoothTip.y, targetColor);
  canvasCtx.restore();
}

/**
 * 対象指先端（TIP）のターゲットマーク描画（shadowBlur全廃・多層二重発光リング＋白熱コア）
 * @param {number} x
 * @param {number} y
 * @param {object} color
 */
function drawTipTargetMark(x, y, color) {
  const strokeColor = color?.stroke || "rgba(0, 229, 255, 0.95)";
  const haloColor = color?.halo || "rgba(0, 229, 255, 0.25)";
  const accentColor = color?.accent || "rgba(0, 229, 255, 0.65)";
  const fillColor = color?.fill || "#00e5ff";

  canvasCtx.save();

  // 外側の極太発光メインリング（多層化：太ハロー＋鮮明コア線）
  // 層1: 外側発光ハローリング（半径13px、線幅 7.5px）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 13, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = haloColor;
  canvasCtx.lineWidth = 7.5;
  canvasCtx.stroke();

  // 層2: 外側メインリング（半径13px、線幅 2.8px）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 13, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = strokeColor;
  canvasCtx.lineWidth = 2.8;
  canvasCtx.stroke();

  // 内側の補助リング（半径8px、線幅 1.8px、事前計算accentColorで正規表現全廃）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 8, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = accentColor;
  canvasCtx.lineWidth = 1.8;
  canvasCtx.stroke();

  // 中心発光ドット（多層描画: 外輪ハロー 半径 7.0px + メイン 半径 4.6px + 白熱コア 半径 2.4px）
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 7.0, 0, 2 * Math.PI);
  canvasCtx.fillStyle = haloColor;
  canvasCtx.fill();

  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 4.6, 0, 2 * Math.PI);
  canvasCtx.fillStyle = fillColor;
  canvasCtx.fill();

  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 2.4, 0, 2 * Math.PI);
  canvasCtx.fillStyle = "#ffffff";
  canvasCtx.fill();

  canvasCtx.restore();
}

/**
 * 人差し指TIPの生座標（X, Y）および相対変位のHUD更新表示（白黒ミニマル）
 * @param {number} x
 * @param {number} y
 * @param {number} ry
 */
function updateDebugMetrics(x, y, ry, vy = 0) {
  if (!isDebugPanelVisible) return;
  tipXVal.textContent = `${x.toFixed(1)}px`;
  tipYVal.textContent = `${y.toFixed(1)}px`;
  if (relYVal) {
    relYVal.textContent = ry !== undefined && ry !== null ? (ry >= 0 ? `+${ry.toFixed(2)}` : ry.toFixed(2)) : "-";
  }
  if (velYVal) {
    velYVal.textContent = vy !== undefined && vy !== null ? (vy >= 0 ? `+${vy.toFixed(0)}` : vy.toFixed(0)) : "-";
  }
}

/**
 * 手ロスト時のHUDメトリクスリセット
 */
function resetDebugMetrics() {
  if (!isDebugPanelVisible) return;
  tipXVal.textContent = "-";
  tipYVal.textContent = "-";
  if (relYVal) {
    relYVal.textContent = "-";
  }
  if (velYVal) {
    velYVal.textContent = "-";
  }
}

/**
 * 打鍵ステート（IDLE / TOUCHED）のHUD更新と打鍵時ハイライト
 * @param {string} state
 * @param {boolean} isHitFlash
 */
function updateStateHud(state, isHitFlash) {
  if (!stateVal) return;
  stateVal.textContent = state;
  if (isHitFlash) {
    stateVal.classList.add("spike-highlight");
    setTimeout(() => {
      if (stateVal && tapState === "TOUCHED") {
        stateVal.classList.remove("spike-highlight");
      }
    }, 120);
  } else {
    stateVal.classList.remove("spike-highlight");
  }
}

/**
 * カメラ実測FPSカウンターの計算（500msごとに更新）
 * @param {number} nowMs
 * @param {number} [presentedFrames]
 */
function updateCameraFps(nowMs, presentedFrames) {
  if (typeof presentedFrames === "number") {
    if (lastPresentedFrames < 0 || presentedFrames < lastPresentedFrames) {
      cameraFrameCount += 1;
    } else {
      cameraFrameCount += presentedFrames - lastPresentedFrames;
    }
    lastPresentedFrames = presentedFrames;
  } else {
    cameraFrameCount += 1;
  }

  const elapsed = nowMs - lastCameraFpsUpdateTime;
  if (elapsed >= 500) {
    const camFps = Math.round((cameraFrameCount * 1000) / elapsed);
    if (telemetryCamFps) {
      telemetryCamFps.textContent = `CAM ${camFps}`;
    }
    cameraFrameCount = 0;
    lastCameraFpsUpdateTime = nowMs;
  }
}

/**
 * アプリ実測FPSカウンターの計算
 */
function updateFps() {
  frameCount++;
  const now = performance.now();
  const elapsed = now - lastFpsUpdateTime;

  if (elapsed >= 500) {
    const fps = ((frameCount * 1000) / elapsed).toFixed(1);
    if (telemetryAppFps) {
      telemetryAppFps.textContent = `APP ${Math.round(parseFloat(fps))} FPS`;
    }
    checkDebugPanelVisibility();
    if (isDebugPanelVisible) {
      fpsCounter.textContent = fps;
    }
    frameCount = 0;
    lastFpsUpdateTime = now;
  }
}

/**
 * ステータス表示の更新
 */
function updateStatus(msg) {
  statusText.textContent = msg;
}

/**
 * カメラエラーの種別に応じたユーザー向けガイダンス文言を生成
 * @param {Error|DOMException} error
 * @returns {string}
 */
function getCameraErrorMessage(error) {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    return "カメラはHTTPS環境でのみ動作します。https:// または localhost でアクセスしてください。";
  }

  const errName = error?.name || "";
  const errMsg = error?.message || "";

  if (
    errName === "NotAllowedError" ||
    errName === "PermissionDeniedError" ||
    errMsg.includes("許可") ||
    errMsg.includes("NotAllowed")
  ) {
    return "カメラの利用が許可されていません。ブラウザのアドレスバーまたは端末の『設定』からカメラを許可し、ページを再読み込みしてください。";
  }

  if (
    errName === "NotReadableError" ||
    errName === "TrackStartError" ||
    errMsg.includes("使用中") ||
    errMsg.includes("NotReadable")
  ) {
    return "カメラを起動できませんでした。他のアプリ（通話・カメラ等）が使用中の可能性があります。他アプリを終了して再読み込みしてください。";
  }

  if (
    errName === "OverconstrainedError" ||
    errName === "ConstraintNotSatisfiedError"
  ) {
    return "カメラの初期化に失敗しました。標準設定で再試行します。";
  }

  return "カメラの初期化に失敗しました。標準設定で再試行します。";
}

/**
 * カメラエラー・ガイダンス通知オーバーレイの表示（白黒ミニマル）
 * @param {string} message
 * @param {string} [title]
 */
function showCameraErrorOverlay(message, title = "カメラを起動できません") {
  if (cameraErrorTitle) cameraErrorTitle.textContent = title;
  if (cameraErrorMessage) cameraErrorMessage.textContent = message;
  if (cameraErrorOverlay) {
    cameraErrorOverlay.classList.remove("hidden");
  }
}

/**
 * エラー表示（デバッグパネル用）
 */
function showError(msg) {
  if (errorBox) {
    errorBox.textContent = `ERROR: ${msg}`;
    errorBox.classList.remove("hidden");
  }
}

// カメラエラーオーバーレイのボタンハンドラ
if (cameraErrorReloadBtn) {
  cameraErrorReloadBtn.addEventListener("click", () => {
    window.location.reload();
  });
}
if (cameraErrorCloseBtn) {
  cameraErrorCloseBtn.addEventListener("click", () => {
    if (cameraErrorOverlay) {
      cameraErrorOverlay.classList.add("hidden");
    }
  });
}

// カメラ選択ボタン・メニュー制御
if (cameraSelectBtn && cameraSelectMenu) {
  cameraSelectBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await refreshCameraList();
    if (songSelectMenu) songSelectMenu.classList.add("hidden");
    cameraSelectMenu.classList.toggle("hidden");
  });

  document.addEventListener("click", (e) => {
    if (!cameraSelectMenu.classList.contains("hidden") && !cameraSelectMenu.contains(e.target) && e.target !== cameraSelectBtn) {
      cameraSelectMenu.classList.add("hidden");
    }
  });
}

// 鏡像（左右反転）トグル制御
if (mirrorToggleBtn) {
  mirrorToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isMirrorEnabled = !isMirrorEnabled;
    try {
      localStorage.setItem(PC_MIRROR_STORAGE_KEY, String(isMirrorEnabled));
    } catch {}
    applyMirrorUI();
  });
}

function syncOneEuroToggleUI() {
  if (!oneEuroToggleBtn) return;
  oneEuroToggleBtn.classList.toggle("off", !isOneEuroEnabled);
  oneEuroToggleBtn.setAttribute("aria-pressed", String(isOneEuroEnabled));
  oneEuroToggleBtn.title = isOneEuroEnabled ? "1 Euro Filter: ON" : "1 Euro Filter: OFF";
}

if (oneEuroToggleBtn) {
  syncOneEuroToggleUI();
  oneEuroToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isOneEuroEnabled = !isOneEuroEnabled;
    try {
      localStorage.setItem(PC_ONE_EURO_STORAGE_KEY, String(isOneEuroEnabled));
    } catch {}
    syncOneEuroToggleUI();
    console.log(`[1 EURO FILTER] ${isOneEuroEnabled ? "ON（平滑化有効）" : "OFF（生座標バイパス）"}`);
  });
}

// 解像度同期イベント
video.addEventListener("resize", updateCanvasResolution);
window.addEventListener("resize", updateCanvasResolution);

// DOM読み込み完了時に自動実行（既に完了している場合は即時実行）
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

// ブラウザのAutoplay Policy対応（初回クリックまたはタップでAudioContextを確実にresume）
["pointerdown", "touchstart", "click", "keydown"].forEach((evt) => {
  window.addEventListener(evt, ensureAudioContext, { once: false, passive: true });
});

// TEST SOUNDボタン押下時の手動テスト発音
if (testSoundBtn) {
  testSoundBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    ensureAudioContext();
    // 右手メロディ（E5）と左手和音伴奏（C3 ＋ [G3, C4]）のテスト発音
    playTapSound("E5", false, { duration: "2n", isBass: false });
    playTapSound(["C3", "G3", "C4"], false, { duration: "1m", isBass: true });
    console.log("[AUDIO] TEST SOUND 和音テスト発音（右手: E5 / 左手伴奏: C3 + G3 + C4）");
  });
}

// 手元CSVファイルの手動追加インプット (+FILE)
if (csvFileInput) {
  csvFileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const loadedList = [];
    for (const file of files) {
      const text = await file.text();
      loadedList.push({ name: file.name, content: text });
    }

    ingestCsvFiles(loadedList);
    csvFileInput.value = ""; // 連続選択可能にするためリセット
  });
}

// 画面右上の丸い楽曲選択ボタンとメニュー
const songSelectBtn = document.getElementById("song-select-btn");
const songSelectMenu = document.getElementById("song-select-menu");

if (songSelectBtn && songSelectMenu) {
  // 丸ボタンタップでメニュー表示/非表示トグル
  songSelectBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    ensureAudioContext(); // ユーザー操作契機でオーディオ初期化
    songSelectMenu.classList.toggle("hidden");
  });

  // メニュー項目タップで曲選択
  document.querySelectorAll(".song-menu-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      ensureAudioContext();
      const songId = item.dataset.song;
      if (songId) {
        selectSong(songId);
      }
    });
  });

  // 画面外タップでメニューを閉じる
  document.addEventListener("click", (e) => {
    if (!songSelectMenu.classList.contains("hidden") && !songSelectMenu.contains(e.target)) {
      songSelectMenu.classList.add("hidden");
    }
  });
}

// 画面右上のトラッキングライン表示切替ボタン
const trackingLineToggleBtn = document.getElementById("tracking-line-toggle-btn");
if (trackingLineToggleBtn) {
  const updateBtnUI = () => {
    trackingLineToggleBtn.classList.toggle("off", !showTrackingLines);
    trackingLineToggleBtn.setAttribute("aria-pressed", showTrackingLines.toString());
    trackingLineToggleBtn.title = showTrackingLines
      ? "トラッキングライン非表示に切替"
      : "トラッキングライン表示に切替";
  };
  updateBtnUI();

  trackingLineToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    ensureAudioContext();
    showTrackingLines = !showTrackingLines;
    try {
      localStorage.setItem(TRACKING_LINES_STORAGE_KEY, showTrackingLines.toString());
    } catch {
      // localStorage制限環境用フォールバック
    }
    updateBtnUI();
  });
}

// ==========================================================================
// BlanchTouch 実験用デバッグオーバーレイ制御
// ==========================================================================
const blanchDebugToggleBtn = document.getElementById("blanch-debug-toggle-btn");
const blanchDebugOverlay = document.getElementById("blanch-debug-overlay");
const blanchRoiPreview = document.getElementById("blanch-roi-preview");
const blanchStatePill = document.getElementById("blanch-state-pill");
const blanchScoreVal = document.getElementById("blanch-score-val");
const blanchThDisplay = document.getElementById("blanch-th-display");
const blanchDaVal = document.getElementById("blanch-da-val");
const blanchDlVal = document.getElementById("blanch-dl-val");
const blanchBarFill = document.getElementById("blanch-bar-fill");
const blanchBarOnMarker = document.getElementById("blanch-bar-on-marker");
const blanchBarOffMarker = document.getElementById("blanch-bar-off-marker");
const blanchThSlider = document.getElementById("blanch-th-slider");
const blanchThVal = document.getElementById("blanch-th-val");
const blanchRoiSlider = document.getElementById("blanch-roi-slider");
const blanchRoiVal = document.getElementById("blanch-roi-val");
const blanchEmaSlider = document.getElementById("blanch-ema-slider");
const blanchEmaVal = document.getElementById("blanch-ema-val");
const blanchResetBtn = document.getElementById("blanch-reset-btn");
const blanchCloseBtn = document.getElementById("blanch-close-btn");
const blanchModeBtns = document.querySelectorAll(".blanch-mode-btn");

const BLANCH_BAR_MAX_SCORE = 12.0;

/**
 * 検証用デバッグオーバーレイのリアルタイム表示更新
 * @param {object} result BlanchDetector.processFrame の返却値
 */
function updateBlanchDebugOverlay(result) {
  if (!isBlanchDebugVisible || !result) return;

  if (blanchRoiPreview) {
    blanchDetector.renderPreview(blanchRoiPreview);
  }
  if (blanchScoreVal) {
    blanchScoreVal.textContent = result.smoothScore.toFixed(2);
  }
  if (blanchDaVal) {
    blanchDaVal.textContent = (result.deltaA >= 0 ? "+" : "") + result.deltaA.toFixed(2);
  }
  if (blanchDlVal) {
    blanchDlVal.textContent = (result.deltaL >= 0 ? "+" : "") + result.deltaL.toFixed(2);
  }
  if (blanchStatePill) {
    blanchStatePill.textContent = result.isBlanching ? "BLANCH" : "IDLE";
    blanchStatePill.classList.toggle("active", result.isBlanching);
  }
  if (blanchBarFill) {
    const pct = Math.min(100, Math.max(0, (result.smoothScore / BLANCH_BAR_MAX_SCORE) * 100));
    blanchBarFill.style.width = `${pct.toFixed(1)}%`;
  }
  if (blanchBarOnMarker) {
    const onPct = Math.min(100, Math.max(0, (result.onThreshold / BLANCH_BAR_MAX_SCORE) * 100));
    blanchBarOnMarker.style.left = `${onPct.toFixed(1)}%`;
  }
  if (blanchBarOffMarker) {
    const offPct = Math.min(100, Math.max(0, (result.offThreshold / BLANCH_BAR_MAX_SCORE) * 100));
    blanchBarOffMarker.style.left = `${offPct.toFixed(1)}%`;
  }
}

function syncBlanchControlUI() {
  if (blanchDebugOverlay) {
    blanchDebugOverlay.classList.toggle("hidden", !isBlanchDebugVisible);
  }
  if (blanchDebugToggleBtn) {
    blanchDebugToggleBtn.classList.toggle("off", !isBlanchDebugVisible);
    blanchDebugToggleBtn.setAttribute("aria-pressed", isBlanchDebugVisible.toString());
  }
  blanchModeBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === detectionMode);
  });
  if (telemetryMode) {
    const modeLabelMap = {
      [DETECTION_MODES.LEGACY]: "LEGACY",
      [DETECTION_MODES.BLANCH_ONLY]: "BLANCH",
      [DETECTION_MODES.HYBRID]: "HYBRID"
    };
    telemetryMode.textContent = modeLabelMap[detectionMode] || "LEGACY";
  }
  if (blanchThSlider && blanchThVal && blanchThDisplay) {
    blanchThSlider.value = blanchDetector.onThreshold.toFixed(1);
    blanchThVal.textContent = blanchDetector.onThreshold.toFixed(1);
    blanchThDisplay.textContent = blanchDetector.onThreshold.toFixed(1);
  }
  if (blanchRoiSlider && blanchRoiVal) {
    blanchRoiSlider.value = String(blanchDetector.roiSize);
    blanchRoiVal.textContent = `${blanchDetector.roiSize}px`;
  }
  if (blanchEmaSlider && blanchEmaVal) {
    blanchEmaSlider.value = blanchDetector.smoothingAlpha.toFixed(2);
    blanchEmaVal.textContent = blanchDetector.smoothingAlpha.toFixed(2);
  }
}

syncBlanchControlUI();

if (blanchDebugToggleBtn) {
  blanchDebugToggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isBlanchDebugVisible = !isBlanchDebugVisible;
    syncBlanchControlUI();
  });
}

if (blanchCloseBtn) {
  blanchCloseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isBlanchDebugVisible = false;
    syncBlanchControlUI();
  });
}

if (blanchResetBtn) {
  blanchResetBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    updateAllBlanchSettings({
      onThreshold: BLANCH_DEFAULTS.onThreshold,
      roiSize: BLANCH_DEFAULTS.roiSize,
      smoothingAlpha: BLANCH_DEFAULTS.smoothingAlpha
    });
    resetAllBlanchBaselines();
    try {
      localStorage.setItem(BLANCH_STORAGE_KEYS.THRESHOLD, String(BLANCH_DEFAULTS.onThreshold));
      localStorage.setItem(BLANCH_STORAGE_KEYS.ROI_SIZE, String(BLANCH_DEFAULTS.roiSize));
      localStorage.setItem(BLANCH_STORAGE_KEYS.SMOOTHING, String(BLANCH_DEFAULTS.smoothingAlpha));
    } catch {
      // localStorage制限環境用フォールバック
    }
    syncBlanchControlUI();
  });
}

blanchModeBtns.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const nextMode = btn.dataset.mode;
    if (!Object.values(DETECTION_MODES).includes(nextMode)) return;
    detectionMode = nextMode;
    tapState = "IDLE";
    resetFreeFingerStates();
    updateStateHud("IDLE", false);
    resetAllBlanchBaselines();
    try {
      localStorage.setItem(BLANCH_STORAGE_KEYS.MODE, nextMode);
    } catch {
      // localStorage制限環境用フォールバック
    }
    syncBlanchControlUI();
  });
});

if (blanchThSlider) {
  blanchThSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    updateAllBlanchSettings({ onThreshold: val });
    try {
      localStorage.setItem(BLANCH_STORAGE_KEYS.THRESHOLD, String(val));
    } catch {}
    syncBlanchControlUI();
  });
}

if (blanchRoiSlider) {
  blanchRoiSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    updateAllBlanchSettings({ roiSize: val });
    try {
      localStorage.setItem(BLANCH_STORAGE_KEYS.ROI_SIZE, String(val));
    } catch {}
    syncBlanchControlUI();
  });
}

if (blanchEmaSlider) {
  blanchEmaSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    updateAllBlanchSettings({ smoothingAlpha: val });
    try {
      localStorage.setItem(BLANCH_STORAGE_KEYS.SMOOTHING, String(val));
    } catch {}
    syncBlanchControlUI();
  });
}

// ==========================================================================
// 画面内の右手運指ガイドウィジェット（画面右下・中サイズ 130px 固定配置）
// ==========================================================================
const handGuideWidget = document.getElementById("hand-guide-widget");
if (handGuideWidget) {
  // 過去のドラッグ移動座標やリサイズ値がlocalStorageに残っている場合はクリーンアップ
  try {
    localStorage.removeItem("piarno_hand_guide_custom_size");
    localStorage.removeItem("piarno_hand_guide_pos_x");
    localStorage.removeItem("piarno_hand_guide_pos_y");
  } catch {
    // localStorage制限環境用フォールバック
  }
  handGuideWidget.style.left = "";
  handGuideWidget.style.top = "";
  handGuideWidget.style.right = "";
  handGuideWidget.style.bottom = "";
  handGuideWidget.style.setProperty("--hand-guide-size", "200px");
}

// 初期ターゲット指（よろこびのうた 第1音: 中指 3 ミ）の設定と楽曲ガイドUIの描画
setTargetFinger(currentSequence[0].fingerKey);
renderSongGuideUI();

// バックグラウンドからの復帰時の安定化（AudioContext再開・Wake Lock再要求）
// AudioContextのサスペンド復帰およびScreen Wake Lockの再要求
async function handleResumeAudioAndWakeLock() {
  try {
    if (Tone && Tone.context && (Tone.context.state === "suspended" || Tone.context.state === "interrupted")) {
      await Tone.context.resume();
      console.log("[AUDIO] 復帰ハンドラ: Tone.context を再開しました");
    }
    const rawCtx = Tone.getContext().rawContext;
    if (rawCtx && (rawCtx.state === "suspended" || rawCtx.state === "interrupted")) {
      await rawCtx.resume();
      console.log("[AUDIO] 復帰ハンドラ: raw AudioContext を再開しました");
    }
  } catch (err) {
    console.warn("[AUDIO] 復帰時のAudioContext再開エラー:", err);
  }

  // カメラ推論・演奏中であれば画面スリープ防止（Wake Lock）を安全に再要求
  if (isPredicting) {
    await requestWakeLock();
  }
}

// タブ表示切り替えイベント（戻ってきた時に再開）
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    handleResumeAudioAndWakeLock();
  }
});

// ページ復元イベント（bfcache復元時等）
window.addEventListener("pageshow", () => {
  handleResumeAudioAndWakeLock();
});



