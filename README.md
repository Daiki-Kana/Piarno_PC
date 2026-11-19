# Piarno PC

PC接続の60fps対応Webカメラを前提とした、ブラウザ完結型のエアピアノ（Hand Landmarker + Tone.js）です。

## 主な特徴

- **60fps Webカメラ対応**: `1280×720 @ 60fps` を優先してキャプチャし、`requestVideoFrameCallback` でフレーム同期推論を実行
- **カメラデバイス選択**: 接続中のWebカメラ一覧から使用カメラを切り替え可能（選択状態を保持）
- **鏡像（Mirror）切替**: 映像・トラッキング描画の左右反転をワンクリックで切り替え可能
- **打鍵検知モード切替**: 従来の1D位置・速度ベース判定（`LEGACY`）を標準とし、検証用に `BLANCH` / `HYBRID` へリアルタイム切替可能
- **60fpsテレメトリ表示**: 左上のピルで実測FPS・解像度・推論時間・判定モードを常時確認可能

## ディレクトリ構成

```
Piarno_PC/
├── index.html          # メインUI（ビデオ/Canvas重畳、テレメトリ、各種アイコンコントロール）
├── main.js             # Webカメラ制御・MediaPipe推論・1D打鍵判定・Tone.js音響合成
├── style.css           # 白黒ミニマル＆ボーダーレスUIスタイル
├── blanchDetector.js   # BlanchTouch（爪先ROI色変化）検出モジュール
├── vite.config.js      # Vite開発サーバー＆静的ビルド設定
└── public/
    └── models/         # MediaPipe HandLandmarker モデルアセット
```

## 起動方法

```bash
npm install
npm run dev
```

本番用静的ビルド（GitHub Pages / Cloudflare Pages / Vercel 等でそのまま公開可能）：

```bash
npm run build
```

