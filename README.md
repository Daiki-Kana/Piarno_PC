# Maracas Web App

MediaPipe Hand Landmarker（および MicroPose WebGPU 切替対応）と Web Audio API による、ブラウザ完結型のエアマラカス Web アプリです。

## 主な特徴

- **パー / グーによる2音色打ち分け**:
  - **パー（開いた手）**: 高域中心の明るいシェイカー音
  - **グー（閉じた手）**: 低〜中域中心の太く低めの音
- **外方向振り検知**: 勢い良く外（小指側）へ振った直後の減速・折り返しを検知して発音
- **トラッキングエンジン切替**: MediaPipe（`MP-GPU`）を標準とし、WebGPU 環境では `MICRO`（`@svenflow/micro-handpose`）へのリアルタイム切替も可能
- **ローカル開発用 HTTPS + QR 表示**: `npm run dev` 起動時にスマホ等からアクセスできる `https://` URL の QR コードを表示

## 起動方法（ローカル開発）

```bash
npm install
npm run dev
```

## Cloudflare Pages での公開方法

静的ホスティング（クライアントサイド完結）としてそのまま公開できます。

- **ビルドコマンド**: `npm run build`
- **ビルド出力ディレクトリ**: `dist`
- **デプロイブランチ**: `maracas`（または公開対象ブランチ）
- **ヘッダ設定**: `public/_headers` が自動的に `dist/_headers` へ出力され、`Permissions-Policy: camera=(self)` 等が適用されます

ローカルでビルド結果を確認する場合：

```bash
npm run build
npm run preview
```
