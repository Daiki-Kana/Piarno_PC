import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { qrcode } from "vite-plugin-qrcode";

// Vite設定ファイル（ローカル開発時はHTTPS+QR、本番ビルドは静的ホスティング向け）
export default defineConfig(({ command }) => ({
  base: "./",
  plugins: command === "serve" ? [basicSsl(), qrcode()] : [],
  server: {
    host: true,
    port: 5173
  }
}));

