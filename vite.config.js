import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { qrcode } from "vite-plugin-qrcode";

// Vite設定ファイル（PCローカルテスト・HTTPS・QR表示・静的ホスティング対応）
export default defineConfig({
  base: "./",
  plugins: [basicSsl(), qrcode()],
  server: {
    host: true,
    port: 5173
  }
});

