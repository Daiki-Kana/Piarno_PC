import { defineConfig } from "vite";

// Vite設定ファイル（PCローカルテスト・静的ホスティング対応）
export default defineConfig({
  base: "./",
  server: {
    port: 5173
  }
});

