import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  plugins: [vue()],
  server: {
    host: "0.0.0.0",
    watch:
      process.env.CODEX_SANDBOX === "seatbelt"
        ? { useFsEvents: false, usePolling: true }
        : undefined,
  },
});
