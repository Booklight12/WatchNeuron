import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  plugins: [vue()],
  server: {
    host: "0.0.0.0",
    watch: {
      // Generated output and local browser profiles can contain locked files on
      // Windows. They are not source inputs, so Vite must never watch them.
      ignored: [
        "**/build/**",
        "**/dist/**",
        "**/deploy/**",
        "**/out/**",
        "**/outputs/**",
        "**/coverage/**",
        "**/.cache/**",
        "**/.zig-cache/**",
        "**/zig-out/**",
        "**/db/**",
      ],
      ...(process.env.CODEX_SANDBOX === "seatbelt"
        ? { useFsEvents: false, usePolling: true }
        : {}),
    },
  },
});
