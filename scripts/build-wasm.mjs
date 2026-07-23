import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const cacheRoot = resolve(".zig-cache");
mkdirSync(resolve(cacheRoot, "global"), { recursive: true });
mkdirSync(resolve(cacheRoot, "local"), { recursive: true });

const args = [
  "build-exe",
  "wasm/neuron_kernel.zig",
  "-target",
  "wasm32-freestanding",
  "-O",
  "ReleaseFast",
  "-fno-entry",
  "--export-memory",
  "--export=matvec",
  "--export=activate",
  "--export=forward_sparse",
  "--export=train_sample",
  "--export=__heap_base",
  "--initial-memory=8388608",
  "-femit-bin=public/neuron_kernel.wasm",
];

const result = spawnSync("zig", args, {
  stdio: "inherit",
  shell: process.platform === "win32",
  env: {
    ...process.env,
    ZIG_GLOBAL_CACHE_DIR: resolve(cacheRoot, "global"),
    ZIG_LOCAL_CACHE_DIR: resolve(cacheRoot, "local"),
  },
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
