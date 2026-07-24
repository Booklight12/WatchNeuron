import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const cacheRoot = resolve(".zig-cache");
mkdirSync(resolve(cacheRoot, "global"), { recursive: true });
mkdirSync(resolve(cacheRoot, "local"), { recursive: true });

const commonArgs = [
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
  "--export=forward_dense_block",
  "--export=forward_dense_training",
  "--export=train_sample",
  "--export=train_dense_from_gradient",
  "--export=conv2d_forward",
  "--export=conv2d_train",
  "--export=conv2d_forward_batch",
  "--export=conv2d_train_batch",
  "--export=pool2d_forward",
  "--export=pool2d_backward",
  "--export=pool2d_forward_batch",
  "--export=pool2d_backward_batch",
  "--export=dense_forward_batch",
  "--export=dense_backward_batch",
  "--export=output_loss_batch",
  "--export=apply_optimizer",
  "--export=simd_enabled",
  "--export=set_math_mode",
  "--export=math_mode",
  "--export=__heap_base",
  "--initial-memory=8388608",
];

function buildKernel(cpu, output) {
  const result = spawnSync("zig", [
    ...commonArgs,
    `-mcpu=${cpu}`,
    `-femit-bin=${output}`,
  ], {
    stdio: "inherit",
    shell: false,
    env: {
      ...process.env,
      ZIG_GLOBAL_CACHE_DIR: resolve(cacheRoot, "global"),
      ZIG_LOCAL_CACHE_DIR: resolve(cacheRoot, "local"),
    },
  });

  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

buildKernel("baseline", "public/neuron_kernel.wasm");
buildKernel("baseline+simd128", "public/neuron_kernel_simd.wasm");
