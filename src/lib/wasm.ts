import type { MathMode } from "../types";

export type WasmKernelFlavor = "simd" | "scalar";

export interface MathModeWasmExports {
  set_math_mode: (mode: number) => void;
  math_mode: () => number;
}

export interface LoadedWasmKernel<Exports> {
  exports: Exports;
  flavor: WasmKernelFlavor;
}

const candidates: Array<{ file: string; flavor: WasmKernelFlavor }> = [
  { file: "neuron_kernel_simd.wasm", flavor: "simd" },
  { file: "neuron_kernel.wasm", flavor: "scalar" },
];

export function applyWasmMathMode(
  exports: MathModeWasmExports,
  mode: MathMode,
) {
  exports.set_math_mode(mode === "full" ? 1 : 0);
  if (exports.math_mode() !== (mode === "full" ? 1 : 0)) {
    throw new Error("Zig/Wasm 数学模式切换失败");
  }
}

export async function loadBestWasmKernel<Exports>(
  validate: (exports: Exports, flavor: WasmKernelFlavor) => boolean,
): Promise<LoadedWasmKernel<Exports>> {
  let lastError: unknown;
  for (const candidate of candidates) {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}${candidate.file}`);
      if (!response.ok) throw new Error(`无法载入 ${candidate.file}`);
      const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
      const exports = instance.exports as unknown as Exports;
      if (!validate(exports, candidate.flavor)) {
        throw new Error(`${candidate.file} 导出不完整`);
      }
      return { exports, flavor: candidate.flavor };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("无法载入 Zig/Wasm 内核");
}
