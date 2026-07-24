import type {
  ActivationKind,
  InferenceResult,
  MathMode,
  NeuralModel,
} from "../types";
import { activateScalar } from "./activations";
import { modelConvolutions } from "./convolution";
import {
  applyWasmMathMode,
  loadBestWasmKernel,
  type MathModeWasmExports,
  type WasmKernelFlavor,
} from "./wasm";

interface WasmExports extends MathModeWasmExports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  matvec: (
    input: number,
    weights: number,
    biases: number,
    output: number,
    inputSize: number,
    outputSize: number,
  ) => void;
  activate: (values: number, length: number, kind: number) => void;
  conv2d_forward: (...args: number[]) => void;
  simd_enabled: () => number;
}

interface PreparedConvolution {
  position: number;
  inputPtr: number;
  weightsPtr: number;
  biasesPtr: number;
  outputPtr: number;
  preactivationPtr: number;
  inputWidth: number;
  inputHeight: number;
  inputChannels: number;
  outputSize: number;
  filters: number;
  kernelSize: number;
  stride: number;
  padding: number;
  activation: ActivationKind;
}

interface PreparedLayer {
  inputPtr: number;
  weightsPtr: number;
  biasesPtr: number;
  outputPtr: number;
  inputSize: number;
  outputSize: number;
  activation: ActivationKind | "linear";
}

interface PreparedModel {
  inputPtr: number;
  convolutions: PreparedConvolution[];
  layers: PreparedLayer[];
}

const activationCodes: Record<ActivationKind | "linear", number> = {
  linear: 0,
  relu: 1,
  leakyRelu: 2,
  sigmoid: 3,
  tanh: 4,
  softmax: 5,
  elu: 6,
  selu: 7,
  gelu: 8,
  swish: 9,
  mish: 10,
  softplus: 11,
  softsign: 12,
  hardSigmoid: 13,
  hardTanh: 14,
  relu6: 15,
};

function align(value: number) {
  return (value + 15) & ~15;
}

function softmax(logits: ArrayLike<number>) {
  let maximum = -Infinity;
  for (let index = 0; index < logits.length; index++) maximum = Math.max(maximum, logits[index]);
  const values = Array.from(logits, (value) => Math.exp(value - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

export class InferenceEngine {
  private wasm: WasmExports | null = null;
  private wasmFlavor: WasmKernelFlavor | null = null;
  private selectedMathMode: MathMode = "fast";
  private prepared = new WeakMap<NeuralModel, PreparedModel>();

  get backend(): "Wasm SIMD" | "Wasm" | "JavaScript" {
    if (!this.wasm) return "JavaScript";
    return this.wasmFlavor === "simd" ? "Wasm SIMD" : "Wasm";
  }

  get mathMode(): MathMode {
    return this.selectedMathMode;
  }

  setMathMode(mode: MathMode) {
    this.selectedMathMode = mode === "full" ? "full" : "fast";
    if (this.wasm) applyWasmMathMode(this.wasm, this.selectedMathMode);
  }

  async initialize() {
    try {
      const loaded = await loadBestWasmKernel<WasmExports>((exports, flavor) =>
        exports.memory instanceof WebAssembly.Memory &&
        typeof exports.matvec === "function" &&
        typeof exports.activate === "function" &&
        typeof exports.conv2d_forward === "function" &&
        typeof exports.set_math_mode === "function" &&
        typeof exports.math_mode === "function" &&
        typeof exports.simd_enabled === "function" &&
        exports.simd_enabled() === (flavor === "simd" ? 1 : 0),
      );
      this.wasm = loaded.exports;
      this.wasmFlavor = loaded.flavor;
      applyWasmMathMode(this.wasm, this.selectedMathMode);
    } catch {
      this.wasm = null;
      this.wasmFlavor = null;
    }
  }

  run(model: NeuralModel, input: Float32Array): InferenceResult {
    const startedAt = performance.now();
    if (this.wasm) applyWasmMathMode(this.wasm, this.selectedMathMode);
    const activations = this.wasm
      ? this.runWasm(model, input)
      : this.runJavaScript(model, input);
    const probabilities = softmax(activations.at(-1) ?? []);
    activations[activations.length - 1] = probabilities;
    return {
      probabilities,
      activations: [Array.from(input), ...activations],
      latencyMs: performance.now() - startedAt,
      backend: this.backend,
    };
  }

  private prepare(model: NeuralModel): PreparedModel {
    const cached = this.prepared.get(model);
    if (cached) return cached;
    if (!this.wasm) throw new Error("Wasm is not initialized");

    let cursor = align(Number(this.wasm.__heap_base.value));
    const inputPtr = cursor;
    cursor += 784 * 4;
    let previousOutputPtr = inputPtr;
    const layers: PreparedLayer[] = [];
    let memory = new Float32Array(this.wasm.memory.buffer);

    const convolutions: PreparedConvolution[] = [];
    const sourceConvolutions = modelConvolutions(model);
    for (let layerIndex = 0; layerIndex < model.layers.length; layerIndex++) {
      for (const layer of sourceConvolutions.filter(({ position }) => position === layerIndex)) {
        const outputSize = layer.outputWidth * layer.outputHeight * layer.filters;
        const weightsPtr = align(cursor);
        cursor = weightsPtr + layer.weights.byteLength;
        const biasesPtr = align(cursor);
        cursor = biasesPtr + layer.biases.byteLength;
        const outputPtr = align(cursor);
        cursor = outputPtr + outputSize * 4;
        const preactivationPtr = align(cursor);
        cursor = preactivationPtr + outputSize * 4;
        if (cursor > this.wasm.memory.buffer.byteLength) {
          this.wasm.memory.grow(Math.ceil((cursor - this.wasm.memory.buffer.byteLength) / 65536));
          memory = new Float32Array(this.wasm.memory.buffer);
        }
        memory.set(layer.weights, weightsPtr / 4);
        memory.set(layer.biases, biasesPtr / 4);
        convolutions.push({
          position: layer.position,
          inputPtr: previousOutputPtr,
          weightsPtr,
          biasesPtr,
          outputPtr,
          preactivationPtr,
          inputWidth: layer.inputWidth,
          inputHeight: layer.inputHeight,
          inputChannels: layer.inputChannels,
          outputSize,
          filters: layer.filters,
          kernelSize: layer.kernelSize,
          stride: layer.stride,
          padding: layer.padding,
          activation: layer.activation,
        });
        previousOutputPtr = outputPtr;
      }
      const layer = model.layers[layerIndex];
      cursor = align(cursor);
      const weightsPtr = cursor;
      cursor += layer.weights.byteLength;
      const biasesPtr = cursor;
      cursor += layer.biases.byteLength;
      const outputPtr = align(cursor);
      cursor = outputPtr + layer.outputSize * 4;
      if (cursor > this.wasm.memory.buffer.byteLength) {
        const missingBytes = cursor - this.wasm.memory.buffer.byteLength;
        this.wasm.memory.grow(Math.ceil(missingBytes / 65536));
        memory = new Float32Array(this.wasm.memory.buffer);
      }
      memory.set(layer.weights, weightsPtr / 4);
      memory.set(layer.biases, biasesPtr / 4);
      layers.push({
        inputPtr: previousOutputPtr,
        weightsPtr,
        biasesPtr,
        outputPtr,
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activation: layer.activation,
      });
      previousOutputPtr = outputPtr;
    }

    const prepared = { inputPtr, convolutions, layers };
    this.prepared.set(model, prepared);
    return prepared;
  }

  private runWasm(model: NeuralModel, input: Float32Array) {
    if (!this.wasm) return this.runJavaScript(model, input);
    const prepared = this.prepare(model);
    const memory = new Float32Array(this.wasm.memory.buffer);
    memory.set(input, prepared.inputPtr / 4);
    const activations: number[][] = [];

    for (let layerIndex = 0; layerIndex < prepared.layers.length; layerIndex++) {
      for (const layer of prepared.convolutions.filter(({ position }) => position === layerIndex)) {
        this.wasm.conv2d_forward(
          layer.inputPtr,
          layer.weightsPtr,
          layer.biasesPtr,
          layer.outputPtr,
          layer.preactivationPtr,
          layer.inputWidth,
          layer.inputHeight,
          layer.inputChannels,
          layer.filters,
          layer.kernelSize,
          layer.stride,
          layer.padding,
          activationCodes[layer.activation],
        );
        activations.push(Array.from(memory.subarray(
          layer.outputPtr / 4,
          layer.outputPtr / 4 + layer.outputSize,
        )));
      }
      const layer = prepared.layers[layerIndex];
      this.wasm.matvec(
        layer.inputPtr,
        layer.weightsPtr,
        layer.biasesPtr,
        layer.outputPtr,
        layer.inputSize,
        layer.outputSize,
      );
      this.wasm.activate(
        layer.outputPtr,
        layer.outputSize,
        activationCodes[layer.activation],
      );
      activations.push(
        Array.from(
          memory.subarray(
            layer.outputPtr / 4,
            layer.outputPtr / 4 + layer.outputSize,
          ),
        ),
      );
    }
    return activations;
  }

  private runJavaScript(model: NeuralModel, input: Float32Array) {
    let current: ArrayLike<number> = input;
    const activations: number[][] = [];
    const convolutions = modelConvolutions(model);
    for (let layerIndex = 0; layerIndex < model.layers.length; layerIndex++) {
      for (const layer of convolutions.filter(({ position }) => position === layerIndex)) {
        const output = new Array<number>(layer.outputWidth * layer.outputHeight * layer.filters);
        for (let filter = 0; filter < layer.filters; filter++) {
          for (let outputY = 0; outputY < layer.outputHeight; outputY++) {
            for (let outputX = 0; outputX < layer.outputWidth; outputX++) {
              let sum = layer.biases[filter];
              for (let channel = 0; channel < layer.inputChannels; channel++) {
                for (let kernelY = 0; kernelY < layer.kernelSize; kernelY++) {
                  const inputY = outputY * layer.stride + kernelY - layer.padding;
                  if (inputY < 0 || inputY >= layer.inputHeight) continue;
                  for (let kernelX = 0; kernelX < layer.kernelSize; kernelX++) {
                    const inputX = outputX * layer.stride + kernelX - layer.padding;
                    if (inputX < 0 || inputX >= layer.inputWidth) continue;
                    const inputIndex = channel * layer.inputWidth * layer.inputHeight + inputY * layer.inputWidth + inputX;
                    const weightIndex = ((filter * layer.inputChannels + channel) * layer.kernelSize + kernelY) * layer.kernelSize + kernelX;
                    sum += layer.weights[weightIndex] * current[inputIndex];
                  }
                }
              }
              const outputIndex = filter * layer.outputWidth * layer.outputHeight + outputY * layer.outputWidth + outputX;
              output[outputIndex] = activateScalar(sum, layer.activation);
            }
          }
        }
        if (layer.activation === "softmax") {
          output.splice(0, output.length, ...softmax(output));
        }
        activations.push(output);
        current = output;
      }
      const layer = model.layers[layerIndex];
      const output = new Array<number>(layer.outputSize);
      for (let row = 0; row < layer.outputSize; row++) {
        let sum = layer.biases[row];
        const offset = row * layer.inputSize;
        for (let column = 0; column < layer.inputSize; column++) {
          sum += layer.weights[offset + column] * current[column];
        }
        output[row] = activateScalar(sum, layer.activation);
      }
      if (layer.activation === "softmax") {
        output.splice(0, output.length, ...softmax(output));
      }
      activations.push(output);
      current = output;
    }
    return activations;
  }
}
