import type {
  ActivationKind,
  InferenceResult,
  NeuralModel,
} from "../types";
import { activateScalar } from "./activations";

interface WasmExports {
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
  private prepared = new WeakMap<NeuralModel, PreparedModel>();

  get backend(): "Wasm" | "JavaScript" {
    return this.wasm ? "Wasm" : "JavaScript";
  }

  async initialize() {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}neuron_kernel.wasm`);
      const bytes = await response.arrayBuffer();
      const instance = await WebAssembly.instantiate(bytes, {});
      this.wasm = instance.instance.exports as unknown as WasmExports;
    } catch {
      this.wasm = null;
    }
  }

  run(model: NeuralModel, input: Float32Array): InferenceResult {
    const startedAt = performance.now();
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

    for (const layer of model.layers) {
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

    const prepared = { inputPtr, layers };
    this.prepared.set(model, prepared);
    return prepared;
  }

  private runWasm(model: NeuralModel, input: Float32Array) {
    if (!this.wasm) return this.runJavaScript(model, input);
    const prepared = this.prepare(model);
    const memory = new Float32Array(this.wasm.memory.buffer);
    memory.set(input, prepared.inputPtr / 4);
    const activations: number[][] = [];

    for (const layer of prepared.layers) {
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
    for (const layer of model.layers) {
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
