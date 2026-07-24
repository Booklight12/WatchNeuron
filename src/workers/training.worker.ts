/// <reference lib="webworker" />

import type {
  ActivationKind,
  ConvolutionConfig,
  ConvolutionLayerData,
  CustomDatasetSample,
  DenseLayerData,
  HiddenLayer,
  MathMode,
  OptimizerConfig,
  OptimizerKind,
  NeuralModel,
  TrainingSettings,
} from "../types";
import {
  convolutionOutputShape,
  convolutionPipeline,
  fitConvolutionsToLayers,
  modelConvolutions,
} from "../lib/convolution";
import {
  applyWasmMathMode,
  loadBestWasmKernel,
  type MathModeWasmExports,
  type WasmKernelFlavor,
} from "../lib/wasm";

interface SparseSample {
  label: number;
  indices: Uint16Array;
  values: Float32Array;
}

interface TrainMessage {
  type: "train";
  datasetUrl: string;
  mnistEnabled?: boolean;
  layers: HiddenLayer[];
  convolutions?: ConvolutionConfig[];
  /** Legacy message field accepted by older callers. */
  convolution?: ConvolutionConfig;
  settings: TrainingSettings;
  customSamples: CustomDatasetSample[];
  initialModel?: NeuralModel;
}

interface CancelMessage {
  type: "cancel";
}

interface PauseMessage {
  type: "pause";
}

interface ResumeMessage {
  type: "resume";
}

interface SnapshotMessage {
  type: "snapshot";
}

interface SnapshotProgress {
  epoch: number;
  epochs: number;
  sample: number;
  samples: number;
  accuracy: number;
  loss: number;
  elapsedMs: number;
}

interface SampleTraceResult {
  loss: number;
  activations: Float32Array[];
  gradients: Float32Array[];
  label: number;
  prediction: number;
}

interface TrainingWasmExports extends MathModeWasmExports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  forward_sparse: (...args: number[]) => void;
  forward_dense_block: (...args: number[]) => void;
  forward_dense_training: (...args: number[]) => void;
  train_sample: (...args: number[]) => number;
  train_dense_from_gradient: (...args: number[]) => void;
  conv2d_forward: (...args: number[]) => void;
  conv2d_train: (...args: number[]) => void;
  simd_enabled: () => number;
}

interface WasmPointers {
  inputSizes: number;
  outputSizes: number;
  activationKinds: number;
  weights: number;
  biases: number;
  activations: number;
  preactivations: number;
  deltas: number;
  weightFirst: number;
  biasFirst: number;
  weightSecond: number;
  biasSecond: number;
  sampleIndices: number;
  sampleValues: number;
  inputGradient: number;
  dropoutRates: number;
  dropoutMasks: number;
}

interface ConvolutionPointers {
  input: number;
  weights: number;
  biases: number;
  output: number;
  preactivation: number;
  delta: number;
  inputGradient: number;
  weightFirst: number;
  biasFirst: number;
  weightSecond: number;
  biasSecond: number;
}

interface WasmConvolutionRuntime {
  model: ConvolutionLayerData;
  pointers: ConvolutionPointers;
  input: Float32Array;
  output: Float32Array;
  preactivation: Float32Array;
  delta: Float32Array;
  inputGradient: Float32Array;
}

interface WasmTrainingRuntime {
  wasm: TrainingWasmExports;
  config: OptimizerConfig;
  pointers: WasmPointers;
  model: DenseLayerData[];
  activations: Float32Array[];
  preactivations: Float32Array[];
  deltas: Float32Array[];
  sampleIndices: Uint16Array;
  sampleValues: Float32Array;
  inputGradient: Float32Array;
  dropoutMasks: Float32Array[];
  convolutions: WasmConvolutionRuntime[];
  backend: "Zig/Wasm SIMD · 快速" | "Zig/Wasm SIMD · 完整" | "Zig/Wasm · 快速" | "Zig/Wasm · 完整";
  mathMode: MathMode;
  step: number;
  beta1Power: number;
  beta2Power: number;
}

let cancelled = false;
let paused = false;
let pausedAt: number | null = null;
let pausedDuration = 0;
let controlWaiters: Array<() => void> = [];
let activeRuntime: WasmTrainingRuntime | null = null;
let activeSnapshotProgress: SnapshotProgress | null = null;
let activeStartedAt: number | null = null;
let trainingWasmPromise: Promise<{
  exports: TrainingWasmExports;
  flavor: WasmKernelFlavor;
}> | null = null;

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

const optimizerCodes: Record<OptimizerKind, number> = {
  sgd: 0,
  momentum: 1,
  adam: 2,
  rmsprop: 3,
  adagrad: 4,
};

function wakeControlWaiters() {
  const waiters = controlWaiters;
  controlWaiters = [];
  for (const resolve of waiters) resolve();
}

function activeElapsed(startedAt: number) {
  const currentPause = pausedAt === null ? 0 : performance.now() - pausedAt;
  return performance.now() - startedAt - pausedDuration - currentPause;
}

async function waitForControl() {
  await new Promise((resolve) => setTimeout(resolve, 0));
  while (paused && !cancelled) {
    await new Promise<void>((resolve) => controlWaiters.push(resolve));
  }
  return !cancelled;
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function architectureSeed(layers: HiddenLayer[], convolutions: ConvolutionConfig[]) {
  let seed = 0x57a7c11;
  for (const layer of layers) {
    seed = Math.imul(seed ^ layer.units, 16777619);
    for (const character of layer.activation) {
      seed = Math.imul(seed ^ character.charCodeAt(0), 16777619);
    }
    seed = Math.imul(seed ^ Math.round(layer.dropout * 10000), 16777619);
  }
  for (const convolution of convolutions) {
    seed = Math.imul(seed ^ convolution.position, 16777619);
    seed = Math.imul(seed ^ convolution.filters, 16777619);
    seed = Math.imul(seed ^ convolution.kernelSize, 16777619);
    seed = Math.imul(seed ^ convolution.stride, 16777619);
    seed = Math.imul(seed ^ convolution.padding, 16777619);
  }
  return seed >>> 0;
}

function initializeConvolutions(
  configs: ConvolutionConfig[],
  hiddenLayers: HiddenLayer[],
): ConvolutionLayerData[] {
  return convolutionPipeline(hiddenLayers, configs).map(({ config, input, output }) => {
    const kernelLength = config.kernelSize * config.kernelSize;
    const weights = new Float32Array(config.filters * input.channels * kernelLength);
    for (let filter = 0; filter < config.filters; filter++) {
      const kernel = config.kernels[filter] ?? [];
      for (let channel = 0; channel < input.channels; channel++) {
        for (let index = 0; index < kernelLength; index++) {
          weights[(filter * input.channels + channel) * kernelLength + index] =
            Number.isFinite(kernel[index]) ? kernel[index] : 0;
        }
      }
    }
    return {
      id: config.id,
      position: config.position,
      inputWidth: input.width,
      inputHeight: input.height,
      inputChannels: input.channels,
      outputWidth: output.width,
      outputHeight: output.height,
      filters: config.filters,
      kernelSize: config.kernelSize,
      stride: config.stride,
      padding: config.padding,
      activation: config.activation,
      weights,
      biases: new Float32Array(config.filters),
    };
  });
}

function denseLayerLayout(
  hiddenLayers: HiddenLayer[],
  convolutions: ConvolutionLayerData[],
) {
  const layout: Array<{
    inputSize: number;
    outputSize: number;
    activation: ActivationKind | "linear";
  }> = [];
  let inputSize = 784;
  for (let index = 0; index < hiddenLayers.length; index++) {
    for (const convolution of convolutions.filter(({ position }) => position === index)) {
      inputSize = convolution.outputWidth * convolution.outputHeight * convolution.filters;
    }
    const layer = hiddenLayers[index];
    layout.push({ inputSize, outputSize: layer.units, activation: layer.activation });
    inputSize = layer.units;
  }
  for (const convolution of convolutions.filter(({ position }) => position === hiddenLayers.length)) {
    inputSize = convolution.outputWidth * convolution.outputHeight * convolution.filters;
  }
  layout.push({ inputSize, outputSize: 10, activation: "linear" });
  return layout;
}

function initializeModel(hiddenLayers: HiddenLayer[], convolutionConfigs: ConvolutionConfig[]): NeuralModel {
  const random = seededRandom(architectureSeed(hiddenLayers, convolutionConfigs));
  const convolutions = initializeConvolutions(convolutionConfigs, hiddenLayers);
  const layers = denseLayerLayout(hiddenLayers, convolutions).map((spec): DenseLayerData => {
    const { inputSize, outputSize, activation } = spec;
    const usesHeScale = [
      "relu",
      "leakyRelu",
      "elu",
      "relu6",
      "gelu",
      "swish",
      "mish",
    ].includes(activation);
    const scale = Math.sqrt((usesHeScale ? 2 : 1) / inputSize);
    return {
      inputSize,
      outputSize,
      activation,
      weights: Float32Array.from(
        { length: inputSize * outputSize },
        () => (random() * 2 - 1) * scale,
      ),
      biases: new Float32Array(outputSize),
    };
  });
  return { convolutions, layers, calibrated: false, trained: false };
}

function validateInitialModel(
  hiddenLayers: HiddenLayer[],
  convolutionConfigs: ConvolutionConfig[],
  initialModel: NeuralModel,
) {
  const expectedConvolutions = initializeConvolutions(convolutionConfigs, hiddenLayers);
  const convolutions = modelConvolutions(initialModel);
  if (expectedConvolutions.length !== convolutions.length) {
    throw new Error("载入模型的卷积结构与当前架构不匹配");
  }
  for (let index = 0; index < convolutions.length; index++) {
    const expectedConvolution = expectedConvolutions[index];
    const convolution = convolutions[index];
    if (
      convolution.position !== expectedConvolution.position ||
      convolution.inputWidth !== expectedConvolution.inputWidth ||
      convolution.inputHeight !== expectedConvolution.inputHeight ||
      convolution.inputChannels !== expectedConvolution.inputChannels ||
      convolution.outputWidth !== expectedConvolution.outputWidth ||
      convolution.outputHeight !== expectedConvolution.outputHeight ||
      convolution.filters !== expectedConvolution.filters ||
      convolution.kernelSize !== expectedConvolution.kernelSize ||
      convolution.stride !== expectedConvolution.stride ||
      convolution.padding !== expectedConvolution.padding ||
      convolution.activation !== expectedConvolution.activation ||
      !(convolution.weights instanceof Float32Array) ||
      !(convolution.biases instanceof Float32Array) ||
      convolution.weights.length !== expectedConvolution.weights.length ||
      convolution.biases.length !== expectedConvolution.biases.length
    ) {
      throw new Error(`载入模型的第 ${index + 1} 个卷积层与当前架构不匹配`);
    }
    for (const value of convolution.weights) {
      if (!Number.isFinite(value)) throw new Error("载入模型包含无效卷积权重");
    }
    for (const value of convolution.biases) {
      if (!Number.isFinite(value)) throw new Error("载入模型包含无效卷积偏置");
    }
  }
  const layout = denseLayerLayout(hiddenLayers, expectedConvolutions);
  if (initialModel.layers.length !== layout.length) {
    throw new Error("载入模型的层数与当前架构不匹配");
  }

  for (let index = 0; index < initialModel.layers.length; index++) {
    const layer = initialModel.layers[index];
    const { activation: expectedActivation, inputSize, outputSize } = layout[index];
    if (
      layer.inputSize !== inputSize ||
      layer.outputSize !== outputSize ||
      layer.activation !== expectedActivation ||
      !(layer.weights instanceof Float32Array) ||
      !(layer.biases instanceof Float32Array) ||
      layer.weights.length !== inputSize * outputSize ||
      layer.biases.length !== outputSize
    ) {
      throw new Error(`载入模型的第 ${index + 1} 层与当前架构不匹配`);
    }
    for (const value of layer.weights) {
      if (!Number.isFinite(value)) throw new Error("载入模型包含无效权重");
    }
    for (const value of layer.biases) {
      if (!Number.isFinite(value)) throw new Error("载入模型包含无效偏置");
    }
  }
  return initialModel;
}

function normalizeOptimizer(config: OptimizerConfig | undefined): OptimizerConfig {
  const kind: OptimizerKind = ["sgd", "momentum", "adam", "rmsprop", "adagrad"].includes(config?.kind ?? "")
    ? config!.kind
    : "sgd";
  const momentum = Number.isFinite(config?.momentum)
    ? Math.min(0.999999, Math.max(0, config!.momentum))
    : 0.9;
  const beta1 = Number.isFinite(config?.beta1)
    ? Math.min(0.999999, Math.max(0.000001, config!.beta1))
    : 0.9;
  const beta2 = Number.isFinite(config?.beta2)
    ? Math.min(0.999999, Math.max(0.000001, config!.beta2))
    : 0.999;
  const decay = Number.isFinite(config?.decay)
    ? Math.min(0.999999, Math.max(0.000001, config!.decay))
    : 0.9;
  const epsilon = Number.isFinite(config?.epsilon) && config!.epsilon > 0
    ? config!.epsilon
    : 1e-8;
  return { kind, momentum, beta1, beta2, decay, epsilon };
}

function align(value: number, alignment = 16) {
  return Math.ceil(value / alignment) * alignment;
}

async function loadTrainingWasm() {
  if (!trainingWasmPromise) {
    trainingWasmPromise = loadBestWasmKernel<TrainingWasmExports>((exports, flavor) =>
      exports.memory instanceof WebAssembly.Memory &&
      typeof exports.forward_sparse === "function" &&
      typeof exports.forward_dense_block === "function" &&
      typeof exports.forward_dense_training === "function" &&
      typeof exports.train_sample === "function" &&
      typeof exports.train_dense_from_gradient === "function" &&
      typeof exports.conv2d_forward === "function" &&
      typeof exports.conv2d_train === "function" &&
      typeof exports.set_math_mode === "function" &&
      typeof exports.math_mode === "function" &&
      typeof exports.simd_enabled === "function" &&
      exports.simd_enabled() === (flavor === "simd" ? 1 : 0),
    );
  }
  return trainingWasmPromise;
}

function createWasmTrainingRuntime(
  wasm: TrainingWasmExports,
  sourceModel: NeuralModel,
  hiddenLayers: HiddenLayer[],
  optimizerConfig: OptimizerConfig | undefined,
  mathMode: MathMode,
  flavor: WasmKernelFlavor,
): WasmTrainingRuntime {
  applyWasmMathMode(wasm, mathMode);
  const sourceLayers = sourceModel.layers;
  const layerCount = sourceLayers.length;
  let cursor = align(Number(wasm.__heap_base.value));
  const allocate = (bytes: number, alignment = 16) => {
    cursor = align(cursor, alignment);
    const pointer = cursor;
    cursor += bytes;
    return pointer;
  };
  const allocateTable = () => allocate(layerCount * 4, 4);

  const pointers: WasmPointers = {
    inputSizes: allocateTable(),
    outputSizes: allocateTable(),
    activationKinds: allocateTable(),
    weights: allocateTable(),
    biases: allocateTable(),
    activations: allocateTable(),
    preactivations: allocateTable(),
    deltas: allocateTable(),
    weightFirst: allocateTable(),
    biasFirst: allocateTable(),
    weightSecond: allocateTable(),
    biasSecond: allocateTable(),
    dropoutRates: allocateTable(),
    dropoutMasks: allocateTable(),
    sampleIndices: 0,
    sampleValues: 0,
    inputGradient: 0,
  };

  const weightPointers: number[] = [];
  const biasPointers: number[] = [];
  const activationPointers: number[] = [];
  const preactivationPointers: number[] = [];
  const deltaPointers: number[] = [];
  const weightFirstPointers: number[] = [];
  const biasFirstPointers: number[] = [];
  const weightSecondPointers: number[] = [];
  const biasSecondPointers: number[] = [];
  const dropoutMaskPointers: number[] = [];

  for (const layer of sourceLayers) {
    weightPointers.push(allocate(layer.weights.byteLength));
    biasPointers.push(allocate(layer.biases.byteLength));
    activationPointers.push(allocate(layer.outputSize * 4));
    preactivationPointers.push(allocate(layer.outputSize * 4));
    deltaPointers.push(allocate(layer.outputSize * 4));
    weightFirstPointers.push(allocate(layer.weights.byteLength));
    biasFirstPointers.push(allocate(layer.biases.byteLength));
    weightSecondPointers.push(allocate(layer.weights.byteLength));
    biasSecondPointers.push(allocate(layer.biases.byteLength));
    dropoutMaskPointers.push(allocate(layer.outputSize * 4));
  }

  const sourceConvolutions = modelConvolutions(sourceModel);
  const convolutionPointers = sourceConvolutions.map((sourceConvolution): ConvolutionPointers => {
    const convolutionOutputSize = sourceConvolution.outputWidth * sourceConvolution.outputHeight * sourceConvolution.filters;
    const convolutionInputSize = sourceConvolution.inputWidth * sourceConvolution.inputHeight * sourceConvolution.inputChannels;
    return {
      input: allocate(convolutionInputSize * 4),
      weights: allocate(sourceConvolution.weights.byteLength),
      biases: allocate(sourceConvolution.biases.byteLength),
      output: allocate(convolutionOutputSize * 4),
      preactivation: allocate(convolutionOutputSize * 4),
      delta: allocate(convolutionOutputSize * 4),
      inputGradient: allocate(convolutionInputSize * 4),
      weightFirst: allocate(sourceConvolution.weights.byteLength),
      biasFirst: allocate(sourceConvolution.biases.byteLength),
      weightSecond: allocate(sourceConvolution.weights.byteLength),
      biasSecond: allocate(sourceConvolution.biases.byteLength),
    };
  });

  const sampleCapacity = Math.max(
    784,
    ...sourceConvolutions.flatMap((convolution) => [
      convolution.inputWidth * convolution.inputHeight * convolution.inputChannels,
      convolution.outputWidth * convolution.outputHeight * convolution.filters,
    ]),
  );
  pointers.sampleIndices = allocate(sampleCapacity * 2, 2);
  pointers.sampleValues = allocate(sampleCapacity * 4);
  pointers.inputGradient = allocate(sampleCapacity * 4);

  if (cursor > wasm.memory.buffer.byteLength) {
    wasm.memory.grow(Math.ceil((cursor - wasm.memory.buffer.byteLength) / 65536));
  }
  const buffer = wasm.memory.buffer;
  new Int32Array(buffer, pointers.inputSizes, layerCount).set(
    sourceLayers.map((layer) => layer.inputSize),
  );
  new Int32Array(buffer, pointers.outputSizes, layerCount).set(
    sourceLayers.map((layer) => layer.outputSize),
  );
  new Int32Array(buffer, pointers.activationKinds, layerCount).set(
    sourceLayers.map((layer) => activationCodes[layer.activation]),
  );
  const setPointerTable = (pointer: number, values: number[]) => {
    new Uint32Array(buffer, pointer, layerCount).set(values);
  };
  setPointerTable(pointers.weights, weightPointers);
  setPointerTable(pointers.biases, biasPointers);
  setPointerTable(pointers.activations, activationPointers);
  setPointerTable(pointers.preactivations, preactivationPointers);
  setPointerTable(pointers.deltas, deltaPointers);
  setPointerTable(pointers.weightFirst, weightFirstPointers);
  setPointerTable(pointers.biasFirst, biasFirstPointers);
  setPointerTable(pointers.weightSecond, weightSecondPointers);
  setPointerTable(pointers.biasSecond, biasSecondPointers);
  setPointerTable(pointers.dropoutMasks, dropoutMaskPointers);
  new Float32Array(buffer, pointers.dropoutRates, layerCount).set(
    sourceLayers.map((_, index) => index < hiddenLayers.length
      ? Math.min(0.95, Math.max(0, hiddenLayers[index].dropout ?? 0))
      : 0),
  );

  const model = sourceLayers.map((layer, index): DenseLayerData => {
    const weights = new Float32Array(buffer, weightPointers[index], layer.weights.length);
    const biases = new Float32Array(buffer, biasPointers[index], layer.biases.length);
    weights.set(layer.weights);
    biases.set(layer.biases);
    new Float32Array(buffer, weightFirstPointers[index], layer.weights.length).fill(0);
    new Float32Array(buffer, biasFirstPointers[index], layer.biases.length).fill(0);
    new Float32Array(buffer, weightSecondPointers[index], layer.weights.length).fill(0);
    new Float32Array(buffer, biasSecondPointers[index], layer.biases.length).fill(0);
    new Float32Array(buffer, dropoutMaskPointers[index], layer.outputSize).fill(1);
    return { ...layer, weights, biases };
  });

  const convolutions = sourceConvolutions.map((sourceConvolution, index): WasmConvolutionRuntime => {
    const pointersForConvolution = convolutionPointers[index];
    const outputSize = sourceConvolution.outputWidth * sourceConvolution.outputHeight * sourceConvolution.filters;
    const inputSize = sourceConvolution.inputWidth * sourceConvolution.inputHeight * sourceConvolution.inputChannels;
    const weights = new Float32Array(buffer, pointersForConvolution.weights, sourceConvolution.weights.length);
    const biases = new Float32Array(buffer, pointersForConvolution.biases, sourceConvolution.biases.length);
    weights.set(sourceConvolution.weights);
    biases.set(sourceConvolution.biases);
    new Float32Array(buffer, pointersForConvolution.weightFirst, sourceConvolution.weights.length).fill(0);
    new Float32Array(buffer, pointersForConvolution.biasFirst, sourceConvolution.biases.length).fill(0);
    new Float32Array(buffer, pointersForConvolution.weightSecond, sourceConvolution.weights.length).fill(0);
    new Float32Array(buffer, pointersForConvolution.biasSecond, sourceConvolution.biases.length).fill(0);
    return {
      pointers: pointersForConvolution,
      model: { ...sourceConvolution, weights, biases },
      input: new Float32Array(buffer, pointersForConvolution.input, inputSize),
      output: new Float32Array(buffer, pointersForConvolution.output, outputSize),
      preactivation: new Float32Array(buffer, pointersForConvolution.preactivation, outputSize),
      delta: new Float32Array(buffer, pointersForConvolution.delta, outputSize),
      inputGradient: new Float32Array(buffer, pointersForConvolution.inputGradient, inputSize),
    };
  });

  const sampleIndices = new Uint16Array(buffer, pointers.sampleIndices, sampleCapacity);

  return {
    wasm,
    config: normalizeOptimizer(optimizerConfig),
    pointers,
    model,
    activations: sourceLayers.map(
      (layer, index) => new Float32Array(buffer, activationPointers[index], layer.outputSize),
    ),
    preactivations: sourceLayers.map(
      (layer, index) => new Float32Array(buffer, preactivationPointers[index], layer.outputSize),
    ),
    deltas: sourceLayers.map(
      (layer, index) => new Float32Array(buffer, deltaPointers[index], layer.outputSize),
    ),
    dropoutMasks: sourceLayers.map(
      (layer, index) => new Float32Array(buffer, dropoutMaskPointers[index], layer.outputSize),
    ),
    sampleIndices,
    sampleValues: new Float32Array(buffer, pointers.sampleValues, sampleCapacity),
    inputGradient: new Float32Array(buffer, pointers.inputGradient, sampleCapacity),
    convolutions,
    backend: `${flavor === "simd" ? "Zig/Wasm SIMD" : "Zig/Wasm"} · ${mathMode === "full" ? "完整" : "快速"}`,
    mathMode,
    step: 0,
    beta1Power: 1,
    beta2Power: 1,
  };
}

function parseDataset(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== "WNDS" || view.getUint16(4, true) !== 1) {
    throw new Error("训练数据格式无效");
  }
  const count = view.getUint32(6, true);
  const training: SparseSample[] = [];
  const validation: SparseSample[] = [];
  let offset = 12;

  for (let sampleIndex = 0; sampleIndex < count; sampleIndex++) {
    const split = view.getUint8(offset);
    const label = view.getUint8(offset + 1);
    const activeCount = view.getUint16(offset + 2, true);
    offset += 4;
    const indices = new Uint16Array(activeCount);
    const values = new Float32Array(activeCount);
    for (let pixel = 0; pixel < activeCount; pixel++) {
      indices[pixel] = view.getUint16(offset, true);
      values[pixel] = view.getUint8(offset + 2) / 255;
      offset += 3;
    }
    const sample = { label, indices, values };
    if (split === 0) training.push(sample);
    else validation.push(sample);
  }
  return { training, validation };
}

function appendCustomSamples(
  customSamples: CustomDatasetSample[],
  training: SparseSample[],
  validation: SparseSample[],
) {
  for (const sample of customSamples) {
    if (
      !Number.isInteger(sample.label) ||
      sample.label < 0 ||
      sample.label > 9 ||
      !Array.isArray(sample.indices) ||
      !Array.isArray(sample.values) ||
      sample.indices.length !== sample.values.length ||
      (sample.split !== "training" && sample.split !== "test") ||
      !sample.indices.every(
        (index, position) =>
          Number.isInteger(index) &&
          index >= 0 &&
          index < 784 &&
          Number.isFinite(sample.values[position]) &&
          sample.values[position] >= 0 &&
          sample.values[position] <= 1,
      )
    ) {
      continue;
    }
    const indices = Uint16Array.from(sample.indices);
    const values = Float32Array.from(sample.values);
    const target = sample.split === "training" ? training : validation;
    target.push({ label: sample.label, indices, values });
  }
}

function stageSample(runtime: WasmTrainingRuntime, sample: SparseSample) {
  runtime.sampleIndices.set(sample.indices);
  runtime.sampleValues.set(sample.values);
}

function stageConvolutionInput(
  convolution: WasmConvolutionRuntime,
  sample: SparseSample,
  denseInput: Float32Array | null,
) {
  convolution.input.fill(0);
  if (denseInput) {
    convolution.input.set(denseInput);
    return;
  }
  for (let index = 0; index < sample.indices.length; index++) {
    convolution.input[sample.indices[index]] = sample.values[index];
  }
}

function forwardConvolution(runtime: WasmTrainingRuntime, convolution: WasmConvolutionRuntime) {
  const { model, pointers } = convolution;
  runtime.wasm.conv2d_forward(
    pointers.input,
    pointers.weights,
    pointers.biases,
    pointers.output,
    pointers.preactivation,
    model.inputWidth,
    model.inputHeight,
    model.inputChannels,
    model.filters,
    model.kernelSize,
    model.stride,
    model.padding,
    activationCodes[model.activation],
  );
}

function tableAt(pointer: number, layerIndex: number) {
  return pointer + layerIndex * 4;
}

function pointerAt(runtime: WasmTrainingRuntime, table: number, layerIndex: number) {
  return new Uint32Array(runtime.wasm.memory.buffer, tableAt(table, layerIndex), 1)[0];
}

function forwardDenseSegment(
  runtime: WasmTrainingRuntime,
  sample: SparseSample,
  start: number,
  count: number,
  inputValuesPointer: number,
  activeCount: number,
  inputIsDense: boolean,
  training: boolean,
) {
  if (count <= 0) return;
  const { pointers } = runtime;
  if (training) {
    runtime.wasm.forward_dense_training(
      count,
      pointers.sampleIndices,
      inputValuesPointer,
      activeCount,
      tableAt(pointers.inputSizes, start),
      tableAt(pointers.outputSizes, start),
      tableAt(pointers.activationKinds, start),
      tableAt(pointers.weights, start),
      tableAt(pointers.biases, start),
      tableAt(pointers.activations, start),
      tableAt(pointers.preactivations, start),
      tableAt(pointers.dropoutRates, start),
      tableAt(pointers.dropoutMasks, start),
      runtime.step >>> 0,
      inputIsDense ? 1 : 0,
    );
    return;
  }
  runtime.wasm.forward_dense_block(
    count,
    pointers.sampleIndices,
    inputValuesPointer,
    activeCount,
    tableAt(pointers.inputSizes, start),
    tableAt(pointers.outputSizes, start),
    tableAt(pointers.activationKinds, start),
    tableAt(pointers.weights, start),
    tableAt(pointers.biases, start),
    tableAt(pointers.activations, start),
    tableAt(pointers.preactivations, start),
    inputIsDense ? 1 : 0,
  );
}

function forwardConvolutionPrefix(
  runtime: WasmTrainingRuntime,
  sample: SparseSample,
  training: boolean,
) {
  let denseStart = 0;
  let inputPointer = runtime.pointers.sampleValues;
  let inputCount = sample.indices.length;
  let inputIsDense = false;
  let inputValues: Float32Array | null = null;

  for (const convolution of runtime.convolutions) {
    const denseCount = convolution.model.position - denseStart;
    if (denseCount > 0) {
      forwardDenseSegment(
        runtime,
        sample,
        denseStart,
        denseCount,
        inputPointer,
        inputCount,
        inputIsDense,
        training,
      );
      const finalDenseIndex = convolution.model.position - 1;
      inputPointer = pointerAt(runtime, runtime.pointers.activations, finalDenseIndex);
      inputCount = runtime.activations[finalDenseIndex].length;
      inputIsDense = true;
      inputValues = runtime.activations[finalDenseIndex];
    }
    stageConvolutionInput(convolution, sample, inputValues);
    forwardConvolution(runtime, convolution);
    denseStart = convolution.model.position;
    inputPointer = convolution.pointers.output;
    inputCount = convolution.output.length;
    inputIsDense = true;
    inputValues = convolution.output;
  }

  return { denseStart, inputPointer, inputCount, inputIsDense };
}

function forwardWithWasm(runtime: WasmTrainingRuntime, sample: SparseSample) {
  stageSample(runtime, sample);
  const suffix = forwardConvolutionPrefix(runtime, sample, false);
  forwardDenseSegment(
    runtime,
    sample,
    suffix.denseStart,
    runtime.model.length - suffix.denseStart,
    suffix.inputPointer,
    suffix.inputCount,
    suffix.inputIsDense,
    false,
  );
}

function denseInput(sample: SparseSample) {
  const input = new Float32Array(784);
  for (let index = 0; index < sample.indices.length; index++) {
    input[sample.indices[index]] = sample.values[index];
  }
  return input;
}

function trainSampleWithWasm(
  runtime: WasmTrainingRuntime,
  sample: SparseSample,
  learningRate: number,
  captureTrace = false,
) {
  stageSample(runtime, sample);
  runtime.step++;
  runtime.beta1Power *= runtime.config.beta1;
  runtime.beta2Power *= runtime.config.beta2;
  const { pointers, config } = runtime;
  const suffix = forwardConvolutionPrefix(runtime, sample, true);
  const denseStart = suffix.denseStart;
  const denseCount = runtime.model.length - denseStart;
  const loss = runtime.wasm.train_sample(
    denseCount,
    pointers.sampleIndices,
    suffix.inputPointer,
    suffix.inputCount,
    sample.label,
    tableAt(pointers.inputSizes, denseStart),
    tableAt(pointers.outputSizes, denseStart),
    tableAt(pointers.activationKinds, denseStart),
    tableAt(pointers.weights, denseStart),
    tableAt(pointers.biases, denseStart),
    tableAt(pointers.activations, denseStart),
    tableAt(pointers.preactivations, denseStart),
    tableAt(pointers.deltas, denseStart),
    tableAt(pointers.weightFirst, denseStart),
    tableAt(pointers.biasFirst, denseStart),
    tableAt(pointers.weightSecond, denseStart),
    tableAt(pointers.biasSecond, denseStart),
    optimizerCodes[config.kind],
    learningRate,
    config.momentum,
    config.decay,
    config.beta1,
    config.beta2,
    config.epsilon,
    1 - runtime.beta1Power,
    1 - runtime.beta2Power,
    captureTrace || runtime.convolutions.length > 0 ? 1 : 0,
    pointers.inputGradient,
    tableAt(pointers.dropoutRates, denseStart),
    tableAt(pointers.dropoutMasks, denseStart),
    runtime.step >>> 0,
    suffix.inputIsDense ? 1 : 0,
  );
  if (!Number.isFinite(loss)) throw new Error("Zig/Wasm 训练产生了无效损失");
  let downstreamGradientPointer = pointers.inputGradient;
  for (let index = runtime.convolutions.length - 1; index >= 0; index--) {
    const convolution = runtime.convolutions[index];
    const model = convolution.model;
    const convPointers = convolution.pointers;
    runtime.wasm.conv2d_train(
      convPointers.input,
      convPointers.weights,
      convPointers.biases,
      convPointers.preactivation,
      downstreamGradientPointer,
      convPointers.inputGradient,
      convPointers.delta,
      convPointers.weightFirst,
      convPointers.biasFirst,
      convPointers.weightSecond,
      convPointers.biasSecond,
      model.inputWidth,
      model.inputHeight,
      model.inputChannels,
      model.filters,
      model.kernelSize,
      model.stride,
      model.padding,
      activationCodes[model.activation],
      optimizerCodes[config.kind],
      learningRate,
      config.momentum,
      config.decay,
      config.beta1,
      config.beta2,
      config.epsilon,
      1 - runtime.beta1Power,
      1 - runtime.beta2Power,
    );
    const previousConvolution = runtime.convolutions[index - 1];
    const segmentStart = previousConvolution?.model.position ?? 0;
    const segmentCount = model.position - segmentStart;
    if (segmentCount > 0) {
      runtime.wasm.train_dense_from_gradient(
        segmentCount,
        pointers.sampleIndices,
        previousConvolution?.pointers.output ?? pointers.sampleValues,
        previousConvolution?.output.length ?? sample.indices.length,
        convPointers.inputGradient,
        tableAt(pointers.inputSizes, segmentStart),
        tableAt(pointers.outputSizes, segmentStart),
        tableAt(pointers.activationKinds, segmentStart),
        tableAt(pointers.weights, segmentStart),
        tableAt(pointers.biases, segmentStart),
        tableAt(pointers.activations, segmentStart),
        tableAt(pointers.preactivations, segmentStart),
        tableAt(pointers.deltas, segmentStart),
        tableAt(pointers.weightFirst, segmentStart),
        tableAt(pointers.biasFirst, segmentStart),
        tableAt(pointers.weightSecond, segmentStart),
        tableAt(pointers.biasSecond, segmentStart),
        optimizerCodes[config.kind],
        learningRate,
        config.momentum,
        config.decay,
        config.beta1,
        config.beta2,
        config.epsilon,
        1 - runtime.beta1Power,
        1 - runtime.beta2Power,
        captureTrace ? 1 : 0,
        pointers.inputGradient,
        tableAt(pointers.dropoutMasks, segmentStart),
        previousConvolution ? 1 : 0,
      );
      downstreamGradientPointer = pointers.inputGradient;
    } else {
      downstreamGradientPointer = convPointers.inputGradient;
    }
  }
  if (!captureTrace) return loss;

  const probabilities = runtime.activations.at(-1)!;
  let prediction = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index] > probabilities[prediction]) prediction = index;
  }
  const traceActivations: Float32Array[] = [denseInput(sample)];
  const firstConvolution = runtime.convolutions[0];
  const inputGradient = firstConvolution
    ? firstConvolution.model.position > 0
      ? Float32Array.from(runtime.inputGradient.subarray(0, 784))
      : Float32Array.from(firstConvolution.inputGradient)
    : Float32Array.from(runtime.inputGradient.subarray(0, 784));
  const traceGradients: Float32Array[] = [inputGradient];
  for (let layerIndex = 0; layerIndex < runtime.model.length; layerIndex++) {
    for (const convolution of runtime.convolutions.filter(({ model }) => model.position === layerIndex)) {
      traceActivations.push(Float32Array.from(convolution.output));
      traceGradients.push(Float32Array.from(convolution.delta));
    }
    traceActivations.push(Float32Array.from(runtime.activations[layerIndex]));
    traceGradients.push(Float32Array.from(runtime.deltas[layerIndex]));
  }
  return {
    loss,
    activations: traceActivations,
    gradients: traceGradients,
    label: sample.label,
    prediction,
  } satisfies SampleTraceResult;
}

function validateWithWasm(runtime: WasmTrainingRuntime, samples: SparseSample[]) {
  if (samples.length === 0) return 0;
  let correct = 0;
  for (const sample of samples) {
    forwardWithWasm(runtime, sample);
    const probabilities = runtime.activations.at(-1)!;
    let prediction = 0;
    for (let digit = 1; digit < 10; digit++) {
      if (probabilities[digit] > probabilities[prediction]) prediction = digit;
    }
    if (prediction === sample.label) correct++;
  }
  return correct / samples.length;
}

function cloneWasmModel(runtime: WasmTrainingRuntime): NeuralModel {
  return {
    convolutions: runtime.convolutions.map((convolution) => ({
      ...convolution.model,
      weights: Float32Array.from(convolution.model.weights),
      biases: Float32Array.from(convolution.model.biases),
    })),
    layers: runtime.model.map((layer): DenseLayerData => ({
      ...layer,
      weights: Float32Array.from(layer.weights),
      biases: Float32Array.from(layer.biases),
    })),
    calibrated: false,
    trained: true,
  };
}

async function train(message: TrainMessage) {
  cancelled = false;
  paused = false;
  pausedAt = null;
  pausedDuration = 0;
  activeRuntime = null;
  activeSnapshotProgress = null;
  wakeControlWaiters();
  const startedAt = performance.now();
  activeStartedAt = startedAt;
  const legacyConvolutions = message.convolution?.enabled ? [message.convolution] : [];
  const convolutionConfigs = fitConvolutionsToLayers(
    message.convolutions ?? legacyConvolutions,
    message.layers,
  );
  const epochs = Number.isFinite(message.settings.epochs)
    ? Math.max(1, Math.floor(message.settings.epochs))
    : 1;
  postMessage({ type: "progress", phase: "loading", epoch: 0, backend: "Zig/Wasm" });
  const datasetPromise = message.mnistEnabled === false
    ? Promise.resolve<ArrayBuffer | null>(null)
    : fetch(message.datasetUrl).then((response) => {
        if (!response.ok) throw new Error("无法载入训练数据");
        return response.arrayBuffer();
      });
  const [datasetBuffer, loadedWasm] = await Promise.all([
    datasetPromise,
    loadTrainingWasm(),
  ]);
  const { training, validation } = datasetBuffer
    ? parseDataset(datasetBuffer)
    : { training: [] as SparseSample[], validation: [] as SparseSample[] };
  appendCustomSamples(message.customSamples ?? [], training, validation);
  if (training.length === 0) {
    throw new Error("训练集为空，请启用 MNIST 或加入自定义训练样本");
  }
  const sourceModel = message.initialModel
    ? validateInitialModel(message.layers, convolutionConfigs, message.initialModel)
    : initializeModel(message.layers, convolutionConfigs);
  const runtime = createWasmTrainingRuntime(
    loadedWasm.exports,
    sourceModel,
    message.layers,
    message.settings.optimizer,
    message.settings.mathMode === "full" ? "full" : "fast",
    loadedWasm.flavor,
  );
  activeRuntime = runtime;
  if (!Number.isFinite(message.settings.learningRate) || message.settings.learningRate <= 0) {
    throw new Error("学习率必须是大于 0 的有限数值");
  }
  const random = seededRandom(architectureSeed(message.layers, convolutionConfigs) ^ 0x9e3779b9);
  let lastAccuracy = 0;
  activeSnapshotProgress = {
    epoch: 0,
    epochs,
    sample: 0,
    samples: training.length,
    accuracy: 0,
    loss: 0,
    elapsedMs: 0,
  };
  let lastTraceAt = -Infinity;
  let lastControlYieldAt = performance.now();
  const traceInterval = Math.max(1, Math.floor(training.length / 6));

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let cursor = training.length - 1; cursor > 0; cursor--) {
      const swap = Math.floor(random() * (cursor + 1));
      [training[cursor], training[swap]] = [training[swap], training[cursor]];
    }
    const learningRate =
      message.settings.learningRate * Math.pow(0.94, epoch);
    let loss = 0;
    for (let sampleIndex = 0; sampleIndex < training.length; sampleIndex++) {
      const sample = training[sampleIndex];
      const now = performance.now();
      const captureTrace =
        sampleIndex === 0 ||
        (sampleIndex % traceInterval === 0 && now - lastTraceAt >= 90);
      const result = trainSampleWithWasm(runtime, sample, learningRate, captureTrace);
      if (typeof result === "number") {
        loss += result;
      } else {
        loss += result.loss;
        lastTraceAt = performance.now();
        postMessage({
          type: "trace",
          epoch: epoch + 1,
          sample: sampleIndex + 1,
          samples: training.length,
          activations: result.activations,
          gradients: result.gradients,
          label: result.label,
          prediction: result.prediction,
          loss: result.loss,
          backend: runtime.backend,
        });
      }

      activeSnapshotProgress.epoch = epoch + 1;
      activeSnapshotProgress.sample = sampleIndex + 1;
      activeSnapshotProgress.accuracy = lastAccuracy;
      activeSnapshotProgress.loss = loss / (sampleIndex + 1);

      const shouldYield = captureTrace || performance.now() - lastControlYieldAt >= 32;
      if (shouldYield) {
        if (!(await waitForControl())) {
          activeRuntime = null;
          activeSnapshotProgress = null;
          activeStartedAt = null;
          postMessage({ type: "cancelled" });
          return;
        }
        lastControlYieldAt = performance.now();
      }
      if (cancelled) {
        activeRuntime = null;
        activeSnapshotProgress = null;
        activeStartedAt = null;
        postMessage({ type: "cancelled" });
        return;
      }
    }
    const accuracy = validateWithWasm(runtime, validation);
    lastAccuracy = accuracy;
    if (activeSnapshotProgress) {
      activeSnapshotProgress.accuracy = accuracy;
      activeSnapshotProgress.loss = loss / training.length;
    }
    postMessage({
      type: "progress",
      phase: "training",
      epoch: epoch + 1,
      epochs,
      accuracy,
      loss: loss / training.length,
      elapsedMs: activeElapsed(startedAt),
      backend: runtime.backend,
    });
    if (!(await waitForControl())) {
      activeRuntime = null;
      activeSnapshotProgress = null;
      activeStartedAt = null;
      postMessage({ type: "cancelled" });
      return;
    }
  }

  const accuracy = validateWithWasm(runtime, validation);
  const completedModel = cloneWasmModel(runtime);
  const transfer = completedModel.layers.flatMap((layer) => [
    layer.weights.buffer,
    layer.biases.buffer,
  ]);
  for (const convolution of completedModel.convolutions) {
    transfer.push(convolution.weights.buffer, convolution.biases.buffer);
  }
  postMessage(
    {
      type: "complete",
      model: completedModel,
      accuracy,
      elapsedMs: activeElapsed(startedAt),
      trainingSamples: training.length,
      testSamples: validation.length,
      backend: runtime.backend,
    },
    transfer,
  );
  activeRuntime = null;
  activeSnapshotProgress = null;
  activeStartedAt = null;
}

self.onmessage = (event: MessageEvent<TrainMessage | CancelMessage | PauseMessage | ResumeMessage | SnapshotMessage>) => {
  if (event.data.type === "cancel") {
    cancelled = true;
    wakeControlWaiters();
    return;
  }
  if (event.data.type === "pause") {
    if (!paused && !cancelled) {
      paused = true;
      pausedAt = performance.now();
      postMessage({ type: "paused" });
    }
    return;
  }
  if (event.data.type === "resume") {
    if (paused && !cancelled) {
      paused = false;
      if (pausedAt !== null) pausedDuration += performance.now() - pausedAt;
      pausedAt = null;
      wakeControlWaiters();
      postMessage({ type: "resumed" });
    }
    return;
  }
  if (event.data.type === "snapshot") {
    if (paused && activeRuntime && activeSnapshotProgress && activeStartedAt !== null) {
      postMessage({
        type: "snapshot",
        model: cloneWasmModel(activeRuntime),
        ...activeSnapshotProgress,
        elapsedMs: activeElapsed(activeStartedAt),
      });
    }
    return;
  }
  train(event.data).catch((error: unknown) => {
    activeRuntime = null;
    activeSnapshotProgress = null;
    activeStartedAt = null;
    postMessage({
      type: "error",
      message: error instanceof Error ? error.stack ?? error.message : "训练失败",
    });
  });
};

export {};
