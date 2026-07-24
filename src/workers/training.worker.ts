/// <reference lib="webworker" />

import type {
  ActivationKind,
  ComputeBackend,
  ConvolutionConfig,
  ConvolutionLayerData,
  CustomDatasetSample,
  DenseLayerData,
  HiddenLayer,
  MathMode,
  OptimizerConfig,
  OptimizerKind,
  NeuralModel,
  OutputHeadKind,
  PoolingConfig,
  PoolingLayerData,
  TrainingSettings,
} from "../types";
import {
  WebGpuBatchExecutor,
  type WebGpuLayerDescriptor,
  type WebGpuTrainingGraph,
} from "../lib/webgpu-batch";
import {
  convolutionOutputShape,
  modelPoolings,
  modelSpatialLayers,
  spatialPipeline,
  fitConvolutionsToLayers,
  fitPoolingsToLayers,
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
  poolings?: PoolingConfig[];
  outputHead?: OutputHeadKind;
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
  convolutionWeights: Float32Array[];
  convolutionBiases: Float32Array[];
  label: number;
  prediction: number;
}

interface FullBatchTraceResult extends SampleTraceResult {
  batchLoss: number;
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
  conv2d_forward_batch: (...args: number[]) => void;
  conv2d_train_batch: (...args: number[]) => void;
  pool2d_forward: (...args: number[]) => void;
  pool2d_backward: (...args: number[]) => void;
  pool2d_forward_batch: (...args: number[]) => void;
  pool2d_backward_batch: (...args: number[]) => void;
  dense_forward_batch: (...args: number[]) => void;
  dense_backward_batch: (...args: number[]) => void;
  output_loss_batch: (...args: number[]) => void;
  apply_optimizer: (...args: number[]) => void;
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
  weightGradients: number;
  biasGradients: number;
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
  weightGradient: number;
  biasGradient: number;
}

interface PoolingPointers {
  input: number;
  output: number;
  indices: number;
  inputGradient: number;
  delta: number;
}

interface WasmPoolingRuntime {
  type: "pool";
  model: PoolingLayerData;
  pointers: PoolingPointers;
  input: Float32Array;
  output: Float32Array;
  inputGradient: Float32Array;
  delta: Float32Array;
}

interface WasmConvolutionRuntime {
  type: "conv";
  model: ConvolutionLayerData;
  pointers: ConvolutionPointers;
  input: Float32Array;
  output: Float32Array;
  preactivation: Float32Array;
  delta: Float32Array;
  inputGradient: Float32Array;
}

type WasmSpatialRuntime = WasmConvolutionRuntime | WasmPoolingRuntime;

interface WasmBatchLayerBase {
  inputPointer: number;
  outputPointer: number;
  preactivationPointer: number;
  deltaPointer: number;
  inputGradientPointer: number;
  dropoutMaskPointer: number;
  inputSize: number;
  outputSize: number;
  input: Float32Array;
  output: Float32Array;
  preactivation: Float32Array;
  delta: Float32Array;
  inputGradient: Float32Array;
  dropoutMask: Float32Array;
}

interface WasmBatchDenseRuntime extends WasmBatchLayerBase {
  type: "dense";
  layerIndex: number;
}

interface WasmBatchConvolutionRuntime extends WasmBatchLayerBase {
  type: "conv";
  convolutionIndex: number;
}

interface WasmBatchPoolingRuntime extends WasmBatchLayerBase {
  type: "pool";
  poolingIndex: number;
  indicesPointer: number;
  indices: Uint32Array;
}

type WasmBatchLayerRuntime =
  | WasmBatchDenseRuntime
  | WasmBatchConvolutionRuntime
  | WasmBatchPoolingRuntime;

interface WasmBatchLayerAllocation {
  type: "dense" | "conv" | "pool";
  layerIndex?: number;
  convolutionIndex?: number;
  poolingIndex?: number;
  indicesPointer: number;
  inputPointer: number;
  outputPointer: number;
  preactivationPointer: number;
  deltaPointer: number;
  inputGradientPointer: number;
  dropoutMaskPointer: number;
  inputSize: number;
  outputSize: number;
}

interface WasmBatchRuntime {
  capacity: number;
  inputPointer: number;
  labelsPointer: number;
  lossesPointer: number;
  input: Float32Array;
  labels: Int32Array;
  losses: Float32Array;
  layers: WasmBatchLayerRuntime[];
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
  poolings: WasmPoolingRuntime[];
  spatialLayers: WasmSpatialRuntime[];
  batch: WasmBatchRuntime;
  outputHead: OutputHeadKind;
  backend: string;
  mathMode: MathMode;
  computeBackend: ComputeBackend;
  webgpu: WebGpuBatchExecutor | null;
  webgpuGraph: WebGpuTrainingGraph | null;
  webgpuDescriptors: WebGpuLayerDescriptor[] | null;
  step: number;
  sampleStep: number;
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

function architectureSeed(
  layers: HiddenLayer[],
  convolutions: ConvolutionConfig[],
  poolings: PoolingConfig[],
  outputHead: OutputHeadKind,
) {
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
  for (const pooling of poolings) {
    seed = Math.imul(seed ^ pooling.position ^ pooling.order, 16777619);
    seed = Math.imul(seed ^ pooling.kernelSize ^ pooling.stride, 16777619);
  }
  seed = Math.imul(seed ^ (outputHead === "sigmoid" ? 0x51 : 0x5f), 16777619);
  return seed >>> 0;
}

function initializeConvolutions(
  configs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
  hiddenLayers: HiddenLayer[],
): ConvolutionLayerData[] {
  return spatialPipeline(hiddenLayers, configs, poolingConfigs)
    .filter((entry) => entry.kind === "conv")
    .map(({ config, input, output }) => {
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
      trainable: config.trainable,
      position: config.position,
      order: config.order,
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
    } as ConvolutionLayerData;
  });
}

function initializePoolings(
  configs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
  hiddenLayers: HiddenLayer[],
): PoolingLayerData[] {
  return spatialPipeline(hiddenLayers, configs, poolingConfigs)
    .filter((entry) => entry.kind === "pool")
    .map(({ config, input, output }) => ({
      id: config.id,
      position: config.position,
      order: config.order,
      kind: config.kind,
      inputWidth: input.width,
      inputHeight: input.height,
      inputChannels: input.channels,
      outputWidth: output.width,
      outputHeight: output.height,
      kernelSize: config.kernelSize,
      stride: config.stride,
      padding: config.padding,
    }));
}

function denseLayerLayout(
  hiddenLayers: HiddenLayer[],
  convolutions: ConvolutionLayerData[],
  poolings: PoolingLayerData[],
) {
  const layout: Array<{
    inputSize: number;
    outputSize: number;
    activation: ActivationKind | "linear";
  }> = [];
  let inputSize = 784;
  for (let index = 0; index < hiddenLayers.length; index++) {
    for (const layer of modelSpatialLayers({ convolutions, poolings } as NeuralModel).filter(({ position }) => position === index)) {
      inputSize = layer.outputWidth * layer.outputHeight * (layer.type === "conv" ? layer.filters : layer.inputChannels);
    }
    const layer = hiddenLayers[index];
    layout.push({ inputSize, outputSize: layer.units, activation: layer.activation });
    inputSize = layer.units;
  }
  for (const layer of modelSpatialLayers({ convolutions, poolings } as NeuralModel).filter(({ position }) => position === hiddenLayers.length)) {
    inputSize = layer.outputWidth * layer.outputHeight * (layer.type === "conv" ? layer.filters : layer.inputChannels);
  }
  layout.push({ inputSize, outputSize: 10, activation: "linear" });
  return layout;
}

function initializeModel(
  hiddenLayers: HiddenLayer[],
  convolutionConfigs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
  outputHead: OutputHeadKind,
): NeuralModel {
  const random = seededRandom(architectureSeed(hiddenLayers, convolutionConfigs, poolingConfigs, outputHead));
  const convolutions = initializeConvolutions(convolutionConfigs, poolingConfigs, hiddenLayers);
  const poolings = initializePoolings(convolutionConfigs, poolingConfigs, hiddenLayers);
  const layers = denseLayerLayout(hiddenLayers, convolutions, poolings).map((spec): DenseLayerData => {
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
  return { convolutions, poolings, layers, outputHead, calibrated: false, trained: false };
}

function validateInitialModel(
  hiddenLayers: HiddenLayer[],
  convolutionConfigs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
  outputHead: OutputHeadKind,
  initialModel: NeuralModel,
) {
  const expectedConvolutions = initializeConvolutions(convolutionConfigs, poolingConfigs, hiddenLayers);
  const expectedPoolings = initializePoolings(convolutionConfigs, poolingConfigs, hiddenLayers);
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
  const poolings = modelPoolings(initialModel);
  if (poolings.length !== expectedPoolings.length || poolings.some((pooling, index) =>
    JSON.stringify(pooling) !== JSON.stringify(expectedPoolings[index])
  )) {
    throw new Error("载入模型的池化结构与当前架构不匹配");
  }
  const layout = denseLayerLayout(hiddenLayers, expectedConvolutions, expectedPoolings);
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
  return {
    ...initialModel,
    outputHead,
    poolings: expectedPoolings,
    convolutions: convolutions.map((convolution, index) => ({
      ...convolution,
      trainable: expectedConvolutions[index].trainable,
    })),
  };
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
  const weightDecay = Number.isFinite(config?.weightDecay) && config!.weightDecay >= 0
    ? config!.weightDecay
    : 0;
  return { kind, momentum, beta1, beta2, decay, epsilon, weightDecay };
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
      typeof exports.conv2d_forward_batch === "function" &&
      typeof exports.conv2d_train_batch === "function" &&
      typeof exports.pool2d_forward === "function" &&
      typeof exports.pool2d_backward === "function" &&
      typeof exports.pool2d_forward_batch === "function" &&
      typeof exports.pool2d_backward_batch === "function" &&
      typeof exports.dense_forward_batch === "function" &&
      typeof exports.dense_backward_batch === "function" &&
      typeof exports.output_loss_batch === "function" &&
      typeof exports.apply_optimizer === "function" &&
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
  batchCapacity: number,
  computeBackend: ComputeBackend,
  webgpu: WebGpuBatchExecutor | null,
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
    weightGradients: allocateTable(),
    biasGradients: allocateTable(),
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
  const weightGradientPointers: number[] = [];
  const biasGradientPointers: number[] = [];
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
    weightGradientPointers.push(allocate(layer.weights.byteLength));
    biasGradientPointers.push(allocate(layer.biases.byteLength));
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
      weightGradient: allocate(sourceConvolution.weights.byteLength),
      biasGradient: allocate(sourceConvolution.biases.byteLength),
    };
  });

  const sourcePoolings = modelPoolings(sourceModel);
  const poolingPointers = sourcePoolings.map((pooling): PoolingPointers => {
    const inputSize = pooling.inputWidth * pooling.inputHeight * pooling.inputChannels;
    const outputSize = pooling.outputWidth * pooling.outputHeight * pooling.inputChannels;
    return {
      input: allocate(inputSize * 4),
      output: allocate(outputSize * 4),
      indices: allocate(outputSize * 4),
      inputGradient: allocate(inputSize * 4),
      delta: allocate(outputSize * 4),
    };
  });

  const sampleCapacity = Math.max(
    784,
    ...sourceConvolutions.flatMap((convolution) => [
      convolution.inputWidth * convolution.inputHeight * convolution.inputChannels,
      convolution.outputWidth * convolution.outputHeight * convolution.filters,
    ]),
    ...sourcePoolings.flatMap((pooling) => [
      pooling.inputWidth * pooling.inputHeight * pooling.inputChannels,
      pooling.outputWidth * pooling.outputHeight * pooling.inputChannels,
    ]),
  );
  pointers.sampleIndices = allocate(sampleCapacity * 2, 2);
  pointers.sampleValues = allocate(sampleCapacity * 4);
  pointers.inputGradient = allocate(sampleCapacity * 4);

  const normalizedBatchCapacity = Math.max(1, Math.floor(batchCapacity));
  const batchInputPointer = allocate(normalizedBatchCapacity * 784 * 4);
  const batchLabelsPointer = allocate(normalizedBatchCapacity * 4, 4);
  const batchLossesPointer = allocate(normalizedBatchCapacity * 4);
  const spatialAllocations = [
    ...sourceConvolutions.map((model, convolutionIndex) => ({
      type: "conv" as const,
      model,
      convolutionIndex,
    })),
    ...sourcePoolings.map((model, poolingIndex) => ({
      type: "pool" as const,
      model,
      poolingIndex,
    })),
  ].sort((left, right) =>
    left.model.position - right.model.position || left.model.order - right.model.order);
  const batchLayerAllocations: WasmBatchLayerAllocation[] = [];
  let batchLayerInputPointer = batchInputPointer;
  for (let layerIndex = 0; layerIndex < sourceLayers.length; layerIndex++) {
    for (const spatial of spatialAllocations.filter(({ model }) => model.position === layerIndex)) {
      const inputSize = spatial.model.inputWidth * spatial.model.inputHeight * spatial.model.inputChannels;
      const outputChannels = spatial.type === "conv" ? spatial.model.filters : spatial.model.inputChannels;
      const outputSize = spatial.model.outputWidth * spatial.model.outputHeight * outputChannels;
      const outputPointer = allocate(normalizedBatchCapacity * outputSize * 4);
      const allocation: WasmBatchLayerAllocation = {
        type: spatial.type,
        convolutionIndex: spatial.type === "conv" ? spatial.convolutionIndex : undefined,
        poolingIndex: spatial.type === "pool" ? spatial.poolingIndex : undefined,
        indicesPointer: spatial.type === "pool"
          ? allocate(normalizedBatchCapacity * outputSize * 4, 4)
          : 0,
        inputPointer: batchLayerInputPointer,
        outputPointer,
        preactivationPointer: allocate(normalizedBatchCapacity * outputSize * 4),
        deltaPointer: allocate(normalizedBatchCapacity * outputSize * 4),
        inputGradientPointer: allocate(normalizedBatchCapacity * inputSize * 4),
        dropoutMaskPointer: allocate(normalizedBatchCapacity * outputSize * 4),
        inputSize,
        outputSize,
      };
      batchLayerAllocations.push(allocation);
      batchLayerInputPointer = outputPointer;
    }
    const layer = sourceLayers[layerIndex];
    const outputPointer = allocate(normalizedBatchCapacity * layer.outputSize * 4);
    batchLayerAllocations.push({
      type: "dense",
      layerIndex,
      indicesPointer: 0,
      inputPointer: batchLayerInputPointer,
      outputPointer,
      preactivationPointer: allocate(normalizedBatchCapacity * layer.outputSize * 4),
      deltaPointer: allocate(normalizedBatchCapacity * layer.outputSize * 4),
      inputGradientPointer: allocate(normalizedBatchCapacity * layer.inputSize * 4),
      dropoutMaskPointer: allocate(normalizedBatchCapacity * layer.outputSize * 4),
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
    });
    batchLayerInputPointer = outputPointer;
  }

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
  setPointerTable(pointers.weightGradients, weightGradientPointers);
  setPointerTable(pointers.biasGradients, biasGradientPointers);
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
    new Float32Array(buffer, weightGradientPointers[index], layer.weights.length).fill(0);
    new Float32Array(buffer, biasGradientPointers[index], layer.biases.length).fill(0);
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
    new Float32Array(buffer, pointersForConvolution.weightGradient, sourceConvolution.weights.length).fill(0);
    new Float32Array(buffer, pointersForConvolution.biasGradient, sourceConvolution.biases.length).fill(0);
    return {
      type: "conv",
      pointers: pointersForConvolution,
      model: { ...sourceConvolution, weights, biases },
      input: new Float32Array(buffer, pointersForConvolution.input, inputSize),
      output: new Float32Array(buffer, pointersForConvolution.output, outputSize),
      preactivation: new Float32Array(buffer, pointersForConvolution.preactivation, outputSize),
      delta: new Float32Array(buffer, pointersForConvolution.delta, outputSize),
      inputGradient: new Float32Array(buffer, pointersForConvolution.inputGradient, inputSize),
    };
  });

  const poolings = sourcePoolings.map((sourcePooling, index): WasmPoolingRuntime => {
    const pointersForPooling = poolingPointers[index];
    const inputSize = sourcePooling.inputWidth * sourcePooling.inputHeight * sourcePooling.inputChannels;
    const outputSize = sourcePooling.outputWidth * sourcePooling.outputHeight * sourcePooling.inputChannels;
    return {
      type: "pool",
      model: { ...sourcePooling },
      pointers: pointersForPooling,
      input: new Float32Array(buffer, pointersForPooling.input, inputSize),
      output: new Float32Array(buffer, pointersForPooling.output, outputSize),
      inputGradient: new Float32Array(buffer, pointersForPooling.inputGradient, inputSize),
      delta: new Float32Array(buffer, pointersForPooling.delta, outputSize),
    };
  });
  const spatialLayers: WasmSpatialRuntime[] = [...convolutions, ...poolings]
    .sort((left, right) => left.model.position - right.model.position || left.model.order - right.model.order);
  const batchLayers = batchLayerAllocations.map((allocation): WasmBatchLayerRuntime => {
    const common: WasmBatchLayerBase = {
      inputPointer: allocation.inputPointer,
      outputPointer: allocation.outputPointer,
      preactivationPointer: allocation.preactivationPointer,
      deltaPointer: allocation.deltaPointer,
      inputGradientPointer: allocation.inputGradientPointer,
      dropoutMaskPointer: allocation.dropoutMaskPointer,
      inputSize: allocation.inputSize,
      outputSize: allocation.outputSize,
      input: new Float32Array(buffer, allocation.inputPointer, normalizedBatchCapacity * allocation.inputSize),
      output: new Float32Array(buffer, allocation.outputPointer, normalizedBatchCapacity * allocation.outputSize),
      preactivation: new Float32Array(buffer, allocation.preactivationPointer, normalizedBatchCapacity * allocation.outputSize),
      delta: new Float32Array(buffer, allocation.deltaPointer, normalizedBatchCapacity * allocation.outputSize),
      inputGradient: new Float32Array(buffer, allocation.inputGradientPointer, normalizedBatchCapacity * allocation.inputSize),
      dropoutMask: new Float32Array(buffer, allocation.dropoutMaskPointer, normalizedBatchCapacity * allocation.outputSize),
    };
    common.dropoutMask.fill(1);
    if (allocation.type === "dense") {
      return { ...common, type: "dense", layerIndex: allocation.layerIndex! };
    }
    if (allocation.type === "conv") {
      return { ...common, type: "conv", convolutionIndex: allocation.convolutionIndex! };
    }
    return {
      ...common,
      type: "pool",
      poolingIndex: allocation.poolingIndex!,
      indicesPointer: allocation.indicesPointer,
      indices: new Uint32Array(
        buffer,
        allocation.indicesPointer,
        normalizedBatchCapacity * allocation.outputSize,
      ),
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
    poolings,
    spatialLayers,
    batch: {
      capacity: normalizedBatchCapacity,
      inputPointer: batchInputPointer,
      labelsPointer: batchLabelsPointer,
      lossesPointer: batchLossesPointer,
      input: new Float32Array(buffer, batchInputPointer, normalizedBatchCapacity * 784),
      labels: new Int32Array(buffer, batchLabelsPointer, normalizedBatchCapacity),
      losses: new Float32Array(buffer, batchLossesPointer, normalizedBatchCapacity),
      layers: batchLayers,
    },
    outputHead: sourceModel.outputHead === "sigmoid" ? "sigmoid" : "softmax",
    backend: webgpu
      ? `Zig/WebGPU · ${mathMode === "full" ? "完整" : "快速"}`
      : `${flavor === "simd" ? "Zig/Wasm SIMD" : "Zig/Wasm"} · ${mathMode === "full" ? "完整" : "快速"}`,
    mathMode,
    computeBackend,
    webgpu,
    webgpuGraph: null,
    webgpuDescriptors: null,
    step: 0,
    sampleStep: 0,
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

function stageSpatialInput(
  layer: WasmSpatialRuntime,
  sample: SparseSample,
  denseInput: Float32Array | null,
) {
  layer.input.fill(0);
  if (denseInput) {
    layer.input.set(denseInput);
    return;
  }
  for (let index = 0; index < sample.indices.length; index++) {
    layer.input[sample.indices[index]] = sample.values[index];
  }
}

function forwardSpatialLayer(runtime: WasmTrainingRuntime, layer: WasmSpatialRuntime) {
  if (layer.type === "pool") {
    const { model, pointers } = layer;
    runtime.wasm.pool2d_forward(
      pointers.input,
      pointers.output,
      pointers.indices,
      model.inputWidth,
      model.inputHeight,
      model.inputChannels,
      model.kernelSize,
      model.stride,
      model.padding,
      model.kind === "max" ? 0 : model.kind === "average" ? 1 : 2,
    );
    return;
  }
  const { model, pointers } = layer;
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
      runtime.sampleStep >>> 0,
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

function forwardSpatialPrefix(
  runtime: WasmTrainingRuntime,
  sample: SparseSample,
  training: boolean,
  nchwInput: Float32Array | null = null,
) {
  let denseStart = 0;
  let inputPointer = runtime.pointers.sampleValues;
  let inputCount = sample.indices.length;
  let inputIsDense = false;
  let inputValues: Float32Array | null = nchwInput;

  for (const layer of runtime.spatialLayers) {
    const denseCount = layer.model.position - denseStart;
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
      const finalDenseIndex = layer.model.position - 1;
      inputPointer = pointerAt(runtime, runtime.pointers.activations, finalDenseIndex);
      inputCount = runtime.activations[finalDenseIndex].length;
      inputIsDense = true;
      inputValues = runtime.activations[finalDenseIndex];
    }
    stageSpatialInput(layer, sample, inputValues);
    forwardSpatialLayer(runtime, layer);
    denseStart = layer.model.position;
    inputPointer = layer.pointers.output;
    inputCount = layer.output.length;
    inputIsDense = true;
    inputValues = layer.output;
  }

  return { denseStart, inputPointer, inputCount, inputIsDense };
}

function forwardWithWasm(runtime: WasmTrainingRuntime, sample: SparseSample) {
  stageSample(runtime, sample);
  const suffix = forwardSpatialPrefix(runtime, sample, false);
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

function packNchwBatch(samples: SparseSample[]) {
  const tensor = new Float32Array(samples.length * 784);
  samples.forEach((sample, batchIndex) => {
    const offset = batchIndex * 784;
    for (let index = 0; index < sample.indices.length; index++) {
      tensor[offset + sample.indices[index]] = sample.values[index];
    }
  });
  return tensor;
}

function forwardFullBatch(
  runtime: WasmTrainingRuntime,
  samples: SparseSample[],
  training: boolean,
) {
  const batchSize = samples.length;
  if (batchSize === 0 || batchSize > runtime.batch.capacity) {
    throw new Error("完整数学批次大小超出 Wasm 缓冲区容量");
  }
  runtime.batch.input.set(packNchwBatch(samples), 0);
  runtime.batch.labels.set(samples.map(({ label }) => label), 0);
  if (training) runtime.sampleStep += batchSize;
  const dropoutRates = new Float32Array(
    runtime.wasm.memory.buffer,
    runtime.pointers.dropoutRates,
    runtime.model.length,
  );

  runtime.batch.layers.forEach((layer, networkIndex) => {
    if (layer.type === "dense") {
      const model = runtime.model[layer.layerIndex];
      runtime.wasm.dense_forward_batch(
        layer.inputPointer,
        pointerAt(runtime, runtime.pointers.weights, layer.layerIndex),
        pointerAt(runtime, runtime.pointers.biases, layer.layerIndex),
        layer.outputPointer,
        layer.preactivationPointer,
        layer.dropoutMaskPointer,
        batchSize,
        layer.inputSize,
        layer.outputSize,
        activationCodes[model.activation],
        dropoutRates[layer.layerIndex],
        runtime.sampleStep >>> 0,
        networkIndex,
        training ? 1 : 0,
      );
      return;
    }
    if (layer.type === "conv") {
      const convolution = runtime.convolutions[layer.convolutionIndex];
      const { model, pointers } = convolution;
      runtime.wasm.conv2d_forward_batch(
        layer.inputPointer,
        pointers.weights,
        pointers.biases,
        layer.outputPointer,
        layer.preactivationPointer,
        batchSize,
        model.inputWidth,
        model.inputHeight,
        model.inputChannels,
        model.filters,
        model.kernelSize,
        model.stride,
        model.padding,
        activationCodes[model.activation],
      );
      return;
    }
    const pooling = runtime.poolings[layer.poolingIndex].model;
    runtime.wasm.pool2d_forward_batch(
      layer.inputPointer,
      layer.outputPointer,
      layer.indicesPointer,
      batchSize,
      pooling.inputWidth,
      pooling.inputHeight,
      pooling.inputChannels,
      pooling.kernelSize,
      pooling.stride,
      pooling.padding,
      pooling.kind === "max" ? 0 : pooling.kind === "average" ? 1 : 2,
    );
  });

  const outputLayer = runtime.batch.layers.at(-1);
  if (!outputLayer || outputLayer.type !== "dense") {
    throw new Error("完整数学路径缺少输出层");
  }
  runtime.wasm.output_loss_batch(
    outputLayer.outputPointer,
    outputLayer.deltaPointer,
    runtime.batch.labelsPointer,
    runtime.batch.lossesPointer,
    batchSize,
    outputLayer.outputSize,
    runtime.outputHead === "sigmoid" ? 1 : 0,
  );
  return outputLayer;
}

function webGpuLayerDescriptors(runtime: WasmTrainingRuntime): WebGpuLayerDescriptor[] {
  const dropoutRates = new Float32Array(
    runtime.wasm.memory.buffer,
    runtime.pointers.dropoutRates,
    runtime.model.length,
  );
  return runtime.batch.layers.map((layer): WebGpuLayerDescriptor => {
    if (layer.type === "dense") {
      const model = runtime.model[layer.layerIndex];
      return {
        type: "dense",
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activationKind: activationCodes[model.activation],
        dropoutRate: dropoutRates[layer.layerIndex],
        weights: model.weights,
        biases: model.biases,
      };
    }
    if (layer.type === "conv") {
      const model = runtime.convolutions[layer.convolutionIndex].model;
      return {
        type: "conv",
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activationKind: activationCodes[model.activation],
        inputWidth: model.inputWidth,
        inputHeight: model.inputHeight,
        inputChannels: model.inputChannels,
        filters: model.filters,
        kernelSize: model.kernelSize,
        stride: model.stride,
        padding: model.padding,
        trainable: model.trainable,
        weights: model.weights,
        biases: model.biases,
      };
    }
    const model = runtime.poolings[layer.poolingIndex].model;
    const global = model.kind === "globalAverage";
    return {
      type: "pool",
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      inputWidth: model.inputWidth,
      inputHeight: model.inputHeight,
      inputChannels: model.inputChannels,
      outputWidth: model.outputWidth,
      outputHeight: model.outputHeight,
      kernelSize: global ? Math.max(model.inputWidth, model.inputHeight) : Math.max(1, model.kernelSize),
      stride: global ? 1 : Math.max(1, model.stride),
      padding: global ? 0 : Math.max(0, model.padding),
      poolingKind: model.kind === "max" ? 0 : model.kind === "average" ? 1 : 2,
    };
  });
}

async function forwardWebGpuBatch(
  runtime: WasmTrainingRuntime,
  samples: SparseSample[],
  _training: boolean,
) {
  const graph = runtime.webgpuGraph;
  if (!graph) throw new Error("Zig/WebGPU 计算图尚未初始化");
  const batchSize = samples.length;
  if (batchSize === 0 || batchSize > runtime.batch.capacity) {
    throw new Error("WebGPU 批次大小超出张量缓冲区容量");
  }
  const input = packNchwBatch(samples);
  runtime.batch.input.set(input, 0);
  runtime.batch.labels.set(samples.map(({ label }) => label), 0);
  const result = await graph.forward(
    input,
    runtime.batch.labels.subarray(0, batchSize),
    batchSize,
  );
  const outputLayer = runtime.batch.layers.at(-1);
  if (!outputLayer || outputLayer.type !== "dense") {
    throw new Error("Zig/WebGPU 路径缺少输出层");
  }
  outputLayer.output.set(result.probabilities);
  runtime.batch.losses.set(result.losses);
  return outputLayer;
}

async function trainWebGpuBatch(
  runtime: WasmTrainingRuntime,
  samples: SparseSample[],
  learningRate: number,
  captureTrace = false,
) {
  const graph = runtime.webgpuGraph;
  if (!graph) throw new Error("Zig/WebGPU 计算图尚未初始化");
  const batchSize = samples.length;
  const input = packNchwBatch(samples);
  runtime.batch.input.set(input, 0);
  runtime.batch.labels.set(samples.map(({ label }) => label), 0);
  runtime.sampleStep += batchSize;
  runtime.step++;
  runtime.beta1Power *= runtime.config.beta1;
  runtime.beta2Power *= runtime.config.beta2;
  const result = await graph.train(
    input,
    runtime.batch.labels.subarray(0, batchSize),
    batchSize,
    runtime.sampleStep >>> 0,
    captureTrace,
    {
      kind: optimizerCodes[runtime.config.kind],
      learningRate,
      momentum: runtime.config.momentum,
      decay: runtime.config.decay,
      beta1: runtime.config.beta1,
      beta2: runtime.config.beta2,
      epsilon: runtime.config.epsilon,
      beta1Correction: 1 - runtime.beta1Power,
      beta2Correction: 1 - runtime.beta2Power,
      gradientScale: 1 / Math.max(1, batchSize),
      weightDecay: runtime.config.weightDecay,
    },
  );
  runtime.batch.losses.set(result.losses);

  let totalLoss = 0;
  for (let index = 0; index < batchSize; index++) {
    const value = result.losses[index];
    if (!Number.isFinite(value)) throw new Error("Zig/WebGPU 批量训练产生了无效损失");
    totalLoss += value;
  }
  if (!captureTrace) return totalLoss;

  const outputLayer = runtime.batch.layers.at(-1)!;
  result.outputs.forEach((output, index) => runtime.batch.layers[index].output.set(output));
  result.deltas.forEach((delta, index) => runtime.batch.layers[index].delta.set(delta));
  if (result.firstInputGradient) {
    runtime.batch.layers[0].inputGradient.set(result.firstInputGradient);
  }
  const probabilities = result.outputs.at(-1)!;
  let prediction = 0;
  for (let index = 1; index < outputLayer.outputSize; index++) {
    if (probabilities[index] > probabilities[prediction]) prediction = index;
  }
  return {
    batchLoss: totalLoss,
    loss: runtime.batch.losses[0],
    activations: [
      Float32Array.from(runtime.batch.input.subarray(0, 784)),
      ...result.outputs,
    ],
    gradients: [
      result.firstInputGradient ?? new Float32Array(runtime.batch.layers[0].inputSize),
      ...result.deltas,
    ],
    convolutionWeights: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.weights),
    ),
    convolutionBiases: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.biases),
    ),
    label: samples[0].label,
    prediction,
  } satisfies FullBatchTraceResult;
}

function trainFullBatchWithWasm(
  runtime: WasmTrainingRuntime,
  samples: SparseSample[],
  captureTrace = false,
) {
  const batchSize = samples.length;
  const outputLayer = forwardFullBatch(runtime, samples, true);
  for (let index = runtime.batch.layers.length - 1; index >= 0; index--) {
    const layer = runtime.batch.layers[index];
    const nextLayer = runtime.batch.layers[index + 1];
    const outputGradientPointer = nextLayer
      ? nextLayer.inputGradientPointer
      : outputLayer.deltaPointer;
    if (layer.type === "dense") {
      const model = runtime.model[layer.layerIndex];
      runtime.wasm.dense_backward_batch(
        layer.inputPointer,
        pointerAt(runtime, runtime.pointers.weights, layer.layerIndex),
        layer.preactivationPointer,
        outputGradientPointer,
        layer.inputGradientPointer,
        layer.deltaPointer,
        pointerAt(runtime, runtime.pointers.weightGradients, layer.layerIndex),
        pointerAt(runtime, runtime.pointers.biasGradients, layer.layerIndex),
        layer.dropoutMaskPointer,
        batchSize,
        layer.inputSize,
        layer.outputSize,
        activationCodes[model.activation],
      );
      continue;
    }
    if (layer.type === "conv") {
      const convolution = runtime.convolutions[layer.convolutionIndex];
      const { model, pointers } = convolution;
      runtime.wasm.conv2d_train_batch(
        layer.inputPointer,
        pointers.weights,
        pointers.biases,
        layer.preactivationPointer,
        outputGradientPointer,
        layer.inputGradientPointer,
        layer.deltaPointer,
        pointers.weightGradient,
        pointers.biasGradient,
        batchSize,
        model.inputWidth,
        model.inputHeight,
        model.inputChannels,
        model.filters,
        model.kernelSize,
        model.stride,
        model.padding,
        activationCodes[model.activation],
        model.trainable ? 1 : 0,
      );
      continue;
    }
    const pooling = runtime.poolings[layer.poolingIndex].model;
    runtime.wasm.pool2d_backward_batch(
      outputGradientPointer,
      layer.inputGradientPointer,
      layer.deltaPointer,
      layer.indicesPointer,
      batchSize,
      pooling.inputWidth,
      pooling.inputHeight,
      pooling.inputChannels,
      pooling.kernelSize,
      pooling.stride,
      pooling.padding,
      pooling.kind === "max" ? 0 : pooling.kind === "average" ? 1 : 2,
    );
  }

  let totalLoss = 0;
  for (let index = 0; index < batchSize; index++) {
    const value = runtime.batch.losses[index];
    if (!Number.isFinite(value)) throw new Error("Zig/Wasm 批量训练产生了无效损失");
    totalLoss += value;
  }
  if (!captureTrace) return totalLoss;

  let prediction = 0;
  for (let index = 1; index < outputLayer.outputSize; index++) {
    if (outputLayer.output[index] > outputLayer.output[prediction]) prediction = index;
  }
  return {
    batchLoss: totalLoss,
    loss: runtime.batch.losses[0],
    activations: [
      Float32Array.from(runtime.batch.input.subarray(0, 784)),
      ...runtime.batch.layers.map((layer) =>
        Float32Array.from(layer.output.subarray(0, layer.outputSize))),
    ],
    gradients: [
      Float32Array.from(runtime.batch.layers[0].inputGradient.subarray(
        0,
        runtime.batch.layers[0].inputSize,
      )),
      ...runtime.batch.layers.map((layer) =>
        Float32Array.from(layer.delta.subarray(0, layer.outputSize))),
    ],
    convolutionWeights: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.weights),
    ),
    convolutionBiases: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.biases),
    ),
    label: samples[0].label,
    prediction,
  } satisfies FullBatchTraceResult;
}

function trainSampleWithWasm(
  runtime: WasmTrainingRuntime,
  sample: SparseSample,
  captureTrace = false,
  nchwInput: Float32Array | null = null,
) {
  stageSample(runtime, sample);
  runtime.sampleStep++;
  const { pointers } = runtime;
  const suffix = forwardSpatialPrefix(runtime, sample, true, nchwInput);
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
    tableAt(pointers.weightGradients, denseStart),
    tableAt(pointers.biasGradients, denseStart),
    runtime.outputHead === "sigmoid" ? 1 : 0,
    captureTrace || runtime.spatialLayers.length > 0 ? 1 : 0,
    pointers.inputGradient,
    tableAt(pointers.dropoutRates, denseStart),
    tableAt(pointers.dropoutMasks, denseStart),
    runtime.sampleStep >>> 0,
    suffix.inputIsDense ? 1 : 0,
  );
  if (!Number.isFinite(loss)) throw new Error("Zig/Wasm 训练产生了无效损失");
  let downstreamGradientPointer = pointers.inputGradient;
  for (let index = runtime.spatialLayers.length - 1; index >= 0; index--) {
    const layer = runtime.spatialLayers[index];
    if (layer.type === "pool") {
      const model = layer.model;
      layer.delta.set(new Float32Array(runtime.wasm.memory.buffer, downstreamGradientPointer, layer.output.length));
      runtime.wasm.pool2d_backward(
        downstreamGradientPointer,
        layer.pointers.inputGradient,
        layer.pointers.indices,
        model.inputWidth,
        model.inputHeight,
        model.inputChannels,
        model.kernelSize,
        model.stride,
        model.padding,
        model.kind === "max" ? 0 : model.kind === "average" ? 1 : 2,
      );
    } else runtime.wasm.conv2d_train(
      layer.pointers.input,
      layer.pointers.weights,
      layer.pointers.biases,
      layer.pointers.preactivation,
      downstreamGradientPointer,
      layer.pointers.inputGradient,
      layer.pointers.delta,
      layer.pointers.weightGradient,
      layer.pointers.biasGradient,
      layer.model.inputWidth,
      layer.model.inputHeight,
      layer.model.inputChannels,
      layer.model.filters,
      layer.model.kernelSize,
      layer.model.stride,
      layer.model.padding,
      activationCodes[layer.model.activation],
      layer.model.trainable ? 1 : 0,
    );
    const previousLayer = runtime.spatialLayers[index - 1];
    const segmentStart = previousLayer?.model.position ?? 0;
    const segmentCount = layer.model.position - segmentStart;
    if (segmentCount > 0) {
      runtime.wasm.train_dense_from_gradient(
        segmentCount,
        pointers.sampleIndices,
        previousLayer?.pointers.output ?? pointers.sampleValues,
        previousLayer?.output.length ?? sample.indices.length,
        layer.pointers.inputGradient,
        tableAt(pointers.inputSizes, segmentStart),
        tableAt(pointers.outputSizes, segmentStart),
        tableAt(pointers.activationKinds, segmentStart),
        tableAt(pointers.weights, segmentStart),
        tableAt(pointers.biases, segmentStart),
        tableAt(pointers.activations, segmentStart),
        tableAt(pointers.preactivations, segmentStart),
        tableAt(pointers.deltas, segmentStart),
        tableAt(pointers.weightGradients, segmentStart),
        tableAt(pointers.biasGradients, segmentStart),
        captureTrace ? 1 : 0,
        pointers.inputGradient,
        tableAt(pointers.dropoutMasks, segmentStart),
        previousLayer ? 1 : 0,
      );
      downstreamGradientPointer = pointers.inputGradient;
    } else {
      downstreamGradientPointer = layer.pointers.inputGradient;
    }
  }
  if (!captureTrace) return loss;

  const probabilities = runtime.activations.at(-1)!;
  let prediction = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index] > probabilities[prediction]) prediction = index;
  }
  const traceActivations: Float32Array[] = [denseInput(sample)];
  const firstSpatialLayer = runtime.spatialLayers[0];
  const inputGradient = firstSpatialLayer
    ? firstSpatialLayer.model.position > 0
      ? Float32Array.from(runtime.inputGradient.subarray(0, 784))
      : Float32Array.from(firstSpatialLayer.inputGradient)
    : Float32Array.from(runtime.inputGradient.subarray(0, 784));
  const traceGradients: Float32Array[] = [inputGradient];
  for (let layerIndex = 0; layerIndex < runtime.model.length; layerIndex++) {
    for (const layer of runtime.spatialLayers.filter(({ model }) => model.position === layerIndex)) {
      traceActivations.push(Float32Array.from(layer.output));
      traceGradients.push(Float32Array.from(layer.delta));
    }
    traceActivations.push(Float32Array.from(runtime.activations[layerIndex]));
    traceGradients.push(Float32Array.from(runtime.deltas[layerIndex]));
  }
  return {
    loss,
    activations: traceActivations,
    gradients: traceGradients,
    convolutionWeights: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.weights),
    ),
    convolutionBiases: runtime.convolutions.map((convolution) =>
      Float32Array.from(convolution.model.biases),
    ),
    label: sample.label,
    prediction,
  } satisfies SampleTraceResult;
}

function applyBatchUpdates(runtime: WasmTrainingRuntime, learningRate: number, batchCount: number) {
  runtime.step++;
  runtime.beta1Power *= runtime.config.beta1;
  runtime.beta2Power *= runtime.config.beta2;
  const { config, pointers } = runtime;
  const apply = (
    parameters: number,
    gradients: number,
    first: number,
    second: number,
    length: number,
    weightDecay: number,
  ) => runtime.wasm.apply_optimizer(
    parameters,
    gradients,
    first,
    second,
    length,
    optimizerCodes[config.kind],
    learningRate,
    config.momentum,
    config.decay,
    config.beta1,
    config.beta2,
    config.epsilon,
    1 - runtime.beta1Power,
    1 - runtime.beta2Power,
    1 / Math.max(1, batchCount),
    weightDecay,
  );
  runtime.model.forEach((layer, index) => {
    apply(pointerAt(runtime, pointers.weights, index), pointerAt(runtime, pointers.weightGradients, index), pointerAt(runtime, pointers.weightFirst, index), pointerAt(runtime, pointers.weightSecond, index), layer.weights.length, config.weightDecay);
    apply(pointerAt(runtime, pointers.biases, index), pointerAt(runtime, pointers.biasGradients, index), pointerAt(runtime, pointers.biasFirst, index), pointerAt(runtime, pointers.biasSecond, index), layer.biases.length, 0);
  });
  runtime.convolutions.forEach((convolution) => {
    if (!convolution.model.trainable) return;
    const { pointers: convPointers, model } = convolution;
    apply(convPointers.weights, convPointers.weightGradient, convPointers.weightFirst, convPointers.weightSecond, model.weights.length, config.weightDecay);
    apply(convPointers.biases, convPointers.biasGradient, convPointers.biasFirst, convPointers.biasSecond, model.biases.length, 0);
  });
}

async function validateWithWasm(runtime: WasmTrainingRuntime, samples: SparseSample[]) {
  if (samples.length === 0) return 0;
  let correct = 0;
  if (runtime.mathMode === "full" || runtime.webgpu) {
    for (let start = 0; start < samples.length; start += runtime.batch.capacity) {
      const batchSamples = samples.slice(start, start + runtime.batch.capacity);
      const outputLayer = runtime.webgpu
        ? await forwardWebGpuBatch(runtime, batchSamples, false)
        : forwardFullBatch(runtime, batchSamples, false);
      for (let batchIndex = 0; batchIndex < batchSamples.length; batchIndex++) {
        const offset = batchIndex * outputLayer.outputSize;
        let prediction = 0;
        for (let digit = 1; digit < outputLayer.outputSize; digit++) {
          if (outputLayer.output[offset + digit] > outputLayer.output[offset + prediction]) {
            prediction = digit;
          }
        }
        if (prediction === batchSamples[batchIndex].label) correct++;
      }
    }
    return correct / samples.length;
  }
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
    poolings: runtime.poolings.map((pooling) => ({ ...pooling.model })),
    layers: runtime.model.map((layer): DenseLayerData => ({
      ...layer,
      weights: Float32Array.from(layer.weights),
      biases: Float32Array.from(layer.biases),
    })),
    outputHead: runtime.outputHead,
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
  const poolingConfigs = fitPoolingsToLayers(message.poolings ?? [], message.layers);
  const outputHead: OutputHeadKind = message.outputHead === "sigmoid" ? "sigmoid" : "softmax";
  const epochs = Number.isFinite(message.settings.epochs)
    ? Math.max(1, Math.floor(message.settings.epochs))
    : 1;
  const requestedComputeBackend: ComputeBackend =
    message.settings.computeBackend === "webgpu" ? "webgpu" : "wasm";
  postMessage({
    type: "progress",
    phase: "loading",
    epoch: 0,
    backend: requestedComputeBackend === "webgpu" ? "Zig/WebGPU" : "Zig/Wasm",
  });
  const datasetPromise = message.mnistEnabled === false
    ? Promise.resolve<ArrayBuffer | null>(null)
    : fetch(message.datasetUrl).then((response) => {
        if (!response.ok) throw new Error("无法载入训练数据");
        return response.arrayBuffer();
      });
  const [datasetBuffer, loadedWasm, webgpu] = await Promise.all([
    datasetPromise,
    loadTrainingWasm(),
    requestedComputeBackend === "webgpu"
      ? WebGpuBatchExecutor.create()
      : Promise.resolve(null),
  ]);
  const { training, validation } = datasetBuffer
    ? parseDataset(datasetBuffer)
    : { training: [] as SparseSample[], validation: [] as SparseSample[] };
  appendCustomSamples(message.customSamples ?? [], training, validation);
  if (training.length === 0) {
    throw new Error("训练集为空，请启用 MNIST 或加入自定义训练样本");
  }
  const sourceModel = message.initialModel
    ? validateInitialModel(message.layers, convolutionConfigs, poolingConfigs, outputHead, message.initialModel)
    : initializeModel(message.layers, convolutionConfigs, poolingConfigs, outputHead);
  const batchSize = Number.isFinite(message.settings.batchSize)
    ? Math.max(1, Math.floor(message.settings.batchSize))
    : 16;
  const mathMode: MathMode =
    requestedComputeBackend === "webgpu" || message.settings.mathMode === "full"
      ? "full"
      : "fast";
  const runtime = createWasmTrainingRuntime(
    loadedWasm.exports,
    sourceModel,
    message.layers,
    message.settings.optimizer,
    mathMode,
    loadedWasm.flavor,
    mathMode === "full" || webgpu ? batchSize : 1,
    requestedComputeBackend,
    webgpu,
  );
  if (webgpu) {
    runtime.webgpuDescriptors = webGpuLayerDescriptors(runtime);
    runtime.webgpuGraph = webgpu.createTrainingGraph(
      runtime.webgpuDescriptors,
      runtime.batch.capacity,
      runtime.outputHead === "sigmoid",
    );
    postMessage({
      type: "progress",
      phase: "loading",
      epoch: 0,
      epochs,
      adapter: webgpu.adapterName,
      backend: "Zig/WebGPU",
    });
  }
  activeRuntime = runtime;
  if (!Number.isFinite(message.settings.learningRate) || message.settings.learningRate <= 0) {
    throw new Error("学习率必须是大于 0 的有限数值");
  }
  const random = seededRandom(architectureSeed(message.layers, convolutionConfigs, poolingConfigs, outputHead) ^ 0x9e3779b9);
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
  const emitTrace = (result: SampleTraceResult, epoch: number, sample: number) => {
    lastTraceAt = performance.now();
    postMessage({
      type: "trace",
      epoch,
      sample,
      samples: training.length,
      activations: result.activations,
      gradients: result.gradients,
      convolutionWeights: result.convolutionWeights,
      convolutionBiases: result.convolutionBiases,
      label: result.label,
      prediction: result.prediction,
      loss: result.loss,
      backend: runtime.backend,
    });
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let cursor = training.length - 1; cursor > 0; cursor--) {
      const swap = Math.floor(random() * (cursor + 1));
      [training[cursor], training[swap]] = [training[swap], training[cursor]];
    }
    const learningRate =
      message.settings.learningRate * Math.pow(0.94, epoch);
    let loss = 0;
    if (runtime.mathMode === "full" || runtime.webgpu) {
      if (runtime.webgpu) {
        const pipelineDepth = 4;
        for (
          let groupStart = 0;
          groupStart < training.length;
          groupStart += batchSize * pipelineDepth
        ) {
          const pending: Array<{
            batchStart: number;
            batchSamples: SparseSample[];
            captureTrace: boolean;
            result: ReturnType<typeof trainWebGpuBatch>;
          }> = [];
          for (
            let batchStart = groupStart;
            batchStart < Math.min(training.length, groupStart + batchSize * pipelineDepth);
            batchStart += batchSize
          ) {
            const batchSamples = training.slice(batchStart, batchStart + batchSize);
            const captureTrace =
              batchStart === 0 ||
              (batchStart === groupStart && performance.now() - lastTraceAt >= 90);
            pending.push({
              batchStart,
              batchSamples,
              captureTrace,
              result: trainWebGpuBatch(runtime, batchSamples, learningRate, captureTrace),
            });
          }
          const results = await Promise.all(pending.map(({ result }) => result));
          results.forEach((result, index) => {
            const { batchStart, batchSamples, captureTrace } = pending[index];
            if (typeof result === "number") loss += result;
            else {
              loss += result.batchLoss;
              emitTrace(result, epoch + 1, batchStart + 1);
            }
            const processed = batchStart + batchSamples.length;
            activeSnapshotProgress!.epoch = epoch + 1;
            activeSnapshotProgress!.sample = processed;
            activeSnapshotProgress!.accuracy = lastAccuracy;
            activeSnapshotProgress!.loss = loss / processed;
            if (captureTrace) lastControlYieldAt = performance.now();
          });
          if (!(await waitForControl()) || cancelled) {
            activeRuntime = null;
            activeSnapshotProgress = null;
            activeStartedAt = null;
            postMessage({ type: "cancelled" });
            return;
          }
          lastControlYieldAt = performance.now();
        }
      } else {
        for (let batchStart = 0; batchStart < training.length; batchStart += batchSize) {
          const batchSamples = training.slice(batchStart, batchStart + batchSize);
          const now = performance.now();
          const captureTrace =
            batchStart === 0 ||
            (batchStart % traceInterval < batchSize && now - lastTraceAt >= 90);
          const result = trainFullBatchWithWasm(runtime, batchSamples, captureTrace);
          applyBatchUpdates(runtime, learningRate, batchSamples.length);
          if (typeof result === "number") loss += result;
          else {
            loss += result.batchLoss;
            emitTrace(result, epoch + 1, batchStart + 1);
          }
          const processed = batchStart + batchSamples.length;
          activeSnapshotProgress.epoch = epoch + 1;
          activeSnapshotProgress.sample = processed;
          activeSnapshotProgress.accuracy = lastAccuracy;
          activeSnapshotProgress.loss = loss / processed;

          if (captureTrace || performance.now() - lastControlYieldAt >= 32) {
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
      }
    } else {
      let currentBatchTensor = new Float32Array(0);
      for (let sampleIndex = 0; sampleIndex < training.length; sampleIndex++) {
        const sample = training[sampleIndex];
        const batchOffset = sampleIndex % batchSize;
        if (batchOffset === 0) {
          currentBatchTensor = packNchwBatch(training.slice(sampleIndex, sampleIndex + batchSize));
        }
        const now = performance.now();
        const captureTrace =
          sampleIndex === 0 ||
          (sampleIndex % traceInterval === 0 && now - lastTraceAt >= 90);
        const result = trainSampleWithWasm(
          runtime,
          sample,
          captureTrace,
          currentBatchTensor.subarray(batchOffset * 784, (batchOffset + 1) * 784),
        );
        const batchCount = (sampleIndex % batchSize) + 1;
        const batchComplete = batchCount === batchSize || sampleIndex === training.length - 1;
        if (batchComplete) applyBatchUpdates(runtime, learningRate, batchCount);
        if (typeof result === "number") loss += result;
        else {
          loss += result.loss;
          emitTrace(result, epoch + 1, sampleIndex + 1);
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
    }
    const accuracy = await validateWithWasm(runtime, validation);
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

  const accuracy = lastAccuracy;
  if (runtime.webgpuGraph && runtime.webgpuDescriptors) {
    await runtime.webgpuGraph.downloadParameters(runtime.webgpuDescriptors);
  }
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

self.onmessage = async (event: MessageEvent<TrainMessage | CancelMessage | PauseMessage | ResumeMessage | SnapshotMessage>) => {
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
      if (activeRuntime.webgpuGraph && activeRuntime.webgpuDescriptors) {
        await activeRuntime.webgpuGraph.downloadParameters(activeRuntime.webgpuDescriptors);
      }
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
