/// <reference lib="webworker" />

import type {
  ActivationKind,
  CustomDatasetSample,
  DenseLayerData,
  HiddenLayer,
  OptimizerConfig,
  OptimizerKind,
  TrainingSettings,
} from "../types";

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
  settings: TrainingSettings;
  customSamples: CustomDatasetSample[];
  initialModel?: DenseLayerData[];
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

interface TrainingWasmExports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  forward_sparse: (...args: number[]) => void;
  train_sample: (...args: number[]) => number;
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
  step: number;
  beta1Power: number;
  beta2Power: number;
}

let cancelled = false;
let paused = false;
let pausedAt: number | null = null;
let pausedDuration = 0;
let controlWaiters: Array<() => void> = [];
let activeModel: DenseLayerData[] | null = null;
let activeSnapshotProgress: SnapshotProgress | null = null;
let activeStartedAt: number | null = null;
let trainingWasmPromise: Promise<TrainingWasmExports> | null = null;

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

function architectureSeed(layers: HiddenLayer[]) {
  let seed = 0x57a7c11;
  for (const layer of layers) {
    seed = Math.imul(seed ^ layer.units, 16777619);
    for (const character of layer.activation) {
      seed = Math.imul(seed ^ character.charCodeAt(0), 16777619);
    }
  }
  return seed >>> 0;
}

function initializeModel(hiddenLayers: HiddenLayer[]) {
  const random = seededRandom(architectureSeed(hiddenLayers));
  const sizes = [784, ...hiddenLayers.map((layer) => layer.units), 10];
  return sizes.slice(1).map((outputSize, index): DenseLayerData => {
    const inputSize = sizes[index];
    const activation =
      index < hiddenLayers.length ? hiddenLayers[index].activation : "linear";
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
}

function validateInitialModel(
  hiddenLayers: HiddenLayer[],
  initialModel: DenseLayerData[],
) {
  const sizes = [784, ...hiddenLayers.map((layer) => layer.units), 10];
  if (initialModel.length !== sizes.length - 1) {
    throw new Error("载入模型的层数与当前架构不匹配");
  }

  for (let index = 0; index < initialModel.length; index++) {
    const layer = initialModel[index];
    const expectedActivation =
      index < hiddenLayers.length ? hiddenLayers[index].activation : "linear";
    const inputSize = sizes[index];
    const outputSize = sizes[index + 1];
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
    trainingWasmPromise = (async () => {
      const response = await fetch(`${import.meta.env.BASE_URL}neuron_kernel.wasm`);
      if (!response.ok) throw new Error("无法载入 Zig/Wasm 训练内核");
      const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
      const exports = instance.exports as unknown as TrainingWasmExports;
      if (
        !(exports.memory instanceof WebAssembly.Memory) ||
        typeof exports.forward_sparse !== "function" ||
        typeof exports.train_sample !== "function"
      ) {
        throw new Error("Zig/Wasm 训练内核缺少必要导出");
      }
      return exports;
    })();
  }
  return trainingWasmPromise;
}

function createWasmTrainingRuntime(
  wasm: TrainingWasmExports,
  sourceModel: DenseLayerData[],
  optimizerConfig: OptimizerConfig | undefined,
): WasmTrainingRuntime {
  const layerCount = sourceModel.length;
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

  for (const layer of sourceModel) {
    weightPointers.push(allocate(layer.weights.byteLength));
    biasPointers.push(allocate(layer.biases.byteLength));
    activationPointers.push(allocate(layer.outputSize * 4));
    preactivationPointers.push(allocate(layer.outputSize * 4));
    deltaPointers.push(allocate(layer.outputSize * 4));
    weightFirstPointers.push(allocate(layer.weights.byteLength));
    biasFirstPointers.push(allocate(layer.biases.byteLength));
    weightSecondPointers.push(allocate(layer.weights.byteLength));
    biasSecondPointers.push(allocate(layer.biases.byteLength));
  }
  pointers.sampleIndices = allocate(784 * 2, 2);
  pointers.sampleValues = allocate(784 * 4);
  pointers.inputGradient = allocate(784 * 4);

  if (cursor > wasm.memory.buffer.byteLength) {
    wasm.memory.grow(Math.ceil((cursor - wasm.memory.buffer.byteLength) / 65536));
  }
  const buffer = wasm.memory.buffer;
  new Int32Array(buffer, pointers.inputSizes, layerCount).set(
    sourceModel.map((layer) => layer.inputSize),
  );
  new Int32Array(buffer, pointers.outputSizes, layerCount).set(
    sourceModel.map((layer) => layer.outputSize),
  );
  new Int32Array(buffer, pointers.activationKinds, layerCount).set(
    sourceModel.map((layer) => activationCodes[layer.activation]),
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

  const model = sourceModel.map((layer, index): DenseLayerData => {
    const weights = new Float32Array(buffer, weightPointers[index], layer.weights.length);
    const biases = new Float32Array(buffer, biasPointers[index], layer.biases.length);
    weights.set(layer.weights);
    biases.set(layer.biases);
    new Float32Array(buffer, weightFirstPointers[index], layer.weights.length).fill(0);
    new Float32Array(buffer, biasFirstPointers[index], layer.biases.length).fill(0);
    new Float32Array(buffer, weightSecondPointers[index], layer.weights.length).fill(0);
    new Float32Array(buffer, biasSecondPointers[index], layer.biases.length).fill(0);
    return { ...layer, weights, biases };
  });

  return {
    wasm,
    config: normalizeOptimizer(optimizerConfig),
    pointers,
    model,
    activations: sourceModel.map(
      (layer, index) => new Float32Array(buffer, activationPointers[index], layer.outputSize),
    ),
    preactivations: sourceModel.map(
      (layer, index) => new Float32Array(buffer, preactivationPointers[index], layer.outputSize),
    ),
    deltas: sourceModel.map(
      (layer, index) => new Float32Array(buffer, deltaPointers[index], layer.outputSize),
    ),
    sampleIndices: new Uint16Array(buffer, pointers.sampleIndices, 784),
    sampleValues: new Float32Array(buffer, pointers.sampleValues, 784),
    inputGradient: new Float32Array(buffer, pointers.inputGradient, 784),
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

function forwardWithWasm(runtime: WasmTrainingRuntime, sample: SparseSample) {
  stageSample(runtime, sample);
  const { pointers } = runtime;
  runtime.wasm.forward_sparse(
    runtime.model.length,
    pointers.sampleIndices,
    pointers.sampleValues,
    sample.indices.length,
    pointers.inputSizes,
    pointers.outputSizes,
    pointers.activationKinds,
    pointers.weights,
    pointers.biases,
    pointers.activations,
    pointers.preactivations,
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
  const loss = runtime.wasm.train_sample(
    runtime.model.length,
    pointers.sampleIndices,
    pointers.sampleValues,
    sample.indices.length,
    sample.label,
    pointers.inputSizes,
    pointers.outputSizes,
    pointers.activationKinds,
    pointers.weights,
    pointers.biases,
    pointers.activations,
    pointers.preactivations,
    pointers.deltas,
    pointers.weightFirst,
    pointers.biasFirst,
    pointers.weightSecond,
    pointers.biasSecond,
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
  );
  if (!Number.isFinite(loss)) throw new Error("Zig/Wasm 训练产生了无效损失");
  if (!captureTrace) return loss;

  const probabilities = runtime.activations.at(-1)!;
  let prediction = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index] > probabilities[prediction]) prediction = index;
  }
  return {
    loss,
    activations: [
      denseInput(sample),
      ...runtime.activations.map((values) => Float32Array.from(values)),
    ],
    gradients: [
      Float32Array.from(runtime.inputGradient),
      ...runtime.deltas.map((values) => Float32Array.from(values)),
    ],
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

function cloneWasmModel(model: DenseLayerData[]) {
  return model.map((layer): DenseLayerData => ({
    ...layer,
    weights: Float32Array.from(layer.weights),
    biases: Float32Array.from(layer.biases),
  }));
}

async function train(message: TrainMessage) {
  cancelled = false;
  paused = false;
  pausedAt = null;
  pausedDuration = 0;
  activeModel = null;
  activeSnapshotProgress = null;
  wakeControlWaiters();
  const startedAt = performance.now();
  activeStartedAt = startedAt;
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
  const [datasetBuffer, wasm] = await Promise.all([
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
    ? validateInitialModel(message.layers, message.initialModel)
    : initializeModel(message.layers);
  const runtime = createWasmTrainingRuntime(wasm, sourceModel, message.settings.optimizer);
  activeModel = runtime.model;
  if (!Number.isFinite(message.settings.learningRate) || message.settings.learningRate <= 0) {
    throw new Error("学习率必须是大于 0 的有限数值");
  }
  const random = seededRandom(architectureSeed(message.layers) ^ 0x9e3779b9);
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
          backend: "Zig/Wasm",
        });
      }

      activeSnapshotProgress.epoch = epoch + 1;
      activeSnapshotProgress.sample = sampleIndex + 1;
      activeSnapshotProgress.accuracy = lastAccuracy;
      activeSnapshotProgress.loss = loss / (sampleIndex + 1);

      const shouldYield = captureTrace || performance.now() - lastControlYieldAt >= 32;
      if (shouldYield) {
        if (!(await waitForControl())) {
          activeModel = null;
          activeSnapshotProgress = null;
          activeStartedAt = null;
          postMessage({ type: "cancelled" });
          return;
        }
        lastControlYieldAt = performance.now();
      }
      if (cancelled) {
        activeModel = null;
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
      backend: "Zig/Wasm",
    });
    if (!(await waitForControl())) {
      activeModel = null;
      activeSnapshotProgress = null;
      activeStartedAt = null;
      postMessage({ type: "cancelled" });
      return;
    }
  }

  const accuracy = validateWithWasm(runtime, validation);
  const completedModel = cloneWasmModel(runtime.model);
  const transfer = completedModel.flatMap((layer) => [
    layer.weights.buffer,
    layer.biases.buffer,
  ]);
  postMessage(
    {
      type: "complete",
      model: completedModel,
      accuracy,
      elapsedMs: activeElapsed(startedAt),
      trainingSamples: training.length,
      testSamples: validation.length,
      backend: "Zig/Wasm",
    },
    transfer,
  );
  activeModel = null;
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
    if (paused && activeModel && activeSnapshotProgress && activeStartedAt !== null) {
      postMessage({
        type: "snapshot",
        model: cloneWasmModel(activeModel),
        ...activeSnapshotProgress,
        elapsedMs: activeElapsed(activeStartedAt),
      });
    }
    return;
  }
  train(event.data).catch((error: unknown) => {
    activeModel = null;
    activeSnapshotProgress = null;
    activeStartedAt = null;
    postMessage({
      type: "error",
      message: error instanceof Error ? error.stack ?? error.message : "训练失败",
    });
  });
};

export {};
