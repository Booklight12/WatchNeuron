<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { Activity, BrainCircuit, Cpu, Database, Layers3, Pause, Play, ScanLine, SlidersHorizontal, Zap } from "@lucide/vue";
import ArchitectureEditor from "./components/ArchitectureEditor.vue";
import DatasetManager from "./components/DatasetManager.vue";
import DigitCanvas from "./components/DigitCanvas.vue";
import ModelManager from "./components/ModelManager.vue";
import NetworkCanvas from "./components/NetworkCanvas.vue";
import OptimizerManager from "./components/OptimizerManager.vue";
import PixelPreview from "./components/PixelPreview.vue";
import PredictionPanel from "./components/PredictionPanel.vue";
import PropagationStepper from "./components/PropagationStepper.vue";
import SignalFlowExplorer from "./components/SignalFlowExplorer.vue";
import TrainingPanel from "./components/TrainingPanel.vue";
import { InferenceEngine } from "./lib/engine";
import { activationLabels, buildModel, countParameters, isActivationKind } from "./lib/model";
import {
  architectureLayerSizes,
  convolutionPipeline,
  createDefaultConvolutionConfig,
  fitConvolutionsToLayers,
  fitPoolingsToLayers,
  modelConvolutions,
  modelPoolings,
  normalizeConvolutionConfig,
  normalizePoolingConfig,
  spatialPipeline,
} from "./lib/convolution";
import {
  deleteModelRecords,
  listSavedModels,
  renameModelRecord,
  saveModelRecord,
} from "./lib/modelLibrary";
import type {
  CustomDatasetSample,
  ConvolutionConfig,
  ConvolutionLayerData,
  DatasetSplit,
  HiddenLayer,
  InferenceResult,
  NeuralModel,
  OptimizerConfig,
  OutputHeadKind,
  PoolingConfig,
  PoolingLayerData,
  PropagationDirection,
  SavedModel,
  SavedModelSource,
  SerializedModel,
  TrainingProgress,
  TrainingMode,
  TrainingProfileSettings,
  TrainingProfiles,
  TrainingSettings,
  TrainingTrace,
} from "./types";

const defaultLayers = (): HiddenLayer[] => [
  { id: "layer-1", units: 32, activation: "relu", dropout: 0 },
];

const defaultOptimizer = (): OptimizerConfig => ({
  kind: "sgd",
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  decay: 0.9,
  epsilon: 1e-8,
  weightDecay: 0,
});

const defaultScratchProfile = (): TrainingProfileSettings => ({
  epochs: 10,
  learningRate: 0.018,
  batchSize: 16,
  optimizer: defaultOptimizer(),
});

const defaultFinetuneProfile = (): TrainingProfileSettings => ({
  epochs: 5,
  learningRate: 0.0002,
  batchSize: 16,
  optimizer: { ...defaultOptimizer(), kind: "adam" },
});

function normalizeTrainingProfile(
  value: unknown,
  fallback: TrainingProfileSettings,
): TrainingProfileSettings {
  const source = value && typeof value === "object"
    ? value as Partial<TrainingProfileSettings>
    : {};
  const optimizer = source.optimizer && typeof source.optimizer === "object"
    ? source.optimizer
    : {} as Partial<OptimizerConfig>;
  const epochs = Number(source.epochs);
  const learningRate = Number(source.learningRate);
  const batchSize = Number(source.batchSize);
  const kind = ["sgd", "momentum", "adam", "rmsprop", "adagrad"].includes(String(optimizer.kind))
    ? optimizer.kind as OptimizerConfig["kind"]
    : fallback.optimizer.kind;
  return {
    epochs: Number.isFinite(epochs) ? Math.max(1, Math.floor(epochs)) : fallback.epochs,
    learningRate: Number.isFinite(learningRate) && learningRate > 0
      ? learningRate
      : fallback.learningRate,
    batchSize: Number.isFinite(batchSize) ? Math.max(1, Math.floor(batchSize)) : fallback.batchSize,
    optimizer: {
      kind,
      momentum:
        Number.isFinite(optimizer.momentum) && optimizer.momentum! >= 0 && optimizer.momentum! < 1
          ? optimizer.momentum!
          : fallback.optimizer.momentum,
      beta1:
        Number.isFinite(optimizer.beta1) && optimizer.beta1! > 0 && optimizer.beta1! < 1
          ? optimizer.beta1!
          : fallback.optimizer.beta1,
      beta2:
        Number.isFinite(optimizer.beta2) && optimizer.beta2! > 0 && optimizer.beta2! < 1
          ? optimizer.beta2!
          : fallback.optimizer.beta2,
      decay:
        Number.isFinite(optimizer.decay) && optimizer.decay! > 0 && optimizer.decay! < 1
          ? optimizer.decay!
          : fallback.optimizer.decay,
      epsilon:
        Number.isFinite(optimizer.epsilon) && optimizer.epsilon! > 0
          ? optimizer.epsilon!
          : fallback.optimizer.epsilon,
      weightDecay:
        Number.isFinite(optimizer.weightDecay) && optimizer.weightDecay! >= 0
          ? optimizer.weightDecay!
          : fallback.optimizer.weightDecay,
    },
  };
}

function storedTrainingProfiles(): TrainingProfiles {
  const fallback: TrainingProfiles = {
    mathMode: "fast",
    computeBackend: "wasm",
    scratch: defaultScratchProfile(),
    finetune: defaultFinetuneProfile(),
  };
  try {
    const parsed = JSON.parse(localStorage.getItem("watchneuron-training-settings") ?? "null");
    if (!parsed || typeof parsed !== "object") return fallback;
    const isProfileFormat = parsed.scratch && parsed.finetune;
    return {
      mathMode: parsed.mathMode === "full" ? "full" : "fast",
      computeBackend: parsed.computeBackend === "webgpu" ? "webgpu" : "wasm",
      scratch: normalizeTrainingProfile(
        isProfileFormat ? parsed.scratch : parsed,
        fallback.scratch,
      ),
      finetune: normalizeTrainingProfile(
        isProfileFormat ? parsed.finetune : null,
        fallback.finetune,
      ),
    };
  } catch {
    return fallback;
  }
}

function storedLayers() {
  try {
    const parsed = JSON.parse(localStorage.getItem("watchneuron-architecture") ?? "null");
    if (!Array.isArray(parsed)) return defaultLayers();
    if (parsed.length === 0) {
      return storedConvolutions().length > 0 || storedPoolings().length > 0
        ? []
        : defaultLayers();
    }
    return parsed.map((layer, index) => {
      const units = Number(layer.units);
      return {
        id: String(layer.id ?? `layer-${index + 1}`),
        units: Number.isFinite(units) ? Math.max(1, Math.floor(units)) : 32,
        activation: isActivationKind(layer.activation) ? layer.activation : "relu",
        dropout: Number.isFinite(Number(layer.dropout))
          ? Math.min(0.95, Math.max(0, Number(layer.dropout)))
          : 0,
      };
    }) as HiddenLayer[];
  } catch {
    return defaultLayers();
  }
}

function storedConvolutions() {
  try {
    const stored = JSON.parse(localStorage.getItem("watchneuron-convolutions") ?? "null");
    if (Array.isArray(stored)) return stored.map(normalizeConvolutionConfig);
    const legacy = normalizeConvolutionConfig(
      JSON.parse(localStorage.getItem("watchneuron-convolution") ?? "null"),
    );
    return legacy.enabled ? [legacy] : [];
  } catch {
    return [];
  }
}

function storedPoolings() {
  try {
    const stored = JSON.parse(localStorage.getItem("watchneuron-poolings") ?? "[]");
    return Array.isArray(stored) ? stored.map(normalizePoolingConfig) : [];
  } catch {
    return [];
  }
}

function storedOutputHead(): OutputHeadKind {
  return localStorage.getItem("watchneuron-output-head") === "sigmoid" ? "sigmoid" : "softmax";
}

function convolutionConfigFromLayer(layer: ConvolutionLayerData | null | undefined) {
  if (!layer) return createDefaultConvolutionConfig();
  const kernelLength = layer.kernelSize * layer.kernelSize;
  return normalizeConvolutionConfig({
    id: layer.id,
    enabled: true,
    trainable: layer.trainable !== false,
    initialization: "he",
    position: Number.isFinite(layer.position) ? layer.position : 0,
    order: Number.isFinite(layer.order) ? layer.order : 0,
    filters: layer.filters,
    kernelSize: layer.kernelSize,
    stride: layer.stride,
    padding: layer.padding,
    activation: layer.activation,
    kernels: Array.from({ length: layer.filters }, (_, filter) =>
      Array.from(layer.weights.slice(
        filter * layer.inputChannels * kernelLength,
        filter * layer.inputChannels * kernelLength + kernelLength,
      )),
    ),
  });
}

function poolingConfigFromLayer(layer: PoolingLayerData): PoolingConfig {
  return normalizePoolingConfig({
    id: layer.id,
    enabled: true,
    position: layer.position,
    order: layer.order,
    kind: layer.kind,
    kernelSize: layer.kernelSize,
    stride: layer.stride,
    padding: layer.padding,
  });
}

function convolutionConfigsFromRecord(record: SavedModel) {
  const configs = Array.isArray(record.convolutionConfigs)
    ? record.convolutionConfigs.map(normalizeConvolutionConfig)
    : record.convolutionConfig
      ? [normalizeConvolutionConfig(record.convolutionConfig)]
      : modelConvolutions(record.model).map(convolutionConfigFromLayer);
  return fitConvolutionsToLayers(configs, record.hiddenLayers);
}

function poolingConfigsFromRecord(record: SavedModel) {
  const configs = Array.isArray(record.poolingConfigs)
    ? record.poolingConfigs.map(normalizePoolingConfig)
    : modelPoolings(record.model).map(poolingConfigFromLayer);
  return fitPoolingsToLayers(configs, record.hiddenLayers);
}

function normalizeSavedModelRecord(record: SavedModel): SavedModel {
  const convolutionConfigs = convolutionConfigsFromRecord(record);
  const poolingConfigs = poolingConfigsFromRecord(record);
  const convolutions = modelConvolutions(record.model).map((convolution, index) => ({
    ...convolution,
    id: convolution.id || convolutionConfigs[index]?.id || `conv-model-${index}`,
    trainable: convolutionConfigs[index]?.trainable !== false,
    order: Number.isFinite(convolution.order) ? convolution.order : convolutionConfigs[index]?.order ?? index,
  }));
  return {
    ...record,
    hiddenLayers: record.hiddenLayers.map((layer) => ({
      ...layer,
      dropout: Number.isFinite(layer.dropout) ? Math.min(0.95, Math.max(0, layer.dropout)) : 0,
    })),
    convolutionConfigs,
    poolingConfigs,
    outputHead: record.outputHead === "sigmoid" || record.model.outputHead === "sigmoid" ? "sigmoid" : "softmax",
    model: {
      ...record.model,
      convolutions,
      poolings: modelPoolings(record.model).map((pooling, index) => ({
        ...pooling,
        order: Number.isFinite(pooling.order) ? pooling.order : poolingConfigs[index]?.order ?? index,
      })),
      outputHead: record.outputHead === "sigmoid" || record.model.outputHead === "sigmoid" ? "sigmoid" : "softmax",
    },
  };
}

function architectureSignature(
  layers: HiddenLayer[],
  convolutionConfigs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
  head: OutputHeadKind,
) {
  return JSON.stringify({
    layers: layers.map(({ units, activation, dropout }) => ({ units, activation, dropout })),
    convolutions: convolutionConfigs
      .filter((config) => config.enabled)
      .map(({ position, order, filters, kernelSize, stride, padding, activation, kernels, trainable, initialization }) => ({
        position,
        order,
        filters,
        kernelSize,
        stride,
        padding,
        activation,
        kernels,
        trainable,
        initialization,
      })),
    poolings: poolingConfigs.map(({ position, order, kind, kernelSize, stride, padding }) => ({
      position, order, kind, kernelSize, stride, padding,
    })),
    outputHead: head,
  });
}

function architectureMatchesRecord(record: SavedModel) {
  return architectureSignature(hiddenLayers.value, convolutions.value, poolings.value, outputHead.value) ===
    architectureSignature(
      record.hiddenLayers,
      convolutionConfigsFromRecord(record),
      poolingConfigsFromRecord(record),
      record.outputHead === "sigmoid" ? "sigmoid" : "softmax",
    );
}

function trainabilityOnlyChanged(
  configs: ConvolutionConfig[],
  previous: ConvolutionConfig[] | undefined,
) {
  if (!previous || configs.length !== previous.length) return false;
  const withoutTrainability = (values: ConvolutionConfig[]) => JSON.stringify(
    values.map(({ trainable: _trainable, ...config }) => config),
  );
  return withoutTrainability(configs) === withoutTrainability(previous) &&
    configs.some((config, index) => config.trainable !== previous[index].trainable);
}

function storedCustomSamples(): CustomDatasetSample[] {
  try {
    const parsed = JSON.parse(localStorage.getItem("watchneuron-custom-samples") ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((sample): sample is CustomDatasetSample => {
      if (
        typeof sample?.id !== "string" ||
        !Number.isInteger(sample.label) ||
        sample.label < 0 ||
        sample.label > 9 ||
        (sample.split !== "training" && sample.split !== "test") ||
        !Array.isArray(sample.indices) ||
        !Array.isArray(sample.values) ||
        sample.indices.length !== sample.values.length
      ) {
        return false;
      }
      return sample.indices.every(
        (index: unknown, position: number) =>
          Number.isInteger(index) &&
          Number(index) >= 0 &&
          Number(index) < 784 &&
          Number.isFinite(sample.values[position]) &&
          sample.values[position] >= 0 &&
          sample.values[position] <= 1,
      );
    });
  } catch {
    return [];
  }
}

function storedMnistEnabled() {
  return localStorage.getItem("watchneuron-mnist-enabled") !== "false";
}

const engine = new InferenceEngine();
type AppView = "lab" | "signals" | "samples" | "models" | "optimizer";
type ModelSaveState = "idle" | "saving" | "saved" | "error";

function viewFromHash(): AppView {
  if (window.location.hash === "#signals") return "signals";
  if (window.location.hash === "#samples") return "samples";
  if (window.location.hash === "#models") return "models";
  if (window.location.hash === "#optimizer") return "optimizer";
  return "lab";
}

const activeView = ref<AppView>(viewFromHash());
const digitCanvas = ref<InstanceType<typeof DigitCanvas> | null>(null);
const serialized = shallowRef<SerializedModel | null>(null);
const bundledDefaultModel = shallowRef<SavedModel | null>(null);
const trainedModel = shallowRef<NeuralModel | null>(null);
const trainedEpochCount = ref(0);
const trainedModelAccuracy = ref<number | null>(null);
const hadStoredArchitecture = [
  "watchneuron-architecture",
  "watchneuron-convolutions",
  "watchneuron-convolution",
  "watchneuron-poolings",
  "watchneuron-output-head",
].some((key) => localStorage.getItem(key) !== null);
const initialHiddenLayers = storedLayers();
const hiddenLayers = ref<HiddenLayer[]>(initialHiddenLayers);
const convolutions = ref<ConvolutionConfig[]>(
  fitConvolutionsToLayers(storedConvolutions(), initialHiddenLayers),
);
const poolings = ref<PoolingConfig[]>(fitPoolingsToLayers(storedPoolings(), initialHiddenLayers));
const outputHead = ref<OutputHeadKind>(storedOutputHead());
const customSamples = ref<CustomDatasetSample[]>(storedCustomSamples());
const mnistEnabled = ref(storedMnistEnabled());
const savedModels = ref<SavedModel[]>([]);
const modelLibraryLoading = ref(true);
const modelLibraryDeleting = ref(false);
const modelSaveState = ref<ModelSaveState>("idle");
const activeTrainingMode = ref<TrainingMode>("scratch");
const optimizerViewMode = ref<TrainingMode>("scratch");
const inputPixels = shallowRef<Float32Array>(new Float32Array(784));
const inputEnergy = ref(0);
const engineBackend = ref<"Wasm SIMD" | "Wasm" | "JavaScript">("JavaScript");
const loading = ref(true);
const animated = ref(true);
const selectedLayer = ref(1);
const stepMode = ref(false);
const stepDirection = ref<PropagationDirection>("forward");
const sampleCursor = ref(8);
const trainingProfiles = ref<TrainingProfiles>(storedTrainingProfiles());
const trainingProgress = ref<TrainingProgress>({
  phase: "idle",
  epoch: 0,
  epochs: trainingProfiles.value.scratch.epochs,
  accuracy: 0,
  loss: 0,
  elapsedMs: 0,
});
const selectedComputeBackendLabel = computed(() => {
  if (trainingProfiles.value.computeBackend === "webgpu") {
    return `WebGPU${trainingProgress.value.adapter ? ` · ${trainingProgress.value.adapter}` : ""}`;
  }
  return engineBackend.value;
});
const trainingTrace = shallowRef<TrainingTrace | null>(null);
let trainingWorker: Worker | null = null;
let modelSaveOperation = 0;
const inference = shallowRef<InferenceResult>({
  probabilities: Array.from({ length: 10 }, () => 0),
  activations: [Array.from({ length: 784 }, () => 0), [], []],
  latencyMs: 0,
  backend: "JavaScript",
});

const generatedModel = computed(() =>
  serialized.value
    ? buildModel(serialized.value, hiddenLayers.value, convolutions.value, poolings.value, outputHead.value)
    : null,
);
const model = computed(() => trainedModel.value ?? generatedModel.value);
const modelStatus = computed(() => {
  if (trainedModel.value) return "trained" as const;
  return model.value?.calibrated ? ("calibrated" as const) : ("experimental" as const);
});
const displayedModelAccuracy = computed(
  () => trainedModelAccuracy.value ?? serialized.value?.accuracy ?? null,
);
const parameterCount = computed(() => (model.value ? countParameters(model.value) : 0));
const customTrainingCount = computed(
  () => customSamples.value.filter((sample) => sample.split === "training").length,
);
const customTestCount = computed(
  () => customSamples.value.filter((sample) => sample.split === "test").length,
);
const datasetLocked = computed(
  () => ["loading", "training", "paused"].includes(trainingProgress.value.phase),
);
const convolutionEntries = computed(() => convolutionPipeline(hiddenLayers.value, convolutions.value, poolings.value));
const spatialEntries = computed(() => spatialPipeline(hiddenLayers.value, convolutions.value, poolings.value));
const layerSizes = computed(() => architectureLayerSizes(hiddenLayers.value, convolutions.value, poolings.value));
const layerNames = computed(() => {
  const names = ["输入层"];
  for (let position = 0; position <= hiddenLayers.value.length; position++) {
    for (const entry of spatialEntries.value.filter((entry) => entry.config.position === position)) {
      if (entry.kind === "conv") names.push(`卷积层 · ${entry.config.filters} 核 · ${activationLabels[entry.config.activation]}`);
      else names.push(entry.config.kind === "globalAverage" ? "全局平均池化 · GAP" : `${entry.config.kind === "max" ? "最大" : "平均"}池化层`);
    }
    if (position < hiddenLayers.value.length) {
      const layer = hiddenLayers.value[position];
      names.push(`隐藏层 ${position + 1} · ${activationLabels[layer.activation]}`);
    }
  }
  names.push(`输出层 · ${outputHead.value === "sigmoid" ? "Sigmoid" : "Softmax"}`);
  return names;
});
const trainingFlowActive = computed(
  () =>
    trainingTrace.value !== null &&
    ["loading", "training", "paused"].includes(trainingProgress.value.phase),
);
const flowActivations = computed(() =>
  trainingFlowActive.value && trainingTrace.value
    ? trainingTrace.value.activations
    : inference.value.activations,
);
const flowGradients = computed(() =>
  trainingFlowActive.value && trainingTrace.value ? trainingTrace.value.gradients : [],
);
const displayedInputPixels = computed(() =>
  trainingFlowActive.value && trainingTrace.value
    ? Float32Array.from(trainingTrace.value.activations[0] ?? [])
    : inputPixels.value,
);
const displayedInputEnergy = computed(() => {
  const pixels = displayedInputPixels.value;
  return pixels.length
    ? pixels.reduce((sum, value) => sum + value, 0) / pixels.length
    : 0;
});
const displayedProbabilities = computed(() => {
  if (
    stepMode.value &&
    stepDirection.value === "forward" &&
    selectedLayer.value < layerSizes.value.length - 1
  ) {
    return Array.from({ length: 10 }, () => 0);
  }
  if (!trainingFlowActive.value || !trainingTrace.value) return inference.value.probabilities;
  return trainingTrace.value.activations.at(-1) ?? inference.value.probabilities;
});
const selectedValues = computed(() => flowActivations.value[selectedLayer.value] ?? []);
const selectedGradients = computed(() => flowGradients.value[selectedLayer.value] ?? []);
const selectedStats = computed(() => {
  const values = selectedValues.value;
  if (!values.length) return { mean: 0, peak: 0, active: 0, gradientMean: 0 };
  const absolute = values.map(Math.abs);
  const absoluteGradients = selectedGradients.value.map(Math.abs);
  return {
    mean: absolute.reduce((sum, value) => sum + value, 0) / absolute.length,
    peak: Math.max(...absolute),
    active: (absolute.filter((value) => value > 0.05).length / absolute.length) * 100,
    gradientMean: absoluteGradients.length
      ? absoluteGradients.reduce((sum, value) => sum + value, 0) / absoluteGradients.length
      : 0,
  };
});
const selectedLayerName = computed(() => {
  return layerNames.value[selectedLayer.value] ?? "网络层";
});

function formatSignal(value: number) {
  if (value === 0) return "0.000";
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

function runInference() {
  if (!model.value) return;
  if (inputEnergy.value < 0.001) {
    inference.value = {
      probabilities: Array.from({ length: 10 }, () => 0),
      activations: layerSizes.value.map((size) => Array.from({ length: size }, () => 0)),
      latencyMs: 0,
      backend: engine.backend,
    };
    return;
  }
  inference.value = engine.run(model.value, inputPixels.value);
}

function handleInput(pixels: Float32Array, energy: number) {
  inputPixels.value = pixels;
  inputEnergy.value = energy;
  runInference();
}

function toggleStepMode() {
  stepMode.value = !stepMode.value;
  if (stepMode.value) {
    stepDirection.value = "forward";
    selectedLayer.value = 0;
  }
}

function updateStepDirection(direction: PropagationDirection) {
  stepDirection.value = direction;
}

function updateStepLayer(layer: number) {
  selectedLayer.value = Math.max(0, Math.min(layerSizes.value.length - 1, layer));
}

function loadNextSample() {
  if (!serialized.value) return;
  sampleCursor.value = (sampleCursor.value + 3) % 10;
  digitCanvas.value?.loadSample(serialized.value.samples[sampleCursor.value]);
}

function addCustomSample(label: number, split: DatasetSplit, pixels: Float32Array) {
  const indices: number[] = [];
  const values: number[] = [];
  for (let index = 0; index < pixels.length; index++) {
    if (pixels[index] <= 0.001) continue;
    indices.push(index);
    values.push(Math.round(pixels[index] * 10000) / 10000);
  }
  customSamples.value.push({
    id: `sample-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    label,
    split,
    indices,
    values,
  });
}

function clearCustomSamples() {
  customSamples.value = [];
}

function updateCustomSample(
  id: string,
  patch: { label?: number; split?: DatasetSplit },
) {
  if (datasetLocked.value) return;
  customSamples.value = customSamples.value.map((sample) => {
    if (sample.id !== id) return sample;
    const label = Number.isInteger(patch.label) && patch.label! >= 0 && patch.label! <= 9
      ? patch.label!
      : sample.label;
    const split = patch.split === "training" || patch.split === "test"
      ? patch.split
      : sample.split;
    return { ...sample, label, split };
  });
}

function removeCustomSamples(ids: string[]) {
  if (datasetLocked.value || ids.length === 0) return;
  const removed = new Set(ids);
  customSamples.value = customSamples.value.filter((sample) => !removed.has(sample.id));
}

function cloneNeuralModel(model: NeuralModel): NeuralModel {
  return {
    calibrated: model.calibrated,
    trained: model.trained,
    outputHead: model.outputHead === "sigmoid" ? "sigmoid" : "softmax",
    convolutions: modelConvolutions(model).map((convolution) => ({
      ...convolution,
      weights: Float32Array.from(convolution.weights),
      biases: Float32Array.from(convolution.biases),
    })),
    poolings: modelPoolings(model).map((pooling) => ({ ...pooling })),
    layers: model.layers.map((layer) => ({
      ...layer,
      weights: Float32Array.from(layer.weights),
      biases: Float32Array.from(layer.biases),
    })),
  };
}

async function applySavedModel(record: SavedModel) {
  hiddenLayers.value = record.hiddenLayers.map((layer) => ({
    ...layer,
    dropout: Number.isFinite(layer.dropout) ? Math.min(0.95, Math.max(0, layer.dropout)) : 0,
  }));
  convolutions.value = convolutionConfigsFromRecord(record);
  poolings.value = poolingConfigsFromRecord(record);
  outputHead.value = record.outputHead === "sigmoid" || record.model.outputHead === "sigmoid"
    ? "sigmoid"
    : "softmax";
  await nextTick();
  trainedModel.value = cloneNeuralModel(record.model);
  trainedEpochCount.value = Math.max(0, Math.floor(record.progress.epoch));
  trainedModelAccuracy.value = Number.isFinite(record.progress.accuracy)
    ? record.progress.accuracy
    : null;
  activeTrainingMode.value = "finetune";
  trainingProgress.value = {
    phase: "idle",
    epoch: 0,
    epochs: trainingProfiles.value.finetune.epochs,
    accuracy: 0,
    loss: 0,
    elapsedMs: 0,
  };
  selectedLayer.value = Math.min(selectedLayer.value, layerSizes.value.length - 1);
}

function modelRecordName(
  source: SavedModelSource,
  trainingMode: TrainingMode,
  createdAt: number,
) {
  const timestamp = new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(createdAt);
  const prefix = source === "complete"
    ? trainingMode === "finetune" ? "微调模型" : "完整训练"
    : trainingMode === "finetune" ? "微调快照" : "暂停快照";
  return `${prefix} · ${timestamp}`;
}

async function persistTrainingModel(
  model: NeuralModel,
  source: SavedModelSource,
  progress: SavedModel["progress"],
  trainingMode: TrainingMode,
) {
  const operation = ++modelSaveOperation;
  modelSaveState.value = "saving";
  const createdAt = Date.now();
  const record: SavedModel = {
    id: `model-${createdAt.toString(36)}-${crypto.randomUUID().slice(0, 8)}`,
    name: modelRecordName(source, trainingMode, createdAt),
    createdAt,
    source,
    trainingMode,
    hiddenLayers: hiddenLayers.value.map((layer) => ({ ...layer })),
    convolutionConfigs: convolutions.value.map((convolution) => ({
      ...convolution,
      kernels: convolution.kernels.map((kernel) => [...kernel]),
    })),
    poolingConfigs: poolings.value.map((pooling) => ({ ...pooling })),
    outputHead: outputHead.value,
    model,
    progress: { ...progress },
  };
  try {
    await saveModelRecord(record);
    savedModels.value = [record, ...savedModels.value];
    if (operation === modelSaveOperation) modelSaveState.value = "saved";
  } catch {
    if (operation === modelSaveOperation) modelSaveState.value = "error";
  }
}

async function renameSavedModel(id: string, name: string) {
  try {
    await renameModelRecord(id, name);
    savedModels.value = savedModels.value.map((model) =>
      model.id === id ? { ...model, name } : model,
    );
  } catch {
    // Keep the existing model name if IndexedDB rejects the update.
  }
}

async function removeSavedModels(ids: string[]) {
  const availableIds = new Set(savedModels.value.map((model) => model.id));
  const targetIds = [...new Set(ids)].filter((id) => availableIds.has(id));
  if (modelLibraryDeleting.value || !targetIds.length) return;

  modelLibraryDeleting.value = true;
  try {
    await deleteModelRecords(targetIds);
    const removedIds = new Set(targetIds);
    savedModels.value = savedModels.value.filter((model) => !removedIds.has(model.id));
  } catch {
    // Keep every model visible when the atomic IndexedDB transaction fails.
  } finally {
    modelLibraryDeleting.value = false;
  }
}

async function loadSavedModel(record: SavedModel) {
  if (trainingWorker) cancelTraining();
  trainingTrace.value = null;
  modelSaveOperation++;
  modelSaveState.value = "idle";
  await applySavedModel(record);
  navigateTo("lab");
  runInference();
}

function navigateTo(view: AppView) {
  activeView.value = view;
  const hash = view === "signals"
    ? "#signals"
    : view === "samples"
      ? "#samples"
    : view === "models"
      ? "#models"
      : view === "optimizer"
        ? "#optimizer"
        : "#lab";
  if (window.location.hash !== hash) window.location.hash = hash;
  window.scrollTo({ top: 0 });
}

function syncViewFromHash() {
  activeView.value = viewFromHash();
}

function openOptimizer(mode: TrainingMode = "scratch") {
  optimizerViewMode.value = mode;
  navigateTo("optimizer");
}

function updateLayers(layers: HiddenLayer[]) {
  const removedIndex = hiddenLayers.value.findIndex(
    (layer) => !layers.some((candidate) => candidate.id === layer.id),
  );
  hiddenLayers.value = layers;
  const shifted = removedIndex < 0
    ? convolutions.value
    : convolutions.value.map((convolution) => ({
        ...convolution,
        position: convolution.position > removedIndex
          ? convolution.position - 1
          : convolution.position,
      }));
  convolutions.value = fitConvolutionsToLayers(shifted, layers);
  const shiftedPoolings = removedIndex < 0
    ? poolings.value
    : poolings.value.map((pooling) => ({
        ...pooling,
        position: pooling.position > removedIndex ? pooling.position - 1 : pooling.position,
      }));
  poolings.value = fitPoolingsToLayers(shiftedPoolings, layers);
}

function updateConvolutions(configs: ConvolutionConfig[]) {
  convolutions.value = fitConvolutionsToLayers(configs, hiddenLayers.value);
}

function updatePoolings(configs: PoolingConfig[]) {
  poolings.value = fitPoolingsToLayers(configs, hiddenLayers.value);
}

function reorderArchitecture(
  layers: HiddenLayer[],
  configs: ConvolutionConfig[],
  poolingConfigs: PoolingConfig[],
) {
  hiddenLayers.value = layers;
  convolutions.value = fitConvolutionsToLayers(configs, layers);
  poolings.value = fitPoolingsToLayers(poolingConfigs, layers);
}

function updateOutputHead(head: OutputHeadKind) {
  outputHead.value = head === "sigmoid" ? "sigmoid" : "softmax";
}

function updateTrainingSettings(mode: TrainingMode, settings: TrainingSettings) {
  const fallback = mode === "finetune" ? defaultFinetuneProfile() : defaultScratchProfile();
  trainingProfiles.value = {
    ...trainingProfiles.value,
    mathMode: settings.mathMode === "full" ? "full" : "fast",
    computeBackend: settings.computeBackend === "webgpu" ? "webgpu" : "wasm",
    [mode]: normalizeTrainingProfile(settings, fallback),
  };
  if (trainingProgress.value.phase === "idle") {
    trainingProgress.value.epochs = trainingProfiles.value[mode].epochs;
  }
}

function startTraining(mode: TrainingMode = "scratch") {
  if (mode === "finetune" && !trainedModel.value) return;
  if (!mnistEnabled.value && customTrainingCount.value === 0) {
    trainingProgress.value = {
      ...trainingProgress.value,
      phase: "error",
      message: "MNIST 已关闭，请先加入至少一个自定义训练样本",
    };
    return;
  }
  const baseEpochs = mode === "finetune" ? trainedEpochCount.value : 0;
  const initialModel = mode === "finetune" ? cloneNeuralModel(trainedModel.value!) : undefined;
  const settings: TrainingSettings = {
    ...trainingProfiles.value[mode],
    mathMode: trainingProfiles.value.mathMode,
    computeBackend: trainingProfiles.value.computeBackend,
    optimizer: { ...trainingProfiles.value[mode].optimizer },
  };
  trainingWorker?.terminate();
  trainingTrace.value = null;
  activeTrainingMode.value = mode;
  modelSaveOperation++;
  modelSaveState.value = "idle";
  trainingProgress.value = {
    phase: "loading",
    epoch: 0,
    epochs: settings.epochs,
    accuracy: 0,
    loss: 0,
    elapsedMs: 0,
  };
  const worker = new Worker(
    new URL("./workers/training.worker.ts", import.meta.url),
    { type: "module" },
  );
  trainingWorker = worker;
  worker.onmessage = (event: MessageEvent) => {
    const message = event.data;
    if (message.type === "trace") {
      trainingTrace.value = {
        id: Number.isFinite(message.id) ? message.id : 0,
        activations: message.activations.map((layer: ArrayLike<number>) => Array.from(layer)),
        gradients: message.gradients.map((layer: ArrayLike<number>) => Array.from(layer)),
        convolutionWeights: (message.convolutionWeights ?? []).map(
          (weights: ArrayLike<number>) => Array.from(weights),
        ),
        convolutionBiases: (message.convolutionBiases ?? []).map(
          (biases: ArrayLike<number>) => Array.from(biases),
        ),
        epoch: message.epoch,
        sample: message.sample,
        samples: message.samples,
        label: message.label,
        prediction: message.prediction,
        loss: message.loss,
      };
      trainingProgress.value.epoch = message.epoch;
      if (trainingProgress.value.phase === "loading") {
        trainingProgress.value.phase = "training";
      }
      return;
    }
    if (message.type === "progress") {
      trainingProgress.value = {
        ...trainingProgress.value,
        ...message,
        epochs: message.epochs ?? settings.epochs,
      };
      return;
    }
    if (message.type === "paused") {
      if (
        trainingTrace.value &&
        Number.isFinite(message.traceId) &&
        message.traceId !== trainingTrace.value.id
      ) {
        return;
      }
      trainingProgress.value.phase = "paused";
      return;
    }
    if (message.type === "resumed") {
      trainingProgress.value.phase = "training";
      modelSaveOperation++;
      modelSaveState.value = "idle";
      return;
    }
    if (message.type === "snapshot") {
      const snapshotModel = cloneNeuralModel(message.model as NeuralModel);
      void persistTrainingModel(snapshotModel, "paused", {
        epoch: baseEpochs + message.epoch,
        epochs: baseEpochs + message.epochs,
        accuracy: message.accuracy,
        loss: message.loss,
        elapsedMs: message.elapsedMs,
      }, mode);
      return;
    }
    if (message.type === "complete") {
      trainingTrace.value = null;
      const completedModel = cloneNeuralModel(message.model as NeuralModel);
      trainedModel.value = completedModel;
      trainedModelAccuracy.value = Number.isFinite(message.accuracy) ? message.accuracy : null;
      const cumulativeEpochs = baseEpochs + settings.epochs;
      trainedEpochCount.value = cumulativeEpochs;
      trainingProgress.value = {
        ...trainingProgress.value,
        phase: "complete",
        epoch: settings.epochs,
        epochs: settings.epochs,
        accuracy: message.accuracy,
        elapsedMs: message.elapsedMs,
      };
      void persistTrainingModel(completedModel, "complete", {
        epoch: cumulativeEpochs,
        epochs: cumulativeEpochs,
        accuracy: message.accuracy,
        loss: trainingProgress.value.loss,
        elapsedMs: message.elapsedMs,
      }, mode);
      trainingWorker?.terminate();
      trainingWorker = null;
      nextTick(runInference);
      return;
    }
    if (message.type === "cancelled") {
      trainingTrace.value = null;
      trainingProgress.value.phase = "cancelled";
      trainingWorker?.terminate();
      trainingWorker = null;
      return;
    }
    if (message.type === "error") {
      trainingTrace.value = null;
      trainingProgress.value = {
        ...trainingProgress.value,
        phase: "error",
        message: message.message,
      };
      trainingWorker?.terminate();
      trainingWorker = null;
    }
  };
  worker.onerror = () => {
    trainingTrace.value = null;
    trainingProgress.value = {
      ...trainingProgress.value,
      phase: "error",
      message: "训练线程启动失败",
    };
    worker.terminate();
    trainingWorker = null;
  };
  try {
    worker.postMessage({
      type: "train",
      datasetUrl: `${import.meta.env.BASE_URL}mnist-training.bin`,
      mnistEnabled: mnistEnabled.value,
      layers: hiddenLayers.value.map((layer) => ({ ...layer })),
      convolutions: convolutions.value.map((convolution) => ({
        ...convolution,
        kernels: convolution.kernels.map((kernel) => [...kernel]),
      })),
      poolings: poolings.value.map((pooling) => ({ ...pooling })),
      outputHead: outputHead.value,
      settings,
      initialModel,
      customSamples: customSamples.value.map((sample) => ({
        ...sample,
        indices: [...sample.indices],
        values: [...sample.values],
      })),
    });
  } catch {
    worker.terminate();
    trainingWorker = null;
    trainingProgress.value = {
      ...trainingProgress.value,
      phase: "error",
      message: "训练参数无法发送到 Worker",
    };
  }
}

function startFineTuning() {
  startTraining("finetune");
}

function cancelTraining() {
  trainingWorker?.terminate();
  trainingWorker = null;
  trainingTrace.value = null;
  trainingProgress.value.phase = "cancelled";
}

function pauseTraining() {
  if (trainingProgress.value.phase !== "training") return;
  trainingWorker?.postMessage({ type: "pause" });
}

function resumeTraining() {
  if (trainingProgress.value.phase !== "paused") return;
  trainingWorker?.postMessage({ type: "resume" });
}

function savePausedSnapshot() {
  if (
    trainingProgress.value.phase !== "paused" ||
    modelSaveState.value === "saving" ||
    modelSaveState.value === "saved"
  ) {
    return;
  }
  modelSaveState.value = "saving";
  trainingWorker?.postMessage({ type: "snapshot" });
}

async function resetArchitecture() {
  if (trainingWorker) cancelTraining();
  if (bundledDefaultModel.value) {
    trainingTrace.value = null;
    await applySavedModel(bundledDefaultModel.value);
    selectedLayer.value = 1;
    runInference();
    return;
  }
  trainedModel.value = null;
  trainedEpochCount.value = 0;
  trainedModelAccuracy.value = null;
  hiddenLayers.value = defaultLayers();
  convolutions.value = [];
  poolings.value = [];
  outputHead.value = "softmax";
  selectedLayer.value = 1;
}

watch(
  hiddenLayers,
  (layers) => {
    if (trainingWorker) cancelTraining();
    trainingTrace.value = null;
    trainedModel.value = null;
    trainedEpochCount.value = 0;
    trainedModelAccuracy.value = null;
    trainingProgress.value = {
      phase: "idle",
      epoch: 0,
      epochs: trainingProfiles.value.scratch.epochs,
      accuracy: 0,
      loss: 0,
      elapsedMs: 0,
    };
    localStorage.setItem("watchneuron-architecture", JSON.stringify(layers));
    selectedLayer.value = Math.min(selectedLayer.value, layerSizes.value.length - 1);
    nextTick(runInference);
  },
  { deep: true },
);

watch(
  convolutions,
  (configs, previous) => {
    const preservesLoadedModel = trainabilityOnlyChanged(configs, previous);
    if (trainingWorker) cancelTraining();
    trainingTrace.value = null;
    if (preservesLoadedModel && trainedModel.value) {
      const currentModel = cloneNeuralModel(trainedModel.value);
      currentModel.convolutions = modelConvolutions(currentModel).map((convolution) => ({
        ...convolution,
        trainable: configs.find((config) => config.id === convolution.id)?.trainable !== false,
      }));
      trainedModel.value = currentModel;
    } else if (!preservesLoadedModel) {
      trainedModel.value = null;
      trainedEpochCount.value = 0;
      trainedModelAccuracy.value = null;
    }
    trainingProgress.value = {
      phase: "idle",
      epoch: 0,
      epochs: trainingProfiles.value.scratch.epochs,
      accuracy: 0,
      loss: 0,
      elapsedMs: 0,
    };
    localStorage.setItem("watchneuron-convolutions", JSON.stringify(configs));
    selectedLayer.value = Math.min(selectedLayer.value, layerSizes.value.length - 1);
    nextTick(runInference);
  },
  { deep: true },
);

watch(
  poolings,
  (configs) => {
    if (trainingWorker) cancelTraining();
    trainingTrace.value = null;
    trainedModel.value = null;
    trainedEpochCount.value = 0;
    trainedModelAccuracy.value = null;
    trainingProgress.value = {
      phase: "idle",
      epoch: 0,
      epochs: trainingProfiles.value.scratch.epochs,
      accuracy: 0,
      loss: 0,
      elapsedMs: 0,
    };
    localStorage.setItem("watchneuron-poolings", JSON.stringify(configs));
    selectedLayer.value = Math.min(selectedLayer.value, layerSizes.value.length - 1);
    nextTick(runInference);
  },
  { deep: true },
);

watch(outputHead, (head) => {
  if (trainingWorker) cancelTraining();
  trainingTrace.value = null;
  trainedModel.value = null;
  trainedEpochCount.value = 0;
  trainedModelAccuracy.value = null;
  localStorage.setItem("watchneuron-output-head", head);
  nextTick(runInference);
});

watch(
  customSamples,
  (samples) => {
    try {
      localStorage.setItem("watchneuron-custom-samples", JSON.stringify(samples));
    } catch {
      // Samples remain available for the current session if browser storage is full.
    }
  },
  { deep: true },
);

watch(mnistEnabled, (enabled) => {
  localStorage.setItem("watchneuron-mnist-enabled", String(enabled));
});

watch(
  trainingProfiles,
  (profiles) => {
    localStorage.setItem("watchneuron-training-settings", JSON.stringify(profiles));
  },
  { deep: true },
);

watch(
  () => trainingProfiles.value.mathMode,
  (mode) => {
    engine.setMathMode(mode);
    nextTick(runInference);
  },
);

watch(trainingFlowActive, (active) => {
  if (!active && stepDirection.value === "backward") {
    stepDirection.value = "forward";
    selectedLayer.value = 0;
  }
});

async function loadModelLibrary() {
  try {
    savedModels.value = (await listSavedModels()).map(normalizeSavedModelRecord);
  } catch {
    savedModels.value = [];
  } finally {
    modelLibraryLoading.value = false;
  }
}

onMounted(async () => {
  window.addEventListener("hashchange", syncViewFromHash);
  try {
    const [modelResponse, bundledResponse] = await Promise.all([
      fetch(`${import.meta.env.BASE_URL}model.json`),
      fetch(`${import.meta.env.BASE_URL}default-model.json`),
      engine.initialize(),
      loadModelLibrary(),
    ]);
    serialized.value = (await modelResponse.json()) as SerializedModel;
    if (bundledResponse.ok) {
      const record = normalizeSavedModelRecord((await bundledResponse.json()) as SavedModel);
      bundledDefaultModel.value = record;
      if (!hadStoredArchitecture || architectureMatchesRecord(record)) {
        await applySavedModel(record);
      }
    }
    engine.setMathMode(trainingProfiles.value.mathMode);
    engineBackend.value = engine.backend;
    await nextTick();
    digitCanvas.value?.loadSample(serialized.value.samples[sampleCursor.value]);
  } finally {
    loading.value = false;
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("hashchange", syncViewFromHash);
  trainingWorker?.terminate();
});
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-lockup">
        <span class="brand-mark" aria-hidden="true">
          <i v-for="index in 9" :key="index" />
        </span>
        <div>
          <strong>WatchNeuron</strong>
          <span>{{ activeView === "lab" ? "数字识别实验台" : activeView === "signals" ? "逐层信号检查" : activeView === "samples" ? "自定义样本库" : activeView === "models" ? "本地模型库" : "优化器配置" }}</span>
        </div>
      </div>

      <div class="topbar-tools">
        <nav class="app-view-tabs" aria-label="应用页面">
          <button
            type="button"
            title="实验台"
            :class="{ active: activeView === 'lab' }"
            :aria-current="activeView === 'lab' ? 'page' : undefined"
            @click="navigateTo('lab')"
          >
            <Activity :size="15" />
            <span>实验台</span>
          </button>
          <button
            type="button"
            title="信号流"
            :class="{ active: activeView === 'signals' }"
            :aria-current="activeView === 'signals' ? 'page' : undefined"
            @click="navigateTo('signals')"
          >
            <ScanLine :size="15" />
            <span>信号流</span>
          </button>
          <button
            type="button"
            title="样本库"
            :class="{ active: activeView === 'samples' }"
            :aria-current="activeView === 'samples' ? 'page' : undefined"
            @click="navigateTo('samples')"
          >
            <Database :size="15" />
            <span>样本库</span>
            <b>{{ customSamples.length }}</b>
          </button>
          <button
            type="button"
            title="模型库"
            :class="{ active: activeView === 'models' }"
            :aria-current="activeView === 'models' ? 'page' : undefined"
            @click="navigateTo('models')"
          >
            <BrainCircuit :size="15" />
            <span>模型库</span>
            <b>{{ savedModels.length }}</b>
          </button>
          <button
            type="button"
            title="优化器配置"
            :class="{ active: activeView === 'optimizer' }"
            :aria-current="activeView === 'optimizer' ? 'page' : undefined"
            @click="navigateTo('optimizer')"
          >
            <SlidersHorizontal :size="15" />
            <span>优化器</span>
          </button>
        </nav>

        <div class="runtime-status">
          <span class="status-item">
            <Cpu :size="15" />
            {{ selectedComputeBackendLabel }} · {{ trainingProfiles.mathMode === "full" ? "完整" : "快速" }}
            <i class="status-dot" />
          </span>
          <span class="status-item latency-item">
            <Zap :size="15" />
            {{ inference.latencyMs.toFixed(2) }} ms
          </span>
        </div>
      </div>
    </header>

    <main v-show="activeView === 'lab'" class="workspace" :class="{ 'is-loading': loading }">
      <section class="input-section" aria-labelledby="input-heading">
        <div class="section-heading">
          <div>
            <span class="eyebrow">01 · INPUT</span>
            <h1 id="input-heading">手写输入</h1>
          </div>
          <span class="dimension-label">280 × 280</span>
        </div>

        <DigitCanvas
          ref="digitCanvas"
          :training-pixels="trainingFlowActive ? trainingTrace?.activations[0] ?? null : null"
          :training-label="trainingFlowActive ? trainingTrace?.label ?? null : null"
          :custom-training-count="customTrainingCount"
          :custom-test-count="customTestCount"
          :dataset-locked="datasetLocked"
          @input="handleInput"
          @sample="loadNextSample"
          @add-sample="addCustomSample"
          @clear-samples="clearCustomSamples"
        />

        <div class="input-metadata">
          <PixelPreview :pixels="displayedInputPixels" />
          <div>
            <b>{{ trainingFlowActive ? "训练采样输入" : "标准化输入" }}</b>
            <span>28 × 28 灰度</span>
          </div>
          <div class="energy-meter">
            <span>能量</span>
            <b>{{ (displayedInputEnergy * 100).toFixed(1) }}%</b>
          </div>
        </div>

        <div class="model-metadata">
          <span>{{ trainingFlowActive ? `训练样本 · 标签 ${trainingTrace?.label ?? "-"}` : "MNIST" }}</span>
          <span>
            {{ trainingFlowActive && trainingTrace
              ? `E${trainingTrace.epoch} · ${trainingTrace.sample.toLocaleString()}/${trainingTrace.samples.toLocaleString()}`
              : displayedModelAccuracy !== null
                ? `${(displayedModelAccuracy * 100).toFixed(1)}% 测试准确率`
                : "载入中" }}
          </span>
        </div>
      </section>

      <section class="flow-section" aria-labelledby="flow-heading">
        <div class="section-heading">
          <div>
            <span class="eyebrow">{{ trainingFlowActive ? "02 · TRAINING TRACE" : "02 · FORWARD PASS" }}</span>
            <h2 id="flow-heading">{{ trainingFlowActive ? "训练信号流" : "神经信号流" }}</h2>
          </div>
          <div class="flow-heading-actions">
            <button
              class="icon-button step-mode-toggle"
              :class="{ active: stepMode }"
              type="button"
              :title="stepMode ? '退出逐层步进' : '进入逐层步进'"
              :aria-label="stepMode ? '退出逐层步进' : '进入逐层步进'"
              :aria-pressed="stepMode"
              data-testid="propagation-step-mode"
              @click="toggleStepMode"
            >
              <Layers3 :size="17" />
            </button>
            <button
              class="icon-button animation-toggle"
              type="button"
              :title="animated ? '暂停信号动画' : '播放信号动画'"
              :aria-label="animated ? '暂停信号动画' : '播放信号动画'"
              @click="animated = !animated"
            >
              <Pause v-if="animated" :size="17" />
              <Play v-else :size="17" />
            </button>
          </div>
        </div>

        <div class="network-stage">
          <NetworkCanvas
            :layers="hiddenLayers"
            :convolutions="convolutions"
            :poolings="poolings"
            :output-head="outputHead"
            :activations="flowActivations"
            :gradients="flowGradients"
            :model="model"
            :trace="trainingTrace"
            :selected-layer="selectedLayer"
            :animated="animated"
            :training="trainingFlowActive"
            :step-enabled="stepMode"
            :step-direction="stepDirection"
            :step-layer="selectedLayer"
            @select="selectedLayer = $event"
          />
        </div>

        <div class="layer-inspector" :class="{ 'is-training': trainingFlowActive }">
          <PropagationStepper
            v-if="stepMode"
            :training="trainingFlowActive"
            :direction="stepDirection"
            :layer="selectedLayer"
            :layer-names="layerNames"
            :values="selectedValues"
            :gradients="selectedGradients"
            @close="stepMode = false"
            @direction="updateStepDirection"
            @layer="updateStepLayer"
          />
          <div class="inspector-title">
            <Activity :size="16" />
            <span>{{ selectedLayerName }}</span>
            <b>{{ layerSizes[selectedLayer] }} 神经元</b>
          </div>
          <div v-if="trainingFlowActive && trainingTrace" class="training-signal-meta" aria-live="polite">
            <span class="forward-signal"><i />前向激活</span>
            <span class="backward-signal"><i />反向梯度</span>
            <b>
              E{{ trainingTrace.epoch }} · {{ trainingTrace.sample.toLocaleString() }}/{{ trainingTrace.samples.toLocaleString() }}
              · 标签 {{ trainingTrace.label }} → 预测 {{ trainingTrace.prediction }}
            </b>
          </div>
          <dl>
            <div><dt>平均激活</dt><dd>{{ selectedStats.mean.toFixed(3) }}</dd></div>
            <div><dt>峰值</dt><dd>{{ selectedStats.peak.toFixed(3) }}</dd></div>
            <div>
              <dt>{{ trainingFlowActive ? "平均 |梯度|" : "活跃占比" }}</dt>
              <dd>{{ trainingFlowActive ? formatSignal(selectedStats.gradientMean) : `${selectedStats.active.toFixed(1)}%` }}</dd>
            </div>
          </dl>
        </div>
      </section>

      <aside class="control-column">
        <PredictionPanel
          :probabilities="displayedProbabilities"
          :output-head="outputHead"
          :status="modelStatus"
          :has-input="
            (trainingFlowActive || inputEnergy >= 0.001) &&
            (!stepMode || stepDirection === 'backward' || selectedLayer === layerSizes.length - 1)
          "
          :training="trainingFlowActive"
        />
        <ArchitectureEditor
          :layers="hiddenLayers"
          :convolutions="convolutions"
          :poolings="poolings"
          :output-head="outputHead"
          :parameter-count="parameterCount"
          @update="updateLayers"
          @update-convolutions="updateConvolutions"
          @update-poolings="updatePoolings"
          @update-output-head="updateOutputHead"
          @reorder="reorderArchitecture"
          @reset="resetArchitecture"
        />
        <TrainingPanel
          :layers="hiddenLayers"
          :convolutions="convolutions"
          :poolings="poolings"
          :backend="engineBackend"
          :profiles="trainingProfiles"
          :progress="trainingProgress"
          :custom-training-count="customTrainingCount"
          :custom-test-count="customTestCount"
          :mnist-enabled="mnistEnabled"
          :save-state="modelSaveState"
          :can-fine-tune="trainedModel !== null"
          :trained-epochs="trainedEpochCount"
          :mode="activeTrainingMode"
          @update="updateTrainingSettings"
          @configure-optimizer="openOptimizer"
          @train="startTraining"
          @fine-tune="startFineTuning"
          @pause="pauseTraining"
          @resume="resumeTraining"
          @save-snapshot="savePausedSnapshot"
          @cancel="cancelTraining"
        />
      </aside>
    </main>

    <SignalFlowExplorer
      v-if="activeView === 'signals'"
      :layers="hiddenLayers"
      :convolutions="convolutions"
      :poolings="poolings"
      :output-head="outputHead"
      :model="model"
      :activations="flowActivations"
      :gradients="flowGradients"
      :training="trainingFlowActive"
      :progress="trainingProgress"
      :trace="trainingTrace"
      @back="navigateTo('lab')"
      @pause-training="pauseTraining"
      @resume-training="resumeTraining"
    />

    <DatasetManager
      v-if="activeView === 'samples'"
      :samples="customSamples"
      :locked="datasetLocked"
      :mnist-enabled="mnistEnabled"
      @back="navigateTo('lab')"
      @update-mnist-enabled="mnistEnabled = $event"
      @update="updateCustomSample"
      @remove="removeCustomSamples([$event])"
      @remove-many="removeCustomSamples"
    />

    <ModelManager
      v-if="activeView === 'models'"
      :models="savedModels"
      :loading="modelLibraryLoading"
      :deleting="modelLibraryDeleting"
      @back="navigateTo('lab')"
      @load="loadSavedModel"
      @rename="renameSavedModel"
      @remove-many="removeSavedModels"
    />

    <OptimizerManager
      v-if="activeView === 'optimizer'"
      :profiles="trainingProfiles"
      :initial-mode="optimizerViewMode"
      :locked="datasetLocked"
      @back="navigateTo('lab')"
      @mode="optimizerViewMode = $event"
      @update="updateTrainingSettings"
    />

    <footer class="statusbar">
      <span><i class="status-dot" /> {{ activeView === "lab" ? "引擎就绪" : activeView === "signals" ? (trainingFlowActive ? "训练信号已接入" : "推理信号已接入") : activeView === "samples" ? `${customSamples.length} 个自定义样本` : activeView === "models" ? `${savedModels.length} 个本地模型` : `重训 ${trainingProfiles.scratch.optimizer.kind.toUpperCase()} · 微调 ${trainingProfiles.finetune.optimizer.kind.toUpperCase()}` }}</span>
      <span>{{ activeView === "lab" ? `FP32 · ${selectedComputeBackendLabel} · ${trainingProfiles.mathMode === "full" ? "完整" : "快速"}` : activeView === "signals" ? `${layerSizes.length} 层 · FP32` : activeView === "samples" ? `训练 ${customTrainingCount} · 测试 ${customTestCount}` : activeView === "models" ? "FP32 · IndexedDB" : `学习率 ${trainingProfiles.scratch.learningRate} / ${trainingProfiles.finetune.learningRate}` }}</span>
      <span>{{ activeView === "lab" ? "本地推理 · 本地训练" : activeView === "signals" ? (trainingProgress.phase === "paused" ? "训练已暂停" : "实时逐层采样") : activeView === "samples" ? (mnistEnabled ? "MNIST 已启用" : "仅自定义样本") : activeView === "models" ? "训练完成自动保存" : "5 种 Zig/Wasm 优化器" }}</span>
    </footer>
  </div>
</template>
