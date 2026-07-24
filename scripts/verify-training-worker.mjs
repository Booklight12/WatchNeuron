import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";

const assets = await readdir(resolve("dist", "assets"));
const workerAsset = assets.find((name) => /^training\.worker-.*\.js$/.test(name));
if (!workerAsset) throw new Error("Production training worker bundle was not found");

const wrapperSource = `
  import { parentPort, workerData } from "node:worker_threads";
  import { readFile } from "node:fs/promises";

  globalThis.self = globalThis;
  globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
  globalThis.fetch = async (url) => {
    const value = String(url);
    const data = await readFile(value.includes("neuron_kernel_simd.wasm")
      ? workerData.simdWasmPath
      : value.includes(".wasm")
        ? workerData.wasmPath
        : workerData.datasetPath);
    return {
      ok: true,
      arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    };
  };

  await import(workerData.workerUrl);
  parentPort.on("message", (data) => globalThis.self.onmessage({ data }));
  parentPort.postMessage({ type: "ready" });
`;

const worker = new Worker(
  new URL(`data:text/javascript,${encodeURIComponent(wrapperSource)}`),
  {
    workerData: {
      datasetPath: resolve("public", "mnist-training.bin"),
      wasmPath: resolve("public", "neuron_kernel.wasm"),
      simdWasmPath: resolve("public", "neuron_kernel_simd.wasm"),
      workerUrl: pathToFileURL(resolve("dist", "assets", workerAsset)).href,
    },
  },
);

const activationKinds = [
  "relu",
  "leakyRelu",
  "elu",
  "selu",
  "relu6",
  "gelu",
  "swish",
  "mish",
  "sigmoid",
  "tanh",
  "hardSigmoid",
  "hardTanh",
  "softplus",
  "softsign",
  "softmax",
  "linear",
];
const optimizerKinds = ["sgd", "momentum", "adam", "rmsprop", "adagrad"];
const optimizerConfig = (kind) => ({
  kind,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  decay: 0.9,
  epsilon: 1e-8,
});
const noConvolution = {
  enabled: false,
  position: 0,
  filters: 4,
  kernelSize: 3,
  stride: 1,
  padding: 1,
  activation: "relu",
  kernels: [],
};
const convolutionConfig = {
  id: "conv-primary",
  enabled: true,
  trainable: true,
  position: 0,
  filters: 2,
  kernelSize: 3,
  stride: 2,
  padding: 1,
  activation: "relu",
  kernels: [
    [0, -1, 0, -1, 4, -1, 0, -1, 0],
    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
  ],
};
const multipleConvolutionConfigs = [
  { ...convolutionConfig, id: "conv-stack-a", position: 0, filters: 2 },
  { ...convolutionConfig, id: "conv-stack-b", position: 0, filters: 2 },
  { ...convolutionConfig, id: "conv-stack-c", position: 1, filters: 2, stride: 1 },
];
const customSamples = [
  {
    id: "custom-training-probe",
    label: 3,
    split: "training",
    indices: [321, 322, 349, 377, 405],
    values: [0.4, 0.9, 1, 0.8, 0.5],
  },
  {
    id: "custom-test-probe",
    label: 7,
    split: "test",
    indices: [180, 181, 208, 236, 264],
    values: [0.5, 1, 0.9, 0.8, 0.4],
  },
];
const batchTrainingSamples = [
  ...customSamples.filter((sample) => sample.split === "training"),
  {
    id: "batch-training-probe-1",
    label: 1,
    split: "training",
    indices: [210, 238, 266, 294, 322],
    values: [0.5, 0.8, 1, 0.8, 0.5],
  },
  {
    id: "batch-training-probe-2",
    label: 6,
    split: "training",
    indices: [320, 321, 349, 377, 405, 406],
    values: [0.4, 0.9, 1, 1, 0.9, 0.4],
  },
  {
    id: "batch-training-probe-3",
    label: 8,
    split: "training",
    indices: [292, 293, 320, 322, 348, 350, 376, 377],
    values: [0.6, 0.6, 0.8, 0.8, 1, 1, 0.7, 0.7],
  },
];
const spatialPoolingConfigs = [
  {
    id: "pool-max-probe",
    enabled: true,
    position: 0,
    order: 1,
    kind: "max",
    kernelSize: 2,
    stride: 2,
    padding: 0,
  },
  {
    id: "pool-gap-probe",
    enabled: true,
    position: 0,
    order: 2,
    kind: "globalAverage",
    kernelSize: 1,
    stride: 1,
    padding: 0,
  },
];

let activationIndex = 0;
let sawValidTrace = false;
let pauseRequested = false;
let pauseVerified = false;
let snapshotVerified = false;
let resumeVerified = false;
let verificationStage = "activations";
let baselineModel = null;
let fineTuneVerified = false;
let customOnlyVerified = false;
let convolutionVerified = false;
let convolutionFineTuneVerified = false;
let frozenConvolutionVerified = false;
let frozenConvolutionGradientVerified = false;
let convolutionParameterTraceVerified = false;
let middleConvolutionVerified = false;
let middleConvolutionTraceVerified = false;
let dropoutVerified = false;
let convolutionBaseline = null;
let multipleConvolutionBaseline = null;
let multipleConvolutionVerified = false;
let multipleConvolutionFineTuneVerified = false;
let multipleConvolutionTraceVerified = false;
let spatialHeadBaseline = null;
let spatialHeadTraceVerified = false;
let spatialHeadVerified = false;
let spatialHeadFineTuneVerified = false;
let wasmBackendVerified = false;
const verifiedOptimizers = new Set();
const accuracies = [];
const trainingDurations = [];
const completed = new Promise((resolvePromise, rejectPromise) => {
  const timeout = setTimeout(
    () => rejectPromise(new Error("Training worker verification timed out")),
    180000,
  );

  function currentActivation() {
    if (verificationStage === "activations") return activationKinds[activationIndex];
    if (verificationStage === "convolution-middle" || verificationStage.startsWith("convolution-multiple")) return "relu";
    if (verificationStage.startsWith("convolution")) return "sigmoid";
    return "relu";
  }

  function expectedTrainingSamples() {
    if (verificationStage.startsWith("spatial-head")) return batchTrainingSamples.length;
    if (verificationStage === "custom-only" || verificationStage.startsWith("convolution")) return 1;
    return 4001;
  }

  function expectedTestSamples() {
    return verificationStage === "custom-only" ||
      verificationStage.startsWith("convolution") ||
      verificationStage.startsWith("spatial-head")
      ? 0
      : 1001;
  }

  function startCurrentActivation() {
    const activation = currentActivation();
    const optimizerKind = optimizerKinds[activationIndex % optimizerKinds.length];
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      layers: [{ id: `${activation}-test`, units: 12, activation, dropout: 0 }],
      convolution: noConvolution,
      settings: {
        epochs: 1,
        learningRate: activationIndex === 0
          ? 0.05
          : ["adam", "rmsprop"].includes(optimizerKind)
            ? 0.003
            : 0.01,
        mathMode: "fast",
        optimizer: optimizerConfig(optimizerKind),
      },
      customSamples,
    });
  }

  function startFineTune() {
    if (!baselineModel) throw new Error("ReLU baseline model was not captured");
    verificationStage = "finetune";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      layers: [{ id: "relu-finetune", units: 12, activation: "relu", dropout: 0 }],
      convolution: noConvolution,
      settings: {
        epochs: 1,
        learningRate: 0.003,
        mathMode: "fast",
        optimizer: optimizerConfig("adam"),
      },
      initialModel: baselineModel,
      customSamples,
    });
  }

  function startCustomOnlyTraining() {
    verificationStage = "custom-only";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [{ id: "custom-only", units: 12, activation: "relu", dropout: 0 }],
      convolution: noConvolution,
      settings: {
        epochs: 1,
        learningRate: 0.01,
        mathMode: "fast",
        optimizer: optimizerConfig("adagrad"),
      },
      customSamples: customSamples.filter((sample) => sample.split === "training"),
    });
  }

  function startConvolutionTraining(initialModel) {
    verificationStage = initialModel ? "convolution-finetune" : "convolution";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [{ id: "conv-dropout", units: 12, activation: "sigmoid", dropout: 0.5 }],
      convolutions: [convolutionConfig],
      settings: {
        epochs: 1,
        learningRate: 0.003,
        mathMode: "fast",
        optimizer: optimizerConfig("adam"),
      },
      initialModel,
      customSamples: customSamples.filter((sample) => sample.split === "training"),
    });
  }

  function startMiddleConvolutionTraining() {
    verificationStage = "convolution-middle";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [
        { id: "before-conv", units: 12, activation: "relu", dropout: 0.25 },
        { id: "after-conv", units: 10, activation: "tanh", dropout: 0 },
      ],
      convolutions: [{ ...convolutionConfig, id: "conv-middle", position: 1 }],
      settings: {
        epochs: 1,
        learningRate: 0.003,
        mathMode: "full",
        optimizer: optimizerConfig("adam"),
      },
      customSamples: customSamples.filter((sample) => sample.split === "training"),
    });
  }

  function startFrozenConvolutionTraining(initialModel) {
    verificationStage = "convolution-frozen";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [{ id: "conv-dropout", units: 12, activation: "sigmoid", dropout: 0.5 }],
      convolutions: [{ ...convolutionConfig, trainable: false }],
      settings: {
        epochs: 1,
        learningRate: 0.003,
        mathMode: "fast",
        optimizer: optimizerConfig("adam"),
      },
      initialModel,
      customSamples: customSamples.filter((sample) => sample.split === "training"),
    });
  }

  function startMultipleConvolutionTraining(initialModel) {
    verificationStage = initialModel ? "convolution-multiple-finetune" : "convolution-multiple";
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [{ id: "between-convs", units: 12, activation: "relu", dropout: 0 }],
      convolutions: multipleConvolutionConfigs,
      settings: {
        epochs: 1,
        learningRate: 0.003,
        mathMode: "fast",
        optimizer: optimizerConfig("adam"),
      },
      initialModel,
      customSamples: customSamples.filter((sample) => sample.split === "training"),
    });
  }

  function startSpatialHeadTraining(initialModel) {
    verificationStage = initialModel ? "spatial-head-finetune" : "spatial-head";
    sawValidTrace = false;
    spatialHeadTraceVerified = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      mnistEnabled: false,
      layers: [{ id: "spatial-head-dense", units: 12, activation: "relu", dropout: 0 }],
      convolutions: [{ ...convolutionConfig, id: "conv-pool-probe", order: 0 }],
      poolings: spatialPoolingConfigs,
      outputHead: "sigmoid",
      settings: {
        epochs: 1,
        learningRate: 0.003,
        batchSize: 4,
        mathMode: "fast",
        optimizer: { ...optimizerConfig("adam"), weightDecay: 0.02 },
      },
      initialModel,
      customSamples: batchTrainingSamples,
    });
  }

  worker.on("error", (error) => {
    clearTimeout(timeout);
    rejectPromise(error);
  });
  worker.on("message", (message) => {
    if (message.type === "ready") {
      startCurrentActivation();
      return;
    }

    if (message.type === "error") {
      clearTimeout(timeout);
      rejectPromise(new Error(message.message));
      return;
    }

    if (message.type === "trace") {
      wasmBackendVerified ||= message.backend?.startsWith("Zig/Wasm SIMD");
      const traceLayerIndex = verificationStage.startsWith("convolution") ? 2 : 1;
      const activations = Array.from(message.activations[traceLayerIndex]);
      const gradients = Array.from(message.gradients[traceLayerIndex]);
      const activation = currentActivation();
      const softmaxIsNormalized =
        activation !== "softmax" ||
        Math.abs(activations.reduce((sum, value) => sum + value, 0) - 1) < 1e-4;
      sawValidTrace ||=
        activations.every(Number.isFinite) &&
        softmaxIsNormalized &&
        message.samples === expectedTrainingSamples() &&
        gradients.every(Number.isFinite) &&
        gradients.some((value) => Math.abs(value) > 1e-10);
      if (verificationStage.startsWith("convolution")) {
        dropoutVerified ||= activations.some((value) => value === 0) && activations.some((value) => value !== 0);
        convolutionParameterTraceVerified ||=
          message.convolutionWeights?.length > 0 &&
          message.convolutionBiases?.length === message.convolutionWeights.length &&
          message.convolutionWeights.every((weights) => Array.from(weights).every(Number.isFinite)) &&
          message.convolutionBiases.every((biases) => Array.from(biases).every(Number.isFinite));
      }
      if (verificationStage === "convolution-frozen") {
        frozenConvolutionGradientVerified ||=
          Array.from(message.gradients[0]).some((value) => Math.abs(value) > 1e-10) &&
          Array.from(message.gradients[1]).some((value) => Math.abs(value) > 1e-10);
      }
      if (verificationStage === "convolution-middle") {
        middleConvolutionTraceVerified ||=
          message.activations.length === 5 &&
          message.gradients.length === 5 &&
          message.activations[0].length === 784 &&
          message.activations[1].length === 12 &&
          message.activations[2].length === 8 &&
          message.activations[3].length === 10 &&
          message.activations[4].length === 10 &&
          Array.from(message.gradients[1]).some((value) => Math.abs(value) > 1e-10) &&
          Array.from(message.gradients[2]).some((value) => Math.abs(value) > 1e-10);
      }
      if (verificationStage.startsWith("convolution-multiple")) {
        multipleConvolutionTraceVerified ||=
          message.activations.length === 6 &&
          message.gradients.length === 6 &&
          [784, 392, 98, 12, 24, 10].every(
            (length, index) => message.activations[index].length === length && message.gradients[index].length === length,
          ) &&
          message.gradients.every((values) => Array.from(values).every(Number.isFinite));
      }
      if (verificationStage.startsWith("spatial-head")) {
        const output = Array.from(message.activations.at(-1));
        const outputSum = output.reduce((sum, value) => sum + value, 0);
        spatialHeadTraceVerified ||=
          message.activations.length === 6 &&
          message.gradients.length === 6 &&
          [784, 392, 98, 2, 12, 10].every(
            (length, index) => message.activations[index].length === length && message.gradients[index].length === length,
          ) &&
          message.gradients[2].some((value) => Math.abs(value) > 1e-10) &&
          message.gradients[3].some((value) => Math.abs(value) > 1e-10) &&
          output.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) &&
          Math.abs(outputSum - 1) > 1e-3;
      }
      if (verificationStage === "activations" && activationIndex === 0 && !pauseRequested) {
        pauseRequested = true;
        worker.postMessage({ type: "pause" });
      }
      return;
    }

    if (message.type === "paused") {
      pauseVerified = true;
      worker.postMessage({ type: "snapshot" });
      return;
    }

    if (message.type === "snapshot") {
      const hiddenLayer = message.model.layers[0];
      snapshotVerified =
        message.epoch >= 1 &&
        message.sample >= 1 &&
        message.samples === 4001 &&
        hiddenLayer.weights.length === hiddenLayer.inputSize * hiddenLayer.outputSize &&
        Array.from(hiddenLayer.weights).every(Number.isFinite);
      setTimeout(() => worker.postMessage({ type: "resume" }), 25);
      return;
    }

    if (message.type === "resumed") {
      resumeVerified = true;
      return;
    }

    if (message.type === "complete") {
      trainingDurations.push(message.elapsedMs);
      const hiddenLayer = message.model.layers[0];
      const valuesAreFinite = [
        ...hiddenLayer.weights,
        ...hiddenLayer.biases,
      ].every(Number.isFinite);
      const activation = currentActivation();
      const trainingSampleCount = expectedTrainingSamples();
      const testSampleCount = expectedTestSamples();
      if (
        !sawValidTrace ||
        hiddenLayer.activation !== activation ||
        !valuesAreFinite ||
        !Number.isFinite(message.accuracy) ||
        message.trainingSamples !== trainingSampleCount ||
        message.testSamples !== testSampleCount ||
        message.backend !== (verificationStage === "convolution-middle"
          ? "Zig/Wasm SIMD · 完整"
          : "Zig/Wasm SIMD · 快速") ||
        (verificationStage === "activations" &&
          activationIndex === 0 &&
          (!pauseVerified || !snapshotVerified || !resumeVerified))
      ) {
        clearTimeout(timeout);
        rejectPromise(new Error(`${activation} hidden-layer training produced an invalid state`));
        return;
      }

      if (verificationStage === "custom-only") {
        customOnlyVerified = true;
        startConvolutionTraining(null);
        return;
      }

      if (verificationStage === "convolution") {
        const convolution = message.model.convolutions[0];
        const initialWeights = convolutionConfig.kernels.flat();
        const weightsChanged = convolution && Array.from(convolution.weights).some(
          (value, index) => Math.abs(value - initialWeights[index]) > 1e-8,
        );
        if (!convolution || !weightsChanged || !dropoutVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Conv2D or Dropout training did not update the expected state"));
          return;
        }
        convolutionVerified = true;
        convolutionBaseline = message.model;
        startConvolutionTraining(convolutionBaseline);
        return;
      }

      if (verificationStage === "convolution-finetune") {
        const changed = Array.from(message.model.convolutions[0].weights).some(
          (value, index) => Math.abs(value - convolutionBaseline.convolutions[0].weights[index]) > 1e-8,
        );
        if (!changed) {
          clearTimeout(timeout);
          rejectPromise(new Error("Loaded Conv2D model did not continue fine-tuning"));
          return;
        }
        convolutionFineTuneVerified = true;
        convolutionBaseline = message.model;
        startFrozenConvolutionTraining(convolutionBaseline);
        return;
      }

      if (verificationStage === "convolution-frozen") {
        const convolution = message.model.convolutions[0];
        const baseline = convolutionBaseline.convolutions[0];
        const parametersUnchanged =
          Array.from(convolution.weights).every((value, index) => value === baseline.weights[index]) &&
          Array.from(convolution.biases).every((value, index) => value === baseline.biases[index]);
        const denseChanged = message.model.layers.some((layer, layerIndex) =>
          Array.from(layer.weights).some(
            (value, index) => Math.abs(value - convolutionBaseline.layers[layerIndex].weights[index]) > 1e-8,
          ),
        );
        frozenConvolutionVerified = Boolean(
          convolution &&
          convolution.trainable === false &&
          parametersUnchanged &&
          denseChanged &&
          frozenConvolutionGradientVerified,
        );
        if (!frozenConvolutionVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Frozen Conv2D changed parameters or stopped gradient propagation"));
          return;
        }
        startMiddleConvolutionTraining();
        return;
      }

      if (verificationStage === "convolution-middle") {
        const convolution = message.model.convolutions[0];
        const initialWeights = convolutionConfig.kernels.flat();
        const weightsChanged = convolution && Array.from(convolution.weights).some(
          (value, index) => Math.abs(value - initialWeights[index]) > 1e-8,
        );
        middleConvolutionVerified = Boolean(
          convolution &&
          convolution.position === 1 &&
          convolution.inputWidth === 4 &&
          convolution.inputHeight === 3 &&
          weightsChanged &&
          middleConvolutionTraceVerified,
        );
        if (!middleConvolutionVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Middle Conv2D training did not propagate through both Dense segments"));
          return;
        }
        startMultipleConvolutionTraining(null);
        return;
      }

      if (verificationStage === "convolution-multiple") {
        const changed = message.model.convolutions.every((convolution, convolutionIndex) => {
          const config = multipleConvolutionConfigs[convolutionIndex];
          const kernelLength = config.kernelSize ** 2;
          return Array.from(convolution.weights).some((value, weightIndex) => {
            const filter = Math.floor(weightIndex / (convolution.inputChannels * kernelLength));
            const kernelIndex = weightIndex % kernelLength;
            return Math.abs(value - config.kernels[filter][kernelIndex]) > 1e-8;
          });
        });
        multipleConvolutionVerified =
          message.model.convolutions.length === 3 && changed && multipleConvolutionTraceVerified;
        if (!multipleConvolutionVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Multiple Conv2D layers did not all receive finite gradients and weight updates"));
          return;
        }
        multipleConvolutionBaseline = message.model;
        startMultipleConvolutionTraining(multipleConvolutionBaseline);
        return;
      }

      if (verificationStage === "convolution-multiple-finetune") {
        const changed = message.model.convolutions.every((convolution, convolutionIndex) =>
          Array.from(convolution.weights).some(
            (value, index) => Math.abs(value - multipleConvolutionBaseline.convolutions[convolutionIndex].weights[index]) > 1e-8,
          ),
        );
        multipleConvolutionFineTuneVerified = changed;
        if (!multipleConvolutionFineTuneVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Loaded multiple-Conv2D model did not continue fine-tuning"));
          return;
        }
        startSpatialHeadTraining(null);
        return;
      }

      if (verificationStage === "spatial-head") {
        const [maxPool, gap] = message.model.poolings ?? [];
        spatialHeadVerified = Boolean(
          spatialHeadTraceVerified &&
          message.model.outputHead === "sigmoid" &&
          message.model.poolings?.length === 2 &&
          maxPool.kind === "max" &&
          maxPool.outputWidth === 7 &&
          maxPool.outputHeight === 7 &&
          gap.kind === "globalAverage" &&
          gap.outputWidth === 1 &&
          gap.outputHeight === 1 &&
          gap.inputChannels === 2 &&
          hiddenLayer.inputSize === 2,
        );
        if (!spatialHeadVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Conv2D/MaxPool2D/GAP mini-batch Sigmoid training produced an invalid model"));
          return;
        }
        spatialHeadBaseline = message.model;
        startSpatialHeadTraining(spatialHeadBaseline);
        return;
      }

      if (verificationStage === "spatial-head-finetune") {
        const denseChanged = message.model.layers.some((layer, layerIndex) =>
          Array.from(layer.weights).some(
            (value, index) => Math.abs(value - spatialHeadBaseline.layers[layerIndex].weights[index]) > 1e-8,
          ),
        );
        const convolutionChanged = Array.from(message.model.convolutions[0].weights).some(
          (value, index) => Math.abs(value - spatialHeadBaseline.convolutions[0].weights[index]) > 1e-8,
        );
        spatialHeadFineTuneVerified = Boolean(
          spatialHeadTraceVerified &&
          denseChanged &&
          convolutionChanged &&
          message.model.outputHead === "sigmoid" &&
          message.model.poolings?.length === 2,
        );
        if (!spatialHeadFineTuneVerified) {
          clearTimeout(timeout);
          rejectPromise(new Error("Loaded pooled Sigmoid model did not continue fine-tuning"));
          return;
        }
        clearTimeout(timeout);
        resolvePromise();
        return;
      }

      if (verificationStage === "finetune") {
        const weightsChanged = message.model.layers.some((layer, layerIndex) => {
          const baselineLayer = baselineModel.layers[layerIndex];
          return Array.from(layer.weights).some(
            (value, index) => Math.abs(value - baselineLayer.weights[index]) > 1e-8,
          ) || Array.from(layer.biases).some(
            (value, index) => Math.abs(value - baselineLayer.biases[index]) > 1e-8,
          );
        });
        if (!weightsChanged) {
          clearTimeout(timeout);
          rejectPromise(new Error("Fine-tuning did not update the loaded model weights"));
          return;
        }
        fineTuneVerified = true;
        verifiedOptimizers.add("adam");
        startCustomOnlyTraining();
        return;
      }

      verifiedOptimizers.add(optimizerKinds[activationIndex % optimizerKinds.length]);

      if (activationIndex === 0) {
        baselineModel = message.model;
      }
      accuracies.push(message.accuracy);
      activationIndex++;
      if (activationIndex < activationKinds.length) {
        startCurrentActivation();
      } else {
        startFineTune();
      }
    }
  });
});

try {
  await completed;
  const meanAccuracy = accuracies.reduce((sum, value) => sum + value, 0) / accuracies.length;
  if (verifiedOptimizers.size !== optimizerKinds.length) {
    throw new Error("Not all optimizer implementations completed a training run");
  }
  if (!wasmBackendVerified) throw new Error("Training traces did not report the Zig/Wasm SIMD backend");
  if (!convolutionParameterTraceVerified) throw new Error("Training traces did not include finite Conv2D parameter snapshots");
  const meanDuration = trainingDurations.reduce((sum, value) => sum + value, 0) / trainingDurations.length;
  console.log(
    `Zig/Wasm SIMD training: fast/full math modes, ${activationKinds.length}/16 activations and ${verifiedOptimizers.size}/5 optimizers passed with pause/save/resume, ${fineTuneVerified ? "dense fine-tune" : "dense fine-tune failure"}, ${customOnlyVerified ? "custom-only dataset" : "custom-only failure"}, ${convolutionVerified ? "trainable Conv2D" : "Conv2D failure"}, ${frozenConvolutionVerified ? "frozen Conv2D with gradient passthrough" : "frozen Conv2D failure"}, ${dropoutVerified ? "Dropout" : "Dropout failure"}, ${convolutionFineTuneVerified ? "Conv2D fine-tune" : "Conv2D fine-tune failure"}, ${middleConvolutionVerified ? "middle-position Conv2D" : "middle-position Conv2D failure"}, ${multipleConvolutionVerified ? "multiple Conv2D" : "multiple Conv2D failure"}, ${multipleConvolutionFineTuneVerified ? "multiple-Conv2D fine-tune" : "multiple-Conv2D fine-tune failure"}, ${spatialHeadVerified ? "NCHW mini-batch Conv2D/MaxPool2D/GAP + Sigmoid/BCE + Weight Decay" : "spatial/output-head failure"}, and ${spatialHeadFineTuneVerified ? "pooled Sigmoid fine-tune" : "pooled Sigmoid fine-tune failure"}; mean validation ${(meanAccuracy * 100).toFixed(1)}%; mean epoch ${meanDuration.toFixed(1)} ms`,
  );
} finally {
  await worker.terminate();
}
