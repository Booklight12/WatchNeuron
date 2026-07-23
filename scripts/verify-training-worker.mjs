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
    const data = await readFile(String(url).includes("neuron_kernel.wasm")
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
    return verificationStage === "activations" ? activationKinds[activationIndex] : "relu";
  }

  function startCurrentActivation() {
    const activation = currentActivation();
    const optimizerKind = optimizerKinds[activationIndex % optimizerKinds.length];
    sawValidTrace = false;
    worker.postMessage({
      type: "train",
      datasetUrl: "/mnist-training.bin",
      layers: [{ id: `${activation}-test`, units: 12, activation }],
      settings: {
        epochs: 1,
        learningRate: activationIndex === 0
          ? 0.05
          : ["adam", "rmsprop"].includes(optimizerKind)
            ? 0.003
            : 0.01,
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
      layers: [{ id: "relu-finetune", units: 12, activation: "relu" }],
      settings: {
        epochs: 1,
        learningRate: 0.003,
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
      layers: [{ id: "custom-only", units: 12, activation: "relu" }],
      settings: {
        epochs: 1,
        learningRate: 0.01,
        optimizer: optimizerConfig("adagrad"),
      },
      customSamples: customSamples.filter((sample) => sample.split === "training"),
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
      wasmBackendVerified ||= message.backend === "Zig/Wasm";
      const activations = Array.from(message.activations[1]);
      const gradients = Array.from(message.gradients[1]);
      const activation = currentActivation();
      const softmaxIsNormalized =
        activation !== "softmax" ||
        Math.abs(activations.reduce((sum, value) => sum + value, 0) - 1) < 1e-4;
      sawValidTrace ||=
        activations.every(Number.isFinite) &&
        softmaxIsNormalized &&
        message.samples === (verificationStage === "custom-only" ? 1 : 4001) &&
        gradients.every(Number.isFinite) &&
        gradients.some((value) => Math.abs(value) > 1e-10);
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
      const hiddenLayer = message.model[0];
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
      const hiddenLayer = message.model[0];
      const valuesAreFinite = [
        ...hiddenLayer.weights,
        ...hiddenLayer.biases,
      ].every(Number.isFinite);
      const activation = currentActivation();
      const expectedTrainingSamples = verificationStage === "custom-only" ? 1 : 4001;
      const expectedTestSamples = verificationStage === "custom-only" ? 0 : 1001;
      if (
        !sawValidTrace ||
        hiddenLayer.activation !== activation ||
        !valuesAreFinite ||
        !Number.isFinite(message.accuracy) ||
        message.trainingSamples !== expectedTrainingSamples ||
        message.testSamples !== expectedTestSamples ||
        message.backend !== "Zig/Wasm" ||
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
        clearTimeout(timeout);
        resolvePromise();
        return;
      }

      if (verificationStage === "finetune") {
        const weightsChanged = message.model.some((layer, layerIndex) => {
          const baselineLayer = baselineModel[layerIndex];
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
        baselineModel = message.model.map((layer) => ({
          ...layer,
          weights: Float32Array.from(layer.weights),
          biases: Float32Array.from(layer.biases),
        }));
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
  if (!wasmBackendVerified) throw new Error("Training traces did not report the Zig/Wasm backend");
  const meanDuration = trainingDurations.reduce((sum, value) => sum + value, 0) / trainingDurations.length;
  console.log(
    `Zig/Wasm training: ${activationKinds.length}/16 activations and ${verifiedOptimizers.size}/5 optimizers passed with pause/save/resume, ${fineTuneVerified ? "fine-tune continuation" : "fine-tune failure"}, and ${customOnlyVerified ? "custom-only dataset" : "custom-only failure"}; mean validation ${(meanAccuracy * 100).toFixed(1)}%; mean epoch ${meanDuration.toFixed(1)} ms`,
  );
} finally {
  await worker.terminate();
}
