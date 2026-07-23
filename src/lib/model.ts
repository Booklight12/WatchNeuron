import type {
  ActivationKind,
  DenseLayerData,
  HiddenLayer,
  NeuralModel,
  SerializedModel,
} from "../types";

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeWeights(
  outputSize: number,
  inputSize: number,
  seed: number,
  activation: ActivationKind = "relu",
) {
  const random = seededRandom(seed);
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
  return Float32Array.from(
    { length: outputSize * inputSize },
    () => (random() * 2 - 1) * scale,
  );
}

function copyRows(
  target: Float32Array,
  source: number[],
  targetRows: number,
  targetColumns: number,
  sourceRows: number,
  sourceColumns: number,
) {
  const rows = Math.min(targetRows, sourceRows);
  const columns = Math.min(targetColumns, sourceColumns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      target[row * targetColumns + column] = source[row * sourceColumns + column];
    }
  }
}

export function buildModel(
  base: SerializedModel,
  hiddenLayers: HiddenLayer[],
): NeuralModel {
  const layers: DenseLayerData[] = [];
  const defaultHiddenSize = base.architecture[1];
  const firstLayer = hiddenLayers[0];
  const firstWeights = makeWeights(
    firstLayer.units,
    784,
    0x1337 + firstLayer.units,
    firstLayer.activation,
  );
  const firstBiases = new Float32Array(firstLayer.units);
  copyRows(firstWeights, base.weights[0], firstLayer.units, 784, defaultHiddenSize, 784);
  firstBiases.set(base.biases[0].slice(0, firstLayer.units));
  layers.push({
    inputSize: 784,
    outputSize: firstLayer.units,
    activation: firstLayer.activation,
    weights: firstWeights,
    biases: firstBiases,
  });

  for (let index = 1; index < hiddenLayers.length; index++) {
    const previousSize = hiddenLayers[index - 1].units;
    const current = hiddenLayers[index];
    const weights = makeWeights(
      current.units,
      previousSize,
      0x2459 + index * 97 + current.units,
      current.activation,
    );
    weights.fill(0);
    for (let unit = 0; unit < Math.min(previousSize, current.units); unit++) {
      weights[unit * previousSize + unit] = 1;
    }
    layers.push({
      inputSize: previousSize,
      outputSize: current.units,
      activation: current.activation,
      weights,
      biases: new Float32Array(current.units),
    });
  }

  const lastSize = hiddenLayers.at(-1)?.units ?? defaultHiddenSize;
  const outputWeights = makeWeights(10, lastSize, 0x7810 + lastSize, "linear");
  copyRows(outputWeights, base.weights[1], 10, lastSize, 10, defaultHiddenSize);
  layers.push({
    inputSize: lastSize,
    outputSize: 10,
    activation: "linear",
    weights: outputWeights,
    biases: Float32Array.from(base.biases[1]),
  });

  return {
    layers,
    calibrated:
      hiddenLayers.length === 1 &&
      hiddenLayers[0].units === defaultHiddenSize &&
      hiddenLayers[0].activation === "relu",
  };
}

export function countParameters(model: NeuralModel) {
  return model.layers.reduce(
    (total, layer) => total + layer.weights.length + layer.biases.length,
    0,
  );
}

export const activationLabels: Record<ActivationKind, string> = {
  relu: "ReLU",
  leakyRelu: "Leaky ReLU",
  elu: "ELU",
  selu: "SELU",
  relu6: "ReLU6",
  gelu: "GELU",
  swish: "SiLU / Swish",
  mish: "Mish",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  hardSigmoid: "Hard Sigmoid",
  hardTanh: "Hard Tanh",
  softplus: "Softplus",
  softsign: "Softsign",
  softmax: "Softmax",
  linear: "Linear",
};

export function isActivationKind(value: unknown): value is ActivationKind {
  return typeof value === "string" && Object.hasOwn(activationLabels, value);
}
