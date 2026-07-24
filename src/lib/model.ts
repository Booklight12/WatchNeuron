import type {
  ActivationKind,
  ConvolutionConfig,
  ConvolutionLayerData,
  DenseLayerData,
  HiddenLayer,
  NeuralModel,
  SerializedModel,
} from "../types";
import {
  convolutionOutputShape,
  convolutionPipeline,
  fitConvolutionsToLayers,
  modelConvolutions,
  type FeatureMapShape,
} from "./convolution";

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
  convolutionConfigs: ConvolutionConfig[] = [],
): NeuralModel {
  const layers: DenseLayerData[] = [];
  const defaultHiddenSize = base.architecture[1];
  const normalizedConvolutions = fitConvolutionsToLayers(convolutionConfigs, hiddenLayers);
  const convolutionEntries = convolutionPipeline(hiddenLayers, normalizedConvolutions);
  const convolutions = convolutionEntries.map(({ config, input }) =>
    buildConvolutionLayer(config, input),
  );
  let previousSize = 784;

  for (let index = 0; index < hiddenLayers.length; index++) {
    for (const convolution of convolutions.filter((item) => item.position === index)) {
      previousSize = convolution.outputWidth * convolution.outputHeight * convolution.filters;
    }
    const current = hiddenLayers[index];
    const weights = makeWeights(
      current.units,
      previousSize,
      0x2459 + index * 97 + current.units,
      current.activation,
    );
    const biases = new Float32Array(current.units);
    if (index === 0 && previousSize === 784) {
      copyRows(weights, base.weights[0], current.units, 784, defaultHiddenSize, 784);
      biases.set(base.biases[0].slice(0, current.units));
    } else if (index > 0) {
      weights.fill(0);
      for (let unit = 0; unit < Math.min(previousSize, current.units); unit++) {
        weights[unit * previousSize + unit] = 1;
      }
    }
    layers.push({
      inputSize: previousSize,
      outputSize: current.units,
      activation: current.activation,
      weights,
      biases,
    });
    previousSize = current.units;
  }

  for (const convolution of convolutions.filter((item) => item.position === hiddenLayers.length)) {
    previousSize = convolution.outputWidth * convolution.outputHeight * convolution.filters;
  }
  const outputWeights = makeWeights(10, previousSize, 0x7810 + previousSize, "linear");
  copyRows(outputWeights, base.weights[1], 10, previousSize, 10, defaultHiddenSize);
  layers.push({
    inputSize: previousSize,
    outputSize: 10,
    activation: "linear",
    weights: outputWeights,
    biases: Float32Array.from(base.biases[1]),
  });

  return {
    convolutions,
    layers,
    calibrated:
      convolutions.length === 0 &&
      hiddenLayers.length === 1 &&
      hiddenLayers[0].units === defaultHiddenSize &&
      hiddenLayers[0].activation === "relu",
  };
}

export function countParameters(model: NeuralModel) {
  const denseParameters = model.layers.reduce(
    (total, layer) => total + layer.weights.length + layer.biases.length,
    0,
  );
  return denseParameters + modelConvolutions(model).reduce(
    (total, convolution) => total + convolution.weights.length + convolution.biases.length,
    0,
  );
}

function buildConvolutionLayer(
  config: ConvolutionConfig,
  input: FeatureMapShape,
): ConvolutionLayerData {
  const output = convolutionOutputShape(config, input);
  const kernelLength = config.kernelSize * config.kernelSize;
  const expectedWeights = config.filters * input.channels * kernelLength;
  const weights = new Float32Array(expectedWeights);
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
