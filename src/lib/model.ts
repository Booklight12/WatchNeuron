import type {
  ActivationKind,
  ConvolutionConfig,
  ConvolutionLayerData,
  DenseLayerData,
  HiddenLayer,
  NeuralModel,
  OutputHeadKind,
  PoolingConfig,
  PoolingLayerData,
  SerializedModel,
} from "../types";
import {
  convolutionOutputShape,
  fitConvolutionsToLayers,
  modelConvolutions,
  spatialPipeline,
  type FeatureMapShape,
} from "./convolution";
import {
  createConvolutionWeights,
  createUniformWeights,
} from "./initialization";

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
  return createUniformWeights(
    outputSize * inputSize,
    inputSize,
    activation,
    random,
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
  poolingConfigs: PoolingConfig[] = [],
  outputHead: OutputHeadKind = "softmax",
): NeuralModel {
  const layers: DenseLayerData[] = [];
  const defaultHiddenSize = base.architecture[1];
  const normalizedConvolutions = fitConvolutionsToLayers(convolutionConfigs, hiddenLayers);
  const spatialEntries = spatialPipeline(hiddenLayers, normalizedConvolutions, poolingConfigs);
  const convolutions = spatialEntries
    .filter((entry) => entry.kind === "conv")
    .map(({ config, input }) => buildConvolutionLayer(config, input));
  const poolings = spatialEntries
    .filter((entry) => entry.kind === "pool")
    .map(({ config, input, output }) => buildPoolingLayer(config, input, output));
  let previousSize = 784;

  for (let index = 0; index < hiddenLayers.length; index++) {
    for (const entry of spatialEntries.filter((item) => item.config.position === index)) {
      previousSize = entry.output.length;
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

  for (const entry of spatialEntries.filter((item) => item.config.position === hiddenLayers.length)) {
    previousSize = entry.output.length;
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
    poolings,
    layers,
    outputHead,
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
  const random = seededRandom(
    0x63f17a2d ^
    config.position * 0x9e3779b9 ^
    config.order * 0x85ebca6b ^
    config.filters * 131 ^
    input.channels * 17 ^
    kernelLength,
  );
  const weights = createConvolutionWeights(config, input.channels, random);
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
  };
}

function buildPoolingLayer(
  config: PoolingConfig,
  input: FeatureMapShape,
  output: FeatureMapShape,
): PoolingLayerData {
  return {
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
