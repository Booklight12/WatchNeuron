import type {
  ActivationKind,
  ConvolutionConfig,
  ConvolutionInitialization,
} from "../types";

const heActivations = new Set<ActivationKind | "linear">([
  "relu",
  "leakyRelu",
  "elu",
  "relu6",
  "gelu",
  "swish",
  "mish",
]);

export function uniformInitializationBoundary(
  fanIn: number,
  activation: ActivationKind | "linear",
) {
  const safeFanIn = Math.max(1, Math.floor(fanIn));
  return Math.sqrt((heActivations.has(activation) ? 6 : 3) / safeFanIn);
}

export function createUniformWeights(
  length: number,
  fanIn: number,
  activation: ActivationKind | "linear",
  random: () => number,
) {
  const boundary = uniformInitializationBoundary(fanIn, activation);
  return Float32Array.from(
    { length },
    () => (random() * 2 - 1) * boundary,
  );
}

function normalizedTemplate(
  values: number[],
  targetDeviation: number,
) {
  if (values.length === 0) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => {
    const difference = value - mean;
    return sum + difference * difference;
  }, 0) / values.length;
  if (!Number.isFinite(variance) || variance <= 1e-12) return null;
  const scale = targetDeviation / Math.sqrt(variance);
  return values.map((value) => (value - mean) * scale);
}

export function createConvolutionWeights(
  config: Pick<
    ConvolutionConfig,
    "filters" | "kernelSize" | "activation" | "kernels" | "initialization"
  >,
  inputChannels: number,
  random: () => number,
) {
  const kernelLength = config.kernelSize * config.kernelSize;
  const fanIn = Math.max(1, inputChannels * kernelLength);
  const boundary = Math.sqrt(6 / fanIn);
  const targetDeviation = boundary / Math.sqrt(3);
  const weights = new Float32Array(config.filters * inputChannels * kernelLength);
  const initialization: ConvolutionInitialization =
    config.initialization === "template" ? "template" : "he";

  for (let filter = 0; filter < config.filters; filter++) {
    const source = config.kernels[filter] ?? [];
    const template = initialization === "template"
      ? normalizedTemplate(
          Array.from({ length: kernelLength }, (_, index) =>
            Number.isFinite(source[index]) ? source[index] : 0),
          targetDeviation,
        )
      : null;
    for (let channel = 0; channel < inputChannels; channel++) {
      for (let index = 0; index < kernelLength; index++) {
        const randomWeight = (random() * 2 - 1) * boundary;
        const value = template
          ? (template[index] + randomWeight) / Math.SQRT2
          : randomWeight;
        weights[(filter * inputChannels + channel) * kernelLength + index] = value;
      }
    }
  }
  return weights;
}
