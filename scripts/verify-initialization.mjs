import {
  createConvolutionWeights,
  createUniformWeights,
  uniformInitializationBoundary,
} from "../src/lib/initialization.ts";
import { shuffleForNextEpoch } from "../src/lib/training-order.ts";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function variance(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => {
    const difference = value - mean;
    return sum + difference * difference;
  }, 0) / values.length;
}

const fanIn = 128;
assert(
  Math.abs(uniformInitializationBoundary(fanIn, "relu") - Math.sqrt(6 / fanIn)) < 1e-12,
  "ReLU He Uniform boundary is not sqrt(6 / fan_in)",
);
assert(
  Math.abs(uniformInitializationBoundary(fanIn, "linear") - Math.sqrt(3 / fanIn)) < 1e-12,
  "Linear Xavier Uniform boundary is not sqrt(3 / fan_in)",
);

const dense = createUniformWeights(250_000, fanIn, "relu", seededRandom(0x51a7));
assert(
  Math.abs(variance(dense) / (2 / fanIn) - 1) < 0.025,
  "Dense He Uniform variance does not approach 2 / fan_in",
);

const baseConvolution = {
  filters: 64,
  kernelSize: 3,
  activation: "relu",
  kernels: Array.from({ length: 64 }, () => [0, -1, 0, -1, 4, -1, 0, -1, 0]),
};
const inputChannels = 16;
const convolutionFanIn = inputChannels * baseConvolution.kernelSize ** 2;
const heConvolution = createConvolutionWeights(
  { ...baseConvolution, initialization: "he" },
  inputChannels,
  seededRandom(0xc011),
);
assert(
  Math.abs(variance(heConvolution) / (2 / convolutionFanIn) - 1) < 0.06,
  "Conv2D He Uniform variance does not approach 2 / fan_in",
);
const kernelLength = baseConvolution.kernelSize ** 2;
assert(
  Array.from({ length: kernelLength }, (_, index) => heConvolution[index])
    .some((value, index) => value !== heConvolution[kernelLength + index]),
  "Conv2D input channels received identical kernels",
);

const templateConvolution = createConvolutionWeights(
  { ...baseConvolution, initialization: "template" },
  inputChannels,
  seededRandom(0x7e1a),
);
assert(
  Math.abs(variance(templateConvolution) / (2 / convolutionFanIn) - 1) < 0.08,
  "Template-guided Conv2D initialization does not preserve He-scale variance",
);
assert(
  Array.from({ length: kernelLength }, (_, index) => templateConvolution[index])
    .some((value, index) => value !== templateConvolution[kernelLength + index]),
  "Template-guided Conv2D initialization copied an identical kernel across channels",
);

const zeroTemplate = createConvolutionWeights(
  {
    ...baseConvolution,
    filters: 2,
    kernels: [new Array(9).fill(0), new Array(9).fill(0)],
    initialization: "template",
  },
  4,
  seededRandom(0x0bad),
);
assert(
  zeroTemplate.some((value) => value !== 0) && zeroTemplate.every(Number.isFinite),
  "A zero template created a dead all-zero Conv2D layer",
);

const order = [0, 1, 2, 3, 4];
shuffleForNextEpoch(order, () => 0.999999);
assert(order.some((value, index) => value !== index), "Epoch shuffle retained the previous order");
const previousOrder = [...order];
shuffleForNextEpoch(order, () => 0.999999);
assert(
  order.some((value, index) => value !== previousOrder[index]),
  "Consecutive epochs retained an identical order",
);

console.log(
  "He/Xavier Uniform boundaries, Dense/Conv2D variance, independent channel kernels, template fallback, and per-epoch shuffle passed",
);
