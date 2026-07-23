import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";

const require = createRequire(import.meta.url);
const mnist = require("mnist");

const INPUTS = 784;
const HIDDEN = 32;
const OUTPUTS = 10;
const EPOCHS = 18;
const { training, test } = mnist.set(9000, 1000);

let seed = 0x5eeda11;
function random() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
}

function createWeights(length, fanIn) {
  const scale = Math.sqrt(2 / fanIn);
  return Float32Array.from({ length }, () => (random() * 2 - 1) * scale);
}

const weights1 = createWeights(INPUTS * HIDDEN, INPUTS);
const biases1 = new Float32Array(HIDDEN);
const weights2 = createWeights(HIDDEN * OUTPUTS, HIDDEN);
const biases2 = new Float32Array(OUTPUTS);

function sparse(sample) {
  const pixels = [];
  for (let index = 0; index < sample.input.length; index++) {
    const value = sample.input[index];
    if (value > 0.008) pixels.push([index, value]);
  }
  return { pixels, label: sample.output.indexOf(1) };
}

const sparseTraining = training.map(sparse);

function infer(input) {
  const hidden = new Float32Array(HIDDEN);
  for (let unit = 0; unit < HIDDEN; unit++) {
    let sum = biases1[unit];
    const offset = unit * INPUTS;
    for (let pixel = 0; pixel < INPUTS; pixel++) sum += weights1[offset + pixel] * input[pixel];
    hidden[unit] = Math.max(0, sum);
  }
  const logits = new Float32Array(OUTPUTS);
  let max = -Infinity;
  for (let output = 0; output < OUTPUTS; output++) {
    let sum = biases2[output];
    const offset = output * HIDDEN;
    for (let unit = 0; unit < HIDDEN; unit++) sum += weights2[offset + unit] * hidden[unit];
    logits[output] = sum;
    max = Math.max(max, sum);
  }
  let total = 0;
  for (let output = 0; output < OUTPUTS; output++) total += Math.exp(logits[output] - max);
  return Array.from(logits, (value) => Math.exp(value - max) / total);
}

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  const learningRate = 0.034 * Math.pow(0.91, epoch);
  for (let cursor = sparseTraining.length - 1; cursor > 0; cursor--) {
    const swap = Math.floor(random() * (cursor + 1));
    [sparseTraining[cursor], sparseTraining[swap]] = [sparseTraining[swap], sparseTraining[cursor]];
  }

  for (const sample of sparseTraining) {
    const hidden = new Float32Array(HIDDEN);
    for (let unit = 0; unit < HIDDEN; unit++) {
      let sum = biases1[unit];
      const offset = unit * INPUTS;
      for (const [pixel, value] of sample.pixels) sum += weights1[offset + pixel] * value;
      hidden[unit] = Math.max(0, sum);
    }

    const probabilities = new Float32Array(OUTPUTS);
    let max = -Infinity;
    for (let output = 0; output < OUTPUTS; output++) {
      let sum = biases2[output];
      const offset = output * HIDDEN;
      for (let unit = 0; unit < HIDDEN; unit++) sum += weights2[offset + unit] * hidden[unit];
      probabilities[output] = sum;
      max = Math.max(max, sum);
    }
    let total = 0;
    for (let output = 0; output < OUTPUTS; output++) total += Math.exp(probabilities[output] - max);
    for (let output = 0; output < OUTPUTS; output++) probabilities[output] = Math.exp(probabilities[output] - max) / total;

    const hiddenGradient = new Float32Array(HIDDEN);
    for (let output = 0; output < OUTPUTS; output++) {
      const delta = probabilities[output] - (output === sample.label ? 1 : 0);
      const offset = output * HIDDEN;
      for (let unit = 0; unit < HIDDEN; unit++) hiddenGradient[unit] += weights2[offset + unit] * delta;
      for (let unit = 0; unit < HIDDEN; unit++) weights2[offset + unit] -= learningRate * delta * hidden[unit];
      biases2[output] -= learningRate * delta;
    }

    for (let unit = 0; unit < HIDDEN; unit++) {
      if (hidden[unit] <= 0) continue;
      const delta = hiddenGradient[unit];
      const offset = unit * INPUTS;
      for (const [pixel, value] of sample.pixels) weights1[offset + pixel] -= learningRate * delta * value;
      biases1[unit] -= learningRate * delta;
    }
  }

  let correct = 0;
  for (const sample of test) {
    const prediction = infer(sample.input);
    if (prediction.indexOf(Math.max(...prediction)) === sample.output.indexOf(1)) correct++;
  }
  console.log(`epoch ${String(epoch + 1).padStart(2, "0")}  accuracy ${(correct / test.length * 100).toFixed(1)}%`);
}

const round = (values) => Array.from(values, (value) => Number(value.toFixed(6)));
const samples = Array.from({ length: 10 }, (_, digit) => mnist[digit].get(3).map((value) => Number(value.toFixed(3))));
let correct = 0;
for (const sample of test) {
  const prediction = infer(sample.input);
  if (prediction.indexOf(Math.max(...prediction)) === sample.output.indexOf(1)) correct++;
}

const model = {
  architecture: [INPUTS, HIDDEN, OUTPUTS],
  weights: [round(weights1), round(weights2)],
  biases: [round(biases1), round(biases2)],
  samples,
  accuracy: Number((correct / test.length).toFixed(4)),
};

await mkdir("public", { recursive: true });
await writeFile("public/model.json", JSON.stringify(model));
console.log(`wrote public/model.json (${(correct / test.length * 100).toFixed(1)}% test accuracy)`);
