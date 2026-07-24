import { readFile } from "node:fs/promises";

const record = JSON.parse(await readFile("public/default-model.json", "utf8"));
const fallback = JSON.parse(await readFile("public/model.json", "utf8"));

if (
  record.name !== "完整训练 · 07/24 15:58" ||
  record.source !== "complete" ||
  record.progress?.epoch !== 100 ||
  record.model?.trained !== true
) {
  throw new Error("Bundled default model metadata is invalid");
}

function finiteVector(values, expectedLength, label) {
  if (
    !Array.isArray(values) ||
    values.length !== expectedLength ||
    !values.every(Number.isFinite)
  ) {
    throw new Error(`${label} has invalid shape or non-finite values`);
  }
}

for (const [index, layer] of record.model.layers.entries()) {
  finiteVector(layer.weights, layer.inputSize * layer.outputSize, `Dense ${index + 1} weights`);
  finiteVector(layer.biases, layer.outputSize, `Dense ${index + 1} biases`);
}

for (const [index, layer] of record.model.convolutions.entries()) {
  const expectedOutputWidth = Math.floor(
    (layer.inputWidth + layer.padding * 2 - layer.kernelSize) / layer.stride,
  ) + 1;
  const expectedOutputHeight = Math.floor(
    (layer.inputHeight + layer.padding * 2 - layer.kernelSize) / layer.stride,
  ) + 1;
  if (layer.outputWidth !== expectedOutputWidth || layer.outputHeight !== expectedOutputHeight) {
    throw new Error(`Conv2D ${index + 1} output shape is inconsistent`);
  }
  finiteVector(
    layer.weights,
    layer.filters * layer.inputChannels * layer.kernelSize ** 2,
    `Conv2D ${index + 1} weights`,
  );
  finiteVector(layer.biases, layer.filters, `Conv2D ${index + 1} biases`);
}

function activate(value, kind) {
  if (kind === "linear") return value;
  if (kind === "relu") return Math.max(0, value);
  if (kind === "leakyRelu") return value >= 0 ? value : value * 0.08;
  throw new Error(`Unsupported activation in bundled default model: ${kind}`);
}

function runConvolution(layer, input) {
  const output = new Array(layer.outputWidth * layer.outputHeight * layer.filters);
  for (let filter = 0; filter < layer.filters; filter++) {
    for (let outputY = 0; outputY < layer.outputHeight; outputY++) {
      for (let outputX = 0; outputX < layer.outputWidth; outputX++) {
        let sum = layer.biases[filter];
        for (let channel = 0; channel < layer.inputChannels; channel++) {
          for (let kernelY = 0; kernelY < layer.kernelSize; kernelY++) {
            const inputY = outputY * layer.stride + kernelY - layer.padding;
            if (inputY < 0 || inputY >= layer.inputHeight) continue;
            for (let kernelX = 0; kernelX < layer.kernelSize; kernelX++) {
              const inputX = outputX * layer.stride + kernelX - layer.padding;
              if (inputX < 0 || inputX >= layer.inputWidth) continue;
              const inputIndex = channel * layer.inputWidth * layer.inputHeight +
                inputY * layer.inputWidth + inputX;
              const weightIndex = ((filter * layer.inputChannels + channel) * layer.kernelSize +
                kernelY) * layer.kernelSize + kernelX;
              sum += layer.weights[weightIndex] * input[inputIndex];
            }
          }
        }
        const outputIndex = filter * layer.outputWidth * layer.outputHeight +
          outputY * layer.outputWidth + outputX;
        output[outputIndex] = activate(sum, layer.activation);
      }
    }
  }
  return output;
}

function infer(input) {
  let current = input;
  for (let layerIndex = 0; layerIndex < record.model.layers.length; layerIndex++) {
    for (const convolution of record.model.convolutions.filter(
      ({ position }) => position === layerIndex,
    )) {
      current = runConvolution(convolution, current);
    }
    const layer = record.model.layers[layerIndex];
    const output = new Array(layer.outputSize);
    for (let row = 0; row < layer.outputSize; row++) {
      let sum = layer.biases[row];
      const offset = row * layer.inputSize;
      for (let column = 0; column < layer.inputSize; column++) {
        sum += layer.weights[offset + column] * current[column];
      }
      output[row] = activate(sum, layer.activation);
    }
    current = output;
  }
  const maximum = Math.max(...current);
  const exponents = current.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

let correct = 0;
for (let digit = 0; digit < 10; digit++) {
  const probabilities = infer(fallback.samples[digit]);
  if (!probabilities.every(Number.isFinite)) {
    throw new Error("Bundled default model returned non-finite probabilities");
  }
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  if (prediction === digit) correct++;
}

if (correct < 8) {
  throw new Error(`Bundled default model recognized only ${correct}/10 demo samples`);
}

const parameters = [
  ...record.model.layers,
  ...record.model.convolutions,
].reduce((total, layer) => total + layer.weights.length + layer.biases.length, 0);

console.log(
  `Bundled default model: ${record.name}; ${parameters.toLocaleString()} parameters; ` +
  `${correct}/10 demo samples recognized; ${(record.progress.accuracy * 100).toFixed(2)}% recorded accuracy`,
);
