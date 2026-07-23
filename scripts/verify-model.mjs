import { readFile } from "node:fs/promises";

const model = JSON.parse(await readFile("public/model.json", "utf8"));
const wasmBytes = await readFile("public/neuron_kernel.wasm");
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const { memory, matvec, activate, __heap_base: heapBase } = instance.exports;
if (
  typeof instance.exports.forward_sparse !== "function" ||
  typeof instance.exports.train_sample !== "function"
) {
  throw new Error("Zig/Wasm training exports are missing");
}
const align = (value) => (value + 15) & ~15;

const initialMemoryBytes = memory.buffer.byteLength;
memory.grow(1);
if (memory.buffer.byteLength !== initialMemoryBytes + 65536) {
  throw new Error("Wasm memory did not grow by one page");
}

const scalarActivationProbes = [
  { name: "Linear", code: 0, input: [-1, 0, 1], expected: [-1, 0, 1] },
  { name: "ReLU", code: 1, input: [-1, 0, 1], expected: [0, 0, 1] },
  { name: "Leaky ReLU", code: 2, input: [-1, 0, 1], expected: [-0.08, 0, 1] },
  { name: "Sigmoid", code: 3, input: [-1, 0, 1], expected: [0.26894, 0.5, 0.73106] },
  { name: "Tanh", code: 4, input: [-1, 0, 1], expected: [-0.76159, 0, 0.76159] },
  { name: "ELU", code: 6, input: [-1, 0, 1], expected: [-0.63212, 0, 1] },
  { name: "SELU", code: 7, input: [-1, 0, 1], expected: [-1.11133, 0, 1.0507] },
  { name: "GELU", code: 8, input: [-1, 0, 1], expected: [-0.15881, 0, 0.84119] },
  { name: "SiLU", code: 9, input: [-1, 0, 1], expected: [-0.26894, 0, 0.73106] },
  { name: "Mish", code: 10, input: [-1, 0, 1], expected: [-0.3034, 0, 0.8651] },
  { name: "Softplus", code: 11, input: [-1, 0, 1], expected: [0.31326, 0.69315, 1.31326] },
  { name: "Softsign", code: 12, input: [-1, 0, 1], expected: [-0.5, 0, 0.5] },
  { name: "Hard Sigmoid", code: 13, input: [-4, 0, 4], expected: [0, 0.5, 1] },
  { name: "Hard Tanh", code: 14, input: [-2, 0, 2], expected: [-1, 0, 1] },
  { name: "ReLU6", code: 15, input: [-1, 3, 8], expected: [0, 3, 6] },
];

const activationProbePtr = align(Number(heapBase.value));
const activationProbe = new Float32Array(memory.buffer, activationProbePtr, 3);
for (const probe of scalarActivationProbes) {
  activationProbe.set(probe.input);
  activate(activationProbePtr, activationProbe.length, probe.code);
  const values = Array.from(activationProbe);
  const valid = values.every(
    (value, index) =>
      Number.isFinite(value) && Math.abs(value - probe.expected[index]) <= 0.08,
  );
  if (!valid) {
    throw new Error(`Wasm ${probe.name} activation probe failed: ${values.join(", ")}`);
  }
}

const softmaxProbePtr = activationProbePtr;
const softmaxProbe = new Float32Array(memory.buffer, softmaxProbePtr, 3);
softmaxProbe.set([-1, 0, 1]);
activate(softmaxProbePtr, softmaxProbe.length, 5);
const softmaxTotal = softmaxProbe[0] + softmaxProbe[1] + softmaxProbe[2];
if (
  !Array.from(softmaxProbe).every(Number.isFinite) ||
  Math.abs(softmaxTotal - 1) > 1e-4 ||
  !(softmaxProbe[0] < softmaxProbe[1] && softmaxProbe[1] < softmaxProbe[2])
) {
  throw new Error("Wasm Softmax activation failed its normalization probe");
}
console.log(`Wasm activation test: ${scalarActivationProbes.length + 1}/16 functions passed`);
console.log("Zig/Wasm training exports: forward_sparse + train_sample");

function infer(input) {
  const floats = new Float32Array(memory.buffer);
  let cursor = align(Number(heapBase.value));
  let inputPtr = cursor;
  floats.set(input, inputPtr / 4);
  cursor += input.length * 4;

  for (let layer = 0; layer < model.weights.length; layer++) {
    const inputSize = model.architecture[layer];
    const outputSize = model.architecture[layer + 1];
    const weights = Float32Array.from(model.weights[layer]);
    const biases = Float32Array.from(model.biases[layer]);
    const weightsPtr = align(cursor);
    cursor = weightsPtr + weights.byteLength;
    const biasesPtr = align(cursor);
    cursor = biasesPtr + biases.byteLength;
    const outputPtr = align(cursor);
    cursor = outputPtr + outputSize * 4;
    floats.set(weights, weightsPtr / 4);
    floats.set(biases, biasesPtr / 4);
    matvec(inputPtr, weightsPtr, biasesPtr, outputPtr, inputSize, outputSize);
    activate(outputPtr, outputSize, layer === model.weights.length - 1 ? 0 : 1);
    inputPtr = outputPtr;
  }

  const logits = Array.from(floats.subarray(inputPtr / 4, inputPtr / 4 + 10));
  const maximum = Math.max(...logits);
  const exponents = logits.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

let correct = 0;
for (let digit = 0; digit < 10; digit++) {
  const probabilities = infer(model.samples[digit]);
  if (!probabilities.every(Number.isFinite)) throw new Error("Wasm inference returned a non-finite value");
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  if (prediction === digit) correct++;
}

if (correct < 8) throw new Error(`Wasm smoke test recognized only ${correct}/10 bundled samples`);
console.log(`Wasm smoke test: ${correct}/10 bundled samples recognized`);
