import { readFile } from "node:fs/promises";

const model = JSON.parse(await readFile("public/model.json", "utf8"));
const wasmBytes = await readFile("public/neuron_kernel.wasm");
const simdWasmBytes = await readFile("public/neuron_kernel_simd.wasm");
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const { instance: simdInstance } = await WebAssembly.instantiate(simdWasmBytes, {});
const { memory, matvec, activate, __heap_base: heapBase } = instance.exports;
if (
  typeof instance.exports.forward_sparse !== "function" ||
  typeof instance.exports.forward_dense_block !== "function" ||
  typeof instance.exports.forward_dense_training !== "function" ||
  typeof instance.exports.train_sample !== "function" ||
  typeof instance.exports.train_dense_from_gradient !== "function" ||
  typeof instance.exports.conv2d_forward !== "function" ||
  typeof instance.exports.conv2d_train !== "function" ||
  typeof instance.exports.conv2d_forward_batch !== "function" ||
  typeof instance.exports.conv2d_train_batch !== "function" ||
  typeof instance.exports.pool2d_forward !== "function" ||
  typeof instance.exports.pool2d_backward !== "function" ||
  typeof instance.exports.pool2d_forward_batch !== "function" ||
  typeof instance.exports.pool2d_backward_batch !== "function" ||
  typeof instance.exports.dense_forward_batch !== "function" ||
  typeof instance.exports.dense_backward_batch !== "function" ||
  typeof instance.exports.output_loss_batch !== "function" ||
  typeof instance.exports.apply_optimizer !== "function" ||
  typeof instance.exports.simd_enabled !== "function" ||
  typeof instance.exports.set_math_mode !== "function" ||
  typeof instance.exports.math_mode !== "function" ||
  typeof simdInstance.exports.simd_enabled !== "function" ||
  typeof simdInstance.exports.set_math_mode !== "function" ||
  typeof simdInstance.exports.math_mode !== "function" ||
  instance.exports.simd_enabled() !== 0 ||
  simdInstance.exports.simd_enabled() !== 1
) {
  throw new Error("Zig/Wasm scalar or SIMD exports are missing");
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

activationProbe.set([-3.25, 0.37, 2.75]);
instance.exports.set_math_mode(0);
activate(activationProbePtr, activationProbe.length, 3);
const fastSigmoid = Array.from(activationProbe);
activationProbe.set([-3.25, 0.37, 2.75]);
instance.exports.set_math_mode(1);
activate(activationProbePtr, activationProbe.length, 3);
const fullSigmoid = Array.from(activationProbe);
const expectedSigmoid = [-3.25, 0.37, 2.75].map((value) => 1 / (1 + Math.exp(-value)));
if (
  instance.exports.math_mode() !== 1 ||
  !fullSigmoid.every((value, index) => Math.abs(value - expectedSigmoid[index]) < 1e-6) ||
  !fullSigmoid.some((value, index) => Math.abs(value - fastSigmoid[index]) > 1e-5)
) {
  throw new Error("Wasm full-precision sigmoid mode did not use the standard implementation");
}
activationProbe.set([-2.4, 0.45, 3.1]);
activate(activationProbePtr, activationProbe.length, 4);
if (!Array.from(activationProbe).every((value, index) =>
  Math.abs(value - Math.tanh([-2.4, 0.45, 3.1][index])) < 1e-6
)) {
  throw new Error("Wasm full-precision tanh mode did not use the standard implementation");
}
simdInstance.exports.set_math_mode(1);
if (simdInstance.exports.math_mode() !== 1) {
  throw new Error("Wasm SIMD full-precision mode could not be selected");
}
instance.exports.set_math_mode(0);
simdInstance.exports.set_math_mode(0);
console.log("Wasm math modes: fast approximation and full precision passed");

function saturatedSigmoidLossProbe(exports) {
  const { memory: probeMemory, __heap_base: probeHeapBase } = exports;
  let cursor = align(Number(probeHeapBase.value));
  const allocate = (bytes, alignment = 16) => {
    cursor = Math.ceil(cursor / alignment) * alignment;
    const pointer = cursor;
    cursor += bytes;
    return pointer;
  };
  const allocateTable = () => allocate(4, 4);
  const inputSizesPtr = allocateTable();
  const outputSizesPtr = allocateTable();
  const activationKindsPtr = allocateTable();
  const weightPointersPtr = allocateTable();
  const biasPointersPtr = allocateTable();
  const activationPointersPtr = allocateTable();
  const preactivationPointersPtr = allocateTable();
  const deltaPointersPtr = allocateTable();
  const weightGradientPointersPtr = allocateTable();
  const biasGradientPointersPtr = allocateTable();
  const dropoutRatesPtr = allocateTable();
  const dropoutMaskPointersPtr = allocateTable();
  const sampleIndicesPtr = allocate(2, 2);
  const sampleValuesPtr = allocate(4);
  const weightsPtr = allocate(2 * 4);
  const biasesPtr = allocate(2 * 4);
  const activationsPtr = allocate(2 * 4);
  const preactivationsPtr = allocate(2 * 4);
  const deltasPtr = allocate(2 * 4);
  const weightGradientsPtr = allocate(2 * 4);
  const biasGradientsPtr = allocate(2 * 4);
  const dropoutMaskPtr = allocate(2 * 4);
  const inputGradientPtr = allocate(4);
  if (cursor > probeMemory.buffer.byteLength) {
    probeMemory.grow(Math.ceil((cursor - probeMemory.buffer.byteLength) / 65536));
  }
  const integers = new Int32Array(probeMemory.buffer);
  const pointers = new Uint32Array(probeMemory.buffer);
  const floats = new Float32Array(probeMemory.buffer);
  integers[inputSizesPtr / 4] = 1;
  integers[outputSizesPtr / 4] = 2;
  integers[activationKindsPtr / 4] = 0;
  pointers[weightPointersPtr / 4] = weightsPtr;
  pointers[biasPointersPtr / 4] = biasesPtr;
  pointers[activationPointersPtr / 4] = activationsPtr;
  pointers[preactivationPointersPtr / 4] = preactivationsPtr;
  pointers[deltaPointersPtr / 4] = deltasPtr;
  pointers[weightGradientPointersPtr / 4] = weightGradientsPtr;
  pointers[biasGradientPointersPtr / 4] = biasGradientsPtr;
  pointers[dropoutMaskPointersPtr / 4] = dropoutMaskPtr;
  new Uint16Array(probeMemory.buffer, sampleIndicesPtr, 1)[0] = 0;
  floats[sampleValuesPtr / 4] = 1;
  floats.set([100, -100], weightsPtr / 4);
  floats.fill(0, biasesPtr / 4, biasesPtr / 4 + 2);
  floats[dropoutRatesPtr / 4] = 0;
  floats.fill(1, dropoutMaskPtr / 4, dropoutMaskPtr / 4 + 2);
  exports.set_math_mode(1);
  const loss = exports.train_sample(
    1,
    sampleIndicesPtr,
    sampleValuesPtr,
    1,
    0,
    inputSizesPtr,
    outputSizesPtr,
    activationKindsPtr,
    weightPointersPtr,
    biasPointersPtr,
    activationPointersPtr,
    preactivationPointersPtr,
    deltaPointersPtr,
    weightGradientPointersPtr,
    biasGradientPointersPtr,
    1,
    1,
    inputGradientPtr,
    dropoutRatesPtr,
    dropoutMaskPointersPtr,
    1,
    0,
  );
  exports.set_math_mode(0);
  return loss;
}

const scalarSaturatedSigmoidLoss = saturatedSigmoidLossProbe(instance.exports);
const simdSaturatedSigmoidLoss = saturatedSigmoidLossProbe(simdInstance.exports);
if (
  !Number.isFinite(scalarSaturatedSigmoidLoss) ||
  !Number.isFinite(simdSaturatedSigmoidLoss)
) {
  throw new Error(
    `Wasm saturated Sigmoid/BCE loss was not finite: scalar=${scalarSaturatedSigmoidLoss}, SIMD=${simdSaturatedSigmoidLoss}`,
  );
}
console.log("Wasm saturated Sigmoid/BCE loss remained finite in scalar and SIMD kernels");
console.log("Zig/Wasm training exports: segmented Dense forward/backprop + train_sample + Conv2D forward/backprop");

const convInputPtr = align(activationProbePtr + 64);
const convWeightsPtr = align(convInputPtr + 25 * 4);
const convBiasPtr = align(convWeightsPtr + 9 * 4);
const convOutputPtr = align(convBiasPtr + 4);
const convPreactivationPtr = align(convOutputPtr + 25 * 4);
const wasmFloats = new Float32Array(memory.buffer);
wasmFloats.fill(0, convInputPtr / 4, convInputPtr / 4 + 25);
wasmFloats[convInputPtr / 4 + 12] = 1;
wasmFloats.set([0, 0, 0, 0, 1, 0, 0, 0, 0], convWeightsPtr / 4);
wasmFloats[convBiasPtr / 4] = 0;
instance.exports.conv2d_forward(
  convInputPtr,
  convWeightsPtr,
  convBiasPtr,
  convOutputPtr,
  convPreactivationPtr,
  5,
  5,
  1,
  1,
  3,
  1,
  1,
  0,
);
const convOutput = Array.from(wasmFloats.subarray(convOutputPtr / 4, convOutputPtr / 4 + 25));
if (Math.abs(convOutput[12] - 1) > 1e-6 || convOutput.some((value, index) => index !== 12 && Math.abs(value) > 1e-6)) {
  throw new Error("Wasm Conv2D identity forward probe failed");
}
console.log("Wasm Conv2D forward probe: 5x5 identity kernel passed");

const poolInputPtr = align(convPreactivationPtr + 25 * 4);
const poolOutputPtr = align(poolInputPtr + 16 * 4);
const poolIndicesPtr = align(poolOutputPtr + 4 * 4);
const poolOutputGradientPtr = align(poolIndicesPtr + 4 * 4);
const poolInputGradientPtr = align(poolOutputGradientPtr + 4 * 4);
const poolInput = new Float32Array(memory.buffer, poolInputPtr, 16);
const poolOutput = new Float32Array(memory.buffer, poolOutputPtr, 4);
const poolIndices = new Uint32Array(memory.buffer, poolIndicesPtr, 4);
const poolOutputGradient = new Float32Array(memory.buffer, poolOutputGradientPtr, 4);
const poolInputGradient = new Float32Array(memory.buffer, poolInputGradientPtr, 16);
poolInput.set(Array.from({ length: 16 }, (_, index) => index + 1));

function assertArrayClose(actual, expected, label, tolerance = 1e-6) {
  if (
    actual.length !== expected.length ||
    actual.some((value, index) => !Number.isFinite(value) || Math.abs(value - expected[index]) > tolerance)
  ) {
    throw new Error(`${label} failed: expected ${expected.join(", ")}; received ${actual.join(", ")}`);
  }
}

instance.exports.pool2d_forward(
  poolInputPtr,
  poolOutputPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  2,
  2,
  0,
  0,
);
assertArrayClose(Array.from(poolOutput), [6, 8, 14, 16], "Wasm MaxPool2D forward probe");
assertArrayClose(Array.from(poolIndices), [5, 7, 13, 15], "Wasm MaxPool2D argmax probe");
poolOutputGradient.fill(1);
instance.exports.pool2d_backward(
  poolOutputGradientPtr,
  poolInputGradientPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  2,
  2,
  0,
  0,
);
assertArrayClose(
  Array.from(poolInputGradient),
  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
  "Wasm MaxPool2D backward probe",
);

instance.exports.pool2d_forward(
  poolInputPtr,
  poolOutputPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  2,
  2,
  0,
  1,
);
assertArrayClose(Array.from(poolOutput), [3.5, 5.5, 11.5, 13.5], "Wasm AvgPool2D forward probe");
instance.exports.pool2d_backward(
  poolOutputGradientPtr,
  poolInputGradientPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  2,
  2,
  0,
  1,
);
assertArrayClose(
  Array.from(poolInputGradient),
  new Array(16).fill(0.25),
  "Wasm AvgPool2D backward probe",
);

instance.exports.pool2d_forward(
  poolInputPtr,
  poolOutputPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  1,
  1,
  0,
  2,
);
assertArrayClose([poolOutput[0]], [8.5], "Wasm GAP forward probe");
poolOutputGradient[0] = 1;
instance.exports.pool2d_backward(
  poolOutputGradientPtr,
  poolInputGradientPtr,
  poolIndicesPtr,
  4,
  4,
  1,
  1,
  1,
  0,
  2,
);
assertArrayClose(
  Array.from(poolInputGradient),
  new Array(16).fill(1 / 16),
  "Wasm GAP backward probe",
);

poolInput[0] = 7;
instance.exports.pool2d_forward(
  poolInputPtr,
  poolOutputPtr,
  poolIndicesPtr,
  1,
  1,
  1,
  4,
  1,
  0,
  0,
);
assertArrayClose([poolOutput[0]], [7], "Wasm oversized MaxPool2D window probe");
poolOutputGradient[0] = 1;
instance.exports.pool2d_backward(
  poolOutputGradientPtr,
  poolInputGradientPtr,
  poolIndicesPtr,
  1,
  1,
  1,
  4,
  1,
  0,
  0,
);
assertArrayClose([poolInputGradient[0]], [1], "Wasm oversized MaxPool2D backward probe");
console.log("Wasm Pool2D probes: MaxPool2D, AvgPool2D, GAP, and oversized windows passed");

let batchProbeCursor = align(poolInputGradientPtr + 16 * 4);
const allocateBatchProbe = (bytes, alignment = 16) => {
  batchProbeCursor = Math.ceil(batchProbeCursor / alignment) * alignment;
  const pointer = batchProbeCursor;
  batchProbeCursor += bytes;
  return pointer;
};
const batchInputPtr = allocateBatchProbe(2 * 16 * 4);
const batchConvWeightsPtr = allocateBatchProbe(4);
const batchConvBiasPtr = allocateBatchProbe(4);
const batchConvOutputPtr = allocateBatchProbe(2 * 16 * 4);
const batchConvPreactivationPtr = allocateBatchProbe(2 * 16 * 4);
const batchPoolOutputPtr = allocateBatchProbe(2 * 4 * 4);
const batchPoolIndicesPtr = allocateBatchProbe(2 * 4 * 4, 4);
const batchDenseWeightsPtr = allocateBatchProbe(2 * 4 * 4);
const batchDenseBiasPtr = allocateBatchProbe(2 * 4);
const batchDenseOutputPtr = allocateBatchProbe(2 * 2 * 4);
const batchDensePreactivationPtr = allocateBatchProbe(2 * 2 * 4);
const batchDenseMaskPtr = allocateBatchProbe(2 * 2 * 4);
const batchDenseDeltaPtr = allocateBatchProbe(2 * 2 * 4);
const batchDenseInputGradientPtr = allocateBatchProbe(2 * 4 * 4);
const batchDenseWeightGradientPtr = allocateBatchProbe(2 * 4 * 4);
const batchDenseBiasGradientPtr = allocateBatchProbe(2 * 4);
const batchLabelsPtr = allocateBatchProbe(2 * 4, 4);
const batchLossesPtr = allocateBatchProbe(2 * 4);
const batchProbeFloats = new Float32Array(memory.buffer);
new Float32Array(memory.buffer, batchInputPtr, 32).set([
  ...Array.from({ length: 16 }, (_, index) => index + 1),
  ...Array.from({ length: 16 }, (_, index) => index + 101),
]);
new Float32Array(memory.buffer, batchConvWeightsPtr, 1)[0] = 2;
new Float32Array(memory.buffer, batchConvBiasPtr, 1)[0] = 1;
instance.exports.set_math_mode(1);
instance.exports.conv2d_forward_batch(
  batchInputPtr,
  batchConvWeightsPtr,
  batchConvBiasPtr,
  batchConvOutputPtr,
  batchConvPreactivationPtr,
  2,
  4,
  4,
  1,
  1,
  1,
  1,
  0,
  0,
);
assertArrayClose(
  Array.from(new Float32Array(memory.buffer, batchConvOutputPtr, 32)),
  [
    ...Array.from({ length: 16 }, (_, index) => (index + 1) * 2 + 1),
    ...Array.from({ length: 16 }, (_, index) => (index + 101) * 2 + 1),
  ],
  "Wasm [B,C,H,W] Conv2D batch isolation probe",
);
instance.exports.pool2d_forward_batch(
  batchConvOutputPtr,
  batchPoolOutputPtr,
  batchPoolIndicesPtr,
  2,
  4,
  4,
  1,
  2,
  2,
  0,
  0,
);
assertArrayClose(
  Array.from(new Float32Array(memory.buffer, batchPoolOutputPtr, 8)),
  [13, 17, 29, 33, 213, 217, 229, 233],
  "Wasm [B,C,H,W] MaxPool2D batch isolation probe",
);
new Float32Array(memory.buffer, batchDenseWeightsPtr, 8).fill(0);
new Float32Array(memory.buffer, batchDenseBiasPtr, 2).fill(0);
new Int32Array(memory.buffer, batchLabelsPtr, 2).set([0, 1]);
instance.exports.dense_forward_batch(
  batchPoolOutputPtr,
  batchDenseWeightsPtr,
  batchDenseBiasPtr,
  batchDenseOutputPtr,
  batchDensePreactivationPtr,
  batchDenseMaskPtr,
  2,
  4,
  2,
  0,
  0,
  1,
  0,
  1,
);
instance.exports.output_loss_batch(
  batchDenseOutputPtr,
  batchDenseDeltaPtr,
  batchLabelsPtr,
  batchLossesPtr,
  2,
  2,
  0,
);
batchProbeFloats.fill(
  0,
  batchDenseWeightGradientPtr / 4,
  batchDenseWeightGradientPtr / 4 + 10,
);
instance.exports.dense_backward_batch(
  batchPoolOutputPtr,
  batchDenseWeightsPtr,
  batchDensePreactivationPtr,
  batchDenseDeltaPtr,
  batchDenseInputGradientPtr,
  batchDenseDeltaPtr,
  batchDenseWeightGradientPtr,
  batchDenseBiasGradientPtr,
  batchDenseMaskPtr,
  2,
  4,
  2,
  0,
);
assertArrayClose(
  Array.from(new Float32Array(memory.buffer, batchDenseWeightGradientPtr, 8)),
  [100, 100, 100, 100, -100, -100, -100, -100],
  "Wasm [B,F] shared Dense batch-gradient probe",
  1e-4,
);
assertArrayClose(
  Array.from(new Float32Array(memory.buffer, batchLossesPtr, 2)),
  [Math.log(2), Math.log(2)],
  "Wasm batch loss probe",
  1e-5,
);
instance.exports.set_math_mode(0);
console.log("Wasm full batch path: [B,C,H,W] Conv2D/Pool2D and [B,F] Dense gradients passed");

const optimizerParametersPtr = align(batchProbeCursor);
const optimizerGradientsPtr = align(optimizerParametersPtr + 3 * 4);
const optimizerFirstPtr = align(optimizerGradientsPtr + 3 * 4);
const optimizerSecondPtr = align(optimizerFirstPtr + 3 * 4);
const optimizerParameters = new Float32Array(memory.buffer, optimizerParametersPtr, 3);
const optimizerGradients = new Float32Array(memory.buffer, optimizerGradientsPtr, 3);
const optimizerFirst = new Float32Array(memory.buffer, optimizerFirstPtr, 3);
const optimizerSecond = new Float32Array(memory.buffer, optimizerSecondPtr, 3);
optimizerParameters.set([2, -4, 3]);
optimizerGradients.fill(0);
optimizerFirst.fill(0);
optimizerSecond.fill(0);
instance.exports.apply_optimizer(
  optimizerParametersPtr,
  optimizerGradientsPtr,
  optimizerFirstPtr,
  optimizerSecondPtr,
  2,
  0,
  0.1,
  0.9,
  0.9,
  0.9,
  0.999,
  1e-8,
  1,
  1,
  1,
  0.5,
);
instance.exports.apply_optimizer(
  optimizerParametersPtr + 2 * 4,
  optimizerGradientsPtr + 2 * 4,
  optimizerFirstPtr + 2 * 4,
  optimizerSecondPtr + 2 * 4,
  1,
  0,
  0.1,
  0.9,
  0.9,
  0.9,
  0.999,
  1e-8,
  1,
  1,
  1,
  0,
);
assertArrayClose(Array.from(optimizerParameters), [1.9, -3.8, 3], "Wasm Weight Decay probe");
console.log("Wasm optimizer probe: decoupled Weight Decay shrinks weights but not biases");

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

function benchmarkMatvec(wasmInstance) {
  const inputSize = 784;
  const outputSize = 128;
  const exports = wasmInstance.exports;
  let cursor = align(Number(exports.__heap_base.value));
  const inputPtr = cursor;
  cursor = align(cursor + inputSize * 4);
  const weightsPtr = cursor;
  cursor = align(cursor + inputSize * outputSize * 4);
  const biasesPtr = cursor;
  cursor = align(cursor + outputSize * 4);
  const outputPtr = cursor;
  cursor += outputSize * 4;
  if (cursor > exports.memory.buffer.byteLength) {
    exports.memory.grow(Math.ceil((cursor - exports.memory.buffer.byteLength) / 65536));
  }
  const floats = new Float32Array(exports.memory.buffer);
  const input = Float32Array.from({ length: inputSize }, (_, index) => ((index * 17) % 101 - 50) / 50);
  const weights = Float32Array.from(
    { length: inputSize * outputSize },
    (_, index) => ((index * 29) % 127 - 63) / 200,
  );
  const biases = Float32Array.from({ length: outputSize }, (_, index) => (index - 64) / 100);
  floats.set(input, inputPtr / 4);
  floats.set(weights, weightsPtr / 4);
  floats.set(biases, biasesPtr / 4);
  for (let index = 0; index < 100; index++) {
    exports.matvec(inputPtr, weightsPtr, biasesPtr, outputPtr, inputSize, outputSize);
  }
  const timings = [];
  for (let round = 0; round < 5; round++) {
    const startedAt = performance.now();
    for (let index = 0; index < 500; index++) {
      exports.matvec(inputPtr, weightsPtr, biasesPtr, outputPtr, inputSize, outputSize);
    }
    timings.push(performance.now() - startedAt);
  }
  timings.sort((left, right) => left - right);
  return {
    output: Array.from(floats.subarray(outputPtr / 4, outputPtr / 4 + outputSize)),
    elapsedMs: timings[2],
  };
}

const scalarBenchmark = benchmarkMatvec(instance);
const simdBenchmark = benchmarkMatvec(simdInstance);
const maximumDifference = scalarBenchmark.output.reduce(
  (maximum, value, index) => Math.max(maximum, Math.abs(value - simdBenchmark.output[index])),
  0,
);
if (maximumDifference > 0.001) {
  throw new Error(`SIMD matvec diverged from scalar output by ${maximumDifference}`);
}
console.log(
  `Wasm SIMD matvec parity passed; median ${scalarBenchmark.elapsedMs.toFixed(1)} ms scalar vs ${simdBenchmark.elapsedMs.toFixed(1)} ms SIMD (${(scalarBenchmark.elapsedMs / simdBenchmark.elapsedMs).toFixed(2)}x)`,
);
