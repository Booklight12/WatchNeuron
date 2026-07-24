import assert from "node:assert/strict";
import { createServer } from "vite";

let submissionCount = 0;
let bufferCount = 0;
let computePassCount = 0;

class MockBuffer {
  constructor(size) {
    this.size = size;
    this.bytes = new Uint8Array(size);
    bufferCount++;
  }

  async mapAsync() {}

  getMappedRange(offset = 0, size = this.size) {
    return this.bytes.buffer.slice(offset, offset + size);
  }

  unmap() {}

  destroy() {}
}

const device = {
  queue: {
    writeBuffer(target, targetOffset, source, sourceOffset = 0, size) {
      const sourceBytes = new Uint8Array(
        source,
        sourceOffset,
        size ?? source.byteLength - sourceOffset,
      );
      target.bytes.set(sourceBytes, targetOffset);
    },
    submit() {
      submissionCount++;
    },
  },
  createBindGroupLayout: () => ({}),
  createPipelineLayout: () => ({}),
  createShaderModule: () => ({
    getCompilationInfo: async () => ({ messages: [] }),
  }),
  createComputePipeline: ({ compute }) => ({
    entryPoint: compute.entryPoint,
    getBindGroupLayout: () => ({}),
  }),
  createBindGroup: ({ entries }) => ({ entries }),
  createBuffer: ({ size }) => new MockBuffer(size),
  createCommandEncoder: () => ({
    beginComputePass: () => {
      computePassCount++;
      return {
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() {},
        end() {},
      };
    },
    copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
      target.bytes.set(
        source.bytes.subarray(sourceOffset, sourceOffset + size),
        targetOffset,
      );
    },
    finish: () => ({}),
  }),
};

Object.defineProperty(globalThis.navigator, "gpu", {
  configurable: true,
  value: {
    requestAdapter: async () => ({
      info: { description: "mock-gpu", isFallbackAdapter: false },
      requestDevice: async () => device,
    }),
  },
});

const vite = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "silent",
});

try {
  const { WebGpuBatchExecutor } = await vite.ssrLoadModule("/src/lib/webgpu-batch.ts");
  const executor = await WebGpuBatchExecutor.create();
  const descriptors = [
    {
      type: "conv",
      inputSize: 784,
      outputSize: 392,
      activationKind: 1,
      inputWidth: 28,
      inputHeight: 28,
      inputChannels: 1,
      filters: 2,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      trainable: true,
      weights: new Float32Array(18),
      biases: new Float32Array(2),
    },
    {
      type: "pool",
      inputSize: 392,
      outputSize: 98,
      inputWidth: 14,
      inputHeight: 14,
      inputChannels: 2,
      outputWidth: 7,
      outputHeight: 7,
      kernelSize: 2,
      stride: 2,
      padding: 0,
      poolingKind: 0,
    },
    {
      type: "dense",
      inputSize: 98,
      outputSize: 10,
      activationKind: 0,
      dropoutRate: 0,
      weights: new Float32Array(980),
      biases: new Float32Array(10),
    },
  ];
  const graph = executor.createTrainingGraph(descriptors, 4, false);
  const persistentBufferCount = bufferCount;
  const input = new Float32Array(2 * 784);
  const labels = new Int32Array([0, 1]);

  const forwardSubmissions = submissionCount;
  const forwardPasses = computePassCount;
  const forward = await graph.forward(input, labels, 2);
  assert.equal(submissionCount - forwardSubmissions, 1, "forward graph must submit once per batch");
  assert.equal(computePassCount - forwardPasses, 1, "forward graph must use one compute pass");
  assert.equal(bufferCount - persistentBufferCount, 1, "forward graph may allocate only one staging buffer");
  assert.equal(forward.probabilities.length, 20);
  assert.equal(forward.losses.length, 2);

  const buffersAfterForward = bufferCount;
  const trainingSubmissions = submissionCount;
  const trainingPasses = computePassCount;
  const training = await graph.train(input, labels, 2, 2, false, {
    kind: 2,
    learningRate: 0.003,
    momentum: 0.9,
    decay: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    beta1Correction: 0.1,
    beta2Correction: 0.001,
    gradientScale: 0.5,
    weightDecay: 0.01,
  });
  assert.equal(submissionCount - trainingSubmissions, 1, "training graph must submit once per batch");
  assert.equal(computePassCount - trainingPasses, 1, "training graph must use one compute pass");
  assert.equal(bufferCount - buffersAfterForward, 1, "training graph may allocate only one staging buffer");
  assert.equal(training.losses.length, 2);

  const buffersBeforeUpload = bufferCount;
  graph.uploadParameters(descriptors);
  assert.equal(bufferCount, buffersBeforeUpload, "parameter upload must reuse persistent GPU buffers");

  console.log("WebGPU persistent graph verification passed");
} finally {
  await vite.close();
}
