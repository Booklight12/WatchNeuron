import shaderSource from "../shaders/neuron-kernels.wgsl?raw";

type NumericArray = Float32Array | Uint32Array;

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const GPU_MAP_MODE_READ = 0x0001;
const GPU_SHADER_STAGE_COMPUTE = 0x0004;
const WORKGROUP_SIZE = 64;

export interface DenseBatchResult {
  output: Float32Array;
  preactivation: Float32Array;
  dropoutMask: Float32Array;
}

export interface LossBatchResult {
  output: Float32Array;
  delta: Float32Array;
  losses: Float32Array;
}

export interface BackwardBatchResult {
  inputGradient: Float32Array;
  delta: Float32Array;
  weightGradient: Float32Array;
  biasGradient: Float32Array;
}

export interface PoolBatchResult {
  output: Float32Array;
  indices: Uint32Array;
}

export type WebGpuLayerDescriptor =
  | {
    type: "dense";
    inputSize: number;
    outputSize: number;
    activationKind: number;
    dropoutRate: number;
    weights: Float32Array;
    biases: Float32Array;
  }
  | {
    type: "conv";
    inputSize: number;
    outputSize: number;
    activationKind: number;
    inputWidth: number;
    inputHeight: number;
    inputChannels: number;
    filters: number;
    kernelSize: number;
    stride: number;
    padding: number;
    trainable: boolean;
    weights: Float32Array;
    biases: Float32Array;
  }
  | {
    type: "pool";
    inputSize: number;
    outputSize: number;
    inputWidth: number;
    inputHeight: number;
    inputChannels: number;
    outputWidth: number;
    outputHeight: number;
    kernelSize: number;
    stride: number;
    padding: number;
    poolingKind: number;
  };

export interface WebGpuGraphResult {
  losses: Float32Array;
  outputs: Float32Array[];
  deltas: Float32Array[];
  firstInputGradient: Float32Array | null;
  gradients: Array<{
    weightGradient: Float32Array;
    biasGradient: Float32Array;
  } | null>;
}

interface DispatchResult {
  buffers: ArrayBuffer[];
}

function alignedSize(byteLength: number) {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function parameterBlock(values: number[]) {
  const block = new Uint32Array(16);
  block.set(values.slice(0, 16));
  return block;
}

function floatBits(value: number) {
  const buffer = new ArrayBuffer(4);
  new Float32Array(buffer)[0] = value;
  return new Uint32Array(buffer)[0];
}

export class WebGpuBatchExecutor {
  private readonly pipelines = new Map<string, any>();

  private constructor(
    readonly device: any,
    readonly shader: any,
    readonly pipelineLayout: any,
    readonly adapterName: string,
  ) {}

  static async create() {
    const gpu = (globalThis.navigator as Navigator & { gpu?: any }).gpu;
    if (!gpu) throw new Error("当前浏览器或 Worker 不支持 WebGPU");
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("无法获取 WebGPU 适配器");
    const device = await adapter.requestDevice();
    const bindGroupLayout = device.createBindGroupLayout({
      label: "WatchNeuron tensor buffers",
      entries: [
        ...Array.from({ length: 4 }, (_, binding) => ({
          binding,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "read-only-storage" },
        })),
        ...Array.from({ length: 4 }, (_, index) => ({
          binding: index + 4,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "storage" },
        })),
        {
          binding: 8,
          visibility: GPU_SHADER_STAGE_COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "WatchNeuron WebGPU pipeline layout",
      bindGroupLayouts: [bindGroupLayout],
    });
    const shader = device.createShaderModule({
      label: "WatchNeuron Zig-derived tensor kernels",
      code: shaderSource,
    });
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message: any) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`WGSL 编译失败：${errors.map((message: any) => message.message).join("；")}`);
    }
    const adapterInfo = adapter.info;
    const adapterName = String(
      adapterInfo?.description || adapterInfo?.device || adapterInfo?.architecture || "GPU",
    );
    return new WebGpuBatchExecutor(device, shader, pipelineLayout, adapterName);
  }

  pipeline(entryPoint: string) {
    let pipeline = this.pipelines.get(entryPoint);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        label: `WatchNeuron ${entryPoint}`,
        layout: this.pipelineLayout,
        compute: { module: this.shader, entryPoint },
      });
      this.pipelines.set(entryPoint, pipeline);
    }
    return pipeline;
  }

  createTrainingGraph(
    descriptors: WebGpuLayerDescriptor[],
    capacity: number,
    sigmoidHead: boolean,
  ) {
    return new WebGpuTrainingGraph(this, descriptors, capacity, sigmoidHead);
  }

  private createStorage(data: NumericArray, writable: boolean) {
    const size = alignedSize(data.byteLength);
    const buffer = this.device.createBuffer({
      size,
      usage: GPU_BUFFER_USAGE.STORAGE |
        GPU_BUFFER_USAGE.COPY_DST |
        (writable ? GPU_BUFFER_USAGE.COPY_SRC : 0),
    });
    if (data.byteLength > 0) {
      this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
    return buffer;
  }

  private async dispatch(
    entryPoint: string,
    items: number,
    readOnly: NumericArray[],
    writable: NumericArray[],
    params: Uint32Array,
    readIndexes: number[],
  ): Promise<DispatchResult> {
    const dummy = new Float32Array(1);
    const inputs = Array.from({ length: 4 }, (_, index) => readOnly[index] ?? dummy);
    const outputs = Array.from({ length: 4 }, (_, index) => writable[index] ?? dummy);
    const inputBuffers = inputs.map((data) => this.createStorage(data, false));
    const outputBuffers = outputs.map((data) => this.createStorage(data, true));
    const parameterBuffer = this.device.createBuffer({
      size: 64,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.device.queue.writeBuffer(parameterBuffer, 0, params.buffer, params.byteOffset, params.byteLength);
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline(entryPoint).getBindGroupLayout(0),
      entries: [
        ...inputBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        ...outputBuffers.map((buffer, index) => ({
          binding: index + 4,
          resource: { buffer },
        })),
        { binding: 8, resource: { buffer: parameterBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: `WatchNeuron ${entryPoint}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline(entryPoint));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(items / WORKGROUP_SIZE)));
    pass.end();
    const staging = readIndexes.map((index) => {
      const size = alignedSize(outputs[index].byteLength);
      const buffer = this.device.createBuffer({
        size,
        usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
      });
      encoder.copyBufferToBuffer(outputBuffers[index], 0, buffer, 0, size);
      return { buffer, size, byteLength: outputs[index].byteLength };
    });
    this.device.queue.submit([encoder.finish()]);
    const buffers = await Promise.all(staging.map(async ({ buffer, size, byteLength }) => {
      await buffer.mapAsync(GPU_MAP_MODE_READ, 0, size);
      const copy = buffer.getMappedRange(0, size).slice(0, byteLength);
      buffer.unmap();
      buffer.destroy();
      return copy;
    }));
    [...inputBuffers, ...outputBuffers, parameterBuffer].forEach((buffer) => buffer.destroy());
    return { buffers };
  }

  async denseForward(
    input: Float32Array,
    weights: Float32Array,
    biases: Float32Array,
    batchSize: number,
    inputSize: number,
    outputSize: number,
    activationKind: number,
    dropoutRate: number,
    dropoutSeed: number,
    layerSeed: number,
    training: boolean,
  ): Promise<DenseBatchResult> {
    const length = batchSize * outputSize;
    const result = await this.dispatch(
      "dense_forward",
      length,
      [input, weights, biases],
      [new Float32Array(length), new Float32Array(length), new Float32Array(length)],
      parameterBlock([
        batchSize,
        inputSize,
        outputSize,
        activationKind,
        training ? 1 : 0,
        dropoutSeed,
        layerSeed,
        floatBits(dropoutRate),
      ]),
      [0, 1, 2],
    );
    return {
      output: new Float32Array(result.buffers[0]),
      preactivation: new Float32Array(result.buffers[1]),
      dropoutMask: new Float32Array(result.buffers[2]),
    };
  }

  async outputLoss(
    logits: Float32Array,
    labels: Int32Array,
    batchSize: number,
    outputSize: number,
    sigmoidHead: boolean,
  ): Promise<LossBatchResult> {
    const length = batchSize * outputSize;
    const labelBits = new Uint32Array(labels.buffer, labels.byteOffset, labels.length);
    const result = await this.dispatch(
      "output_loss",
      length,
      [logits, new Float32Array(1), new Float32Array(1), labelBits],
      [new Float32Array(length), new Float32Array(length), new Float32Array(batchSize)],
      parameterBlock([batchSize, outputSize, sigmoidHead ? 1 : 0]),
      [0, 1, 2],
    );
    return {
      output: new Float32Array(result.buffers[0]),
      delta: new Float32Array(result.buffers[1]),
      losses: new Float32Array(result.buffers[2]),
    };
  }

  async denseBackward(
    layerInput: Float32Array,
    weights: Float32Array,
    preactivation: Float32Array,
    dropoutMask: Float32Array,
    upstream: Float32Array,
    batchSize: number,
    inputSize: number,
    outputSize: number,
    activationKind: number,
  ): Promise<BackwardBatchResult> {
    const deltaResult = await this.dispatch(
      "dense_delta",
      batchSize * outputSize,
      [upstream, preactivation, dropoutMask],
      [new Float32Array(batchSize * outputSize)],
      parameterBlock([batchSize, inputSize, outputSize, activationKind]),
      [0],
    );
    const delta = new Float32Array(deltaResult.buffers[0]);
    const inputGradientResult = await this.dispatch(
      "dense_input_gradient",
      batchSize * inputSize,
      [delta, weights],
      [new Float32Array(batchSize * inputSize)],
      parameterBlock([batchSize, inputSize, outputSize]),
      [0],
    );
    const parameterResult = await this.dispatch(
      "dense_parameter_gradient",
      Math.max(inputSize * outputSize, outputSize),
      [layerInput, new Float32Array(1), new Float32Array(1), delta],
      [
        new Float32Array(1),
        new Float32Array(1),
        new Float32Array(inputSize * outputSize),
        new Float32Array(outputSize),
      ],
      parameterBlock([batchSize, inputSize, outputSize]),
      [2, 3],
    );
    return {
      delta,
      inputGradient: new Float32Array(inputGradientResult.buffers[0]),
      weightGradient: new Float32Array(parameterResult.buffers[0]),
      biasGradient: new Float32Array(parameterResult.buffers[1]),
    };
  }

  async convolutionForward(
    input: Float32Array,
    weights: Float32Array,
    biases: Float32Array,
    params: number[],
    outputSize: number,
  ) {
    const length = params[0] * outputSize;
    const result = await this.dispatch(
      "conv_forward",
      length,
      [input, weights, biases],
      [new Float32Array(length), new Float32Array(length)],
      parameterBlock(params),
      [0, 1],
    );
    return {
      output: new Float32Array(result.buffers[0]),
      preactivation: new Float32Array(result.buffers[1]),
    };
  }

  async convolutionBackward(
    layerInput: Float32Array,
    weights: Float32Array,
    preactivation: Float32Array,
    upstream: Float32Array,
    params: number[],
    inputSize: number,
    outputSize: number,
    weightSize: number,
    biasSize: number,
  ): Promise<BackwardBatchResult> {
    const batchSize = params[0];
    const deltaResult = await this.dispatch(
      "conv_delta",
      batchSize * outputSize,
      [upstream, preactivation],
      [new Float32Array(batchSize * outputSize)],
      parameterBlock(params),
      [0],
    );
    const delta = new Float32Array(deltaResult.buffers[0]);
    const inputGradientResult = await this.dispatch(
      "conv_input_gradient",
      batchSize * inputSize,
      [delta, weights],
      [new Float32Array(batchSize * inputSize)],
      parameterBlock(params),
      [0],
    );
    const parameterResult = await this.dispatch(
      "conv_parameter_gradient",
      Math.max(weightSize, biasSize),
      [layerInput, new Float32Array(1), new Float32Array(1), delta],
      [
        new Float32Array(1),
        new Float32Array(1),
        new Float32Array(weightSize),
        new Float32Array(biasSize),
      ],
      parameterBlock(params),
      [2, 3],
    );
    return {
      delta,
      inputGradient: new Float32Array(inputGradientResult.buffers[0]),
      weightGradient: new Float32Array(parameterResult.buffers[0]),
      biasGradient: new Float32Array(parameterResult.buffers[1]),
    };
  }

  async poolingForward(
    input: Float32Array,
    params: number[],
    outputSize: number,
  ): Promise<PoolBatchResult> {
    const length = params[0] * outputSize;
    const result = await this.dispatch(
      "pool_forward",
      length,
      [input],
      [
        new Float32Array(length),
        new Float32Array(1),
        new Uint32Array(length),
      ],
      parameterBlock(params),
      [0, 2],
    );
    return {
      output: new Float32Array(result.buffers[0]),
      indices: new Uint32Array(result.buffers[1]),
    };
  }

  async poolingBackward(
    upstream: Float32Array,
    indices: Uint32Array,
    params: number[],
    inputSize: number,
    outputSize: number,
  ) {
    const batchSize = params[0];
    const result = await this.dispatch(
      "pool_backward",
      Math.max(batchSize * inputSize, batchSize * outputSize),
      [upstream, new Float32Array(1), new Float32Array(1), indices],
      [new Float32Array(batchSize * inputSize), new Float32Array(batchSize * outputSize)],
      parameterBlock(params),
      [0, 1],
    );
    return {
      inputGradient: new Float32Array(result.buffers[0]),
      delta: new Float32Array(result.buffers[1]),
    };
  }
}

interface GraphOperation {
  pipeline: any;
  bindGroup: any;
  parameterBuffer: any;
  items: (batchSize: number) => number;
  parameters: (batchSize: number, sampleStep: number, training: boolean) => Uint32Array;
}

interface GraphLayer {
  descriptor: WebGpuLayerDescriptor;
  input: any;
  output: any;
  preactivation: any;
  delta: any;
  inputGradient: any;
  dropoutMask: any;
  indices: any;
  weights: any;
  biases: any;
  weightGradient: any;
  biasGradient: any;
}

interface ReadSection {
  buffer: any;
  byteLength: number;
  offset: number;
}

export class WebGpuTrainingGraph {
  private readonly dummy: any;
  private readonly dummyBuffers: any[];
  private readonly input: any;
  private readonly labels: any;
  private readonly losses: any;
  private readonly probabilities: any;
  private readonly lossDelta: any;
  private readonly layers: GraphLayer[];
  private readonly forwardOperations: GraphOperation[] = [];
  private readonly backwardOperations: GraphOperation[] = [];
  private readonly lossOperation: GraphOperation;

  constructor(
    private readonly executor: WebGpuBatchExecutor,
    descriptors: WebGpuLayerDescriptor[],
    readonly capacity: number,
    private readonly sigmoidHead: boolean,
  ) {
    if (descriptors.length === 0) throw new Error("WebGPU 计算图至少需要一个网络层");
    this.dummy = {};
    this.dummyBuffers = Array.from({ length: 8 }, () => this.createBuffer(4));
    this.input = this.createBuffer(capacity * 784 * 4);
    this.labels = this.createBuffer(capacity * 4);
    this.losses = this.createBuffer(capacity * 4);
    let previousOutput = this.input;
    this.layers = descriptors.map((descriptor) => {
      const output = this.createBuffer(capacity * descriptor.outputSize * 4);
      const preactivation = this.createBuffer(capacity * descriptor.outputSize * 4);
      const delta = this.createBuffer(capacity * descriptor.outputSize * 4);
      const inputGradient = this.createBuffer(capacity * descriptor.inputSize * 4);
      const dropoutMask = this.createBuffer(capacity * descriptor.outputSize * 4);
      const indices = this.createBuffer(capacity * descriptor.outputSize * 4);
      const weights = "weights" in descriptor
        ? this.createBuffer(descriptor.weights.byteLength, descriptor.weights)
        : this.dummy;
      const biases = "biases" in descriptor
        ? this.createBuffer(descriptor.biases.byteLength, descriptor.biases)
        : this.dummy;
      const weightGradient = "weights" in descriptor
        ? this.createBuffer(descriptor.weights.byteLength)
        : this.dummy;
      const biasGradient = "biases" in descriptor
        ? this.createBuffer(descriptor.biases.byteLength)
        : this.dummy;
      const layer = {
        descriptor,
        input: previousOutput,
        output,
        preactivation,
        delta,
        inputGradient,
        dropoutMask,
        indices,
        weights,
        biases,
        weightGradient,
        biasGradient,
      };
      previousOutput = output;
      return layer;
    });
    const finalLayer = this.layers.at(-1)!;
    this.probabilities = this.createBuffer(capacity * finalLayer.descriptor.outputSize * 4);
    this.lossDelta = this.createBuffer(capacity * finalLayer.descriptor.outputSize * 4);
    this.buildOperations();
    this.lossOperation = this.operation(
      "output_loss",
      [finalLayer.output, this.dummy, this.dummy, this.labels],
      [this.probabilities, this.lossDelta, this.losses, this.dummy],
      (batchSize) => batchSize * finalLayer.descriptor.outputSize,
      (batchSize) => parameterBlock([
        batchSize,
        finalLayer.descriptor.outputSize,
        this.sigmoidHead ? 1 : 0,
      ]),
    );
  }

  private createBuffer(byteLength: number, initial?: NumericArray) {
    const buffer = this.executor.device.createBuffer({
      size: alignedSize(byteLength),
      usage: GPU_BUFFER_USAGE.STORAGE |
        GPU_BUFFER_USAGE.COPY_SRC |
        GPU_BUFFER_USAGE.COPY_DST,
    });
    if (initial && initial.byteLength > 0) {
      this.executor.device.queue.writeBuffer(
        buffer,
        0,
        initial.buffer,
        initial.byteOffset,
        initial.byteLength,
      );
    }
    return buffer;
  }

  private operation(
    entryPoint: string,
    readOnly: any[],
    writable: any[],
    items: GraphOperation["items"],
    parameters: GraphOperation["parameters"],
  ): GraphOperation {
    const parameterBuffer = this.executor.device.createBuffer({
      size: 64,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    const pipeline = this.executor.pipeline(entryPoint);
    const bindGroup = this.executor.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...Array.from({ length: 4 }, (_, binding) => ({
          binding,
          resource: {
            buffer: readOnly[binding] && readOnly[binding] !== this.dummy
              ? readOnly[binding]
              : this.dummyBuffers[binding],
          },
        })),
        ...Array.from({ length: 4 }, (_, index) => ({
          binding: index + 4,
          resource: {
            buffer: writable[index] && writable[index] !== this.dummy
              ? writable[index]
              : this.dummyBuffers[index + 4],
          },
        })),
        { binding: 8, resource: { buffer: parameterBuffer } },
      ],
    });
    return { pipeline, bindGroup, parameterBuffer, items, parameters };
  }

  private convolutionParams(descriptor: Extract<WebGpuLayerDescriptor, { type: "conv" }>, batchSize: number) {
    return [
      batchSize,
      descriptor.inputWidth,
      descriptor.inputHeight,
      descriptor.inputChannels,
      descriptor.filters,
      descriptor.kernelSize,
      descriptor.stride,
      descriptor.padding,
      descriptor.activationKind,
      descriptor.trainable ? 1 : 0,
    ];
  }

  private poolingParams(descriptor: Extract<WebGpuLayerDescriptor, { type: "pool" }>, batchSize: number) {
    const global = descriptor.poolingKind === 2;
    return [
      batchSize,
      descriptor.inputWidth,
      descriptor.inputHeight,
      descriptor.inputChannels,
      descriptor.outputWidth,
      descriptor.outputHeight,
      global ? Math.max(descriptor.inputWidth, descriptor.inputHeight) : descriptor.kernelSize,
      global ? 1 : descriptor.stride,
      global ? 0 : descriptor.padding,
      descriptor.poolingKind,
    ];
  }

  private buildOperations() {
    this.layers.forEach((layer, layerIndex) => {
      const descriptor = layer.descriptor;
      if (descriptor.type === "dense") {
        this.forwardOperations.push(this.operation(
          "dense_forward",
          [layer.input, layer.weights, layer.biases, this.dummy],
          [layer.output, layer.preactivation, layer.dropoutMask, this.dummy],
          (batchSize) => batchSize * descriptor.outputSize,
          (batchSize, sampleStep, training) => parameterBlock([
            batchSize,
            descriptor.inputSize,
            descriptor.outputSize,
            descriptor.activationKind,
            training ? 1 : 0,
            sampleStep,
            layerIndex,
            floatBits(descriptor.dropoutRate),
          ]),
        ));
      } else if (descriptor.type === "conv") {
        this.forwardOperations.push(this.operation(
          "conv_forward",
          [layer.input, layer.weights, layer.biases, this.dummy],
          [layer.output, layer.preactivation, this.dummy, this.dummy],
          (batchSize) => batchSize * descriptor.outputSize,
          (batchSize) => parameterBlock(this.convolutionParams(descriptor, batchSize)),
        ));
      } else {
        this.forwardOperations.push(this.operation(
          "pool_forward",
          [layer.input, this.dummy, this.dummy, this.dummy],
          [layer.output, this.dummy, layer.indices, this.dummy],
          (batchSize) => batchSize * descriptor.outputSize,
          (batchSize) => parameterBlock(this.poolingParams(descriptor, batchSize)),
        ));
      }
    });

    for (let index = this.layers.length - 1; index >= 0; index--) {
      const layer = this.layers[index];
      const descriptor = layer.descriptor;
      const upstream = this.layers[index + 1]?.inputGradient ?? this.lossDelta;
      if (descriptor.type === "dense") {
        this.backwardOperations.push(
          this.operation(
            "dense_delta",
            [upstream, layer.preactivation, layer.dropoutMask, this.dummy],
            [layer.delta, this.dummy, this.dummy, this.dummy],
            (batchSize) => batchSize * descriptor.outputSize,
            (batchSize) => parameterBlock([
              batchSize,
              descriptor.inputSize,
              descriptor.outputSize,
              descriptor.activationKind,
            ]),
          ),
          this.operation(
            "dense_input_gradient",
            [layer.delta, layer.weights, this.dummy, this.dummy],
            [layer.inputGradient, this.dummy, this.dummy, this.dummy],
            (batchSize) => batchSize * descriptor.inputSize,
            (batchSize) => parameterBlock([
              batchSize,
              descriptor.inputSize,
              descriptor.outputSize,
            ]),
          ),
          this.operation(
            "dense_parameter_gradient",
            [layer.input, this.dummy, this.dummy, layer.delta],
            [this.dummy, this.dummy, layer.weightGradient, layer.biasGradient],
            () => Math.max(descriptor.weights.length, descriptor.biases.length),
            (batchSize) => parameterBlock([
              batchSize,
              descriptor.inputSize,
              descriptor.outputSize,
            ]),
          ),
        );
      } else if (descriptor.type === "conv") {
        this.backwardOperations.push(
          this.operation(
            "conv_delta",
            [upstream, layer.preactivation, this.dummy, this.dummy],
            [layer.delta, this.dummy, this.dummy, this.dummy],
            (batchSize) => batchSize * descriptor.outputSize,
            (batchSize) => parameterBlock(this.convolutionParams(descriptor, batchSize)),
          ),
          this.operation(
            "conv_input_gradient",
            [layer.delta, layer.weights, this.dummy, this.dummy],
            [layer.inputGradient, this.dummy, this.dummy, this.dummy],
            (batchSize) => batchSize * descriptor.inputSize,
            (batchSize) => parameterBlock(this.convolutionParams(descriptor, batchSize)),
          ),
        );
        if (descriptor.trainable) {
          this.backwardOperations.push(this.operation(
            "conv_parameter_gradient",
            [layer.input, this.dummy, this.dummy, layer.delta],
            [this.dummy, this.dummy, layer.weightGradient, layer.biasGradient],
            () => Math.max(descriptor.weights.length, descriptor.biases.length),
            (batchSize) => parameterBlock(this.convolutionParams(descriptor, batchSize)),
          ));
        }
      } else {
        this.backwardOperations.push(this.operation(
          "pool_backward",
          [upstream, this.dummy, this.dummy, layer.indices],
          [layer.inputGradient, layer.delta, this.dummy, this.dummy],
          (batchSize) => batchSize * Math.max(descriptor.inputSize, descriptor.outputSize),
          (batchSize) => parameterBlock(this.poolingParams(descriptor, batchSize)),
        ));
      }
    }
  }

  private encodeOperation(
    encoder: any,
    operation: GraphOperation,
    batchSize: number,
    sampleStep: number,
    training: boolean,
  ) {
    const parameters = operation.parameters(batchSize, sampleStep, training);
    this.executor.device.queue.writeBuffer(
      operation.parameterBuffer,
      0,
      parameters.buffer,
      parameters.byteOffset,
      parameters.byteLength,
    );
    const pass = encoder.beginComputePass();
    pass.setPipeline(operation.pipeline);
    pass.setBindGroup(0, operation.bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(operation.items(batchSize) / WORKGROUP_SIZE)));
    pass.end();
  }

  private async readSections(encoder: any, sections: Omit<ReadSection, "offset">[]) {
    let byteLength = 0;
    const positioned = sections.map((section) => {
      const positionedSection = { ...section, offset: byteLength };
      byteLength += alignedSize(section.byteLength);
      return positionedSection;
    });
    const staging = this.executor.device.createBuffer({
      size: alignedSize(byteLength),
      usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    });
    positioned.forEach(({ buffer, byteLength: size, offset }) => {
      encoder.copyBufferToBuffer(buffer, 0, staging, offset, alignedSize(size));
    });
    this.executor.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPU_MAP_MODE_READ, 0, alignedSize(byteLength));
    const copy = staging.getMappedRange(0, alignedSize(byteLength)).slice(0);
    staging.unmap();
    staging.destroy();
    return positioned.map(({ byteLength: size, offset }) => copy.slice(offset, offset + size));
  }

  private uploadBatch(input: Float32Array, labels: Int32Array) {
    this.executor.device.queue.writeBuffer(
      this.input,
      0,
      input.buffer,
      input.byteOffset,
      input.byteLength,
    );
    this.executor.device.queue.writeBuffer(
      this.labels,
      0,
      labels.buffer,
      labels.byteOffset,
      labels.byteLength,
    );
  }

  uploadParameters(descriptors: WebGpuLayerDescriptor[]) {
    descriptors.forEach((descriptor, index) => {
      if (!("weights" in descriptor)) return;
      const layer = this.layers[index];
      this.executor.device.queue.writeBuffer(
        layer.weights,
        0,
        descriptor.weights.buffer,
        descriptor.weights.byteOffset,
        descriptor.weights.byteLength,
      );
      this.executor.device.queue.writeBuffer(
        layer.biases,
        0,
        descriptor.biases.buffer,
        descriptor.biases.byteOffset,
        descriptor.biases.byteLength,
      );
    });
  }

  async forward(
    input: Float32Array,
    labels: Int32Array,
    batchSize: number,
  ) {
    this.uploadBatch(input, labels);
    const encoder = this.executor.device.createCommandEncoder({
      label: "WatchNeuron WebGPU forward graph",
    });
    this.forwardOperations.forEach((operation) =>
      this.encodeOperation(encoder, operation, batchSize, 0, false));
    this.encodeOperation(encoder, this.lossOperation, batchSize, 0, false);
    const finalSize = this.layers.at(-1)!.descriptor.outputSize;
    const [probabilities, losses] = await this.readSections(encoder, [
      { buffer: this.probabilities, byteLength: batchSize * finalSize * 4 },
      { buffer: this.losses, byteLength: batchSize * 4 },
    ]);
    return {
      probabilities: new Float32Array(probabilities),
      losses: new Float32Array(losses),
    };
  }

  async train(
    input: Float32Array,
    labels: Int32Array,
    batchSize: number,
    sampleStep: number,
    captureTrace: boolean,
  ): Promise<WebGpuGraphResult> {
    this.uploadBatch(input, labels);
    const encoder = this.executor.device.createCommandEncoder({
      label: "WatchNeuron WebGPU training graph",
    });
    this.forwardOperations.forEach((operation) =>
      this.encodeOperation(encoder, operation, batchSize, sampleStep, true));
    this.encodeOperation(encoder, this.lossOperation, batchSize, sampleStep, true);
    this.backwardOperations.forEach((operation) =>
      this.encodeOperation(encoder, operation, batchSize, sampleStep, true));

    const sections: Omit<ReadSection, "offset">[] = [
      { buffer: this.losses, byteLength: batchSize * 4 },
    ];
    const gradientLayerIndexes: number[] = [];
    this.layers.forEach((layer, index) => {
      if (!("weights" in layer.descriptor) || (layer.descriptor.type === "conv" && !layer.descriptor.trainable)) {
        return;
      }
      gradientLayerIndexes.push(index);
      sections.push(
        { buffer: layer.weightGradient, byteLength: layer.descriptor.weights.byteLength },
        { buffer: layer.biasGradient, byteLength: layer.descriptor.biases.byteLength },
      );
    });
    if (captureTrace) {
      sections.push({
        buffer: this.layers[0].inputGradient,
        byteLength: this.layers[0].descriptor.inputSize * 4,
      });
      this.layers.forEach((layer, index) => {
        sections.push(
          {
            buffer: index === this.layers.length - 1 ? this.probabilities : layer.output,
            byteLength: layer.descriptor.outputSize * 4,
          },
          { buffer: layer.delta, byteLength: layer.descriptor.outputSize * 4 },
        );
      });
    }
    const buffers = await this.readSections(encoder, sections);
    let cursor = 0;
    const losses = new Float32Array(buffers[cursor++]);
    const gradients: WebGpuGraphResult["gradients"] = this.layers.map(() => null);
    gradientLayerIndexes.forEach((layerIndex) => {
      gradients[layerIndex] = {
        weightGradient: new Float32Array(buffers[cursor++]),
        biasGradient: new Float32Array(buffers[cursor++]),
      };
    });
    let firstInputGradient: Float32Array | null = null;
    const outputs: Float32Array[] = [];
    const deltas: Float32Array[] = [];
    if (captureTrace) {
      firstInputGradient = new Float32Array(buffers[cursor++]);
      this.layers.forEach(() => {
        outputs.push(new Float32Array(buffers[cursor++]));
        deltas.push(new Float32Array(buffers[cursor++]));
      });
    }
    return { losses, outputs, deltas, firstInputGradient, gradients };
  }
}
