import type {
  ConvolutionConfig,
  ConvolutionLayerData,
  HiddenLayer,
  NeuralModel,
} from "../types";

export interface FeatureMapShape {
  width: number;
  height: number;
  channels: number;
  length: number;
}

export const inputImageShape: FeatureMapShape = {
  width: 28,
  height: 28,
  channels: 1,
  length: 784,
};

export interface ConvolutionKernelPreset {
  id: string;
  name: string;
  description: string;
  values: number[];
}

export const convolutionKernelPresets: ConvolutionKernelPreset[] = [
  {
    id: "identity",
    name: "保留中心",
    description: "突出当前位置，适合作为稳定的可训练起点。",
    values: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  },
  {
    id: "edge-horizontal",
    name: "水平边缘",
    description: "响应数字笔画的水平明暗变化。",
    values: [-1, -1, -1, 0, 0, 0, 1, 1, 1],
  },
  {
    id: "edge-vertical",
    name: "垂直边缘",
    description: "响应数字笔画的垂直明暗变化。",
    values: [-1, 0, 1, -1, 0, 1, -1, 0, 1],
  },
  {
    id: "outline",
    name: "轮廓",
    description: "增强四邻域轮廓与局部转折。",
    values: [0, -1, 0, -1, 4, -1, 0, -1, 0],
  },
  {
    id: "sharpen",
    name: "锐化",
    description: "保留中心笔画并抑制周围像素。",
    values: [0, -1, 0, -1, 5, -1, 0, -1, 0],
  },
  {
    id: "soften",
    name: "柔化",
    description: "聚合邻域像素，降低手写抖动。",
    values: [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16],
  },
];

function defaultKernel(filter: number) {
  return [...convolutionKernelPresets[filter % convolutionKernelPresets.length].values];
}

function convolutionId() {
  return `conv-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function createDefaultConvolutionConfig(
  position = 0,
  enabled = false,
): ConvolutionConfig {
  const filters = 4;
  return {
    id: convolutionId(),
    enabled,
    position,
    filters,
    kernelSize: 3,
    stride: 2,
    padding: 1,
    activation: "relu",
    kernels: Array.from({ length: filters }, (_, filter) => defaultKernel(filter)),
  };
}

export function denseFeatureMapShape(units: number): FeatureMapShape {
  const length = Math.max(1, Math.floor(units));
  let height = Math.max(1, Math.floor(Math.sqrt(length)));
  while (height > 1 && length % height !== 0) height--;
  const width = length / height;
  return { width, height, channels: 1, length };
}

export function convolutionInputShape(
  layers: HiddenLayer[],
  position: number,
): FeatureMapShape {
  const normalizedPosition = Math.min(
    layers.length,
    Math.max(0, Math.floor(position)),
  );
  return normalizedPosition === 0
    ? inputImageShape
    : denseFeatureMapShape(layers[normalizedPosition - 1]?.units ?? 1);
}

export interface ConvolutionPipelineEntry {
  config: ConvolutionConfig;
  input: FeatureMapShape;
  output: FeatureMapShape;
}

export function fitConvolutionsToLayers(
  values: unknown,
  layers: HiddenLayer[],
): ConvolutionConfig[] {
  if (!Array.isArray(values)) return [];
  const normalized = values
    .map((value, index) => ({ config: normalizeConvolutionConfig(value), index }))
    .filter(({ config }) => config.enabled)
    .sort((left, right) => left.config.position - right.config.position || left.index - right.index);
  let previousPosition = -1;
  let currentShape = inputImageShape;
  return normalized.map(({ config }) => {
    const position = Math.min(layers.length, Math.max(0, config.position));
    if (position !== previousPosition) {
      currentShape = convolutionInputShape(layers, position);
      previousPosition = position;
    }
    const minimumPadding = Math.max(
      0,
      Math.ceil((config.kernelSize - Math.min(currentShape.width, currentShape.height)) / 2),
    );
    const fitted = {
      ...config,
      enabled: true,
      position,
      padding: Math.max(minimumPadding, config.padding),
    };
    currentShape = convolutionOutputShape(fitted, currentShape);
    return fitted;
  });
}

export function convolutionPipeline(
  layers: HiddenLayer[],
  convolutions: ConvolutionConfig[],
): ConvolutionPipelineEntry[] {
  const fitted = fitConvolutionsToLayers(convolutions, layers);
  let previousPosition = -1;
  let currentShape = inputImageShape;
  return fitted.map((config) => {
    if (config.position !== previousPosition) {
      currentShape = convolutionInputShape(layers, config.position);
      previousPosition = config.position;
    }
    const input = currentShape;
    const output = convolutionOutputShape(config, input);
    currentShape = output;
    return { config, input, output };
  });
}

export function architectureLayerSizes(
  layers: HiddenLayer[],
  convolutions: ConvolutionConfig[],
) {
  const pipeline = convolutionPipeline(layers, convolutions);
  const sizes = [784];
  for (let position = 0; position <= layers.length; position++) {
    for (const entry of pipeline.filter(({ config }) => config.position === position)) {
      sizes.push(entry.output.length);
    }
    if (position < layers.length) sizes.push(layers[position].units);
  }
  sizes.push(10);
  return sizes;
}

export function modelConvolutions(model: NeuralModel | null | undefined): ConvolutionLayerData[] {
  if (!model) return [];
  if (Array.isArray(model.convolutions)) return model.convolutions;
  return model.convolution ? [model.convolution] : [];
}

export function convolutionOutputShape(
  config: ConvolutionConfig,
  input: FeatureMapShape = inputImageShape,
): FeatureMapShape {
  const width = Math.floor(
    (input.width + config.padding * 2 - config.kernelSize) / config.stride,
  ) + 1;
  const height = Math.floor(
    (input.height + config.padding * 2 - config.kernelSize) / config.stride,
  ) + 1;
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  return {
    width: safeWidth,
    height: safeHeight,
    channels: config.filters,
    length: safeWidth * safeHeight * config.filters,
  };
}

export function fitConvolutionToLayers(
  value: unknown,
  layers: HiddenLayer[],
): ConvolutionConfig {
  const config = normalizeConvolutionConfig(value);
  const position = Math.min(layers.length, Math.max(0, config.position));
  const input = convolutionInputShape(layers, position);
  const minimumPadding = Math.max(
    0,
    Math.ceil((config.kernelSize - Math.min(input.width, input.height)) / 2),
  );
  return {
    ...config,
    position,
    padding: Math.max(minimumPadding, config.padding),
  };
}

function resizedKernel(source: unknown, size: number, filter: number) {
  const values = Array.isArray(source) ? source : [];
  const result = new Array<number>(size * size).fill(0);
  if (values.length === size * size) {
    return result.map((_, index) => Number.isFinite(Number(values[index])) ? Number(values[index]) : 0);
  }
  if (size === 3) return defaultKernel(filter);
  result[Math.floor(result.length / 2)] = 1;
  return result;
}

export function normalizeConvolutionConfig(value: unknown): ConvolutionConfig {
  const fallback = createDefaultConvolutionConfig();
  if (!value || typeof value !== "object") return fallback;
  const source = value as Partial<ConvolutionConfig>;
  const positionValue = Number(source.position);
  const position = Number.isFinite(positionValue)
    ? Math.max(0, Math.floor(positionValue))
    : fallback.position;
  const filtersValue = Number(source.filters);
  const filters = Number.isFinite(filtersValue)
    ? Math.min(32, Math.max(1, Math.floor(filtersValue)))
    : fallback.filters;
  const kernelValue = Number(source.kernelSize);
  const kernelSize = [1, 3, 5, 7].includes(kernelValue) ? kernelValue : fallback.kernelSize;
  const strideValue = Number(source.stride);
  const stride = Number.isFinite(strideValue)
    ? Math.min(4, Math.max(1, Math.floor(strideValue)))
    : fallback.stride;
  const paddingValue = Number(source.padding);
  const padding = Number.isFinite(paddingValue)
    ? Math.min(6, Math.max(0, Math.floor(paddingValue)))
    : Math.floor(kernelSize / 2);
  const activation = typeof source.activation === "string" && [
    "relu", "leakyRelu", "elu", "selu", "relu6", "gelu", "swish", "mish",
    "sigmoid", "tanh", "hardSigmoid", "hardTanh", "softplus", "softsign",
    "softmax", "linear",
  ].includes(source.activation) ? source.activation : fallback.activation;
  const sourceKernels = Array.isArray(source.kernels) ? source.kernels : [];
  return {
    id: typeof source.id === "string" && source.id ? source.id : fallback.id,
    enabled: source.enabled === true,
    position,
    filters,
    kernelSize,
    stride,
    padding,
    activation: activation as ConvolutionConfig["activation"],
    kernels: Array.from(
      { length: filters },
      (_, filter) => resizedKernel(sourceKernels[filter], kernelSize, filter),
    ),
  };
}
