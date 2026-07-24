<script setup lang="ts">
import { ChevronLeft, ChevronRight, X } from "@lucide/vue";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { activationLabels } from "../lib/model";
import { convolutionPipeline, modelConvolutions, spatialPipeline } from "../lib/convolution";
import type { ConvolutionConfig, HiddenLayer, NeuralModel, OutputHeadKind, PoolingConfig, PropagationDirection, TrainingTrace } from "../types";
import KernelMatrixView from "./KernelMatrixView.vue";

interface VisibleNode {
  layerIndex: number;
  neuronIndex: number;
  value: number;
  x: number;
  y: number;
}

interface InspectedNode {
  layerIndex: number;
  neuronIndex: number;
  x: number;
  y: number;
}

interface KernelSnapshot {
  weights: number[];
  inputChannels: number;
  kernelSize: number;
  filter: number;
  trainable: boolean;
}

interface NeuronDetail {
  eyebrow: string;
  title: string;
  badge: string;
  details: Array<{ label: string; value: string }>;
  note: string;
  kernel?: KernelSnapshot;
}

const props = defineProps<{
  layers: HiddenLayer[];
  convolutions: ConvolutionConfig[];
  poolings: PoolingConfig[];
  outputHead: OutputHeadKind;
  activations: number[][];
  gradients: number[][];
  model: NeuralModel | null;
  trace: TrainingTrace | null;
  selectedLayer: number;
  animated: boolean;
  training: boolean;
  stepEnabled: boolean;
  stepDirection: PropagationDirection;
  stepLayer: number;
}>();

const emit = defineEmits<{ select: [layer: number] }>();
const canvas = ref<HTMLCanvasElement | null>(null);
const popover = ref<HTMLElement | null>(null);
const nodeTargets = ref<VisibleNode[]>([]);
const inspectedNode = ref<InspectedNode | null>(null);
const convolutionEntries = computed(() => convolutionPipeline(props.layers, props.convolutions, props.poolings));
const spatialEntries = computed(() => spatialPipeline(props.layers, props.convolutions, props.poolings));
const visualLayers = computed(() => {
  const result: Array<
    | { kind: "input"; size: number }
    | { kind: "conv"; size: number; convolutionIndex: number }
    | { kind: "pool"; size: number; spatialIndex: number }
    | { kind: "dense"; size: number; denseIndex: number }
    | { kind: "output"; size: number; denseIndex: number }
  > = [{ kind: "input", size: 784 }];
  for (let position = 0; position <= props.layers.length; position++) {
    spatialEntries.value.forEach((entry, spatialIndex) => {
      if (entry.config.position === position) {
        if (entry.kind === "conv") {
          const convolutionIndex = convolutionEntries.value.findIndex(({ config }) => config.id === entry.config.id);
          result.push({ kind: "conv", size: entry.output.length, convolutionIndex });
        } else result.push({ kind: "pool", size: entry.output.length, spatialIndex });
      }
    });
    if (position < props.layers.length) {
      result.push({ kind: "dense", size: props.layers[position].units, denseIndex: position });
    }
  }
  result.push({ kind: "output", size: 10, denseIndex: props.layers.length });
  return result;
});
const layerSizes = computed(() => visualLayers.value.map(({ size }) => size));
let resizeObserver: ResizeObserver | null = null;
let frame = 0;
let lastTime = 0;
let phase = 0;
let targetSignature = "";

function sampled(values: number[] | undefined, size: number, count: number) {
  const output: Array<{ neuronIndex: number; value: number }> = [];
  for (let slot = 0; slot < count; slot++) {
    const start = Math.floor((slot / count) * size);
    const end = Math.max(start + 1, Math.floor(((slot + 1) / count) * size));
    let neuronIndex = Math.min(size - 1, Math.floor((start + end - 1) / 2));
    let value = values?.[neuronIndex] ?? 0;
    for (let index = start; index < Math.min(end, values?.length ?? 0); index++) {
      if (Math.abs(values?.[index] ?? 0) > Math.abs(value)) {
        neuronIndex = index;
        value = values?.[index] ?? 0;
      }
    }
    output.push({ neuronIndex, value });
  }
  return output;
}

function formatNumber(value: number) {
  const absolute = Math.abs(value);
  if (absolute === 0) return "0.00000";
  if (absolute >= 100) return value.toFixed(2);
  if (absolute >= 1) return value.toFixed(4);
  if (absolute >= 0.001) return value.toFixed(5);
  return value.toExponential(3);
}

const neuronDetails = computed<NeuronDetail | null>(() => {
  const selected = inspectedNode.value;
  if (!selected) return null;
  const size = layerSizes.value[selected.layerIndex];
  if (!size || selected.neuronIndex < 0 || selected.neuronIndex >= size) return null;

  const value = props.activations[selected.layerIndex]?.[selected.neuronIndex] ?? 0;
  const gradient = props.gradients[selected.layerIndex]?.[selected.neuronIndex] ?? 0;
  const sampledLayer = size > (selected.layerIndex === 0 ? 14 : 12);
  const sampleNote = sampledLayer
    ? `画布对本层进行可视采样，可用左右按钮查看全部 ${size} 个神经元。`
    : "点击其他节点可切换查看。";
  const note = props.training
    ? `当前数值来自训练 Worker 快照。${sampleNote}`
    : sampleNote;

  if (selected.layerIndex === 0) {
    const row = Math.floor(selected.neuronIndex / 28);
    const column = selected.neuronIndex % 28;
    return {
      eyebrow: "INPUT NEURON",
      title: `输入神经元 #${selected.neuronIndex}`,
      badge: "像素",
      details: [
        { label: "当前灰度", value: formatNumber(value) },
        { label: "像素坐标", value: `行 ${row + 1} · 列 ${column + 1}` },
        ...(props.training
          ? [
              { label: "反向梯度", value: formatNumber(gradient) },
              { label: "梯度绝对值", value: formatNumber(Math.abs(gradient)) },
            ]
          : []),
        { label: "归一化范围", value: "0.00000 - 1.0000" },
        { label: "可训练参数", value: "无" },
      ],
      note,
    };
  }

  const layerMetadata = visualLayers.value[selected.layerIndex];
  if (layerMetadata?.kind === "conv") {
    const entry = convolutionEntries.value[layerMetadata.convolutionIndex];
    const config = entry.config;
    const shape = entry.output;
    const mapSize = shape.width * shape.height;
    const filter = Math.floor(selected.neuronIndex / mapSize);
    const position = selected.neuronIndex % mapSize;
    const row = Math.floor(position / shape.width);
    const column = position % shape.width;
    const layer = modelConvolutions(props.model)[layerMetadata.convolutionIndex];
    const weights = props.training
      ? props.trace?.convolutionWeights[layerMetadata.convolutionIndex] ?? layer?.weights
      : layer?.weights;
    const biases = props.training
      ? props.trace?.convolutionBiases[layerMetadata.convolutionIndex] ?? layer?.biases
      : layer?.biases;
    const kernelLength = config.kernelSize ** 2;
    const inputChannels = layer?.inputChannels ?? entry.input.channels;
    const kernelOffset = filter * inputChannels * kernelLength;
    const kernel = Array.from(weights ?? []).slice(
      kernelOffset,
      kernelOffset + inputChannels * kernelLength,
    );
    const absoluteMean = kernel.length
      ? Array.from(kernel).reduce((sum, weight) => sum + Math.abs(weight), 0) / kernel.length
      : 0;
    return {
      eyebrow: props.training ? "CONV2D TRAINING TRACE" : "CONV2D FEATURE",
      title: `特征图 ${filter + 1} · 激活 #${position}`,
      badge: `${activationLabels[config.activation]} · ${config.kernelSize}×${config.kernelSize}`,
      details: [
        { label: "当前激活", value: formatNumber(value) },
        { label: "特征坐标", value: `行 ${row + 1} · 列 ${column + 1}` },
        { label: "参数状态", value: config.trainable ? "参与训练" : "已冻结" },
        ...(props.training
          ? [
              { label: "反向梯度", value: formatNumber(gradient) },
              { label: "梯度绝对值", value: formatNumber(Math.abs(gradient)) },
            ]
          : [
              { label: "卷积偏置", value: formatNumber(biases?.[filter] ?? 0) },
              { label: "核权重绝对均值", value: formatNumber(absoluteMean) },
              { label: "步幅 / 填充", value: `${config.stride} / ${config.padding}` },
            ]),
      ],
      kernel: {
        weights: Array.from(weights ?? []),
        inputChannels,
        kernelSize: config.kernelSize,
        filter,
        trainable: config.trainable,
      },
      note,
    };
  }

  if (layerMetadata?.kind === "pool") {
    const entry = spatialEntries.value[layerMetadata.spatialIndex];
    if (!entry || entry.kind !== "pool") return null;
    const mapSize = entry.output.width * entry.output.height;
    const channel = Math.floor(selected.neuronIndex / mapSize);
    const position = selected.neuronIndex % mapSize;
    return {
      eyebrow: "POOLING SIGNAL",
      title: `通道 ${channel + 1} · 池化激活 #${position}`,
      badge: entry.config.kind === "globalAverage" ? "GAP" : entry.config.kind === "max" ? "MaxPool2D" : "AvgPool2D",
      details: [
        { label: "当前激活", value: formatNumber(value) },
        { label: "输出坐标", value: `行 ${Math.floor(position / entry.output.width) + 1} · 列 ${(position % entry.output.width) + 1}` },
        { label: "输入 / 输出", value: `${entry.input.width}×${entry.input.height} → ${entry.output.width}×${entry.output.height}` },
        ...(props.training ? [{ label: "反向梯度", value: formatNumber(gradient) }] : []),
        { label: "可训练参数", value: "无" },
      ],
      note,
    };
  }

  const denseIndex = layerMetadata?.kind === "dense" || layerMetadata?.kind === "output"
    ? layerMetadata.denseIndex
    : -1;
  const denseLayer = props.model?.layers[denseIndex];
  const isOutput = layerMetadata?.kind === "output";
  const activation = isOutput
    ? props.outputHead === "sigmoid" ? "Sigmoid" : "Softmax"
    : activationLabels[props.layers[denseIndex]?.activation ?? "relu"];
  if (props.training) {
    return {
      eyebrow: isOutput ? "OUTPUT TRAINING TRACE" : `DENSE ${denseIndex + 1} TRAINING TRACE`,
      title: isOutput
        ? `数字 ${selected.neuronIndex} 输出神经元`
        : `隐藏层 ${denseIndex + 1} · 神经元 #${selected.neuronIndex}`,
      badge: `${activation} · 训练快照`,
      details: [
        {
          label: isOutput ? (props.outputHead === "sigmoid" ? "独立分数" : "当前概率") : "当前激活",
          value: isOutput ? `${(value * 100).toFixed(2)}%` : formatNumber(value),
        },
        { label: "反向梯度", value: formatNumber(gradient) },
        { label: "梯度绝对值", value: formatNumber(Math.abs(gradient)) },
        {
          label: "梯度符号",
          value: gradient > 0 ? "正" : gradient < 0 ? "负" : "零",
        },
      ],
      note,
    };
  }
  if (!denseLayer || selected.neuronIndex >= denseLayer.outputSize) {
    return {
      eyebrow: isOutput ? "OUTPUT NEURON" : `DENSE ${denseIndex + 1}`,
      title: isOutput
        ? `数字 ${selected.neuronIndex} 输出神经元`
        : `隐藏层 ${denseIndex + 1} · 神经元 #${selected.neuronIndex}`,
      badge: activation,
      details: [{ label: "当前激活", value: formatNumber(value) }],
      note: sampleNote,
    };
  }

  const offset = selected.neuronIndex * denseLayer.inputSize;
  let sum = 0;
  let absoluteSum = 0;
  let squaredSum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < denseLayer.inputSize; index++) {
    const weight = denseLayer.weights[offset + index];
    sum += weight;
    absoluteSum += Math.abs(weight);
    squaredSum += weight * weight;
    minimum = Math.min(minimum, weight);
    maximum = Math.max(maximum, weight);
  }

  return {
    eyebrow: isOutput ? "OUTPUT NEURON" : `DENSE ${denseIndex + 1}`,
    title: isOutput
      ? `数字 ${selected.neuronIndex} 输出神经元`
      : `隐藏层 ${denseIndex + 1} · 神经元 #${selected.neuronIndex}`,
    badge: activation,
    details: [
      {
        label: isOutput ? (props.outputHead === "sigmoid" ? "独立分数" : "当前概率") : "当前激活",
        value: isOutput ? `${(value * 100).toFixed(2)}%` : formatNumber(value),
      },
      { label: "偏置", value: formatNumber(denseLayer.biases[selected.neuronIndex] ?? 0) },
      { label: "输入连接", value: denseLayer.inputSize.toLocaleString("zh-CN") },
      { label: "权重均值", value: formatNumber(sum / denseLayer.inputSize) },
      { label: "绝对均值", value: formatNumber(absoluteSum / denseLayer.inputSize) },
      { label: "权重范围", value: `${formatNumber(minimum)} - ${formatNumber(maximum)}` },
      { label: "L2 范数", value: formatNumber(Math.sqrt(squaredSum)) },
    ],
    note: sampleNote,
  };
});

function nodeLabel(target: VisibleNode) {
  if (target.layerIndex === 0) return `输入神经元 ${target.neuronIndex}，点击查看参数`;
  const metadata = visualLayers.value[target.layerIndex];
  if (metadata?.kind === "conv") {
    return `卷积激活 ${target.neuronIndex}，点击查看卷积参数`;
  }
  if (metadata?.kind === "pool") return `池化激活 ${target.neuronIndex}，点击查看状态`;
  if (metadata?.kind === "output") {
    return `数字 ${target.neuronIndex} 输出神经元，点击查看参数`;
  }
  const denseIndex = metadata?.kind === "dense" ? metadata.denseIndex + 1 : target.layerIndex;
  return `隐藏层 ${denseIndex} 神经元 ${target.neuronIndex}，点击查看参数`;
}

function popoverPosition(target: VisibleNode) {
  const element = canvas.value;
  if (!element) return { x: target.x, y: target.y };
  const bounds = element.getBoundingClientRect();
  const width = Math.min(276, Math.max(0, bounds.width - 16));
  const estimatedHeight = Math.min(300, Math.max(0, bounds.height - 16));
  const x = target.x < bounds.width / 2 ? target.x + 16 : target.x - width - 16;
  return {
    x: Math.max(8, Math.min(x, bounds.width - width - 8)),
    y: Math.max(8, Math.min(target.y - 82, bounds.height - estimatedHeight - 8)),
  };
}

function inspectNode(target: VisibleNode) {
  emit("select", target.layerIndex);
  const position = popoverPosition(target);
  inspectedNode.value = {
    layerIndex: target.layerIndex,
    neuronIndex: target.neuronIndex,
    ...position,
  };
  nextTick(() => {
    fitPopover();
    popover.value?.focus();
  });
}

function stepNeuron(direction: -1 | 1) {
  const selected = inspectedNode.value;
  if (!selected) return;
  const lastIndex = layerSizes.value[selected.layerIndex] - 1;
  selected.neuronIndex = Math.max(0, Math.min(lastIndex, selected.neuronIndex + direction));
  nextTick(fitPopover);
}

function closeInspector() {
  inspectedNode.value = null;
}

function fitPopover() {
  const selected = inspectedNode.value;
  const panel = popover.value;
  const shell = canvas.value?.parentElement;
  if (!selected || !panel || !shell) return;
  selected.x = Math.max(8, Math.min(selected.x, shell.clientWidth - panel.offsetWidth - 8));
  selected.y = Math.max(8, Math.min(selected.y, shell.clientHeight - panel.offsetHeight - 8));
}

function draw(time = performance.now()) {
  const element = canvas.value;
  if (!element) return;
  const bounds = element.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(bounds.width * ratio);
  const pixelHeight = Math.round(bounds.height * ratio);
  if (element.width !== pixelWidth || element.height !== pixelHeight) {
    element.width = pixelWidth;
    element.height = pixelHeight;
  }
  const ctx = element.getContext("2d")!;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  ctx.clearRect(0, 0, width, height);
  if (props.animated) phase = (phase + Math.min(40, time - lastTime) * 0.00034) % 1;
  lastTime = time;

  const sizes = layerSizes.value;
  const count = sizes.length;
  const sidePadding = width < 560 ? 38 : 58;
  const topPadding = 60;
  const bottomPadding = 58;
  const columns = sizes.map((_, index) =>
    sidePadding + (index * (width - sidePadding * 2)) / Math.max(1, count - 1),
  );
  const positions = sizes.map((size, layerIndex) => {
    const visibleCount = Math.min(layerIndex === 0 ? 14 : 12, size);
    const samples = sampled(props.activations[layerIndex], size, visibleCount);
    return samples.map((sample, nodeIndex) => ({
      layerIndex,
      neuronIndex: sample.neuronIndex,
      value: sample.value,
      x: columns[layerIndex],
      y:
        visibleCount === 1
          ? height / 2
          : topPadding +
            (nodeIndex * (height - topPadding - bottomPadding)) / (visibleCount - 1),
    }));
  });
  const signals = positions.map((nodes) => nodes.map((node) => node.value));
  const gradientSignals = positions.map((nodes) =>
    nodes.map((node) => props.gradients[node.layerIndex]?.[node.neuronIndex] ?? 0),
  );
  const gradientMaximums = gradientSignals.map((values) =>
    Math.max(1e-8, ...values.map((value) => Math.abs(value))),
  );
  const nextTargets = positions.flat();
  const nextTargetSignature = nextTargets
    .map((target) => `${target.layerIndex}:${target.neuronIndex}:${target.x.toFixed(1)}:${target.y.toFixed(1)}`)
    .join("|");
  if (nextTargetSignature !== targetSignature) {
    targetSignature = nextTargetSignature;
    nodeTargets.value = nextTargets;
  }

  const selectedX = columns[props.selectedLayer] ?? columns[0];
  ctx.fillStyle = "rgba(115, 201, 191, 0.055)";
  ctx.fillRect(selectedX - 25, 36, 50, height - 72);
  ctx.strokeStyle = "rgba(115, 201, 191, 0.24)";
  ctx.beginPath();
  ctx.moveTo(selectedX, 36);
  ctx.lineTo(selectedX, height - 36);
  ctx.stroke();

  for (let layer = 0; layer < positions.length - 1; layer++) {
    const stepConnectionActive =
      !props.stepEnabled ||
      (props.stepDirection === "forward"
        ? layer < props.stepLayer
        : layer >= props.stepLayer);
    const stepFrontier =
      props.stepEnabled &&
      (props.stepDirection === "forward"
        ? layer === props.stepLayer - 1
        : layer === props.stepLayer);
    positions[layer].forEach((source, sourceIndex) => {
      positions[layer + 1].forEach((target, targetIndex) => {
        const signal = Math.min(1, Math.abs(signals[layer][sourceIndex] ?? 0));
        const backwardSignal = props.training
          ? Math.min(
              1,
              Math.abs(gradientSignals[layer + 1]?.[targetIndex] ?? 0) /
                (gradientMaximums[layer + 1] ?? 1),
            )
          : 0;
        const positive = (sourceIndex * 7 + targetIndex * 3 + layer) % 5 !== 0;
        ctx.strokeStyle = stepConnectionActive
          ? positive
            ? `rgba(115, 201, 191, ${0.055 + signal * 0.12})`
            : `rgba(217, 119, 87, ${0.045 + signal * 0.09})`
          : "rgba(127, 124, 117, 0.025)";
        ctx.lineWidth = stepFrontier ? 1.1 : signal > 0.45 ? 0.9 : 0.55;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();

        if (
          props.animated &&
          stepConnectionActive &&
          (!props.stepEnabled || stepFrontier) &&
          signal > 0.08 &&
          (sourceIndex * 7 + targetIndex * 11 + layer) % 19 === 0
        ) {
          const progress = (phase + sourceIndex * 0.071 + targetIndex * 0.037) % 1;
          ctx.fillStyle = positive ? "#73c9bf" : "#d97757";
          ctx.beginPath();
          ctx.arc(
            source.x + (target.x - source.x) * progress,
            source.y + (target.y - source.y) * progress,
            1.7,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }

        if (
          props.training &&
          props.animated &&
          stepConnectionActive &&
          (!props.stepEnabled || stepFrontier) &&
          backwardSignal > 0.04 &&
          (sourceIndex * 5 + targetIndex * 7 + layer) % 13 === 0
        ) {
          const progress = 1 - ((phase * 1.18 + sourceIndex * 0.043 + targetIndex * 0.061) % 1);
          const gradient = gradientSignals[layer + 1]?.[targetIndex] ?? 0;
          ctx.fillStyle = gradient >= 0 ? "#d97757" : "#e3b75b";
          ctx.beginPath();
          ctx.arc(
            source.x + (target.x - source.x) * progress,
            source.y + (target.y - source.y) * progress,
            1.9,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
      });
    });
  }

  positions.forEach((nodes, layerIndex) => {
    const stepLayerActive =
      !props.stepEnabled ||
      (props.stepDirection === "forward"
        ? layerIndex <= props.stepLayer
        : layerIndex >= props.stepLayer);
    const layerSignals = signals[layerIndex];
    const maximum = Math.max(0.001, ...layerSignals.map(Math.abs));
    nodes.forEach((node, nodeIndex) => {
      const normalized = Math.min(1, Math.abs(layerSignals[nodeIndex] ?? 0) / maximum);
      const radius = 3.8 + normalized * 2.5;
      ctx.fillStyle = !stepLayerActive
        ? "#302f2b"
        : layerIndex === positions.length - 1 && normalized > 0.9
          ? "#e3b75b"
          : normalized > 0.1
            ? "#73c9bf"
            : "#45433d";
      ctx.strokeStyle = layerIndex === props.selectedLayer ? "#f8f8f6" : "#77746d";
      ctx.lineWidth = layerIndex === props.selectedLayer ? 1.2 : 0.7;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      const gradient = gradientSignals[layerIndex]?.[nodeIndex] ?? 0;
      const gradientStrength = props.training
        ? Math.min(1, Math.abs(gradient) / (gradientMaximums[layerIndex] ?? 1))
        : 0;
      if (stepLayerActive && gradientStrength > 0.025) {
        ctx.strokeStyle = gradient >= 0
          ? `rgba(217, 119, 87, ${0.35 + gradientStrength * 0.6})`
          : `rgba(227, 183, 91, ${0.35 + gradientStrength * 0.6})`;
        ctx.lineWidth = 0.8 + gradientStrength * 1.1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    ctx.textAlign = "center";
    ctx.fillStyle = layerIndex === props.selectedLayer
      ? "#f8f8f6"
      : stepLayerActive
        ? "#b8b5ad"
        : "#56544e";
    ctx.font = "600 11px 'Segoe UI', sans-serif";
    const metadata = visualLayers.value[layerIndex];
    const label = metadata?.kind === "input"
      ? "INPUT"
      : metadata?.kind === "output"
        ? "OUTPUT"
        : metadata?.kind === "conv"
          ? "CONV2D"
          : `DENSE ${(metadata?.kind === "dense" ? metadata.denseIndex : layerIndex) + 1}`;
    ctx.fillText(label, columns[layerIndex], 20);
    ctx.fillStyle = "#7f7c75";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(String(sizes[layerIndex]), columns[layerIndex], height - 17);
  });
}

function animate(time: number) {
  draw(time);
  frame = props.animated ? requestAnimationFrame(animate) : 0;
}

function selectLayer(event: MouseEvent) {
  closeInspector();
  const element = canvas.value;
  if (!element) return;
  const bounds = element.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const sidePadding = bounds.width < 560 ? 38 : 58;
  const count = layerSizes.value.length;
  let nearest = 0;
  let distance = Infinity;
  for (let index = 0; index < count; index++) {
    const column = sidePadding + (index * (bounds.width - sidePadding * 2)) / Math.max(1, count - 1);
    if (Math.abs(column - x) < distance) {
      distance = Math.abs(column - x);
      nearest = index;
    }
  }
  emit("select", nearest);
}

watch(
  () => [
    props.layers,
    props.activations,
    props.gradients,
    props.selectedLayer,
    props.animated,
    props.training,
    props.stepEnabled,
    props.stepDirection,
    props.stepLayer,
  ],
  () => {
    draw();
    if (props.animated && frame === 0) {
      lastTime = performance.now();
      frame = requestAnimationFrame(animate);
    }
  },
  { deep: true },
);

onMounted(() => {
  resizeObserver = new ResizeObserver(() => {
    draw();
    if (inspectedNode.value) nextTick(fitPopover);
  });
  if (canvas.value) resizeObserver.observe(canvas.value);
  if (props.animated) frame = requestAnimationFrame(animate);
  else draw();
});

onBeforeUnmount(() => {
  cancelAnimationFrame(frame);
  resizeObserver?.disconnect();
});
</script>

<template>
  <div class="network-canvas-shell" @keydown.esc="closeInspector">
    <canvas
      ref="canvas"
      class="network-canvas"
      role="img"
      :aria-label="training
        ? '训练神经网络前向激活与反向梯度可视化，点击神经元可查看训练快照'
        : '神经网络逐层激活可视化，点击神经元可查看参数，点击列可查看层统计'"
      @click="selectLayer"
    />

    <button
      v-for="target in nodeTargets"
      :key="`${target.layerIndex}-${target.neuronIndex}-${target.y}`"
      class="neuron-hit-target"
      :class="{
        'is-inspected':
          inspectedNode?.layerIndex === target.layerIndex &&
          inspectedNode?.neuronIndex === target.neuronIndex,
      }"
      type="button"
      :style="{ left: `${target.x}px`, top: `${target.y}px` }"
      :aria-label="nodeLabel(target)"
      @click.stop="inspectNode(target)"
    />

    <aside
      v-if="inspectedNode && neuronDetails"
      ref="popover"
      class="neuron-popover"
      :style="{ left: `${inspectedNode.x}px`, top: `${inspectedNode.y}px` }"
      role="dialog"
      :aria-label="neuronDetails.title"
      tabindex="-1"
      @click.stop
    >
      <header class="neuron-popover-header">
        <div>
          <span>{{ neuronDetails.eyebrow }}</span>
          <h3>{{ neuronDetails.title }}</h3>
        </div>
        <button
          class="neuron-popover-close"
          type="button"
          title="关闭参数面板"
          aria-label="关闭参数面板"
          @click="closeInspector"
        >
          <X :size="15" />
        </button>
      </header>

      <div class="neuron-popover-toolbar">
        <span>{{ neuronDetails.badge }}</span>
        <div>
          <button
            type="button"
            title="上一个神经元"
            aria-label="上一个神经元"
            :disabled="inspectedNode.neuronIndex === 0"
            @click="stepNeuron(-1)"
          >
            <ChevronLeft :size="15" />
          </button>
          <b>
            索引 #{{ inspectedNode.neuronIndex }} ·
            {{ inspectedNode.neuronIndex + 1 }}/{{ layerSizes[inspectedNode.layerIndex] }}
          </b>
          <button
            type="button"
            title="下一个神经元"
            aria-label="下一个神经元"
            :disabled="inspectedNode.neuronIndex === layerSizes[inspectedNode.layerIndex] - 1"
            @click="stepNeuron(1)"
          >
            <ChevronRight :size="15" />
          </button>
        </div>
      </div>

      <dl class="neuron-parameters">
        <div v-for="detail in neuronDetails.details" :key="detail.label">
          <dt>{{ detail.label }}</dt>
          <dd>{{ detail.value }}</dd>
        </div>
      </dl>
      <KernelMatrixView
        v-if="neuronDetails.kernel"
        :weights="neuronDetails.kernel.weights"
        :input-channels="neuronDetails.kernel.inputChannels"
        :kernel-size="neuronDetails.kernel.kernelSize"
        :filter="neuronDetails.kernel.filter"
        :trainable="neuronDetails.kernel.trainable"
      />
      <p>{{ neuronDetails.note }}</p>
    </aside>
  </div>
</template>
