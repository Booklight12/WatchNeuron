<script setup lang="ts">
import {
  Activity,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Layers3,
  Pause,
  Play,
  Radio,
  ScanLine,
  Snowflake,
  X,
} from "@lucide/vue";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { activationLabels } from "../lib/model";
import { convolutionPipeline, modelConvolutions, spatialPipeline } from "../lib/convolution";
import type {
  ConvolutionConfig,
  HiddenLayer,
  NeuralModel,
  OutputHeadKind,
  PoolingConfig,
  TrainingProgress,
  TrainingTrace,
} from "../types";
import SignalMap from "./SignalMap.vue";
import KernelMatrixView from "./KernelMatrixView.vue";

interface MapShape {
  width: number;
  height: number;
  channels: number;
}

interface LayerDescriptor {
  index: number;
  kind: "input" | "conv" | "pool" | "dense" | "output";
  code: string;
  name: string;
  activation: string;
  count: number;
  shape?: MapShape;
  convolutionIndex?: number;
  spatialIndex?: number;
  denseIndex?: number;
}

type SignalKind = "input" | "activation" | "gradient";

interface Selection {
  panelLayerIndex: number;
  sourceLayerIndex: number;
  kind: SignalKind;
  neuronIndex: number;
}

interface SignalMapSelection {
  index: number;
  clientX: number;
  clientY: number;
}

interface KernelSnapshot {
  weights: number[];
  inputChannels: number;
  kernelSize: number;
  filter: number;
  trainable: boolean;
}

interface FrozenMeta {
  capturedAt: number;
  training: boolean;
  trace: Omit<TrainingTrace, "activations" | "gradients"> | null;
}

const props = defineProps<{
  layers: HiddenLayer[];
  convolutions: ConvolutionConfig[];
  poolings: PoolingConfig[];
  outputHead: OutputHeadKind;
  model: NeuralModel | null;
  activations: number[][];
  gradients: number[][];
  training: boolean;
  progress: TrainingProgress;
  trace: TrainingTrace | null;
}>();

const emit = defineEmits<{
  back: [];
  pauseTraining: [];
  resumeTraining: [];
}>();

const frozenActivations = shallowRef<number[][] | null>(null);
const frozenGradients = shallowRef<number[][] | null>(null);
const frozenMeta = ref<FrozenMeta | null>(null);
const pauseRequested = ref(false);
const inspector = ref<HTMLElement | null>(null);
const inspectorOpen = ref(false);
const inspectorAnchor = ref({ x: 0, y: 0 });
const inspectorPosition = ref({ x: 0, y: 0 });
const selection = ref<Selection>({
  panelLayerIndex: 0,
  sourceLayerIndex: 0,
  kind: "activation",
  neuronIndex: 0,
});

const convolutionEntries = computed(() => convolutionPipeline(props.layers, props.convolutions, props.poolings));
const spatialEntries = computed(() => spatialPipeline(props.layers, props.convolutions, props.poolings));
const layerDescriptors = computed<LayerDescriptor[]>(() => {
  const result: LayerDescriptor[] = [
    {
      index: 0,
      kind: "input",
      code: "INPUT",
      name: "输入层",
      activation: "恒等映射",
      count: 784,
      shape: { width: 28, height: 28, channels: 1 },
    },
  ];
  for (let position = 0; position <= props.layers.length; position++) {
    for (const entry of spatialEntries.value.filter(({ config }) => config.position === position)) {
      const spatialIndex = spatialEntries.value.indexOf(entry);
      if (entry.kind === "conv") {
        const convolutionIndex = convolutionEntries.value.findIndex(({ config }) => config.id === entry.config.id);
        result.push({
        index: result.length,
        kind: "conv",
        code: `CONV2D ${convolutionIndex + 1}`,
        name: `卷积层 ${convolutionIndex + 1}`,
        activation: activationLabels[entry.config.activation],
        count: entry.output.length,
        shape: {
          width: entry.output.width,
          height: entry.output.height,
          channels: entry.output.channels,
        },
        convolutionIndex,
        spatialIndex,
      });
      } else result.push({
        index: result.length,
        kind: "pool",
        code: entry.config.kind === "globalAverage" ? "GAP" : entry.config.kind === "max" ? "MAXPOOL2D" : "AVGPOOL2D",
        name: entry.config.kind === "globalAverage" ? "全局平均池化" : entry.config.kind === "max" ? "最大池化层" : "平均池化层",
        activation: "无参数聚合",
        count: entry.output.length,
        shape: { width: entry.output.width, height: entry.output.height, channels: entry.output.channels },
        spatialIndex,
      });
    }
    if (position < props.layers.length) {
      const layer = props.layers[position];
      result.push({
        index: result.length,
        kind: "dense",
        code: `DENSE ${position + 1}`,
        name: `全连接层 ${position + 1}`,
        activation: activationLabels[layer.activation],
        count: layer.units,
        denseIndex: position,
      });
    }
  }
  result.push({
    index: result.length,
    kind: "output",
    code: "OUTPUT",
    name: "输出层",
      activation: props.outputHead === "sigmoid" ? "Sigmoid + BCE" : "Softmax + CE",
    count: 10,
    denseIndex: props.layers.length,
  });
  return result;
});

const displayedActivations = computed(() => frozenActivations.value ?? props.activations);
const displayedGradients = computed(() => frozenGradients.value ?? props.gradients);
const displayingTraining = computed(() => frozenMeta.value?.training ?? props.training);
const viewerPaused = computed(
  () => frozenActivations.value !== null || props.progress.phase === "paused",
);
const hasGradients = computed(
  () => displayingTraining.value && displayedGradients.value.some((values) => values?.length > 0),
);
const currentTrace = computed(() => frozenMeta.value?.trace ?? props.trace);
const predictedDigit = computed(() => {
  const probabilities = displayedActivations.value.at(-1) ?? [];
  if (!probabilities.length) return -1;
  let maximum = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index] > probabilities[maximum]) maximum = index;
  }
  return maximum;
});

const selectedDescriptor = computed(
  () => layerDescriptors.value[selection.value.sourceLayerIndex] ?? layerDescriptors.value[0],
);
const selectedValues = computed(() => {
  if (selection.value.kind === "gradient") {
    return displayedGradients.value[selection.value.sourceLayerIndex] ?? [];
  }
  return displayedActivations.value[selection.value.sourceLayerIndex] ?? [];
});
const selectedActivation = computed(
  () => displayedActivations.value[selection.value.sourceLayerIndex]?.[selection.value.neuronIndex] ?? 0,
);
const selectedGradient = computed(
  () => displayedGradients.value[selection.value.sourceLayerIndex]?.[selection.value.neuronIndex] ?? 0,
);

function formatValue(value: number) {
  if (value === 0) return "0.000000";
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 1000) return value.toExponential(4);
  return value.toFixed(6);
}

function vectorStats(values: number[]) {
  if (!values.length) return { mean: 0, absoluteMean: 0, minimum: 0, maximum: 0, active: 0 };
  let sum = 0;
  let absoluteSum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let active = 0;
  for (const rawValue of values) {
    const value = Number.isFinite(rawValue) ? rawValue : 0;
    sum += value;
    absoluteSum += Math.abs(value);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (Math.abs(value) > 0.05) active++;
  }
  return {
    mean: sum / values.length,
    absoluteMean: absoluteSum / values.length,
    minimum,
    maximum,
    active: (active / values.length) * 100,
  };
}

function statsForLayer(index: number) {
  return vectorStats(displayedActivations.value[index] ?? []);
}

function captureSnapshot() {
  frozenActivations.value = displayedActivations.value.map((values) => [...values]);
  frozenGradients.value = displayedGradients.value.map((values) => [...values]);
  const trace = props.trace
    ? {
        epoch: props.trace.epoch,
        sample: props.trace.sample,
        samples: props.trace.samples,
        label: props.trace.label,
        prediction: props.trace.prediction,
        loss: props.trace.loss,
        convolutionWeights: props.trace.convolutionWeights.map((weights) => [...weights]),
        convolutionBiases: props.trace.convolutionBiases.map((biases) => [...biases]),
      }
    : null;
  frozenMeta.value = { capturedAt: Date.now(), training: props.training, trace };
}

function togglePause() {
  if (viewerPaused.value) {
    frozenActivations.value = null;
    frozenGradients.value = null;
    frozenMeta.value = null;
    if (props.progress.phase === "paused" || pauseRequested.value) emit("resumeTraining");
    pauseRequested.value = false;
    return;
  }
  captureSnapshot();
  if (props.progress.phase === "training") {
    pauseRequested.value = true;
    emit("pauseTraining");
  }
}

function selectNeuron(
  panelLayerIndex: number,
  sourceLayerIndex: number,
  kind: SignalKind,
  target: SignalMapSelection,
) {
  selection.value = { panelLayerIndex, sourceLayerIndex, kind, neuronIndex: target.index };
  inspectorAnchor.value = { x: target.clientX, y: target.clientY };
  inspectorPosition.value = { x: target.clientX + 14, y: target.clientY + 14 };
  inspectorOpen.value = true;
  nextTick(() => {
    fitInspector();
    inspector.value?.focus();
  });
}

function fitInspector() {
  const panel = inspector.value;
  if (!panel) return;
  const margin = 12;
  const gap = 14;
  const anchor = inspectorAnchor.value;
  const width = panel.offsetWidth;
  const height = panel.offsetHeight;
  let x = anchor.x + gap;
  let y = anchor.y + gap;
  if (x + width > window.innerWidth - margin) x = anchor.x - width - gap;
  if (y + height > window.innerHeight - margin) y = anchor.y - height - gap;
  inspectorPosition.value = {
    x: Math.max(margin, Math.min(x, window.innerWidth - width - margin)),
    y: Math.max(margin, Math.min(y, window.innerHeight - height - margin)),
  };
}

function closeInspector() {
  inspectorOpen.value = false;
}

function handleDocumentPointerDown(event: PointerEvent) {
  if (!inspectorOpen.value) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (inspector.value?.contains(target) || target.closest(".signal-map-surface")) return;
  closeInspector();
}

function handleWindowResize() {
  if (inspectorOpen.value) fitInspector();
}

function handleWindowScroll() {
  if (inspectorOpen.value) closeInspector();
}

function selectedIndex(panelLayerIndex: number, kind: SignalKind) {
  return selection.value.panelLayerIndex === panelLayerIndex && selection.value.kind === kind
    ? selection.value.neuronIndex
    : null;
}

function stepNeuron(amount: -1 | 1) {
  const length = selectedValues.value.length;
  if (!length) return;
  selection.value.neuronIndex = Math.max(
    0,
    Math.min(length - 1, selection.value.neuronIndex + amount),
  );
  nextTick(fitInspector);
}

function jumpToNeuron(input: HTMLInputElement) {
  const value = Math.floor(Number(input.value));
  const length = selectedValues.value.length;
  if (!Number.isFinite(value) || !length) {
    input.value = String(selection.value.neuronIndex);
    return;
  }
  selection.value.neuronIndex = Math.max(0, Math.min(length - 1, value));
  nextTick(fitInspector);
}

const neuronDetails = computed(() => {
  const descriptor = selectedDescriptor.value;
  const index = selection.value.neuronIndex;
  const details: Array<{ label: string; value: string }> = [
    {
      label: selection.value.kind === "gradient"
        ? "选中梯度"
        : selection.value.kind === "input" ? "输入信号" : "激活值",
      value: formatValue(selectedValues.value[index] ?? 0),
    },
    { label: "本层激活", value: formatValue(selectedActivation.value) },
    ...(hasGradients.value
      ? [
          { label: "反向梯度", value: formatValue(selectedGradient.value) },
          {
            label: "梯度符号",
            value: selectedGradient.value > 0 ? "正" : selectedGradient.value < 0 ? "负" : "零",
          },
        ]
      : []),
  ];

  if (descriptor.kind === "input") {
    details.push(
      { label: "像素坐标", value: `行 ${Math.floor(index / 28) + 1} · 列 ${(index % 28) + 1}` },
      { label: "归一化范围", value: "0 - 1" },
      { label: "可训练参数", value: "无" },
    );
    return details;
  }

  if (descriptor.kind === "conv") {
    const shape = descriptor.shape!;
    const mapSize = shape.width * shape.height;
    const filter = Math.floor(index / mapSize);
    const position = index % mapSize;
    const convolution = modelConvolutions(props.model)[descriptor.convolutionIndex ?? 0];
    const convolutionConfig = convolutionEntries.value[descriptor.convolutionIndex ?? 0]?.config;
    const weightsSource = displayingTraining.value
      ? currentTrace.value?.convolutionWeights[descriptor.convolutionIndex ?? 0] ?? convolution?.weights
      : convolution?.weights;
    const biasesSource = displayingTraining.value
      ? currentTrace.value?.convolutionBiases[descriptor.convolutionIndex ?? 0] ?? convolution?.biases
      : convolution?.biases;
    const kernelLength = (convolution?.inputChannels ?? 1) * (convolution?.kernelSize ?? 1) ** 2;
    const start = filter * kernelLength;
    const weights = Array.from(weightsSource ?? []).slice(start, start + kernelLength);
    let absoluteSum = 0;
    let squaredSum = 0;
    for (const weight of weights) {
      absoluteSum += Math.abs(weight);
      squaredSum += weight * weight;
    }
    details.push(
      { label: "特征图", value: `${filter + 1} / ${shape.channels}` },
      {
        label: "特征坐标",
        value: `行 ${Math.floor(position / shape.width) + 1} · 列 ${(position % shape.width) + 1}`,
      },
      { label: "卷积偏置", value: formatValue(biasesSource?.[filter] ?? 0) },
      { label: "参数状态", value: convolutionConfig?.trainable === false ? "已冻结" : "参与训练" },
      {
        label: "核权重绝对均值",
        value: formatValue(weights.length ? absoluteSum / weights.length : 0),
      },
      { label: "核权重 L2", value: formatValue(Math.sqrt(squaredSum)) },
    );
    return details;
  }

  if (descriptor.kind === "pool") {
    const entry = spatialEntries.value[descriptor.spatialIndex ?? 0];
    if (!entry || entry.kind !== "pool") return details;
    const shape = descriptor.shape!;
    const mapSize = shape.width * shape.height;
    const channel = Math.floor(index / mapSize);
    const position = index % mapSize;
    details.push(
      { label: "通道", value: `${channel + 1} / ${shape.channels}` },
      { label: "输出坐标", value: `行 ${Math.floor(position / shape.width) + 1} · 列 ${(position % shape.width) + 1}` },
      { label: "聚合方式", value: entry?.config.kind === "globalAverage" ? "全局平均" : entry?.config.kind === "max" ? "窗口最大值" : "窗口平均值" },
      { label: "可训练参数", value: "无" },
    );
    return details;
  }

  const denseLayer = props.model?.layers[descriptor.denseIndex ?? 0];
  if (!denseLayer || index >= denseLayer.outputSize) return details;
  const offset = index * denseLayer.inputSize;
  let sum = 0;
  let absoluteSum = 0;
  let squaredSum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let weightIndex = 0; weightIndex < denseLayer.inputSize; weightIndex++) {
    const weight = denseLayer.weights[offset + weightIndex];
    sum += weight;
    absoluteSum += Math.abs(weight);
    squaredSum += weight * weight;
    minimum = Math.min(minimum, weight);
    maximum = Math.max(maximum, weight);
  }
  details.push(
    { label: "偏置", value: formatValue(denseLayer.biases[index] ?? 0) },
    { label: "输入连接", value: denseLayer.inputSize.toLocaleString("zh-CN") },
    { label: "权重均值", value: formatValue(sum / denseLayer.inputSize) },
    { label: "权重绝对均值", value: formatValue(absoluteSum / denseLayer.inputSize) },
    { label: "权重范围", value: `${formatValue(minimum)} - ${formatValue(maximum)}` },
    { label: "权重 L2", value: formatValue(Math.sqrt(squaredSum)) },
  );
  return details;
});

const selectedKernel = computed<KernelSnapshot | null>(() => {
  const descriptor = selectedDescriptor.value;
  if (descriptor.kind !== "conv") return null;
  const convolutionIndex = descriptor.convolutionIndex ?? 0;
  const config = convolutionEntries.value[convolutionIndex]?.config;
  const convolution = modelConvolutions(props.model)[convolutionIndex];
  if (!config || !convolution) return null;
  const shape = descriptor.shape!;
  const filter = Math.floor(selection.value.neuronIndex / (shape.width * shape.height));
  const weights = displayingTraining.value
    ? currentTrace.value?.convolutionWeights[convolutionIndex] ?? convolution.weights
    : convolution.weights;
  return {
    weights: Array.from(weights),
    inputChannels: convolution.inputChannels,
    kernelSize: convolution.kernelSize,
    filter,
    trainable: config.trainable,
  };
});

const selectionContext = computed(() => {
  const panel = layerDescriptors.value[selection.value.panelLayerIndex];
  if (selection.value.kind !== "input" || selection.value.panelLayerIndex === 0) {
    return selectedDescriptor.value.name;
  }
  return `${selectedDescriptor.value.name} → ${panel?.name ?? "下一层"}`;
});

watch(layerDescriptors, (layers) => {
  if (!layers.length) return;
  const source = Math.min(selection.value.sourceLayerIndex, layers.length - 1);
  const values = displayedActivations.value[source] ?? [];
  selection.value.sourceLayerIndex = source;
  selection.value.panelLayerIndex = Math.min(selection.value.panelLayerIndex, layers.length - 1);
  selection.value.neuronIndex = Math.min(selection.value.neuronIndex, Math.max(0, values.length - 1));
});

watch(
  () => props.progress.phase,
  (phase) => {
    if (phase === "paused") pauseRequested.value = false;
  },
);

onMounted(() => {
  document.addEventListener("pointerdown", handleDocumentPointerDown);
  window.addEventListener("resize", handleWindowResize);
  window.addEventListener("scroll", handleWindowScroll);
});

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", handleDocumentPointerDown);
  window.removeEventListener("resize", handleWindowResize);
  window.removeEventListener("scroll", handleWindowScroll);
});
</script>

<template>
  <main
    class="signal-flow-manager"
    aria-labelledby="signal-flow-heading"
    @scroll="closeInspector"
    @keydown.esc="closeInspector"
  >
    <header class="signal-flow-heading">
      <button class="tool-button manager-back-button" type="button" @click="emit('back')">
        <ArrowLeft :size="17" />
        <span>返回实验台</span>
      </button>
      <div>
        <span class="eyebrow">SIGNAL FLOW · LAYER STATE</span>
        <h1 id="signal-flow-heading">神经信号流</h1>
      </div>
      <div class="signal-flow-heading-actions">
        <span class="signal-capture-state" :class="{ paused: viewerPaused }">
          <Snowflake v-if="viewerPaused" :size="14" />
          <Radio v-else :size="14" />
          {{ viewerPaused ? "快照已冻结" : "实时采样" }}
        </span>
        <button
          class="signal-pause-button"
          type="button"
          data-testid="signal-flow-pause"
          :aria-pressed="viewerPaused"
          @click="togglePause"
        >
          <Play v-if="viewerPaused" :size="16" />
          <Pause v-else :size="16" />
          {{ viewerPaused ? (progress.phase === "paused" ? "继续训练" : "继续采样") : "暂停并检查" }}
        </button>
      </div>
    </header>

    <section class="signal-flow-summary" aria-label="信号流状态">
      <div>
        <ScanLine :size="16" />
        <span>来源</span>
        <b>{{ displayingTraining ? "训练追踪" : "推理前向传播" }}</b>
      </div>
      <div>
        <Layers3 :size="16" />
        <span>网络层</span>
        <b>{{ layerDescriptors.length }}</b>
      </div>
      <div>
        <Activity :size="16" />
        <span>{{ displayingTraining ? "训练位置" : "当前预测" }}</span>
        <b v-if="displayingTraining && currentTrace">
          E{{ currentTrace.epoch }} · {{ currentTrace.sample.toLocaleString("zh-CN") }}/{{ currentTrace.samples.toLocaleString("zh-CN") }}
        </b>
        <b v-else>{{ predictedDigit < 0 ? "-" : predictedDigit }}</b>
      </div>
      <div>
        <span>{{ displayingTraining ? "样本 / 损失" : "输出置信度" }}</span>
        <b v-if="displayingTraining && currentTrace">
          {{ currentTrace.label }} → {{ currentTrace.prediction }} · {{ formatValue(currentTrace.loss) }}
        </b>
        <b v-else>
          {{ predictedDigit < 0 ? "0.00%" : `${((displayedActivations.at(-1)?.[predictedDigit] ?? 0) * 100).toFixed(2)}%` }}
        </b>
      </div>
    </section>

    <div class="signal-flow-layout">
      <section class="signal-layer-list" aria-label="全部网络层状态">
        <article
          v-for="layer in layerDescriptors"
          :key="`${layer.code}-${layer.index}`"
          class="signal-layer-panel"
          :class="{ selected: selection.panelLayerIndex === layer.index }"
        >
          <header class="signal-layer-heading">
            <span class="signal-layer-index">{{ String(layer.index).padStart(2, "0") }}</span>
            <div>
              <small>{{ layer.code }}</small>
              <h2>{{ layer.name }}</h2>
            </div>
            <span class="signal-layer-activation">{{ layer.activation }}</span>
            <dl>
              <div><dt>神经元</dt><dd>{{ layer.count.toLocaleString("zh-CN") }}</dd></div>
              <div><dt>平均激活</dt><dd>{{ formatValue(statsForLayer(layer.index).mean) }}</dd></div>
              <div><dt>活跃</dt><dd>{{ statsForLayer(layer.index).active.toFixed(1) }}%</dd></div>
            </dl>
          </header>

          <div class="signal-layer-maps">
            <SignalMap
              v-if="layer.index > 0"
              :values="displayedActivations[layer.index - 1] ?? []"
              :shape="layerDescriptors[layer.index - 1]?.shape"
              label="输入信号"
              :selected-index="selectedIndex(layer.index, 'input')"
              @select="selectNeuron(layer.index, layer.index - 1, 'input', $event)"
            />
            <SignalMap
              :values="displayedActivations[layer.index] ?? []"
              :shape="layer.shape"
              :label="layer.index === 0 ? '输入信号 · 恒等激活' : '激活状态'"
              :selected-index="selectedIndex(layer.index, 'activation')"
              @select="selectNeuron(layer.index, layer.index, 'activation', $event)"
            />
            <SignalMap
              v-if="hasGradients"
              :values="displayedGradients[layer.index] ?? []"
              :shape="layer.shape"
              label="反向梯度"
              :selected-index="selectedIndex(layer.index, 'gradient')"
              @select="selectNeuron(layer.index, layer.index, 'gradient', $event)"
            />
          </div>
        </article>
      </section>

      <Teleport to="body">
        <Transition name="signal-inspector">
          <aside
            v-if="inspectorOpen"
            ref="inspector"
            class="signal-neuron-inspector"
            :style="{ left: `${inspectorPosition.x}px`, top: `${inspectorPosition.y}px` }"
            role="dialog"
            aria-live="polite"
            aria-label="神经元详细信息"
            tabindex="-1"
            @keydown.esc="closeInspector"
          >
            <header>
              <div>
                <span>{{ selectedDescriptor.code }} · {{ selection.kind.toUpperCase() }}</span>
                <h2>{{ selectedDescriptor.kind === "output" ? `数字 ${selection.neuronIndex}` : `神经元 #${selection.neuronIndex}` }}</h2>
                <p>{{ selectionContext }}</p>
              </div>
              <button
                class="signal-neuron-close"
                type="button"
                title="关闭神经元详情"
                aria-label="关闭神经元详情"
                @click="closeInspector"
              >
                <X :size="15" />
              </button>
            </header>

            <div class="signal-neuron-navigation">
              <button
                type="button"
                title="上一个神经元"
                aria-label="上一个神经元"
                :disabled="selection.neuronIndex <= 0"
                @click="stepNeuron(-1)"
              >
                <ChevronLeft :size="16" />
              </button>
              <label>
                <span>索引</span>
                <input
                  type="number"
                  min="0"
                  :max="Math.max(0, selectedValues.length - 1)"
                  :value="selection.neuronIndex"
                  @change="jumpToNeuron($event.target as HTMLInputElement)"
                />
              </label>
              <b>/ {{ Math.max(0, selectedValues.length - 1).toLocaleString("zh-CN") }}</b>
              <button
                type="button"
                title="下一个神经元"
                aria-label="下一个神经元"
                :disabled="selection.neuronIndex >= selectedValues.length - 1"
                @click="stepNeuron(1)"
              >
                <ChevronRight :size="16" />
              </button>
            </div>

            <dl class="signal-neuron-details">
              <div v-for="detail in neuronDetails" :key="detail.label">
                <dt>{{ detail.label }}</dt>
                <dd>{{ detail.value }}</dd>
              </div>
            </dl>

            <KernelMatrixView
              v-if="selectedKernel"
              :weights="selectedKernel.weights"
              :input-channels="selectedKernel.inputChannels"
              :kernel-size="selectedKernel.kernelSize"
              :filter="selectedKernel.filter"
              :trainable="selectedKernel.trainable"
            />

            <footer>
              <span>{{ viewerPaused ? "FROZEN FRAME" : "LIVE FRAME" }}</span>
              <time v-if="frozenMeta" :datetime="new Date(frozenMeta.capturedAt).toISOString()">
                {{ new Date(frozenMeta.capturedAt).toLocaleTimeString("zh-CN", { hour12: false }) }}
              </time>
            </footer>
          </aside>
        </Transition>
      </Teleport>
    </div>
  </main>
</template>
