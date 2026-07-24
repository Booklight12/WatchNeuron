<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Check, ChevronDown, Database, Eraser, Shuffle, Trash2, Undo2 } from "@lucide/vue";
import type { DatasetSplit } from "../types";
import SegmentedControl, { type SegmentedControlOption } from "./SegmentedControl.vue";

const props = withDefaults(
  defineProps<{
    trainingPixels?: number[] | null;
    trainingLabel?: number | null;
    customTrainingCount?: number;
    customTestCount?: number;
    datasetLocked?: boolean;
  }>(),
  {
    trainingPixels: null,
    trainingLabel: null,
    customTrainingCount: 0,
    customTestCount: 0,
    datasetLocked: false,
  },
);

const emit = defineEmits<{
  input: [pixels: Float32Array, energy: number];
  sample: [];
  addSample: [label: number, split: DatasetSplit, pixels: Float32Array];
  clearSamples: [];
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
const trainingCanvas = ref<HTMLCanvasElement | null>(null);
const drawing = ref(false);
const canUndo = ref(false);
const inputEnergy = ref(0);
const sampleLabel = ref(0);
const labelPicker = ref<HTMLElement | null>(null);
const labelTrigger = ref<HTMLButtonElement | null>(null);
const labelMenuOpen = ref(false);
const sampleSplit = ref<DatasetSplit>("training");
const sampleSplitOptions: SegmentedControlOption[] = [
  { value: "training", label: "训练" },
  { value: "test", label: "测试" },
];
const captureMessage = ref("");
let captureMessageTimer: ReturnType<typeof setTimeout> | undefined;
const history: ImageData[] = [];
const showingTrainingSample = computed(
  () => props.trainingPixels !== null && props.trainingPixels.length === 784,
);
const datasetControlsDisabled = computed(
  () => showingTrainingSample.value || props.datasetLocked,
);

function labelOptionButtons() {
  return labelPicker.value?.querySelectorAll<HTMLButtonElement>(".sample-label-option") ?? [];
}

function focusLabelOption(index: number) {
  nextTick(() => labelOptionButtons()[index]?.focus());
}

function openLabelMenu(index = sampleLabel.value) {
  if (datasetControlsDisabled.value) return;
  labelMenuOpen.value = true;
  focusLabelOption(index);
}

function closeLabelMenu(restoreFocus = false) {
  labelMenuOpen.value = false;
  if (restoreFocus) nextTick(() => labelTrigger.value?.focus());
}

function chooseLabel(label: number) {
  sampleLabel.value = label;
  closeLabelMenu(true);
}

function handleLabelTriggerKeydown(event: KeyboardEvent) {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    openLabelMenu(sampleLabel.value);
  } else if (event.key === "Home") {
    event.preventDefault();
    openLabelMenu(0);
  } else if (event.key === "End") {
    event.preventDefault();
    openLabelMenu(9);
  } else if (event.key === "Escape") {
    closeLabelMenu();
  }
}

function handleLabelOptionKeydown(event: KeyboardEvent, index: number) {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    focusLabelOption((index + 1) % 10);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    focusLabelOption((index + 9) % 10);
  } else if (event.key === "Home") {
    event.preventDefault();
    focusLabelOption(0);
  } else if (event.key === "End") {
    event.preventDefault();
    focusLabelOption(9);
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeLabelMenu(true);
  }
}

function handleLabelPickerPointerDown(event: PointerEvent) {
  if (!labelPicker.value?.contains(event.target as Node)) closeLabelMenu();
}

function context() {
  return canvas.value?.getContext("2d", { willReadFrequently: true }) ?? null;
}

function point(event: PointerEvent) {
  const element = canvas.value!;
  const bounds = element.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / bounds.width) * element.width,
    y: ((event.clientY - bounds.top) / bounds.height) * element.height,
  };
}

function beginStroke(event: PointerEvent) {
  if (showingTrainingSample.value) return;
  const element = canvas.value;
  const ctx = context();
  if (!element || !ctx) return;
  history.push(ctx.getImageData(0, 0, element.width, element.height));
  if (history.length > 16) history.shift();
  canUndo.value = true;
  drawing.value = true;
  element.setPointerCapture(event.pointerId);
  const position = point(event);
  ctx.beginPath();
  ctx.moveTo(position.x, position.y);
  ctx.lineTo(position.x + 0.01, position.y + 0.01);
  ctx.stroke();
}

function continueStroke(event: PointerEvent) {
  if (!drawing.value || showingTrainingSample.value) return;
  const ctx = context();
  if (!ctx) return;
  const position = point(event);
  ctx.lineTo(position.x, position.y);
  ctx.stroke();
}

function endStroke(event: PointerEvent) {
  if (!drawing.value) return;
  drawing.value = false;
  canvas.value?.releasePointerCapture(event.pointerId);
  publish();
}

function renderTrainingSample() {
  const target = trainingCanvas.value;
  const pixels = props.trainingPixels;
  if (!target) return;
  const ctx = target.getContext("2d")!;
  ctx.clearRect(0, 0, target.width, target.height);
  if (!pixels?.length) return;

  ctx.fillStyle = "#171716";
  ctx.fillRect(0, 0, target.width, target.height);
  const source = document.createElement("canvas");
  source.width = 28;
  source.height = 28;
  const sourceContext = source.getContext("2d")!;
  const image = sourceContext.createImageData(28, 28);
  pixels.forEach((value, index) => {
    const offset = index * 4;
    image.data[offset] = 243;
    image.data[offset + 1] = 244;
    image.data[offset + 2] = 239;
    image.data[offset + 3] = Math.round(Math.max(0, Math.min(1, value)) * 255);
  });
  sourceContext.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, target.width, target.height);
}

function extractPixels() {
  const element = canvas.value;
  const ctx = context();
  if (!element || !ctx) return new Float32Array(784);
  const image = ctx.getImageData(0, 0, element.width, element.height);
  let left = element.width;
  let right = 0;
  let top = element.height;
  let bottom = 0;
  let hasInk = false;

  for (let y = 0; y < element.height; y++) {
    for (let x = 0; x < element.width; x++) {
      const alpha = image.data[(y * element.width + x) * 4 + 3];
      if (alpha < 12) continue;
      hasInk = true;
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
  }
  if (!hasInk) return new Float32Array(784);

  const sourceWidth = Math.max(1, right - left + 1);
  const sourceHeight = Math.max(1, bottom - top + 1);
  const scale = Math.min(20 / sourceWidth, 20 / sourceHeight);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const normalized = document.createElement("canvas");
  normalized.width = 28;
  normalized.height = 28;
  const normalizedContext = normalized.getContext("2d", { willReadFrequently: true })!;
  normalizedContext.imageSmoothingEnabled = true;
  normalizedContext.drawImage(
    element,
    left,
    top,
    sourceWidth,
    sourceHeight,
    Math.round((28 - width) / 2),
    Math.round((28 - height) / 2),
    width,
    height,
  );
  const pixels = normalizedContext.getImageData(0, 0, 28, 28).data;
  return Float32Array.from({ length: 784 }, (_, index) => {
    const offset = index * 4;
    return (pixels[offset] / 255) * (pixels[offset + 3] / 255);
  });
}

function publish(explicitPixels?: Float32Array) {
  const pixels = explicitPixels ?? extractPixels();
  const energy = pixels.reduce((sum, value) => sum + value, 0) / pixels.length;
  inputEnergy.value = energy;
  emit("input", pixels, energy);
}

function showCaptureMessage(message: string) {
  captureMessage.value = message;
  clearTimeout(captureMessageTimer);
  captureMessageTimer = setTimeout(() => {
    captureMessage.value = "";
  }, 2200);
}

function addCurrentSample() {
  const pixels = extractPixels();
  const energy = pixels.reduce((sum, value) => sum + value, 0) / pixels.length;
  if (energy < 0.001) {
    showCaptureMessage("请先写一个数字");
    return;
  }
  emit("addSample", sampleLabel.value, sampleSplit.value, pixels);
  showCaptureMessage(
    `标签 ${sampleLabel.value} 已加入${sampleSplit.value === "training" ? "训练集" : "测试集"}`,
  );
}

function clearCustomSamples() {
  emit("clearSamples");
  showCaptureMessage("自定义样本已清空");
}

function clear() {
  const element = canvas.value;
  const ctx = context();
  if (!element || !ctx) return;
  history.push(ctx.getImageData(0, 0, element.width, element.height));
  ctx.clearRect(0, 0, element.width, element.height);
  canUndo.value = true;
  publish();
}

function undo() {
  const ctx = context();
  const previous = history.pop();
  if (!ctx || !previous) return;
  ctx.putImageData(previous, 0, 0);
  canUndo.value = history.length > 0;
  publish();
}

function loadSample(sample: number[]) {
  const element = canvas.value;
  const ctx = context();
  if (!element || !ctx) return;
  history.length = 0;
  canUndo.value = false;
  ctx.clearRect(0, 0, element.width, element.height);
  const source = document.createElement("canvas");
  source.width = 28;
  source.height = 28;
  const sourceContext = source.getContext("2d")!;
  const image = sourceContext.createImageData(28, 28);
  sample.forEach((value, index) => {
    const offset = index * 4;
    image.data[offset] = 243;
    image.data[offset + 1] = 244;
    image.data[offset + 2] = 239;
    image.data[offset + 3] = Math.round(value * 255);
  });
  sourceContext.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, element.width, element.height);
  publish(Float32Array.from(sample));
}

onMounted(() => {
  document.addEventListener("pointerdown", handleLabelPickerPointerDown);
  const ctx = context();
  if (!ctx) return;
  ctx.strokeStyle = "#f3f4ef";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  renderTrainingSample();
});

watch(
  () => props.trainingPixels,
  () => {
    drawing.value = false;
    renderTrainingSample();
  },
);

watch(datasetControlsDisabled, (disabled) => {
  if (disabled) closeLabelMenu();
});

onBeforeUnmount(() => {
  clearTimeout(captureMessageTimer);
  document.removeEventListener("pointerdown", handleLabelPickerPointerDown);
});

defineExpose({ loadSample, clear });
</script>

<template>
  <div class="digit-input">
    <div class="canvas-frame" :class="{ 'is-training-sample': showingTrainingSample }">
      <span class="corner corner-top-left" />
      <span class="corner corner-top-right" />
      <span class="corner corner-bottom-left" />
      <span class="corner corner-bottom-right" />
      <canvas
        ref="canvas"
        class="draw-canvas"
        :class="{ 'is-readonly': showingTrainingSample }"
        width="280"
        height="280"
        aria-label="手写数字画布"
        @pointerdown.prevent="beginStroke"
        @pointermove.prevent="continueStroke"
        @pointerup.prevent="endStroke"
        @pointercancel.prevent="endStroke"
      />
      <canvas
        v-show="showingTrainingSample"
        ref="trainingCanvas"
        class="training-sample-canvas"
        width="280"
        height="280"
        :aria-label="`训练采样数字 ${trainingLabel ?? '未知'}`"
      />
      <span v-if="showingTrainingSample" class="training-sample-badge">
        <i />训练样本
        <b>标签 {{ trainingLabel ?? "-" }}</b>
      </span>
    </div>
    <div class="canvas-toolbar" aria-label="画布工具">
      <button class="icon-button" type="button" :disabled="!canUndo || showingTrainingSample" title="撤销" aria-label="撤销" @click="undo">
        <Undo2 :size="17" />
      </button>
      <button class="icon-button" type="button" :disabled="showingTrainingSample" title="清空画布" aria-label="清空画布" @click="clear">
        <Eraser :size="17" />
      </button>
      <span class="toolbar-spacer" />
      <button class="tool-button" type="button" :disabled="showingTrainingSample" @click="emit('sample')">
        <Shuffle :size="16" />
        样本
      </button>
    </div>
    <div class="dataset-capture" aria-label="自定义数据集">
      <div class="dataset-capture-controls">
        <div
          ref="labelPicker"
          class="sample-label-picker"
          :class="{ 'is-open': labelMenuOpen, disabled: datasetControlsDisabled }"
        >
          <button
            ref="labelTrigger"
            class="sample-label-field"
            type="button"
            aria-haspopup="listbox"
            :aria-expanded="labelMenuOpen"
            aria-controls="sample-label-menu"
            :aria-label="`样本标签，当前 ${sampleLabel}`"
            :disabled="datasetControlsDisabled"
            @click="labelMenuOpen ? closeLabelMenu() : openLabelMenu()"
            @keydown="handleLabelTriggerKeydown"
          >
            <span>标签</span>
            <b>{{ sampleLabel }}</b>
            <ChevronDown :size="13" aria-hidden="true" />
          </button>
          <Transition name="label-menu">
            <div
              v-if="labelMenuOpen"
              id="sample-label-menu"
              class="sample-label-menu"
              role="listbox"
              aria-label="选择数字标签"
            >
              <button
                v-for="digit in 10"
                :key="digit - 1"
                class="sample-label-option"
                :class="{ selected: sampleLabel === digit - 1 }"
                type="button"
                role="option"
                :aria-selected="sampleLabel === digit - 1"
                :tabindex="sampleLabel === digit - 1 ? 0 : -1"
                @click="chooseLabel(digit - 1)"
                @keydown="handleLabelOptionKeydown($event, digit - 1)"
              >
                <span>数字 {{ digit - 1 }}</span>
                <Check v-if="sampleLabel === digit - 1" :size="13" aria-hidden="true" />
              </button>
            </div>
          </Transition>
        </div>
        <SegmentedControl
          v-model="sampleSplit"
          class="sample-split-toggle"
          :options="sampleSplitOptions"
          label="加入的数据集"
          :disabled="datasetControlsDisabled"
        />
        <button
          class="add-sample-button"
          type="button"
          data-testid="add-custom-sample"
          :title="`将标签 ${sampleLabel} 加入${sampleSplit === 'training' ? '训练集' : '测试集'}`"
          :aria-label="`将标签 ${sampleLabel} 加入${sampleSplit === 'training' ? '训练集' : '测试集'}`"
          :disabled="inputEnergy < 0.001 || datasetControlsDisabled"
          @click="addCurrentSample"
        >
          <Database :size="15" />
          加入
        </button>
      </div>
      <div class="dataset-capture-status">
        <span>训练 +{{ customTrainingCount }} · 测试 +{{ customTestCount }}</span>
        <b aria-live="polite">{{ captureMessage }}</b>
        <button
          type="button"
          title="清空自定义样本"
          aria-label="清空自定义样本"
          :disabled="customTrainingCount + customTestCount === 0 || datasetControlsDisabled"
          @click="clearCustomSamples"
        >
          <Trash2 :size="14" />
        </button>
      </div>
    </div>
  </div>
</template>
