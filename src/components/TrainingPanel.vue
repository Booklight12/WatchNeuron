<script setup lang="ts">
import { BrainCircuit, Pause, Play, RefreshCcw, Save, SlidersHorizontal, Square, TrendingUp } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { architectureLayerSizes } from "../lib/convolution";
import SegmentedControl, { type SegmentedControlOption } from "./SegmentedControl.vue";
import type {
  ConvolutionConfig,
  HiddenLayer,
  PoolingConfig,
  TrainingMode,
  TrainingProfiles,
  TrainingProgress,
  TrainingSettings,
} from "../types";

const props = defineProps<{
  layers: HiddenLayer[];
  convolutions: ConvolutionConfig[];
  poolings: PoolingConfig[];
  backend: "Wasm SIMD" | "Wasm" | "JavaScript";
  profiles: TrainingProfiles;
  progress: TrainingProgress;
  customTrainingCount: number;
  customTestCount: number;
  mnistEnabled: boolean;
  saveState: "idle" | "saving" | "saved" | "error";
  canFineTune: boolean;
  trainedEpochs: number;
  mode: TrainingMode;
}>();

const emit = defineEmits<{
  update: [mode: TrainingMode, settings: TrainingSettings];
  train: [];
  fineTune: [];
  configureOptimizer: [mode: TrainingMode];
  pause: [];
  resume: [];
  saveSnapshot: [];
  cancel: [];
}>();

const editingMode = ref<TrainingMode>("scratch");
const trainingProfileOptions: SegmentedControlOption[] = [
  { value: "scratch", label: "重头训练", testId: "training-profile-scratch" },
  { value: "finetune", label: "微调训练", testId: "training-profile-finetune" },
];
const mathModeOptions: SegmentedControlOption[] = [
  { value: "fast", label: "快速", title: "使用近似数学函数以提高速度" },
  { value: "full", label: "完整", title: "使用标准精度数学函数" },
];
const computeBackendOptions: SegmentedControlOption[] = [
  { value: "wasm", label: "Zig/Wasm", title: "使用 Zig 编译的 Wasm SIMD/标量内核" },
  { value: "webgpu", label: "Zig/WebGPU", title: "使用 WGSL 在 GPU 上执行批量张量计算" },
];
const settings = computed<TrainingSettings>(() => ({
  ...props.profiles[editingMode.value],
  mathMode: props.profiles.mathMode,
  computeBackend: props.profiles.computeBackend,
  optimizer: { ...props.profiles[editingMode.value].optimizer },
}));

function setSetting(key: "epochs" | "learningRate", value: number) {
  emit("update", editingMode.value, { ...settings.value, [key]: value });
}

function setMathMode(mathMode: "fast" | "full") {
  emit("update", editingMode.value, { ...settings.value, mathMode });
}

function setComputeBackend(computeBackend: "wasm" | "webgpu") {
  emit("update", editingMode.value, {
    ...settings.value,
    computeBackend,
    mathMode: computeBackend === "webgpu" ? "full" : settings.value.mathMode,
  });
}

function setEpochs(value: string) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return;
  setSetting("epochs", Math.max(1, Math.floor(parsed)));
}

function setLearningRate(input: HTMLInputElement) {
  const parsed = Number(input.value);
  if (!Number.isFinite(parsed) || parsed <= 0) return;
  setSetting("learningRate", parsed);
}

function normalizeLearningRate(input: HTMLInputElement) {
  const parsed = Number(input.value);
  if (!Number.isFinite(parsed) || parsed <= 0) input.value = String(settings.value.learningRate);
}

function start(mode: TrainingMode) {
  editingMode.value = mode;
  if (mode === "finetune") emit("fineTune");
  else emit("train");
}

const optimizerLabels = {
  sgd: "SGD",
  momentum: "Momentum",
  adam: "Adam",
  rmsprop: "RMSProp",
  adagrad: "AdaGrad",
};

const trainingLayerSizes = computed(() => architectureLayerSizes(props.layers, props.convolutions, props.poolings));
const controlsLocked = computed(
  () => props.progress.phase === "loading" ||
    props.progress.phase === "training" ||
    props.progress.phase === "paused",
);

watch(
  () => props.canFineTune,
  (canFineTune) => {
    if (!canFineTune) editingMode.value = "scratch";
  },
);

function progressLabel() {
  if (props.progress.phase === "loading") {
    return props.mode === "finetune" ? "载入模型与数据" : "载入数据";
  }
  if (props.progress.phase === "paused") {
    return `${props.mode === "finetune" ? "微调" : "训练"}暂停 · Epoch ${props.progress.epoch} / ${props.progress.epochs}`;
  }
  if (props.progress.phase === "complete") {
    return props.mode === "finetune" ? "微调完成" : "训练完成";
  }
  if (props.progress.phase === "cancelled") return "已取消";
  if (props.progress.phase === "error") return props.mode === "finetune" ? "微调失败" : "训练失败";
  return `${props.mode === "finetune" ? "微调" : "训练"} · Epoch ${props.progress.epoch} / ${props.progress.epochs}`;
}
</script>

<template>
  <section class="training-section" aria-labelledby="training-heading">
    <div class="section-heading compact-heading">
      <div>
        <span class="eyebrow">TRAIN</span>
        <h2 id="training-heading">模型训练</h2>
      </div>
      <span class="training-runtime">
        <BrainCircuit :size="14" />
        {{ settings.computeBackend === "webgpu" ? "Zig/WebGPU" : backend === "Wasm SIMD" ? "Zig/Wasm SIMD" : "Zig/Wasm" }} · {{ settings.mathMode === "full" ? "完整" : "快速" }}
      </span>
    </div>

    <div class="training-architecture" aria-label="训练网络结构">
      <span
        v-for="(size, index) in trainingLayerSizes"
        :key="`${index}-${size}`"
      >
        <b>{{ size }}</b>
        <i v-if="index < trainingLayerSizes.length - 1">→</i>
      </span>
    </div>

    <SegmentedControl
      v-if="canFineTune"
      v-model="editingMode"
      class="training-profile-toggle"
      :options="trainingProfileOptions"
      label="训练配置"
      :disabled="controlsLocked"
    />

    <div class="math-mode-setting">
      <span>张量后端</span>
      <SegmentedControl
        class="math-mode-toggle"
        :model-value="settings.computeBackend"
        :options="computeBackendOptions"
        label="Zig 张量计算后端"
        :disabled="controlsLocked"
        @update:model-value="setComputeBackend($event as 'wasm' | 'webgpu')"
      />
    </div>

    <div class="math-mode-setting">
      <span>数学实现</span>
      <SegmentedControl
        class="math-mode-toggle"
        :model-value="settings.mathMode"
        :options="mathModeOptions"
        label="Zig 数学实现"
        :disabled="controlsLocked || settings.computeBackend === 'webgpu'"
        @update:model-value="setMathMode($event as 'fast' | 'full')"
      />
    </div>

    <div v-if="canFineTune" class="fine-tune-source">
      <span><BrainCircuit :size="14" />当前模型</span>
      <b>累计 {{ trainedEpochs.toLocaleString() }} 轮</b>
    </div>

    <div class="training-settings">
      <label class="epoch-setting">
        <span>{{ editingMode === 'finetune' ? '微调轮数' : '重训轮数' }}</span>
        <input
          class="epoch-input"
          type="number"
          min="1"
          step="1"
          inputmode="numeric"
          aria-label="训练轮数"
          :value="settings.epochs"
          :disabled="progress.phase === 'loading' || progress.phase === 'training' || progress.phase === 'paused'"
          @input="setEpochs(($event.target as HTMLInputElement).value)"
        />
      </label>
      <label>
        <span>学习率 <b>{{ settings.learningRate }}</b></span>
        <input
          class="learning-rate-input"
          type="number"
          step="any"
          inputmode="decimal"
          :value="settings.learningRate"
          :disabled="progress.phase === 'loading' || progress.phase === 'training' || progress.phase === 'paused'"
          data-testid="training-learning-rate"
          @input="setLearningRate($event.target as HTMLInputElement)"
          @blur="normalizeLearningRate($event.target as HTMLInputElement)"
        />
      </label>
    </div>

    <button class="optimizer-config-link" type="button" @click="emit('configureOptimizer', editingMode)">
      <span>
        <SlidersHorizontal :size="14" />
        {{ editingMode === 'finetune' ? '微调优化器' : '重训优化器' }}
      </span>
      <b>{{ optimizerLabels[settings.optimizer.kind] }} · B{{ settings.batchSize }}</b>
    </button>

    <div class="dataset-row">
      <span>训练集 <b>{{ ((mnistEnabled ? 4000 : 0) + customTrainingCount).toLocaleString() }}</b></span>
      <span>测试集 <b>{{ ((mnistEnabled ? 1000 : 0) + customTestCount).toLocaleString() }}</b></span>
    </div>

    <p v-if="!mnistEnabled && customTrainingCount === 0" class="dataset-source-warning">
      MNIST 已关闭，需要至少一个自定义训练样本
    </p>

    <div v-if="progress.phase !== 'idle'" class="training-progress" aria-live="polite">
      <div class="progress-heading">
        <span>
          {{ progressLabel() }}
        </span>
        <b v-if="progress.phase === 'training' || progress.phase === 'paused' || progress.phase === 'complete'">
          {{ (progress.accuracy * 100).toFixed(1) }}%
        </b>
      </div>
      <div class="progress-track">
        <span :style="{ width: `${progress.epochs ? (progress.epoch / progress.epochs) * 100 : 4}%` }" />
      </div>
      <div v-if="progress.phase === 'training' || progress.phase === 'paused' || progress.phase === 'complete'" class="progress-metrics">
        <span>损失 {{ progress.loss.toFixed(3) }}</span>
        <span>{{ (progress.elapsedMs / 1000).toFixed(1) }} s</span>
      </div>
      <p v-if="progress.message">{{ progress.message }}</p>
    </div>

    <p v-if="saveState !== 'idle'" class="model-save-status" :class="saveState" aria-live="polite">
      {{ saveState === 'saving' ? '正在写入本地模型库' : saveState === 'saved' ? (progress.phase === 'complete' ? '已自动存入模型库' : '当前暂停快照已保存') : '模型保存失败' }}
    </p>

    <div
      v-if="progress.phase !== 'loading' && progress.phase !== 'training' && progress.phase !== 'paused'"
      class="training-start-actions"
    >
      <button
        class="train-button"
        :class="{ secondary: canFineTune }"
        type="button"
        data-testid="train-model"
        :disabled="!mnistEnabled && customTrainingCount === 0"
        @click="start('scratch')"
      >
        <RefreshCcw v-if="canFineTune" :size="15" />
        <Play v-else :size="16" />
        {{ canFineTune ? `从头重新训练 · ${profiles.scratch.epochs} 轮` : '开始重新训练' }}
      </button>
      <button
        v-if="canFineTune"
        class="fine-tune-button"
        type="button"
        data-testid="fine-tune-model"
        :disabled="!mnistEnabled && customTrainingCount === 0"
        @click="start('finetune')"
      >
        <TrendingUp :size="16" />
        <span>追加微调 <b>+{{ profiles.finetune.epochs.toLocaleString() }} 轮</b></span>
      </button>
    </div>
    <div v-else class="training-actions">
      <button
        class="pause-training-button"
        type="button"
        :disabled="progress.phase === 'loading'"
        :title="progress.phase === 'paused' ? '从当前样本继续训练' : '保留当前状态并暂停训练'"
        @click="progress.phase === 'paused' ? emit('resume') : emit('pause')"
      >
        <Play v-if="progress.phase === 'paused'" :size="15" />
        <Pause v-else :size="15" />
        {{ progress.phase === 'paused' ? '继续' : '暂停' }}
      </button>
      <button
        v-if="progress.phase === 'paused'"
        class="save-model-button"
        type="button"
        :disabled="saveState === 'saving' || saveState === 'saved'"
        title="将当前暂停状态存入模型库"
        @click="emit('saveSnapshot')"
      >
        <Save :size="14" />
        {{ saveState === 'saving' ? '保存中' : saveState === 'saved' ? '已保存' : '保存' }}
      </button>
      <button class="cancel-training-button" type="button" @click="emit('cancel')">
        <Square :size="14" />
        取消
      </button>
    </div>
  </section>
</template>
