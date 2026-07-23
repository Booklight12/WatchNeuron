<script setup lang="ts">
import { BrainCircuit, Pause, Play, RefreshCcw, Save, SlidersHorizontal, Square, TrendingUp } from "@lucide/vue";
import type {
  HiddenLayer,
  TrainingMode,
  TrainingProgress,
  TrainingSettings,
} from "../types";

const props = defineProps<{
  layers: HiddenLayer[];
  settings: TrainingSettings;
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
  update: [settings: TrainingSettings];
  train: [];
  fineTune: [];
  configureOptimizer: [];
  pause: [];
  resume: [];
  saveSnapshot: [];
  cancel: [];
}>();

function setSetting(key: "epochs" | "learningRate", value: number) {
  emit("update", { ...props.settings, [key]: value });
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
  if (!Number.isFinite(parsed) || parsed <= 0) input.value = String(props.settings.learningRate);
}

const optimizerLabels = {
  sgd: "SGD",
  momentum: "Momentum",
  adam: "Adam",
  rmsprop: "RMSProp",
  adagrad: "AdaGrad",
};

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
        Zig/Wasm Worker
      </span>
    </div>

    <div class="training-architecture" aria-label="训练网络结构">
      <span v-for="(size, index) in [784, ...layers.map((layer) => layer.units), 10]" :key="`${index}-${size}`">
        <b>{{ size }}</b>
        <i v-if="index < layers.length + 1">→</i>
      </span>
    </div>

    <div v-if="canFineTune" class="fine-tune-source">
      <span><BrainCircuit :size="14" />当前模型</span>
      <b>累计 {{ trainedEpochs.toLocaleString() }} 轮</b>
    </div>

    <div class="training-settings">
      <label class="epoch-setting">
        <span>{{ canFineTune ? '本次追加轮数' : '训练轮数' }}</span>
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

    <button class="optimizer-config-link" type="button" @click="emit('configureOptimizer')">
      <span>
        <SlidersHorizontal :size="14" />
        优化器
      </span>
      <b>{{ optimizerLabels[settings.optimizer.kind] }}</b>
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
        @click="emit('train')"
      >
        <RefreshCcw v-if="canFineTune" :size="15" />
        <Play v-else :size="16" />
        {{ canFineTune ? '从头重新训练' : '开始重新训练' }}
      </button>
      <button
        v-if="canFineTune"
        class="fine-tune-button"
        type="button"
        data-testid="fine-tune-model"
        :disabled="!mnistEnabled && customTrainingCount === 0"
        @click="emit('fineTune')"
      >
        <TrendingUp :size="16" />
        <span>追加微调 <b>+{{ settings.epochs.toLocaleString() }} 轮</b></span>
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
