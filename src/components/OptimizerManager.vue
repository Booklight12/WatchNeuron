<script setup lang="ts">
import { ArrowLeft, Check, Gauge, Lightbulb, RotateCcw, SlidersHorizontal } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import AppSelect, { type AppSelectOption } from "./AppSelect.vue";
import SegmentedControl, { type SegmentedControlOption } from "./SegmentedControl.vue";
import type {
  OptimizerConfig,
  OptimizerKind,
  TrainingMode,
  TrainingProfiles,
  TrainingSettings,
} from "../types";

const props = defineProps<{
  profiles: TrainingProfiles;
  initialMode: TrainingMode;
  locked: boolean;
}>();

const emit = defineEmits<{
  back: [];
  mode: [mode: TrainingMode];
  update: [mode: TrainingMode, settings: TrainingSettings];
}>();

const definitions: Array<{
  kind: OptimizerKind;
  name: string;
  label: string;
  formula: string;
  summary: string;
  trait: string;
  recommendation: {
    epochs: number;
    learningRate: number;
    optimizer: Partial<Omit<OptimizerConfig, "kind">>;
    parameters: Array<{ label: string; value: string }>;
    note: string;
  };
}> = [
  {
    kind: "sgd",
    name: "随机梯度下降",
    label: "SGD",
    formula: "w = w - lr * g",
    summary: "直接沿当前样本梯度的反方向更新参数，状态最少，变化直观。",
    trait: "低内存 · 响应直接",
    recommendation: {
      epochs: 10,
      learningRate: 0.018,
      optimizer: {},
      parameters: [
        { label: "轮数", value: "10" },
        { label: "学习率", value: "0.018" },
      ],
      note: "适合从头训练默认 ReLU 网络，更新轨迹最容易观察。",
    },
  },
  {
    kind: "momentum",
    name: "动量梯度下降",
    label: "Momentum",
    formula: "v = mu * v + g; w = w - lr * v",
    summary: "累积历史梯度形成速度，在方向稳定时加速，并减弱来回震荡。",
    trait: "平滑轨迹 · 适合稳定方向",
    recommendation: {
      epochs: 12,
      learningRate: 0.003,
      optimizer: { momentum: 0.9 },
      parameters: [
        { label: "轮数", value: "12" },
        { label: "学习率", value: "0.003" },
        { label: "动量系数", value: "0.9" },
      ],
      note: "已针对默认 32 神经元 ReLU 网络校验，可避免动量放大导致的震荡。",
    },
  },
  {
    kind: "adam",
    name: "自适应矩估计",
    label: "Adam",
    formula: "m, v = EMA(g), EMA(g^2); w = w - lr * m / sqrt(v)",
    summary: "同时估计梯度均值和平方均值，为每个参数生成自适应更新尺度。",
    trait: "自适应步长 · 收敛较快",
    recommendation: {
      epochs: 8,
      learningRate: 0.001,
      optimizer: { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
      parameters: [
        { label: "轮数", value: "8" },
        { label: "学习率", value: "0.001" },
        { label: "beta1", value: "0.9" },
        { label: "beta2", value: "0.999" },
        { label: "epsilon", value: "1e-8" },
      ],
      note: "通用稳定起点，适合层数较多或不同层梯度尺度差异较大的网络。",
    },
  },
  {
    kind: "rmsprop",
    name: "均方根传播",
    label: "RMSProp",
    formula: "s = rho * s + (1 - rho) * g^2; w = w - lr * g / sqrt(s)",
    summary: "跟踪近期平方梯度，并按参数各自的梯度尺度调节更新幅度。",
    trait: "近期自适应 · 抑制震荡",
    recommendation: {
      epochs: 10,
      learningRate: 0.001,
      optimizer: { decay: 0.9, epsilon: 1e-8 },
      parameters: [
        { label: "轮数", value: "10" },
        { label: "学习率", value: "0.001" },
        { label: "衰减率", value: "0.9" },
        { label: "epsilon", value: "1e-8" },
      ],
      note: "适合梯度波动明显的架构，衰减率 0.9 会更关注近期训练状态。",
    },
  },
  {
    kind: "adagrad",
    name: "自适应梯度",
    label: "AdaGrad",
    formula: "s = s + g^2; w = w - lr * g / sqrt(s)",
    summary: "累计每个参数的历史平方梯度，让频繁更新的参数逐渐减小步长。",
    trait: "累计缩放 · 适合稀疏输入",
    recommendation: {
      epochs: 10,
      learningRate: 0.01,
      optimizer: { epsilon: 1e-8 },
      parameters: [
        { label: "轮数", value: "10" },
        { label: "学习率", value: "0.01" },
        { label: "epsilon", value: "1e-8" },
      ],
      note: "适合当前稀疏像素输入；累计量只增不减，长时间微调时可降低学习率。",
    },
  },
];

type OptimizerRecommendation = (typeof definitions)[number]["recommendation"];
const optimizerSelectOptions: AppSelectOption[] = definitions.map((definition) => ({
  value: definition.kind,
  label: `${definition.label} · ${definition.name}`,
}));

const fineTuneRecommendations: Record<OptimizerKind, OptimizerRecommendation> = {
  sgd: {
    epochs: 5,
    learningRate: 0.002,
    optimizer: {},
    parameters: [
      { label: "轮数", value: "5" },
      { label: "学习率", value: "0.002" },
    ],
    note: "微调使用较小固定步长，降低已训练权重被快速改写的风险。",
  },
  momentum: {
    epochs: 5,
    learningRate: 0.0005,
    optimizer: { momentum: 0.9 },
    parameters: [
      { label: "轮数", value: "5" },
      { label: "学习率", value: "0.0005" },
      { label: "动量系数", value: "0.9" },
    ],
    note: "保留 0.9 动量并降低学习率，使已有方向信息平缓适应新增样本。",
  },
  adam: {
    epochs: 5,
    learningRate: 0.0002,
    optimizer: { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
    parameters: [
      { label: "轮数", value: "5" },
      { label: "学习率", value: "0.0002" },
      { label: "beta1", value: "0.9" },
      { label: "beta2", value: "0.999" },
      { label: "epsilon", value: "1e-8" },
    ],
    note: "默认微调组合，以较低自适应步长细化当前模型而不过度偏离已有权重。",
  },
  rmsprop: {
    epochs: 5,
    learningRate: 0.00025,
    optimizer: { decay: 0.95, epsilon: 1e-8 },
    parameters: [
      { label: "轮数", value: "5" },
      { label: "学习率", value: "0.00025" },
      { label: "衰减率", value: "0.95" },
      { label: "epsilon", value: "1e-8" },
    ],
    note: "更慢的平方梯度衰减配合低学习率，适合稳定已有特征。",
  },
  adagrad: {
    epochs: 4,
    learningRate: 0.001,
    optimizer: { epsilon: 1e-8 },
    parameters: [
      { label: "轮数", value: "4" },
      { label: "学习率", value: "0.001" },
      { label: "epsilon", value: "1e-8" },
    ],
    note: "使用较短轮次和较低初始步长，避免累计平方梯度令后期更新过早衰减。",
  },
};

const selectedMode = ref<TrainingMode>(props.initialMode);
const trainingModeOptions: SegmentedControlOption[] = [
  { value: "scratch", label: "重头训练", testId: "optimizer-profile-scratch" },
  { value: "finetune", label: "微调训练", testId: "optimizer-profile-finetune" },
];
const settings = computed<TrainingSettings>(() => ({
  ...props.profiles[selectedMode.value],
  mathMode: props.profiles.mathMode,
  optimizer: { ...props.profiles[selectedMode.value].optimizer },
}));

const selected = computed(
  () => definitions.find((item) => item.kind === settings.value.optimizer.kind) ?? definitions[0],
);

const recommendation = computed(() => selectedMode.value === "scratch"
  ? selected.value.recommendation
  : fineTuneRecommendations[selected.value.kind]);

const recommendationApplied = computed(() => {
  const values = recommendation.value;
  if (settings.value.epochs !== values.epochs || settings.value.learningRate !== values.learningRate) {
    return false;
  }
  if (settings.value.batchSize !== 16 || settings.value.optimizer.weightDecay !== (selectedMode.value === "scratch" ? 1e-4 : 1e-5)) return false;
  return Object.entries(values.optimizer).every(
    ([key, value]) => settings.value.optimizer[key as keyof OptimizerConfig] === value,
  );
});

function updateSettings(patch: Partial<TrainingSettings>) {
  emit("update", selectedMode.value, { ...settings.value, ...patch });
}

function selectMode(mode: TrainingMode) {
  selectedMode.value = mode;
  emit("mode", mode);
}

function selectOptimizer(kind: OptimizerKind) {
  if (props.locked) return;
  updateSettings({ optimizer: { ...settings.value.optimizer, kind } });
}

function applyRecommendation() {
  if (props.locked) return;
  const values = recommendation.value;
  emit("update", selectedMode.value, {
    ...settings.value,
    epochs: values.epochs,
    learningRate: values.learningRate,
    batchSize: 16,
    optimizer: {
      ...settings.value.optimizer,
      ...values.optimizer,
      kind: selected.value.kind,
      weightDecay: selectedMode.value === "scratch" ? 1e-4 : 1e-5,
    },
  });
}

function applyDefaultProfiles() {
  if (props.locked) return;
  const baseOptimizer: OptimizerConfig = {
    kind: "sgd",
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    decay: 0.9,
    epsilon: 1e-8,
    weightDecay: 0.0001,
  };
  emit("update", "scratch", {
    epochs: 10,
    learningRate: 0.018,
    batchSize: 16,
    mathMode: props.profiles.mathMode,
    optimizer: baseOptimizer,
  });
  emit("update", "finetune", {
    epochs: 5,
    learningRate: 0.0002,
    batchSize: 16,
    mathMode: props.profiles.mathMode,
    optimizer: { ...baseOptimizer, kind: "adam" },
  });
}

function setEpochs(input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) return;
  updateSettings({ epochs: Math.max(1, Math.floor(value)) });
}

function setLearningRate(input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value <= 0) return;
  updateSettings({ learningRate: value });
}

function normalizeLearningRate(input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value <= 0) input.value = String(settings.value.learningRate);
}

function setBatchSize(input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) return;
  updateSettings({ batchSize: Math.max(1, Math.floor(value)) });
}

function setOptimizerParameter(
  key: "momentum" | "beta1" | "beta2" | "decay" | "epsilon" | "weightDecay",
  input: HTMLInputElement,
) {
  const value = Number(input.value);
  const valid = key === "weightDecay"
    ? Number.isFinite(value) && value >= 0
    : key === "epsilon"
    ? Number.isFinite(value) && value > 0
    : key === "momentum"
      ? Number.isFinite(value) && value >= 0 && value < 1
      : Number.isFinite(value) && value > 0 && value < 1;
  if (!valid) {
    input.value = String(settings.value.optimizer[key]);
    return;
  }
  updateSettings({
    optimizer: { ...settings.value.optimizer, [key]: value },
  });
}

watch(
  () => props.initialMode,
  (mode) => {
    selectedMode.value = mode;
  },
);
</script>

<template>
  <main class="sample-manager optimizer-manager" aria-labelledby="optimizer-manager-heading">
    <header class="sample-manager-heading">
      <button
        class="tool-button manager-back-button"
        type="button"
        title="返回实验台"
        aria-label="返回实验台"
        @click="emit('back')"
      >
        <ArrowLeft :size="16" />
        <span>返回实验台</span>
      </button>
      <div>
        <span class="eyebrow">OPTIMIZER</span>
        <h1 id="optimizer-manager-heading">优化器与训练参数</h1>
      </div>
      <span class="optimizer-active-state">
        <SlidersHorizontal :size="14" />
        {{ selectedMode === "scratch" ? "重训" : "微调" }} · {{ selected.label }}
      </span>
    </header>

    <section class="optimizer-profile-band" aria-label="训练配置档">
      <SegmentedControl
        class="optimizer-profile-tabs"
        :model-value="selectedMode"
        :options="trainingModeOptions"
        label="训练方式"
        :disabled="locked"
        @update:model-value="selectMode($event as TrainingMode)"
      />
      <button
        class="tool-button optimizer-default-profiles"
        type="button"
        :disabled="locked"
        data-testid="optimizer-default-profiles"
        @click="applyDefaultProfiles"
      >
        <RotateCcw :size="15" />
        恢复两套默认推荐
      </button>
    </section>

    <section class="optimizer-recommendation" aria-label="当前优化器建议参数" aria-live="polite">
      <div class="optimizer-recommendation-copy">
        <Lightbulb :size="18" />
        <div>
          <span class="eyebrow">RECOMMENDED</span>
          <strong>{{ selectedMode === "scratch" ? "重头训练" : "微调训练" }} · {{ selected.label }} 建议参数</strong>
          <p>{{ recommendation.note }}</p>
        </div>
      </div>
      <div class="optimizer-recommendation-values">
        <span v-for="parameter in recommendation.parameters" :key="parameter.label">
          <small>{{ parameter.label }}</small>
          <b>{{ parameter.value }}</b>
        </span>
        <span><small>批大小</small><b>16</b></span>
        <span><small>Weight Decay</small><b>{{ selectedMode === 'scratch' ? '1e-4' : '1e-5' }}</b></span>
      </div>
      <button
        class="tool-button optimizer-recommendation-apply"
        type="button"
        :disabled="locked || recommendationApplied"
        @click="applyRecommendation"
      >
        <Check :size="15" />
        {{ recommendationApplied ? "已应用" : `应用${selectedMode === "scratch" ? "重训" : "微调"}建议` }}
      </button>
    </section>

    <section class="optimizer-list" aria-label="可用优化器">
      <label class="optimizer-select-field">
        <span>选择优化器</span>
        <AppSelect
          :model-value="settings.optimizer.kind"
          :options="optimizerSelectOptions"
          label="选择优化器"
          :disabled="locked"
          mono
          @update:model-value="selectOptimizer($event as OptimizerKind)"
        />
      </label>
      <article
        class="optimizer-card optimizer-card-current active"
        :data-testid="`optimizer-${selected.kind}`"
      >
        <span class="optimizer-card-heading">
          <i>{{ selected.label }}</i>
          <Check :size="16" />
        </span>
        <strong>{{ selected.name }}</strong>
        <p>{{ selected.summary }}</p>
        <code>{{ selected.formula }}</code>
        <small>{{ selected.trait }}</small>
      </article>
    </section>

    <section class="optimizer-config-band" aria-labelledby="optimizer-config-heading">
      <header>
        <div>
          <span class="eyebrow">PARAMETERS</span>
          <h2 id="optimizer-config-heading">{{ selectedMode === "scratch" ? "重头训练" : "微调训练" }} · {{ selected.label }}</h2>
        </div>
        <span v-if="locked" class="optimizer-locked">训练期间已锁定</span>
      </header>

      <div class="optimizer-fields">
        <label class="optimizer-field">
          <span>训练轮数 <small>正整数，不设上限</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              min="1"
              step="1"
              inputmode="numeric"
              :value="settings.epochs"
              :disabled="locked"
              :data-testid="`optimizer-epochs-${selectedMode}`"
              @input="setEpochs($event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <label class="optimizer-field">
          <span>学习率 <small>仅要求大于 0，不设上限</small></span>
          <span class="optimizer-input-shell">
            <Gauge :size="15" />
            <input
              type="number"
              step="any"
              inputmode="decimal"
              :value="settings.learningRate"
              :disabled="locked"
              :data-testid="`optimizer-learning-rate-${selectedMode}`"
              @input="setLearningRate($event.target as HTMLInputElement)"
              @blur="normalizeLearningRate($event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <label class="optimizer-field">
          <span>批大小 <small>NCHW mini-batch，正整数</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              min="1"
              step="1"
              inputmode="numeric"
              :value="settings.batchSize"
              :disabled="locked"
              :data-testid="`optimizer-batch-size-${selectedMode}`"
              @change="setBatchSize($event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <label class="optimizer-field">
          <span>Weight Decay <small>大于等于 0，偏置不衰减</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              min="0"
              step="any"
              inputmode="decimal"
              :value="settings.optimizer.weightDecay"
              :disabled="locked"
              data-testid="optimizer-weight-decay"
              @change="setOptimizerParameter('weightDecay', $event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <label v-if="settings.optimizer.kind === 'momentum'" class="optimizer-field">
          <span>动量系数 mu <small>范围 [0, 1)</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              min="0"
              max="0.999999"
              step="0.01"
              :value="settings.optimizer.momentum"
              :disabled="locked"
              :data-testid="`optimizer-momentum-value-${selectedMode}`"
              @change="setOptimizerParameter('momentum', $event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <template v-if="settings.optimizer.kind === 'adam'">
          <label class="optimizer-field">
            <span>一阶矩衰减 beta1 <small>范围 (0, 1)</small></span>
            <span class="optimizer-input-shell">
              <input
                type="number"
                min="0.000001"
                max="0.999999"
                step="0.01"
                :value="settings.optimizer.beta1"
                :disabled="locked"
                data-testid="optimizer-beta1"
                @change="setOptimizerParameter('beta1', $event.target as HTMLInputElement)"
              />
            </span>
          </label>
          <label class="optimizer-field">
            <span>二阶矩衰减 beta2 <small>范围 (0, 1)</small></span>
            <span class="optimizer-input-shell">
              <input
                type="number"
                min="0.000001"
                max="0.999999"
                step="0.001"
                :value="settings.optimizer.beta2"
                :disabled="locked"
                data-testid="optimizer-beta2"
                @change="setOptimizerParameter('beta2', $event.target as HTMLInputElement)"
              />
            </span>
          </label>
        </template>

        <label v-if="settings.optimizer.kind === 'rmsprop'" class="optimizer-field">
          <span>平方梯度衰减率 rho <small>范围 (0, 1)</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              min="0.000001"
              max="0.999999"
              step="0.01"
              :value="settings.optimizer.decay"
              :disabled="locked"
              data-testid="optimizer-decay"
              @change="setOptimizerParameter('decay', $event.target as HTMLInputElement)"
            />
          </span>
        </label>

        <label v-if="['adam', 'rmsprop', 'adagrad'].includes(settings.optimizer.kind)" class="optimizer-field">
          <span>数值稳定项 epsilon <small>必须大于 0</small></span>
          <span class="optimizer-input-shell">
            <input
              type="number"
              step="any"
              :value="settings.optimizer.epsilon"
              :disabled="locked"
              data-testid="optimizer-epsilon"
              @change="setOptimizerParameter('epsilon', $event.target as HTMLInputElement)"
            />
          </span>
        </label>
      </div>
    </section>
  </main>
</template>
