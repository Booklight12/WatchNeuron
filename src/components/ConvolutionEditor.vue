<script setup lang="ts">
import { RotateCcw, Trash2 } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import AppSelect, { type AppSelectOption, type AppSelectValue } from "./AppSelect.vue";
import type { ActivationKind, ConvolutionConfig } from "../types";
import {
  convolutionKernelPresets,
  convolutionOutputShape,
  normalizeConvolutionConfig,
  type FeatureMapShape,
} from "../lib/convolution";
import { activationLabels } from "../lib/model";

const props = defineProps<{
  config: ConvolutionConfig;
  input: FeatureMapShape;
  displayIndex: number;
  removable: boolean;
}>();
const emit = defineEmits<{
  update: [config: ConvolutionConfig];
  remove: [];
}>();
const selectedFilter = ref(0);
const selectedPreset = ref(convolutionKernelPresets[0].id);
const activationOptions = Object.entries(activationLabels) as [ActivationKind, string][];
const activationSelectOptions: AppSelectOption[] = activationOptions.map(([value, label]) => ({
  value,
  label,
}));
const kernelSizeOptions: AppSelectOption[] = [1, 3, 5, 7].map((size) => ({
  value: size,
  label: `${size} × ${size}`,
}));
const filterOptions = computed<AppSelectOption[]>(() =>
  Array.from({ length: props.config.filters }, (_, index) => ({
    value: index,
    label: `卷积核 ${index + 1}`,
  })),
);
const presetOptions: AppSelectOption[] = convolutionKernelPresets.map((preset) => ({
  value: preset.id,
  label: preset.name,
  description: preset.description,
}));
const output = computed(() => convolutionOutputShape(props.config, props.input));
const currentKernel = computed(() => props.config.kernels[selectedFilter.value] ?? []);

watch(
  () => props.config.filters,
  (filters) => {
    selectedFilter.value = Math.min(selectedFilter.value, filters - 1);
  },
);

function updateConfig(patch: Partial<ConvolutionConfig>) {
  emit("update", normalizeConvolutionConfig({ ...props.config, ...patch }));
}

function setInteger(key: "filters" | "stride" | "padding", input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    input.value = String(props.config[key]);
    return;
  }
  updateConfig({ [key]: Math.floor(value) });
}

function setKernelSize(value: AppSelectValue) {
  updateConfig({ kernelSize: Number(value) });
}

function setKernelValue(index: number, input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    input.value = String(currentKernel.value[index] ?? 0);
    return;
  }
  const kernels = props.config.kernels.map((kernel) => [...kernel]);
  kernels[selectedFilter.value][index] = value;
  updateConfig({ kernels });
}

function applyPreset() {
  const preset = convolutionKernelPresets.find((item) => item.id === selectedPreset.value);
  if (!preset) return;
  const next = normalizeConvolutionConfig({ ...props.config, kernelSize: 3 });
  next.kernels[selectedFilter.value] = [...preset.values];
  emit("update", next);
}

function clearKernel() {
  const kernels = props.config.kernels.map((kernel) => [...kernel]);
  kernels[selectedFilter.value] = new Array(props.config.kernelSize ** 2).fill(0);
  kernels[selectedFilter.value][Math.floor(kernels[selectedFilter.value].length / 2)] = 1;
  updateConfig({ kernels });
}
</script>

<template>
  <div class="convolution-editor enabled">
    <div class="convolution-heading">
      <span class="layer-index">{{ String(displayIndex).padStart(2, '0') }}</span>
      <slot name="drag-handle" />
      <div class="layer-copy">
        <b>二维卷积层</b>
        <small>Conv2D · {{ input.width }} × {{ input.height }} × {{ input.channels }}</small>
      </div>
      <button class="delete-layer" type="button" title="删除卷积层" aria-label="删除卷积层" :disabled="!removable" @click="emit('remove')">
        <Trash2 :size="15" />
      </button>
    </div>

    <div class="convolution-training-control">
      <div>
        <span>参与训练</span>
        <small>{{ config.trainable ? "反向传播会修正卷积核与偏置" : "参数已冻结，梯度仍会继续向前传播" }}</small>
      </div>
      <label class="switch-control" :title="config.trainable ? '冻结卷积参数' : '启用卷积参数训练'">
        <input
          type="checkbox"
          role="switch"
          :checked="config.trainable"
          :aria-label="`卷积层 ${displayIndex} 参与训练`"
          :data-testid="`convolution-trainable-${config.id}`"
          @change="updateConfig({ trainable: ($event.target as HTMLInputElement).checked })"
        />
        <span aria-hidden="true" />
      </label>
    </div>

    <div class="convolution-settings">
        <label>
          <span>卷积核</span>
          <input type="number" min="1" max="32" :value="config.filters" @change="setInteger('filters', $event.target as HTMLInputElement)" />
        </label>
        <label>
          <span>核尺寸</span>
          <AppSelect
            :model-value="config.kernelSize"
            :options="kernelSizeOptions"
            label="卷积核尺寸"
            mono
            @update:model-value="setKernelSize"
          />
        </label>
        <label>
          <span>步幅</span>
          <input type="number" min="1" max="4" :value="config.stride" @change="setInteger('stride', $event.target as HTMLInputElement)" />
        </label>
        <label>
          <span>填充</span>
          <input type="number" min="0" max="6" :value="config.padding" @change="setInteger('padding', $event.target as HTMLInputElement)" />
        </label>
        <label class="convolution-activation">
          <span>激活</span>
          <AppSelect
            :model-value="config.activation"
            :options="activationSelectOptions"
            label="卷积层激活函数"
            mono
            @update:model-value="updateConfig({ activation: $event as ActivationKind })"
          />
        </label>
    </div>

    <div class="convolution-output">
      <span>输出特征图</span>
      <b>{{ output.width }} × {{ output.height }} × {{ config.filters }}</b>
      <small>{{ output.length.toLocaleString('zh-CN') }} 个激活</small>
    </div>

    <div class="kernel-workbench">
      <div class="kernel-toolbar">
        <label>
          <span>编辑卷积核</span>
          <AppSelect
            :model-value="selectedFilter"
            :options="filterOptions"
            label="选择要编辑的卷积核"
            mono
            @update:model-value="selectedFilter = Number($event)"
          />
        </label>
        <button type="button" class="icon-button" title="重置当前卷积核" aria-label="重置当前卷积核" @click="clearKernel">
          <RotateCcw :size="14" />
        </button>
      </div>

      <div class="kernel-matrix" :style="{ gridTemplateColumns: `repeat(${config.kernelSize}, minmax(0, 1fr))` }">
        <input
          v-for="(value, index) in currentKernel"
          :key="index"
          type="number"
          step="any"
          :value="Number(value.toFixed(4))"
          :aria-label="`卷积核 ${selectedFilter + 1} 系数 ${index + 1}`"
          @input="setKernelValue(index, $event.target as HTMLInputElement)"
        />
      </div>

      <div class="kernel-library">
        <div class="kernel-library-title">
          <span>卷积核库</span>
          <small>{{ convolutionKernelPresets.find((item) => item.id === selectedPreset)?.description }}</small>
        </div>
        <div class="kernel-library-controls">
          <AppSelect
            :model-value="selectedPreset"
            :options="presetOptions"
            label="卷积核预设"
            placement="top"
            @update:model-value="selectedPreset = String($event)"
          />
          <button type="button" @click="applyPreset">应用到当前核</button>
        </div>
      </div>
    </div>
  </div>
</template>
