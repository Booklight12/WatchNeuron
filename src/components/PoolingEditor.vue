<script setup lang="ts">
import { Trash2 } from "@lucide/vue";
import { computed } from "vue";
import type { PoolingConfig } from "../types";
import {
  normalizePoolingConfig,
  poolingOutputShape,
  type FeatureMapShape,
} from "../lib/convolution";
import AppSelect, { type AppSelectOption, type AppSelectValue } from "./AppSelect.vue";

const props = defineProps<{
  config: PoolingConfig;
  input: FeatureMapShape;
  displayIndex: number;
}>();

const emit = defineEmits<{
  update: [config: PoolingConfig];
  remove: [];
}>();

const kindOptions: AppSelectOption[] = [
  { value: "max", label: "MaxPool2D", description: "保留窗口内最强响应" },
  { value: "average", label: "AvgPool2D", description: "取窗口内有效值平均" },
  { value: "globalAverage", label: "Global Average Pooling", description: "每个通道压缩为一个数值" },
];
const kernelOptions: AppSelectOption[] = [2, 3, 4].map((value) => ({ value, label: `${value} × ${value}` }));
const output = computed(() => poolingOutputShape(props.config, props.input));

function updateConfig(patch: Partial<PoolingConfig>) {
  emit("update", normalizePoolingConfig({ ...props.config, ...patch }));
}

function setKind(value: AppSelectValue) {
  updateConfig({ kind: value as PoolingConfig["kind"] });
}

function setKernelSize(value: AppSelectValue) {
  updateConfig({ kernelSize: Number(value) });
}

function setInteger(key: "stride" | "padding", input: HTMLInputElement) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    input.value = String(props.config[key]);
    return;
  }
  updateConfig({ [key]: Math.floor(value) });
}
</script>

<template>
  <div class="convolution-editor pooling-editor enabled">
    <div class="convolution-heading">
      <span class="layer-index">{{ String(displayIndex).padStart(2, '0') }}</span>
      <slot name="drag-handle" />
      <div class="layer-copy">
        <b>{{ config.kind === 'globalAverage' ? '全局平均池化' : config.kind === 'max' ? '最大池化层' : '平均池化层' }}</b>
        <small>{{ config.kind === 'globalAverage' ? 'GAP' : config.kind === 'max' ? 'MaxPool2D' : 'AvgPool2D' }} · {{ input.width }} × {{ input.height }} × {{ input.channels }}</small>
      </div>
      <button class="delete-layer" type="button" title="删除池化层" aria-label="删除池化层" @click="emit('remove')">
        <Trash2 :size="15" />
      </button>
    </div>

    <div class="convolution-settings pooling-settings">
      <label>
        <span>池化方式</span>
        <AppSelect :model-value="config.kind" :options="kindOptions" label="池化方式" @update:model-value="setKind" />
      </label>
      <label v-if="config.kind !== 'globalAverage'">
        <span>窗口</span>
        <AppSelect :model-value="config.kernelSize" :options="kernelOptions" label="池化窗口" mono @update:model-value="setKernelSize" />
      </label>
      <label v-if="config.kind !== 'globalAverage'">
        <span>步幅</span>
        <input type="number" min="1" step="1" :value="config.stride" @change="setInteger('stride', $event.target as HTMLInputElement)" />
      </label>
      <label v-if="config.kind !== 'globalAverage'">
        <span>填充</span>
        <input type="number" min="0" step="1" :value="config.padding" @change="setInteger('padding', $event.target as HTMLInputElement)" />
      </label>
    </div>

    <div class="convolution-output">
      <span>输出</span>
      <b>{{ output.width }} × {{ output.height }} × {{ output.channels }}</b>
      <small>{{ output.length.toLocaleString('zh-CN') }} 个激活 · 无可训练参数</small>
    </div>
  </div>
</template>
