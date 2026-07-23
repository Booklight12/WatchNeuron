<script setup lang="ts">
import { ChevronLeft, ChevronRight, Layers3, X } from "@lucide/vue";
import { computed } from "vue";
import type { PropagationDirection } from "../types";

const props = defineProps<{
  training: boolean;
  direction: PropagationDirection;
  layer: number;
  layerNames: string[];
  values: number[];
  gradients: number[];
}>();

const emit = defineEmits<{
  close: [];
  direction: [direction: PropagationDirection];
  layer: [layer: number];
}>();

const order = computed(() => {
  const layers = props.layerNames.map((_, index) => index);
  return props.direction === "backward" ? layers.reverse() : layers;
});
const position = computed(() => Math.max(0, order.value.indexOf(props.layer)));
const canPrevious = computed(() => position.value > 0);
const canNext = computed(() => position.value < order.value.length - 1);
const rankedValues = computed(() => {
  const rankByGradient = props.training && props.direction === "backward";
  return props.values
    .map((value, index) => ({
      index,
      value,
      gradient: props.gradients[index] ?? 0,
    }))
    .sort((left, right) =>
      Math.abs(rankByGradient ? right.gradient : right.value) -
      Math.abs(rankByGradient ? left.gradient : left.value),
    )
    .slice(0, 10);
});

function formatNumber(value: number) {
  if (value === 0) return "0.000000";
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 1000) return value.toExponential(4);
  return value.toFixed(6);
}

function move(amount: -1 | 1) {
  const next = order.value[position.value + amount];
  if (next !== undefined) emit("layer", next);
}

function setDirection(direction: PropagationDirection) {
  if (direction === "backward" && !props.training) return;
  emit("direction", direction);
  emit("layer", direction === "backward" ? props.layerNames.length - 1 : 0);
}
</script>

<template>
  <section class="propagation-stepper" :class="direction" aria-labelledby="stepper-heading">
    <header class="stepper-header">
      <div>
        <span class="stepper-kicker"><Layers3 :size="14" />逐层步进</span>
        <h3 id="stepper-heading">{{ layerNames[layer] }}</h3>
      </div>
      <button class="stepper-close" type="button" title="退出步进模式" aria-label="退出步进模式" @click="emit('close')">
        <X :size="15" />
      </button>
    </header>

    <div class="stepper-toolbar">
      <div class="stepper-direction" role="group" aria-label="传播方向">
        <button
          type="button"
          :class="{ active: direction === 'forward' }"
          :aria-pressed="direction === 'forward'"
          @click="setDirection('forward')"
        >
          前向传播
        </button>
        <button
          type="button"
          :class="{ active: direction === 'backward' }"
          :aria-pressed="direction === 'backward'"
          :disabled="!training"
          @click="setDirection('backward')"
        >
          反向传播
        </button>
      </div>
      <div class="stepper-navigation">
        <button type="button" title="上一步" aria-label="上一步" :disabled="!canPrevious" @click="move(-1)">
          <ChevronLeft :size="16" />
        </button>
        <span>{{ position + 1 }} / {{ layerNames.length }}</span>
        <button type="button" title="下一步" aria-label="下一步" :disabled="!canNext" data-testid="propagation-next" @click="move(1)">
          <ChevronRight :size="16" />
        </button>
      </div>
    </div>

    <div class="stepper-state" aria-live="polite">
      <span>{{ direction === "forward" ? "当前层激活已展开" : "当前层损失梯度已展开" }}</span>
      <b>{{ values.length.toLocaleString() }} 个神经元</b>
    </div>

    <div class="stepper-values" role="region" aria-label="当前层具体数值">
      <table>
        <thead>
          <tr>
            <th>神经元</th>
            <th>激活值</th>
            <th v-if="training">反向梯度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="entry in rankedValues" :key="entry.index">
            <th>#{{ entry.index }}</th>
            <td>{{ formatNumber(entry.value) }}</td>
            <td v-if="training" :class="{ emphasized: direction === 'backward' }">
              {{ formatNumber(entry.gradient) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p>显示当前方向绝对值最高的 {{ Math.min(10, values.length) }} 个神经元。</p>
  </section>
</template>
