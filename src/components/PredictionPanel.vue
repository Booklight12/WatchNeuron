<script setup lang="ts">
import { computed } from "vue";
import { Activity, BadgeCheck, FlaskConical } from "@lucide/vue";
import { BrainCircuit } from "@lucide/vue";
import type { ModelStatus } from "../types";

const props = defineProps<{
  probabilities: number[];
  status: ModelStatus;
  hasInput: boolean;
  training?: boolean;
}>();

const prediction = computed(() => {
  if (!props.hasInput) return -1;
  const maximum = Math.max(...props.probabilities);
  return props.probabilities.indexOf(maximum);
});
const confidence = computed(() => Math.max(...props.probabilities) || 0);
</script>

<template>
  <section class="result-section" aria-labelledby="result-heading">
    <div class="section-heading compact-heading">
      <div>
        <span class="eyebrow">OUTPUT</span>
        <h2 id="result-heading">识别结果</h2>
      </div>
      <span class="model-state" :class="{ experimental: status === 'experimental', trained: status === 'trained', training }">
        <Activity v-if="training" :size="14" />
        <BadgeCheck v-else-if="status === 'calibrated'" :size="14" />
        <BrainCircuit v-else-if="status === 'trained'" :size="14" />
        <FlaskConical v-else :size="14" />
        {{ training ? "训练采样" : status === 'calibrated' ? "已校准" : status === 'trained' ? "已训练" : "待训练" }}
      </span>
    </div>

    <div class="prediction-readout" aria-live="polite">
      <strong>{{ prediction < 0 ? '—' : prediction }}</strong>
      <div>
        <span>置信度</span>
        <b>{{ (confidence * 100).toFixed(1) }}%</b>
      </div>
    </div>

    <div class="probability-list" aria-label="数字概率">
      <div
        v-for="(probability, digit) in probabilities"
        :key="digit"
        class="probability-row"
        :class="{ winner: hasInput && digit === prediction }"
      >
        <span class="digit-label">{{ digit }}</span>
        <div class="probability-track">
          <span :style="{ width: `${Math.max(1, probability * 100)}%` }" />
        </div>
        <span class="probability-value">{{ (probability * 100).toFixed(1) }}</span>
      </div>
    </div>
  </section>
</template>
