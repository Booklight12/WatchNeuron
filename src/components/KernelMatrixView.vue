<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  weights: number[] | Float32Array;
  inputChannels: number;
  kernelSize: number;
  filter: number;
  trainable: boolean;
}>();

const kernelLength = computed(() => props.kernelSize ** 2);
const channelMatrices = computed(() => {
  const filterOffset = props.filter * props.inputChannels * kernelLength.value;
  return Array.from({ length: props.inputChannels }, (_, channel) => {
    const start = filterOffset + channel * kernelLength.value;
    return {
      channel,
      values: Array.from(
        { length: kernelLength.value },
        (_, index) => Number(props.weights[start + index] ?? 0),
      ),
    };
  });
});
const maximumAbsolute = computed(() => Math.max(
  1e-12,
  ...channelMatrices.value.flatMap(({ values }) => values.map(Math.abs)),
));

function formatWeight(value: number) {
  const absolute = Math.abs(value);
  if (absolute === 0) return "0";
  if (absolute >= 100 || absolute < 0.001) return value.toExponential(1);
  if (absolute >= 10) return value.toFixed(1);
  if (absolute >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function weightStyle(value: number) {
  const strength = Math.min(1, Math.abs(value) / maximumAbsolute.value);
  if (value > 0) {
    return { backgroundColor: `rgba(115, 201, 191, ${0.06 + strength * 0.3})` };
  }
  if (value < 0) {
    return { backgroundColor: `rgba(217, 119, 87, ${0.06 + strength * 0.3})` };
  }
  return { backgroundColor: "rgba(248, 248, 246, 0.025)" };
}
</script>

<template>
  <section
    class="kernel-parameter-view"
    data-testid="kernel-parameter-view"
    :data-filter="filter"
    :aria-label="`卷积核 ${filter + 1} 权重矩阵`"
  >
    <header>
      <div>
        <span>共享卷积核矩阵</span>
        <small>该特征图的每个空间神经元共用此组参数</small>
      </div>
      <b :class="{ frozen: !trainable }">F{{ filter + 1 }} · {{ trainable ? "可训练" : "已冻结" }}</b>
    </header>

    <div class="kernel-channel-list">
      <div v-for="matrix in channelMatrices" :key="matrix.channel" class="kernel-channel-matrix">
        <span v-if="inputChannels > 1">输入通道 {{ matrix.channel + 1 }}</span>
        <div
          class="kernel-weight-grid"
          role="grid"
          :aria-label="`输入通道 ${matrix.channel + 1}，${kernelSize} 乘 ${kernelSize}`"
          :style="{
            gridTemplateColumns: `repeat(${kernelSize}, minmax(0, 1fr))`,
            width: `min(100%, ${kernelSize * 36}px)`,
          }"
        >
          <code
            v-for="(weight, index) in matrix.values"
            :key="index"
            role="gridcell"
            :class="{ positive: weight > 0, negative: weight < 0 }"
            :style="weightStyle(weight)"
            :title="`行 ${Math.floor(index / kernelSize) + 1}，列 ${(index % kernelSize) + 1}：${weight.toPrecision(8)}`"
          >{{ formatWeight(weight) }}</code>
        </div>
      </div>
    </div>
  </section>
</template>
