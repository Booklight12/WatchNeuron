<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import type { CustomDatasetSample } from "../types";

const props = defineProps<{
  sample: CustomDatasetSample;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);

function render() {
  const target = canvas.value;
  if (!target) return;
  const context = target.getContext("2d");
  if (!context) return;

  const pixels = new Float32Array(784);
  props.sample.indices.forEach((index, position) => {
    pixels[index] = props.sample.values[position] ?? 0;
  });

  const source = document.createElement("canvas");
  source.width = 28;
  source.height = 28;
  const sourceContext = source.getContext("2d")!;
  const image = sourceContext.createImageData(28, 28);
  pixels.forEach((value, index) => {
    const intensity = Math.round(23 + Math.max(0, Math.min(1, value)) * 220);
    const offset = index * 4;
    image.data[offset] = intensity;
    image.data[offset + 1] = intensity;
    image.data[offset + 2] = Math.max(23, intensity - 4);
    image.data[offset + 3] = 255;
  });
  sourceContext.putImageData(image, 0, 0);

  context.fillStyle = "#171716";
  context.fillRect(0, 0, target.width, target.height);
  context.imageSmoothingEnabled = false;
  context.drawImage(source, 0, 0, target.width, target.height);
}

onMounted(render);
watch(() => props.sample.id, render);
</script>

<template>
  <canvas
    ref="canvas"
    class="sample-thumbnail"
    width="112"
    height="112"
    :aria-label="`手写数字样本，标签 ${sample.label}`"
  />
</template>
