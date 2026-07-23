<script setup lang="ts">
import { onMounted, ref, watch } from "vue";

const props = defineProps<{ pixels: Float32Array }>();
const canvas = ref<HTMLCanvasElement | null>(null);

function render() {
  const ctx = canvas.value?.getContext("2d");
  if (!ctx) return;
  const image = ctx.createImageData(28, 28);
  props.pixels.forEach((value, index) => {
    const offset = index * 4;
    image.data[offset] = 73;
    image.data[offset + 1] = 212;
    image.data[offset + 2] = 200;
    image.data[offset + 3] = Math.round(value * 255);
  });
  ctx.clearRect(0, 0, 28, 28);
  ctx.putImageData(image, 0, 0);
}

watch(() => props.pixels, render);
onMounted(render);
</script>

<template>
  <canvas ref="canvas" class="pixel-preview" width="28" height="28" aria-label="28 乘 28 标准化输入" />
</template>
