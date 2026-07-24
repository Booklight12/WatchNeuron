<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";

interface SignalMapShape {
  width: number;
  height: number;
  channels: number;
}

const props = defineProps<{
  values: number[];
  label: string;
  shape?: SignalMapShape;
  selectedIndex?: number | null;
}>();

const emit = defineEmits<{
  select: [selection: { index: number; clientX: number; clientY: number }];
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
const hoverIndex = ref<number | null>(null);

const layout = computed(() => {
  const length = Math.max(1, props.values.length);
  const spatial = props.shape &&
    props.shape.width > 0 &&
    props.shape.height > 0 &&
    props.shape.channels > 0;
  if (spatial && props.shape) {
    const width = Math.max(1, Math.floor(props.shape.width));
    const height = Math.max(1, Math.floor(props.shape.height));
    const channels = Math.max(1, Math.floor(props.shape.channels));
    const gap = channels > 1 ? 1 : 0;
    const tileColumns = Math.max(
      1,
      Math.min(channels, Math.ceil(Math.sqrt((channels * height * 1.45) / width))),
    );
    const tileRows = Math.ceil(channels / tileColumns);
    return {
      kind: "spatial" as const,
      width: tileColumns * width + Math.max(0, tileColumns - 1) * gap,
      height: tileRows * height + Math.max(0, tileRows - 1) * gap,
      mapWidth: width,
      mapHeight: height,
      channels,
      gap,
      tileColumns,
      columns: width,
    };
  }
  const columns = Math.max(1, Math.ceil(Math.sqrt(length * 1.55)));
  return {
    kind: "vector" as const,
    width: columns,
    height: Math.max(1, Math.ceil(length / columns)),
    mapWidth: columns,
    mapHeight: Math.max(1, Math.ceil(length / columns)),
    channels: 1,
    gap: 0,
    tileColumns: 1,
    columns,
  };
});

const statistics = computed(() => {
  if (!props.values.length) return { minimum: 0, maximum: 0, mean: 0 };
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let valid = 0;
  for (const rawValue of props.values) {
    const value = Number.isFinite(rawValue) ? rawValue : 0;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    valid++;
  }
  return {
    minimum: valid ? minimum : 0,
    maximum: valid ? maximum : 0,
    mean: valid ? sum / valid : 0,
  };
});

const canvasStyle = computed(() => {
  const ratio = Math.max(1.15, Math.min(2.8, layout.value.width / layout.value.height));
  return {
    aspectRatio: String(ratio),
    "--signal-map-ratio": String(ratio),
  };
});

function formatValue(value: number) {
  if (value === 0) return "0.0000";
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 1000) return value.toExponential(3);
  return value.toFixed(4);
}

function cellForIndex(index: number) {
  const current = layout.value;
  if (current.kind === "vector") {
    return { x: index % current.columns, y: Math.floor(index / current.columns) };
  }
  const mapSize = current.mapWidth * current.mapHeight;
  const channel = Math.floor(index / mapSize);
  const position = index % mapSize;
  const tileX = channel % current.tileColumns;
  const tileY = Math.floor(channel / current.tileColumns);
  return {
    x: tileX * (current.mapWidth + current.gap) + (position % current.mapWidth),
    y: tileY * (current.mapHeight + current.gap) + Math.floor(position / current.mapWidth),
  };
}

function indexForCell(x: number, y: number) {
  const current = layout.value;
  if (x < 0 || y < 0 || x >= current.width || y >= current.height) return -1;
  if (current.kind === "vector") {
    const index = y * current.columns + x;
    return index < props.values.length ? index : -1;
  }
  const strideX = current.mapWidth + current.gap;
  const strideY = current.mapHeight + current.gap;
  const tileX = Math.floor(x / strideX);
  const tileY = Math.floor(y / strideY);
  const localX = x % strideX;
  const localY = y % strideY;
  if (
    tileX >= current.tileColumns ||
    localX >= current.mapWidth ||
    localY >= current.mapHeight
  ) {
    return -1;
  }
  const channel = tileY * current.tileColumns + tileX;
  if (channel >= current.channels) return -1;
  const index = channel * current.mapWidth * current.mapHeight + localY * current.mapWidth + localX;
  return index < props.values.length ? index : -1;
}

function draw() {
  const element = canvas.value;
  if (!element) return;
  const current = layout.value;
  element.width = current.width;
  element.height = current.height;
  const context = element.getContext("2d");
  if (!context) return;
  const image = context.createImageData(current.width, current.height);
  const background = 20;
  for (let offset = 0; offset < image.data.length; offset += 4) {
    image.data[offset] = background;
    image.data[offset + 1] = background;
    image.data[offset + 2] = background;
    image.data[offset + 3] = 255;
  }
  const { minimum, maximum } = statistics.value;
  const range = maximum - minimum;
  for (let index = 0; index < props.values.length; index++) {
    const value = Number.isFinite(props.values[index]) ? props.values[index] : 0;
    const normalized = range > 1e-12
      ? (value - minimum) / range
      : Math.abs(value) > 1e-12 ? 0.72 : 0;
    const shade = Math.round(27 + Math.max(0, Math.min(1, normalized)) * 218);
    const cell = cellForIndex(index);
    const offset = (cell.y * current.width + cell.x) * 4;
    image.data[offset] = shade;
    image.data[offset + 1] = shade;
    image.data[offset + 2] = shade;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function indexFromPointer(event: PointerEvent) {
  const element = canvas.value;
  if (!element) return -1;
  const bounds = element.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return -1;
  const x = Math.min(
    layout.value.width - 1,
    Math.max(0, Math.floor(((event.clientX - bounds.left) / bounds.width) * layout.value.width)),
  );
  const y = Math.min(
    layout.value.height - 1,
    Math.max(0, Math.floor(((event.clientY - bounds.top) / bounds.height) * layout.value.height)),
  );
  return indexForCell(x, y);
}

function handlePointerMove(event: PointerEvent) {
  const index = indexFromPointer(event);
  hoverIndex.value = index >= 0 ? index : null;
}

function handlePointerDown(event: PointerEvent) {
  const index = indexFromPointer(event);
  if (index >= 0) {
    emit("select", { index, clientX: event.clientX, clientY: event.clientY });
  }
}

function anchorForIndex(index: number) {
  const element = canvas.value;
  if (!element) return { clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 };
  const bounds = element.getBoundingClientRect();
  const cell = cellForIndex(index);
  return {
    clientX: bounds.left + ((cell.x + 0.5) / layout.value.width) * bounds.width,
    clientY: bounds.top + ((cell.y + 0.5) / layout.value.height) * bounds.height,
  };
}

function handleKeydown(event: KeyboardEvent) {
  if (!props.values.length) return;
  const current = props.selectedIndex ?? 0;
  const rowStep = layout.value.kind === "spatial"
    ? layout.value.mapWidth
    : layout.value.columns;
  const amount = event.key === "ArrowLeft"
    ? -1
    : event.key === "ArrowRight"
      ? 1
      : event.key === "ArrowUp"
        ? -rowStep
        : event.key === "ArrowDown"
          ? rowStep
          : 0;
  if (!amount) return;
  event.preventDefault();
  const index = Math.max(0, Math.min(props.values.length - 1, current + amount));
  emit("select", { index, ...anchorForIndex(index) });
}

function markerStyle(index: number | null | undefined) {
  if (index === null || index === undefined || index < 0 || index >= props.values.length) return null;
  const cell = cellForIndex(index);
  return {
    left: `${(cell.x / layout.value.width) * 100}%`,
    top: `${(cell.y / layout.value.height) * 100}%`,
    width: `${100 / layout.value.width}%`,
    height: `${100 / layout.value.height}%`,
  };
}

watch(
  [() => props.values, layout],
  () => nextTick(draw),
  { deep: true },
);

onMounted(draw);
</script>

<template>
  <div class="signal-map">
    <div class="signal-map-heading">
      <span>{{ label }}</span>
      <b>{{ values.length.toLocaleString("zh-CN") }}</b>
    </div>
    <div
      class="signal-map-surface"
      :style="canvasStyle"
      role="grid"
      tabindex="0"
      :aria-label="`${label}，${values.length} 个数值`"
      @keydown="handleKeydown"
      @pointerleave="hoverIndex = null"
    >
      <canvas
        ref="canvas"
        :aria-label="label"
        @pointermove="handlePointerMove"
        @pointerdown="handlePointerDown"
      />
      <i
        v-if="markerStyle(selectedIndex)"
        class="signal-map-selection"
        :style="markerStyle(selectedIndex) ?? undefined"
        aria-hidden="true"
      />
      <output v-if="hoverIndex !== null" class="signal-map-hover">
        #{{ hoverIndex }} · {{ formatValue(values[hoverIndex] ?? 0) }}
      </output>
      <span v-if="values.length === 0" class="signal-map-empty">无采样</span>
    </div>
    <div class="signal-map-scale" aria-hidden="true">
      <span>{{ formatValue(statistics.minimum) }}</span>
      <i />
      <span>{{ formatValue(statistics.maximum) }}</span>
    </div>
  </div>
</template>
