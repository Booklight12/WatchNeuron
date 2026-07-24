<script setup lang="ts">
import {
  ArrowRight,
  Check,
  Copy,
  Grid3X3,
  GripVertical,
  Layers3,
  Minus,
  Plus,
  RotateCcw,
  Trash2,
} from "@lucide/vue";
import { type CSSProperties, computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import ActivationGuide, { type ActivationGuideKind } from "./ActivationGuide.vue";
import AppSelect, { type AppSelectOption } from "./AppSelect.vue";
import ConvolutionEditor from "./ConvolutionEditor.vue";
import type { ActivationKind, ConvolutionConfig, HiddenLayer } from "../types";
import {
  convolutionPipeline,
  createDefaultConvolutionConfig,
  fitConvolutionsToLayers,
} from "../lib/convolution";
import { activationLabels } from "../lib/model";

const props = defineProps<{
  layers: HiddenLayer[];
  convolutions: ConvolutionConfig[];
  parameterCount: number;
}>();

const emit = defineEmits<{
  update: [layers: HiddenLayer[]];
  updateConvolutions: [configs: ConvolutionConfig[]];
  reorder: [layers: HiddenLayer[], configs: ConvolutionConfig[]];
  reset: [];
}>();

const activationOptions = Object.entries(activationLabels) as [ActivationKind, string][];
const activationSelectOptions: AppSelectOption[] = activationOptions.map(([value, label]) => ({
  value,
  label,
}));
const guideKind = ref<ActivationGuideKind>("relu");
const descriptionMode = ref<"brief" | "detailed">("brief");
const copyState = ref<"idle" | "copied" | "error">("idle");
const addMenuOpen = ref(false);
const addMenu = ref<HTMLElement | null>(null);
const layerStack = ref<HTMLElement | null>(null);
const draggedLayerKey = ref<string | null>(null);
const dropIndex = ref<number | null>(null);
const reorderAnnouncement = ref("");
const dragPreviewStyle = ref<CSSProperties>({ transform: "translate3d(-320px, -160px, 0)" });
let copyFeedbackTimer: ReturnType<typeof setTimeout> | undefined;
let activePointer: { id: number; key: string; startY: number; handle: HTMLElement } | null = null;
let dragPreviewFrame: number | null = null;
let dragPreviewX = -320;
let dragPreviewY = -160;
let dragPreviewTargetX = -320;
let dragPreviewTargetY = -160;

const convolutionEntries = computed(() => convolutionPipeline(props.layers, props.convolutions));
type OrderedLayer =
  | { kind: "dense"; key: string; denseIndex: number; layer: HiddenLayer; displayIndex: number }
  | {
      kind: "conv";
      key: string;
      config: ConvolutionConfig;
      input: (typeof convolutionEntries.value)[number]["input"];
      output: (typeof convolutionEntries.value)[number]["output"];
      displayIndex: number;
    };

const orderedLayers = computed(() => {
  const result: OrderedLayer[] = [];
  for (let position = 0; position <= props.layers.length; position++) {
    for (const entry of convolutionEntries.value.filter(({ config }) => config.position === position)) {
      result.push({
        kind: "conv",
        key: entry.config.id,
        config: entry.config,
        input: entry.input,
        output: entry.output,
        displayIndex: result.length + 1,
      });
    }
    if (position < props.layers.length) {
      result.push({
        kind: "dense",
        key: props.layers[position].id,
        denseIndex: position,
        layer: props.layers[position],
        displayIndex: result.length + 1,
      });
    }
  }
  return result;
});

const dragTargetIndex = computed(() => {
  if (!draggedLayerKey.value) return -1;
  const sourceIndex = orderedLayers.value.findIndex((item) => item.key === draggedLayerKey.value);
  if (sourceIndex < 0 || dropIndex.value === null) return sourceIndex;
  return Math.max(
    0,
    Math.min(
      orderedLayers.value.length - 1,
      dropIndex.value - (sourceIndex < dropIndex.value ? 1 : 0),
    ),
  );
});

const dragPreview = computed(() => {
  const item = orderedLayers.value.find((candidate) => candidate.key === draggedLayerKey.value);
  if (!item) return null;
  return item.kind === "dense"
    ? {
        kind: item.kind,
        type: "DENSE",
        name: "全连接层",
        detail: `${item.layer.units.toLocaleString("zh-CN")} 神经元 · ${activationLabels[item.layer.activation]}`,
        from: item.displayIndex,
        to: dragTargetIndex.value + 1,
      }
    : {
        kind: item.kind,
        type: "CONV2D",
        name: "二维卷积层",
        detail: `${item.config.filters} 核 · ${item.config.kernelSize}×${item.config.kernelSize} · ${activationLabels[item.config.activation]}`,
        from: item.displayIndex,
        to: dragTargetIndex.value + 1,
      };
});

function describeNetwork(mode: "brief" | "detailed") {
  const totalLayers = orderedLayers.value.length + 2;
  const briefLayers = ["输入层"];
  const structure: Array<string | number> = [784];
  const layerDescriptions = ["第 1 层：输入层，784 个像素神经元。"];
  let layerNumber = 2;

  for (const item of orderedLayers.value) {
    if (item.kind === "conv") {
      briefLayers.push(`Conv2D（${item.config.filters} 个 ${item.config.kernelSize}×${item.config.kernelSize} 卷积核）`);
      structure.push(item.output.length);
      layerDescriptions.push(
        `第 ${layerNumber++} 层：二维卷积层，将 ${item.input.width}×${item.input.height}×${item.input.channels} 输入映射为 ${item.output.width}×${item.output.height}×${item.config.filters}，使用 ${item.config.filters} 个 ${item.config.kernelSize}×${item.config.kernelSize} 可训练卷积核，步幅 ${item.config.stride}，填充 ${item.config.padding}，激活函数为 ${activationLabels[item.config.activation]}。`,
      );
      continue;
    }
    briefLayers.push(`隐藏层 ${item.denseIndex + 1}（全连接 / ${activationLabels[item.layer.activation]}）`);
    structure.push(item.layer.units);
    layerDescriptions.push(
      `第 ${layerNumber++} 层：全连接隐藏层，${item.layer.units} 个神经元，使用 ${activationLabels[item.layer.activation]} 激活，Dropout ${Math.round(item.layer.dropout * 100)}%。`,
    );
  }
  briefLayers.push("Softmax 输出层");
  structure.push(10);
  if (mode === "brief") return `${totalLayers} 层神经网络：${briefLayers.join(" → ")}。`;
  layerDescriptions.push(`第 ${totalLayers} 层：全连接输出层，10 个神经元，使用 Softmax 激活。`);
  return [
    `神经网络共 ${totalLayers} 层，连接结构为 ${structure.join(" → ")}，共有 ${props.parameterCount.toLocaleString("zh-CN")} 个可训练参数。`,
    ...layerDescriptions,
  ].join("\n");
}

function fallbackCopy(text: string) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Copy command failed");
}

async function copyDescription() {
  const description = describeNetwork(descriptionMode.value);
  try {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(description);
      } catch {
        fallbackCopy(description);
      }
    } else {
      fallbackCopy(description);
    }
    copyState.value = "copied";
  } catch {
    copyState.value = "error";
  }
  clearTimeout(copyFeedbackTimer);
  copyFeedbackTimer = setTimeout(() => {
    copyState.value = "idle";
  }, 1800);
}

function closeAddMenu(event?: PointerEvent) {
  if (event && addMenu.value?.contains(event.target as Node)) return;
  addMenuOpen.value = false;
}

function handleEscape(event: KeyboardEvent) {
  if (event.key === "Escape") addMenuOpen.value = false;
}

onMounted(() => {
  document.addEventListener("pointerdown", closeAddMenu);
  document.addEventListener("keydown", handleEscape);
  window.addEventListener("pointermove", handlePointerMove, { passive: false });
  window.addEventListener("pointerup", handlePointerUp);
  window.addEventListener("pointercancel", handlePointerCancel);
});

onBeforeUnmount(() => {
  clearTimeout(copyFeedbackTimer);
  if (dragPreviewFrame !== null) cancelAnimationFrame(dragPreviewFrame);
  document.removeEventListener("pointerdown", closeAddMenu);
  document.removeEventListener("keydown", handleEscape);
  window.removeEventListener("pointermove", handlePointerMove);
  window.removeEventListener("pointerup", handlePointerUp);
  window.removeEventListener("pointercancel", handlePointerCancel);
});

function updateLayer(id: string, patch: Partial<HiddenLayer>) {
  emit("update", props.layers.map((layer) => (layer.id === id ? { ...layer, ...patch } : layer)));
}

function step(id: string, amount: number) {
  const layer = props.layers.find((candidate) => candidate.id === id);
  if (!layer) return;
  updateLayer(id, { units: Math.max(1, layer.units + amount) });
}

function setUnits(id: string, input: HTMLInputElement) {
  const parsed = Number(input.value);
  if (!Number.isFinite(parsed)) {
    const layer = props.layers.find((candidate) => candidate.id === id);
    input.value = String(layer?.units ?? 1);
    return;
  }
  updateLayer(id, { units: Math.max(1, Math.floor(parsed)) });
}

function addDenseLayer() {
  emit("update", [
    ...props.layers,
    {
      id: `layer-${Date.now().toString(36)}`,
      units: props.layers.at(-1)?.units ?? 32,
      activation: "relu",
      dropout: 0,
    },
  ]);
  addMenuOpen.value = false;
}

function addConvolutionLayer() {
  const next = createDefaultConvolutionConfig(props.layers.length, true);
  emit("updateConvolutions", fitConvolutionsToLayers([...props.convolutions, next], props.layers));
  addMenuOpen.value = false;
}

function updateConvolution(config: ConvolutionConfig) {
  emit(
    "updateConvolutions",
    fitConvolutionsToLayers(
      props.convolutions.map((candidate) => candidate.id === config.id ? config : candidate),
      props.layers,
    ),
  );
}

function removeConvolution(id: string) {
  emit("updateConvolutions", props.convolutions.filter((config) => config.id !== id));
}

function removeLayer(id: string) {
  if (props.layers.length <= 1) return;
  emit("update", props.layers.filter((layer) => layer.id !== id));
}

function selectActivation(id: string, activation: ActivationKind) {
  updateLayer(id, { activation });
  guideKind.value = activation;
}

function setDropout(id: string, input: HTMLInputElement) {
  const parsed = Number(input.value);
  if (!Number.isFinite(parsed)) return;
  updateLayer(id, { dropout: Math.min(0.95, Math.max(0, parsed / 100)) });
}

function layerName(item: OrderedLayer) {
  return item.kind === "dense"
    ? `全连接层 ${item.denseIndex + 1}`
    : "二维卷积层";
}

function emitReorderedLayers(items: OrderedLayer[], movedKey: string) {
  const layers = items
    .filter((item): item is Extract<OrderedLayer, { kind: "dense" }> => item.kind === "dense")
    .map((item) => item.layer);
  const convolutions: ConvolutionConfig[] = [];
  let densePosition = 0;
  for (const item of items) {
    if (item.kind === "dense") {
      densePosition++;
    } else {
      convolutions.push({ ...item.config, position: densePosition });
    }
  }
  emit("reorder", layers, fitConvolutionsToLayers(convolutions, layers));
  const index = items.findIndex((item) => item.key === movedKey);
  const moved = items[index];
  reorderAnnouncement.value = moved
    ? `${layerName(moved)}已移动到第 ${index + 1} 个隐藏层级`
    : "层级顺序已更新";
}

function moveLayerToSlot(key: string, slot: number, restoreFocus = false) {
  const items = [...orderedLayers.value];
  const sourceIndex = items.findIndex((item) => item.key === key);
  if (sourceIndex < 0) return;
  const targetIndex = Math.max(
    0,
    Math.min(items.length - 1, slot - (sourceIndex < slot ? 1 : 0)),
  );
  if (sourceIndex === targetIndex) return;
  const [moved] = items.splice(sourceIndex, 1);
  items.splice(targetIndex, 0, moved);
  emitReorderedLayers(items, key);
  if (restoreFocus) {
    nextTick(() => {
      const handles = layerStack.value?.querySelectorAll<HTMLButtonElement>(".layer-drag-handle") ?? [];
      Array.from(handles).find((handle) => handle.dataset.layerKey === key)?.focus();
    });
  }
}

function moveLayerByKeyboard(item: OrderedLayer, amount: number) {
  const sourceIndex = orderedLayers.value.findIndex((candidate) => candidate.key === item.key);
  const targetIndex = Math.max(0, Math.min(orderedLayers.value.length - 1, sourceIndex + amount));
  if (sourceIndex === targetIndex) return;
  moveLayerToSlot(item.key, amount > 0 ? targetIndex + 1 : targetIndex, true);
}

function updatePointerDropIndex(clientY: number) {
  const items = layerStack.value?.querySelectorAll<HTMLElement>(".sortable-layer") ?? [];
  let slot = items.length;
  for (let index = 0; index < items.length; index++) {
    const bounds = items[index].getBoundingClientRect();
    if (clientY < bounds.top + bounds.height / 2) {
      slot = index;
      break;
    }
  }
  dropIndex.value = slot;

  const scroller = layerStack.value?.closest<HTMLElement>(".control-column");
  if (!scroller) return;
  const bounds = scroller.getBoundingClientRect();
  if (clientY < bounds.top + 44) scroller.scrollTop -= 14;
  else if (clientY > bounds.bottom - 44) scroller.scrollTop += 14;
}

function renderDragPreview() {
  dragPreviewFrame = null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const easing = reducedMotion ? 1 : 0.34;
  dragPreviewX += (dragPreviewTargetX - dragPreviewX) * easing;
  dragPreviewY += (dragPreviewTargetY - dragPreviewY) * easing;
  dragPreviewStyle.value = {
    transform: `translate3d(${dragPreviewX.toFixed(2)}px, ${dragPreviewY.toFixed(2)}px, 0)`,
  };
  if (
    Math.abs(dragPreviewTargetX - dragPreviewX) > 0.2 ||
    Math.abs(dragPreviewTargetY - dragPreviewY) > 0.2
  ) {
    dragPreviewFrame = requestAnimationFrame(renderDragPreview);
  }
}

function updateDragPreviewPosition(clientX: number, clientY: number, immediate = false) {
  const previewWidth = Math.min(246, window.innerWidth - 24);
  const previewHeight = 72;
  dragPreviewTargetX = clientX + previewWidth + 30 > window.innerWidth
    ? Math.max(12, clientX - previewWidth - 18)
    : clientX + 18;
  dragPreviewTargetY = clientY + previewHeight + 24 > window.innerHeight
    ? Math.max(12, clientY - previewHeight - 16)
    : clientY + 16;
  if (immediate) {
    dragPreviewX = dragPreviewTargetX;
    dragPreviewY = dragPreviewTargetY;
  }
  if (dragPreviewFrame === null) dragPreviewFrame = requestAnimationFrame(renderDragPreview);
}

function handlePointerDown(event: PointerEvent, item: OrderedLayer) {
  if (event.button !== 0) return;
  const handle = event.currentTarget as HTMLElement;
  activePointer = { id: event.pointerId, key: item.key, startY: event.clientY, handle };
  try {
    handle.setPointerCapture(event.pointerId);
  } catch {
    // Global pointer listeners keep dragging functional when capture is unavailable.
  }
}

function handlePointerMove(event: PointerEvent) {
  if (!activePointer || event.pointerId !== activePointer.id) return;
  if (!draggedLayerKey.value && Math.abs(event.clientY - activePointer.startY) < 4) return;
  event.preventDefault();
  const startsDragging = !draggedLayerKey.value;
  draggedLayerKey.value = activePointer.key;
  updateDragPreviewPosition(event.clientX, event.clientY, startsDragging);
  updatePointerDropIndex(event.clientY);
}

function handlePointerUp(event: PointerEvent) {
  if (!activePointer || event.pointerId !== activePointer.id) return;
  const { handle } = activePointer;
  if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
  const key = draggedLayerKey.value;
  const slot = dropIndex.value;
  if (key && slot !== null) moveLayerToSlot(key, slot);
  handlePointerCancel();
}

function handlePointerCancel() {
  activePointer = null;
  draggedLayerKey.value = null;
  dropIndex.value = null;
  if (dragPreviewFrame !== null) cancelAnimationFrame(dragPreviewFrame);
  dragPreviewFrame = null;
}
</script>

<template>
  <section class="architecture-section" aria-labelledby="architecture-heading">
    <div class="section-heading compact-heading">
      <div>
        <span class="eyebrow">ARCHITECTURE</span>
        <h2 id="architecture-heading">网络结构</h2>
      </div>
      <div class="architecture-heading-actions">
        <div class="description-mode" role="group" aria-label="网络描述详细程度">
          <button type="button" :aria-pressed="descriptionMode === 'brief'" :class="{ active: descriptionMode === 'brief' }" @click="descriptionMode = 'brief'">简述</button>
          <button type="button" :aria-pressed="descriptionMode === 'detailed'" :class="{ active: descriptionMode === 'detailed' }" @click="descriptionMode = 'detailed'">详细</button>
        </div>
        <button
          class="icon-button copy-description-button"
          type="button"
          data-testid="copy-network-description"
          :title="copyState === 'copied' ? '已复制网络描述' : copyState === 'error' ? '复制失败' : `复制${descriptionMode === 'brief' ? '简述' : '详细描述'}`"
          :aria-label="copyState === 'copied' ? '已复制网络描述' : copyState === 'error' ? '复制失败' : `复制${descriptionMode === 'brief' ? '简述' : '详细描述'}`"
          @click="copyDescription"
        >
          <Check v-if="copyState === 'copied'" :size="16" />
          <Copy v-else :size="16" />
        </button>
        <button class="icon-button" type="button" title="恢复校准模型" aria-label="恢复校准模型" @click="emit('reset')">
          <RotateCcw :size="16" />
        </button>
        <span class="copy-status" aria-live="polite">{{ copyState === "copied" ? "网络描述已复制" : copyState === "error" ? "网络描述复制失败" : "" }}</span>
      </div>
    </div>

    <div ref="layerStack" class="layer-stack">
      <div class="fixed-layer">
        <span class="layer-index">00</span>
        <div><b>输入层</b><small>像素神经元</small></div>
        <strong>784</strong>
      </div>

      <div
        v-for="(item, itemIndex) in orderedLayers"
        :key="item.key"
        class="sortable-layer"
        :class="{
          dragging: draggedLayerKey === item.key,
          'drop-before': draggedLayerKey && dropIndex === itemIndex,
          'drop-after': draggedLayerKey && itemIndex === orderedLayers.length - 1 && dropIndex === orderedLayers.length,
        }"
      >
        <ConvolutionEditor
          v-if="item.kind === 'conv'"
          :config="item.config"
          :input="item.input"
          :display-index="item.displayIndex"
          @update="updateConvolution"
          @remove="removeConvolution(item.config.id)"
        >
          <template #drag-handle>
            <button
              class="layer-drag-handle"
              type="button"
              :data-layer-key="item.key"
              title="调整层级"
              :aria-label="`拖动${layerName(item)}，当前位于第 ${item.displayIndex} 个隐藏层级；按上下方向键移动`"
              @pointerdown="handlePointerDown($event, item)"
              @keydown.up.prevent="moveLayerByKeyboard(item, -1)"
              @keydown.down.prevent="moveLayerByKeyboard(item, 1)"
            >
              <GripVertical :size="15" />
            </button>
          </template>
        </ConvolutionEditor>

        <div v-else class="editable-layer">
          <div class="layer-row">
            <span class="layer-index">{{ String(item.displayIndex).padStart(2, '0') }}</span>
            <button
              class="layer-drag-handle"
              type="button"
              :data-layer-key="item.key"
              title="调整层级"
              :aria-label="`拖动${layerName(item)}，当前位于第 ${item.displayIndex} 个隐藏层级；按上下方向键移动`"
              @pointerdown="handlePointerDown($event, item)"
              @keydown.up.prevent="moveLayerByKeyboard(item, -1)"
              @keydown.down.prevent="moveLayerByKeyboard(item, 1)"
            >
              <GripVertical :size="15" />
            </button>
            <div class="layer-copy"><b>全连接层</b><small>Dense</small></div>
            <div class="unit-stepper" aria-label="神经元数量">
              <button type="button" title="减少神经元" aria-label="减少神经元" :disabled="item.layer.units <= 1" @click="step(item.layer.id, -8)"><Minus :size="14" /></button>
              <input
                type="number"
                min="1"
                step="1"
                inputmode="numeric"
                :value="item.layer.units"
                :aria-label="`隐藏层 ${item.denseIndex + 1} 神经元数量`"
                @change="setUnits(item.layer.id, $event.target as HTMLInputElement)"
              />
              <button type="button" title="增加神经元" aria-label="增加神经元" @click="step(item.layer.id, 8)"><Plus :size="14" /></button>
            </div>
            <button class="delete-layer" type="button" title="删除层" aria-label="删除层" :disabled="layers.length <= 1" @click="removeLayer(item.layer.id)"><Trash2 :size="15" /></button>
          </div>
          <label class="activation-select">
            <span>激活</span>
            <AppSelect
              :model-value="item.layer.activation"
              :options="activationSelectOptions"
              :label="`隐藏层 ${item.denseIndex + 1} 激活函数`"
              mono
              @update:model-value="selectActivation(item.layer.id, $event as ActivationKind)"
            />
          </label>
          <label class="dropout-setting">
            <span>Dropout</span>
            <div>
              <input type="number" min="0" max="95" step="1" :value="Math.round(item.layer.dropout * 100)" :aria-label="`隐藏层 ${item.denseIndex + 1} Dropout 百分比`" @input="setDropout(item.layer.id, $event.target as HTMLInputElement)" />
              <i>%</i>
            </div>
          </label>
        </div>
      </div>

      <div class="fixed-layer output-layer">
        <span class="layer-index">{{ String(orderedLayers.length + 1).padStart(2, '0') }}</span>
        <div><b>输出层</b><small>Softmax</small></div>
        <strong>10</strong>
      </div>
    </div>
    <span class="layer-order-status" aria-live="polite">{{ reorderAnnouncement }}</span>

    <div class="architecture-footer">
      <div ref="addMenu" class="add-layer-menu">
        <button
          class="add-layer-button"
          type="button"
          data-testid="add-hidden-layer"
          :aria-expanded="addMenuOpen"
          aria-haspopup="menu"
          @click="addMenuOpen = !addMenuOpen"
        >
          <Plus :size="16" />
          添加隐藏层
        </button>
        <div v-if="addMenuOpen" class="layer-type-menu" role="menu">
          <button type="button" role="menuitem" data-testid="add-dense-layer" @click="addDenseLayer">
            <Layers3 :size="17" />
            <span><b>全连接层</b><small>Dense</small></span>
          </button>
          <button type="button" role="menuitem" data-testid="add-convolution-layer" @click="addConvolutionLayer">
            <Grid3X3 :size="17" />
            <span><b>二维卷积层</b><small>Conv2D</small></span>
          </button>
        </div>
      </div>
      <span>{{ parameterCount.toLocaleString() }} 参数</span>
    </div>

    <ActivationGuide :selected="guideKind" @select="guideKind = $event" />
  </section>

  <Teleport to="body">
    <div
      v-if="dragPreview"
      class="layer-drag-preview-anchor"
      :style="dragPreviewStyle"
      aria-hidden="true"
    >
      <div class="layer-drag-preview" :class="dragPreview.kind">
        <span class="layer-drag-preview-icon">
          <Layers3 v-if="dragPreview.kind === 'dense'" :size="18" />
          <Grid3X3 v-else :size="18" />
        </span>
        <span class="layer-drag-preview-copy">
          <small>{{ dragPreview.type }} · REORDER</small>
          <strong>{{ dragPreview.name }}</strong>
          <i>{{ dragPreview.detail }}</i>
        </span>
        <span class="layer-drag-preview-position">
          <b>{{ String(dragPreview.from).padStart(2, '0') }}</b>
          <ArrowRight :size="13" />
          <strong>{{ String(dragPreview.to).padStart(2, '0') }}</strong>
        </span>
      </div>
    </div>
  </Teleport>
</template>
