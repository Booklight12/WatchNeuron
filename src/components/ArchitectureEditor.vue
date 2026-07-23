<script setup lang="ts">
import { Check, Copy, Plus, Minus, RotateCcw, Trash2 } from "@lucide/vue";
import { onBeforeUnmount, ref } from "vue";
import ActivationGuide, { type ActivationGuideKind } from "./ActivationGuide.vue";
import type { ActivationKind, HiddenLayer } from "../types";
import { activationLabels } from "../lib/model";

const props = defineProps<{
  layers: HiddenLayer[];
  parameterCount: number;
}>();

const emit = defineEmits<{
  update: [layers: HiddenLayer[]];
  reset: [];
}>();

const activationOptions = Object.entries(activationLabels) as [ActivationKind, string][];
const guideKind = ref<ActivationGuideKind>("relu");
const descriptionMode = ref<"brief" | "detailed">("brief");
const copyState = ref<"idle" | "copied" | "error">("idle");
let copyFeedbackTimer: ReturnType<typeof setTimeout> | undefined;

function describeNetwork(mode: "brief" | "detailed") {
  const totalLayers = props.layers.length + 2;
  const hiddenSizes = props.layers.map((layer) => layer.units);
  if (mode === "brief") {
    const hiddenDescriptions = props.layers.map(
      (layer, index) =>
        `隐藏层 ${index + 1}（全连接 / ${activationLabels[layer.activation]}）`,
    );
    return `${totalLayers} 层神经网络：输入层 → ${hiddenDescriptions.join(" → ")} → Softmax 输出层。`;
  }

  const layerDescriptions = [
    "第 1 层：输入层，784 个像素神经元。",
    ...props.layers.map(
      (layer, index) =>
        `第 ${index + 2} 层：全连接隐藏层，${layer.units} 个神经元，使用 ${activationLabels[layer.activation]} 激活。`,
    ),
    `第 ${totalLayers} 层：全连接输出层，10 个神经元，使用 Softmax 激活。`,
  ];
  const structure = [784, ...hiddenSizes, 10].join(" → ");
  return [
    `神经网络共 ${totalLayers} 层，连接结构为 ${structure}，共有 ${props.parameterCount.toLocaleString("zh-CN")} 个可训练参数。`,
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

onBeforeUnmount(() => clearTimeout(copyFeedbackTimer));

function updateLayer(id: string, patch: Partial<HiddenLayer>) {
  emit(
    "update",
    props.layers.map((layer) => (layer.id === id ? { ...layer, ...patch } : layer)),
  );
}

function step(id: string, amount: number) {
  const layer = props.layers.find((candidate) => candidate.id === id);
  if (!layer) return;
  const units = Math.max(8, layer.units + amount);
  updateLayer(id, { units });
}

function setUnits(id: string, input: HTMLInputElement) {
  const parsed = Number(input.value);
  if (!Number.isFinite(parsed)) {
    const layer = props.layers.find((candidate) => candidate.id === id);
    input.value = String(layer?.units ?? 8);
    return;
  }
  updateLayer(id, { units: Math.max(8, Math.floor(parsed)) });
}

function addLayer() {
  if (props.layers.length >= 4) return;
  emit("update", [
    ...props.layers,
    {
      id: `layer-${Date.now()}`,
      units: props.layers.at(-1)?.units ?? 32,
      activation: "relu",
    },
  ]);
}

function removeLayer(id: string) {
  if (props.layers.length <= 1) return;
  emit("update", props.layers.filter((layer) => layer.id !== id));
}

function selectActivation(id: string, activation: ActivationKind) {
  updateLayer(id, { activation });
  guideKind.value = activation;
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
          <button
            type="button"
            :aria-pressed="descriptionMode === 'brief'"
            :class="{ active: descriptionMode === 'brief' }"
            @click="descriptionMode = 'brief'"
          >
            简述
          </button>
          <button
            type="button"
            :aria-pressed="descriptionMode === 'detailed'"
            :class="{ active: descriptionMode === 'detailed' }"
            @click="descriptionMode = 'detailed'"
          >
            详细
          </button>
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
        <span class="copy-status" aria-live="polite">
          {{ copyState === "copied" ? "网络描述已复制" : copyState === "error" ? "网络描述复制失败" : "" }}
        </span>
      </div>
    </div>

    <div class="layer-stack">
      <div class="fixed-layer">
        <span class="layer-index">00</span>
        <div><b>输入层</b><small>像素神经元</small></div>
        <strong>784</strong>
      </div>

      <div v-for="(layer, index) in layers" :key="layer.id" class="editable-layer">
        <div class="layer-row">
          <span class="layer-index">{{ String(index + 1).padStart(2, '0') }}</span>
          <div><b>全连接层</b><small>Dense</small></div>
          <div class="unit-stepper" aria-label="神经元数量">
            <button type="button" title="减少神经元" aria-label="减少神经元" :disabled="layer.units <= 8" @click="step(layer.id, -8)">
              <Minus :size="14" />
            </button>
            <input
              type="number"
              min="8"
              step="1"
              inputmode="numeric"
              :value="layer.units"
              :aria-label="`隐藏层 ${index + 1} 神经元数量`"
              @change="setUnits(layer.id, $event.target as HTMLInputElement)"
            />
            <button type="button" title="增加神经元" aria-label="增加神经元" @click="step(layer.id, 8)">
              <Plus :size="14" />
            </button>
          </div>
          <button class="delete-layer" type="button" title="删除层" aria-label="删除层" :disabled="layers.length <= 1" @click="removeLayer(layer.id)">
            <Trash2 :size="15" />
          </button>
        </div>
        <label class="activation-select">
          <span>激活</span>
          <select :value="layer.activation" @change="selectActivation(layer.id, ($event.target as HTMLSelectElement).value as ActivationKind)">
            <option v-for="([value, label]) in activationOptions" :key="value" :value="value">{{ label }}</option>
          </select>
        </label>
      </div>

      <div class="fixed-layer output-layer">
        <span class="layer-index">{{ String(layers.length + 1).padStart(2, '0') }}</span>
        <div><b>输出层</b><small>Softmax</small></div>
        <strong>10</strong>
      </div>
    </div>

    <div class="architecture-footer">
      <button class="add-layer-button" type="button" :disabled="layers.length >= 4" @click="addLayer">
        <Plus :size="16" />
        添加隐藏层
      </button>
      <span>{{ parameterCount.toLocaleString() }} 参数</span>
    </div>

    <ActivationGuide :selected="guideKind" @select="guideKind = $event" />
  </section>
</template>
