<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import {
  ArrowLeft,
  BrainCircuit,
  Check,
  Database,
  ListChecks,
  PencilLine,
  Play,
  Trash2,
  X,
} from "@lucide/vue";
import { activationLabels } from "../lib/model";
import { modelConvolutions, modelSpatialLayers } from "../lib/convolution";
import type { SavedModel, SavedModelSource } from "../types";
import SegmentedControl, { type SegmentedControlOption } from "./SegmentedControl.vue";

const props = defineProps<{
  models: SavedModel[];
  loading: boolean;
  deleting: boolean;
}>();

const emit = defineEmits<{
  back: [];
  load: [model: SavedModel];
  rename: [id: string, name: string];
  removeMany: [ids: string[]];
}>();

type ModelFilter = "all" | SavedModelSource;

const filter = ref<ModelFilter>("all");
const modelFilterOptions: SegmentedControlOption[] = [
  { value: "all", label: "全部" },
  { value: "complete", label: "完整训练" },
  { value: "paused", label: "暂停快照" },
];
const editingId = ref<string | null>(null);
const draftName = ref("");
const selectionMode = ref(false);
const selectedIds = ref<Set<string>>(new Set());
const pendingDeleteIds = ref<string[]>([]);
const deleteSubmitted = ref(false);

const filteredModels = computed(() =>
  props.models.filter((model) => filter.value === "all" || model.source === filter.value),
);
const visibleIds = computed(() => filteredModels.value.map((model) => model.id));
const allVisibleSelected = computed(
  () =>
    visibleIds.value.length > 0 &&
    visibleIds.value.every((id) => selectedIds.value.has(id)),
);
const someVisibleSelected = computed(
  () =>
    !allVisibleSelected.value &&
    visibleIds.value.some((id) => selectedIds.value.has(id)),
);
const selectedCount = computed(() => selectedIds.value.size);
const pendingDeleteModels = computed(() => {
  const ids = new Set(pendingDeleteIds.value);
  return props.models.filter((model) => ids.has(model.id));
});
const pendingDeleteLabel = computed(() => {
  if (pendingDeleteModels.value.length === 1) return `“${pendingDeleteModels.value[0].name}”`;
  return `${pendingDeleteModels.value.length} 个模型`;
});
const completeCount = computed(
  () => props.models.filter((model) => model.source === "complete").length,
);
const pausedCount = computed(
  () => props.models.filter((model) => model.source === "paused").length,
);
const totalBytes = computed(() =>
  props.models.reduce((total, model) => total + modelBytes(model), 0),
);

function parameterCount(model: SavedModel) {
  const dense = model.model.layers.reduce(
    (total, layer) => total + layer.weights.length + layer.biases.length,
    0,
  );
  return dense + modelConvolutions(model.model).reduce(
    (total, convolution) => total + convolution.weights.length + convolution.biases.length,
    0,
  );
}

function modelBytes(model: SavedModel) {
  const dense = model.model.layers.reduce(
    (total, layer) => total + layer.weights.byteLength + layer.biases.byteLength,
    0,
  );
  return dense + modelConvolutions(model.model).reduce(
    (total, convolution) => total + convolution.weights.byteLength + convolution.biases.byteLength,
    0,
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function modelSourceLabel(model: SavedModel) {
  if (model.trainingMode === "finetune") {
    return model.source === "complete" ? "微调完成" : "微调快照";
  }
  return model.source === "complete" ? "完整训练" : "暂停快照";
}

function architectureItems(model: SavedModel) {
  const items = ["784"];
  for (let position = 0; position <= model.hiddenLayers.length; position++) {
    for (const layer of modelSpatialLayers(model.model).filter((item) => item.position === position)) {
      items.push(layer.type === "conv"
        ? `Conv2D ${layer.filters}×${layer.kernelSize}²`
        : layer.kind === "globalAverage" ? "GAP" : layer.kind === "max" ? `MaxPool ${layer.kernelSize}²` : `AvgPool ${layer.kernelSize}²`);
    }
    if (position < model.hiddenLayers.length) {
      const layer = model.hiddenLayers[position];
      const dropout = layer.dropout > 0 ? ` · D${Math.round(layer.dropout * 100)}%` : "";
      items.push(`${layer.units} · ${activationLabels[layer.activation]}${dropout}`);
    }
  }
  items.push(`10 · ${model.model.outputHead === "sigmoid" ? "Sigmoid" : "Softmax"}`);
  return items;
}

function beginRename(model: SavedModel) {
  editingId.value = model.id;
  draftName.value = model.name;
  nextTick(() => {
    const input = document.querySelector<HTMLInputElement>(".model-name-editor input");
    input?.focus();
    input?.select();
  });
}

function finishRename(model: SavedModel) {
  const name = draftName.value.trim();
  if (name && name !== model.name) emit("rename", model.id, name);
  editingId.value = null;
}

function setSelection(id: string, selected: boolean) {
  const next = new Set(selectedIds.value);
  if (selected) next.add(id);
  else next.delete(id);
  selectedIds.value = next;
}

function toggleVisibleSelection() {
  const next = new Set(selectedIds.value);
  if (allVisibleSelected.value) visibleIds.value.forEach((id) => next.delete(id));
  else visibleIds.value.forEach((id) => next.add(id));
  selectedIds.value = next;
}

function toggleSelectionMode() {
  selectionMode.value = !selectionMode.value;
  if (!selectionMode.value) {
    selectedIds.value = new Set();
    pendingDeleteIds.value = [];
  }
}

function clearSelection() {
  selectedIds.value = new Set();
}

function requestDelete(ids: string[]) {
  const available = new Set(props.models.map((model) => model.id));
  pendingDeleteIds.value = [...new Set(ids)].filter((id) => available.has(id));
  deleteSubmitted.value = false;
}

function confirmDelete() {
  if (props.deleting || !pendingDeleteIds.value.length) return;
  deleteSubmitted.value = true;
  emit("removeMany", [...pendingDeleteIds.value]);
}

watch(
  () => props.models.map((model) => model.id),
  (ids) => {
    const available = new Set(ids);
    selectedIds.value = new Set([...selectedIds.value].filter((id) => available.has(id)));
    pendingDeleteIds.value = pendingDeleteIds.value.filter((id) => available.has(id));
  },
);

watch(
  () => props.deleting,
  (deleting) => {
    if (!deleting && deleteSubmitted.value) {
      pendingDeleteIds.value = [];
      deleteSubmitted.value = false;
    }
  },
);
</script>

<template>
  <main class="sample-manager model-manager" aria-labelledby="model-manager-heading">
    <header class="sample-manager-heading">
      <button
        class="tool-button manager-back-button"
        type="button"
        title="返回实验台"
        aria-label="返回实验台"
        @click="emit('back')"
      >
        <ArrowLeft :size="16" />
        <span>返回实验台</span>
      </button>
      <div>
        <span class="eyebrow">MODEL LIBRARY</span>
        <h1 id="model-manager-heading">本地模型库</h1>
      </div>
      <span class="model-storage-state">
        <Database :size="14" />
        IndexedDB
      </span>
    </header>

    <section class="sample-summary" aria-label="模型统计">
      <div><span>全部模型</span><strong>{{ models.length }}</strong></div>
      <div><span>完整训练</span><strong>{{ completeCount }}</strong></div>
      <div><span>暂停快照</span><strong>{{ pausedCount }}</strong></div>
      <div><span>权重占用</span><strong>{{ formatBytes(totalBytes) }}</strong></div>
    </section>

    <section class="sample-manager-toolbar model-manager-toolbar" aria-label="模型筛选和批量操作">
      <SegmentedControl
        v-model="filter"
        class="sample-filter-tabs"
        :options="modelFilterOptions"
        label="模型类型筛选"
      />

      <span class="model-toolbar-result">{{ filteredModels.length }} 个结果</span>

      <button
        v-if="!selectionMode"
        class="tool-button model-selection-toggle"
        type="button"
        :disabled="!models.length || deleting"
        @click="toggleSelectionMode"
      >
        <ListChecks :size="15" />
        批量管理
      </button>
      <template v-else>
        <label class="select-visible-control" :class="{ disabled: visibleIds.length === 0 || deleting }">
          <input
            type="checkbox"
            :checked="allVisibleSelected"
            :indeterminate="someVisibleSelected"
            :disabled="visibleIds.length === 0 || deleting"
            @change="toggleVisibleSelection"
          />
          <span>选择当前</span>
        </label>
        <span class="toolbar-selection-count">已选 {{ selectedCount }}</span>
        <button
          class="tool-button model-clear-selection"
          type="button"
          :disabled="selectedCount === 0 || deleting"
          @click="clearSelection"
        >
          <X :size="15" />
          清除
        </button>
        <button
          class="tool-button remove-selected-button"
          type="button"
          :disabled="selectedCount === 0 || deleting"
          @click="requestDelete([...selectedIds])"
        >
          <Trash2 :size="15" />
          删除所选
        </button>
        <button
          class="tool-button model-selection-done"
          type="button"
          :disabled="deleting"
          @click="toggleSelectionMode"
        >
          <Check :size="15" />
          完成
        </button>
      </template>
    </section>

    <section
      v-if="pendingDeleteIds.length"
      class="model-delete-confirm"
      aria-live="polite"
      aria-label="确认删除模型"
    >
      <div class="model-delete-confirm-copy">
        <span><Trash2 :size="16" /></span>
        <div>
          <strong>删除 {{ pendingDeleteLabel }}？</strong>
          <p>模型权重和训练进度将从此浏览器永久移除。</p>
        </div>
      </div>
      <div class="model-delete-confirm-actions">
        <button type="button" :disabled="deleting" @click="pendingDeleteIds = []">
          <X :size="15" />
          取消
        </button>
        <button class="confirm" type="button" :disabled="deleting" @click="confirmDelete">
          <Trash2 :size="15" />
          {{ deleting ? "正在删除" : "确认删除" }}
        </button>
      </div>
    </section>

    <section v-if="filteredModels.length" class="model-list" aria-label="已保存模型列表">
      <article
        v-for="model in filteredModels"
        :key="model.id"
        class="model-card"
        :class="{ selected: selectedIds.has(model.id), 'selection-mode': selectionMode }"
      >
        <div class="model-card-main">
          <div class="model-card-meta">
            <label v-if="selectionMode" class="model-select-control">
              <input
                type="checkbox"
                :checked="selectedIds.has(model.id)"
                :disabled="deleting"
                :aria-label="`选择模型 ${model.name}`"
                @change="setSelection(model.id, ($event.target as HTMLInputElement).checked)"
              />
              <span>{{ selectedIds.has(model.id) ? "已选择" : "选择" }}</span>
            </label>
            <span
              class="model-source"
              :class="[model.source, { finetune: model.trainingMode === 'finetune' }]"
            >
              <Check v-if="model.source === 'complete'" :size="13" />
              <BrainCircuit v-else :size="13" />
              {{ modelSourceLabel(model) }}
            </span>
          </div>

          <div v-if="editingId === model.id" class="model-name-editor">
            <input
              v-model="draftName"
              maxlength="48"
              aria-label="模型名称"
              @keydown.enter="finishRename(model)"
              @keydown.esc="editingId = null"
            />
            <button type="button" title="确认名称" aria-label="确认名称" @click="finishRename(model)">
              <Check :size="15" />
            </button>
            <button type="button" title="取消重命名" aria-label="取消重命名" @click="editingId = null">
              <X :size="15" />
            </button>
          </div>
          <div v-else class="model-card-title">
            <h2>{{ model.name }}</h2>
            <button type="button" title="重命名模型" aria-label="重命名模型" @click="beginRename(model)">
              <PencilLine :size="15" />
            </button>
          </div>

          <div class="model-architecture" aria-label="模型架构">
            <template v-for="(item, index) in architectureItems(model)" :key="`${index}-${item}`">
              <i v-if="index">→</i>
              <span>{{ item }}</span>
            </template>
          </div>

          <dl class="model-card-metrics">
            <div>
              <dt>训练进度</dt>
              <dd>E{{ model.progress.epoch }} / {{ model.progress.epochs }}</dd>
            </div>
            <div>
              <dt>验证准确率</dt>
              <dd>{{ (model.progress.accuracy * 100).toFixed(1) }}%</dd>
            </div>
            <div>
              <dt>参数</dt>
              <dd>{{ parameterCount(model).toLocaleString() }}</dd>
            </div>
            <div>
              <dt>权重大小</dt>
              <dd>{{ formatBytes(modelBytes(model)) }}</dd>
            </div>
          </dl>
        </div>

        <div class="model-card-side">
          <time :datetime="new Date(model.createdAt).toISOString()">{{ formatDate(model.createdAt) }}</time>
          <button class="load-model-button" type="button" @click="emit('load', model)">
            <Play :size="15" />
            载入模型
          </button>
          <button
            class="delete-model-button"
            type="button"
            title="删除模型"
            aria-label="删除模型"
            :disabled="deleting"
            @click="requestDelete([model.id])"
          >
            <Trash2 :size="15" />
          </button>
        </div>
      </article>
    </section>

    <section v-else class="sample-empty-state" aria-live="polite">
      <span><BrainCircuit :size="24" /></span>
      <h2>{{ loading ? "正在读取本地模型" : models.length ? "没有符合筛选条件的模型" : "还没有本地模型" }}</h2>
      <button v-if="!loading && !models.length" class="tool-button" type="button" @click="emit('back')">
        <Play :size="16" />
        开始训练
      </button>
    </section>
  </main>
</template>
