<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import {
  ArrowLeft,
  BrainCircuit,
  Check,
  Database,
  PencilLine,
  Play,
  Trash2,
  X,
} from "@lucide/vue";
import { activationLabels } from "../lib/model";
import type { SavedModel, SavedModelSource } from "../types";

const props = defineProps<{
  models: SavedModel[];
  loading: boolean;
}>();

const emit = defineEmits<{
  back: [];
  load: [model: SavedModel];
  rename: [id: string, name: string];
  remove: [id: string];
}>();

type ModelFilter = "all" | SavedModelSource;

const filter = ref<ModelFilter>("all");
const editingId = ref<string | null>(null);
const draftName = ref("");

const filteredModels = computed(() =>
  props.models.filter((model) => filter.value === "all" || model.source === filter.value),
);
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
  return model.model.layers.reduce(
    (total, layer) => total + layer.weights.length + layer.biases.length,
    0,
  );
}

function modelBytes(model: SavedModel) {
  return model.model.layers.reduce(
    (total, layer) => total + layer.weights.byteLength + layer.biases.byteLength,
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

function removeModel(model: SavedModel) {
  if (!window.confirm(`删除模型“${model.name}”？`)) return;
  emit("remove", model.id);
}
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

    <section class="sample-manager-toolbar model-manager-toolbar" aria-label="模型筛选">
      <div class="sample-filter-tabs" role="group" aria-label="模型类型筛选">
        <button type="button" :class="{ active: filter === 'all' }" @click="filter = 'all'">全部</button>
        <button type="button" :class="{ active: filter === 'complete' }" @click="filter = 'complete'">完整训练</button>
        <button type="button" :class="{ active: filter === 'paused' }" @click="filter = 'paused'">暂停快照</button>
      </div>
      <span>{{ filteredModels.length }} 个结果</span>
    </section>

    <section v-if="filteredModels.length" class="model-list" aria-label="已保存模型列表">
      <article v-for="model in filteredModels" :key="model.id" class="model-card">
        <div class="model-card-main">
          <span
            class="model-source"
            :class="[model.source, { finetune: model.trainingMode === 'finetune' }]"
          >
            <Check v-if="model.source === 'complete'" :size="13" />
            <BrainCircuit v-else :size="13" />
            {{ modelSourceLabel(model) }}
          </span>

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
            <span>784</span>
            <template v-for="layer in model.hiddenLayers" :key="layer.id">
              <i>→</i>
              <span>{{ layer.units }} · {{ activationLabels[layer.activation] }}</span>
            </template>
            <i>→</i>
            <span>10 · Softmax</span>
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
            @click="removeModel(model)"
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
