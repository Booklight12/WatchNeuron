<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { ArrowLeft, Database, HardDrive, LockKeyhole, PencilLine, Trash2 } from "@lucide/vue";
import AppSelect, { type AppSelectOption, type AppSelectValue } from "./AppSelect.vue";
import SampleThumbnail from "./SampleThumbnail.vue";
import SegmentedControl, { type SegmentedControlOption } from "./SegmentedControl.vue";
import type { CustomDatasetSample, DatasetSplit } from "../types";

const props = defineProps<{
  samples: CustomDatasetSample[];
  locked: boolean;
  mnistEnabled: boolean;
}>();

const emit = defineEmits<{
  back: [];
  updateMnistEnabled: [enabled: boolean];
  update: [id: string, patch: { label?: number; split?: DatasetSplit }];
  remove: [id: string];
  removeMany: [ids: string[]];
}>();

type SplitFilter = "all" | DatasetSplit;

const splitFilter = ref<SplitFilter>("all");
const splitFilterOptions: SegmentedControlOption[] = [
  { value: "all", label: "全部" },
  { value: "training", label: "训练集" },
  { value: "test", label: "测试集" },
];
const sampleSplitOptions: SegmentedControlOption[] = [
  { value: "training", label: "训练集" },
  { value: "test", label: "测试集" },
];
const labelFilter = ref<AppSelectValue>("all");
const selectedIds = ref<Set<string>>(new Set());
const digitOptions: AppSelectOption[] = Array.from({ length: 10 }, (_, digit) => ({
  value: digit,
  label: String(digit),
}));
const labelFilterOptions: AppSelectOption[] = [
  { value: "all", label: "全部" },
  ...digitOptions,
];

const trainingCount = computed(
  () => props.samples.filter((sample) => sample.split === "training").length,
);
const testCount = computed(
  () => props.samples.filter((sample) => sample.split === "test").length,
);
const filteredSamples = computed(() =>
  props.samples
    .filter(
      (sample) =>
        (splitFilter.value === "all" || sample.split === splitFilter.value) &&
        (labelFilter.value === "all" || sample.label === Number(labelFilter.value)),
    )
    .slice()
    .reverse(),
);
const visibleIds = computed(() => filteredSamples.value.map((sample) => sample.id));
const allVisibleSelected = computed(
  () =>
    visibleIds.value.length > 0 &&
    visibleIds.value.every((id) => selectedIds.value.has(id)),
);
const selectedCount = computed(() => selectedIds.value.size);

function sampleEnergy(sample: CustomDatasetSample) {
  const total = sample.values.reduce((sum, value) => sum + value, 0);
  return (total / 784) * 100;
}

function shortId(sample: CustomDatasetSample) {
  return sample.id.replace(/^sample-/, "").slice(0, 13).toUpperCase();
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

function removeSample(id: string) {
  if (props.locked || !window.confirm("删除这个自定义样本？")) return;
  emit("remove", id);
}

function removeSelected() {
  if (props.locked || selectedCount.value === 0) return;
  if (!window.confirm(`删除选中的 ${selectedCount.value} 个自定义样本？`)) return;
  emit("removeMany", [...selectedIds.value]);
  selectedIds.value = new Set();
}

function removeFiltered() {
  if (props.locked || visibleIds.value.length === 0) return;
  if (!window.confirm(`删除当前筛选结果中的 ${visibleIds.value.length} 个样本？`)) return;
  emit("removeMany", [...visibleIds.value]);
  selectedIds.value = new Set();
}

watch(
  () => props.samples.map((sample) => sample.id),
  (ids) => {
    const available = new Set(ids);
    selectedIds.value = new Set([...selectedIds.value].filter((id) => available.has(id)));
  },
  { deep: true },
);
</script>

<template>
  <main class="sample-manager" aria-labelledby="sample-manager-heading">
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
        <span class="eyebrow">CUSTOM DATASET</span>
        <h1 id="sample-manager-heading">自定义样本库</h1>
      </div>
      <span v-if="locked" class="manager-lock-state">
        <LockKeyhole :size="14" />
        训练中只读
      </span>
    </header>

    <section class="dataset-source-control" aria-labelledby="dataset-source-heading">
      <div class="dataset-source-copy">
        <HardDrive :size="18" />
        <div>
          <span class="eyebrow">TRAINING SOURCE</span>
          <strong id="dataset-source-heading">MNIST 基础样本</strong>
          <p>{{ mnistEnabled ? "内置 4,000 个训练样本和 1,000 个测试样本会与自定义样本合并。" : "训练和测试将只使用下方的自定义样本。" }}</p>
        </div>
      </div>
      <dl class="dataset-source-counts">
        <div><dt>训练</dt><dd>{{ mnistEnabled ? "4,000" : "0" }}</dd></div>
        <div><dt>测试</dt><dd>{{ mnistEnabled ? "1,000" : "0" }}</dd></div>
      </dl>
      <label class="dataset-source-switch" :class="{ disabled: locked }">
        <span>{{ mnistEnabled ? "已启用" : "已关闭" }}</span>
        <input
          type="checkbox"
          role="switch"
          aria-label="启用 MNIST 基础样本"
          :checked="mnistEnabled"
          :disabled="locked"
          data-testid="mnist-dataset-switch"
          @change="emit('updateMnistEnabled', ($event.target as HTMLInputElement).checked)"
        />
        <i aria-hidden="true"><b /></i>
      </label>
    </section>

    <section class="sample-summary" aria-label="样本统计">
      <div>
        <span>全部样本</span>
        <strong>{{ samples.length.toLocaleString() }}</strong>
      </div>
      <div>
        <span>训练集</span>
        <strong>{{ trainingCount.toLocaleString() }}</strong>
      </div>
      <div>
        <span>测试集</span>
        <strong>{{ testCount.toLocaleString() }}</strong>
      </div>
      <div>
        <span>当前筛选</span>
        <strong>{{ filteredSamples.length.toLocaleString() }}</strong>
      </div>
    </section>

    <section class="sample-manager-toolbar" aria-label="样本筛选和批量操作">
      <SegmentedControl
        v-model="splitFilter"
        class="sample-filter-tabs"
        :options="splitFilterOptions"
        label="数据集筛选"
      />

      <label class="sample-label-filter">
        <span>标签</span>
        <AppSelect
          v-model="labelFilter"
          :options="labelFilterOptions"
          label="按数字标签筛选"
          mono
        />
      </label>

      <label class="select-visible-control" :class="{ disabled: visibleIds.length === 0 }">
        <input
          type="checkbox"
          :checked="allVisibleSelected"
          :disabled="visibleIds.length === 0"
          @change="toggleVisibleSelection"
        />
        <span>选择当前</span>
      </label>

      <span class="toolbar-selection-count">已选 {{ selectedCount }}</span>

      <button
        class="tool-button remove-selected-button"
        type="button"
        :disabled="locked || selectedCount === 0"
        @click="removeSelected"
      >
        <Trash2 :size="15" />
        删除所选
      </button>
      <button
        class="tool-button remove-filtered-button"
        type="button"
        :disabled="locked || visibleIds.length === 0"
        @click="removeFiltered"
      >
        <Trash2 :size="15" />
        清空当前
      </button>
    </section>

    <section v-if="filteredSamples.length" class="sample-grid" aria-label="自定义样本列表">
      <article v-for="sample in filteredSamples" :key="sample.id" class="sample-card">
        <div class="sample-card-visual">
          <label class="sample-select-control" :title="`选择标签 ${sample.label} 的样本`">
            <input
              type="checkbox"
              :checked="selectedIds.has(sample.id)"
              @change="setSelection(sample.id, ($event.target as HTMLInputElement).checked)"
            />
          </label>
          <SampleThumbnail :sample="sample" />
          <strong>{{ sample.label }}</strong>
        </div>

        <div class="sample-card-body">
          <div class="sample-card-title">
            <span>{{ shortId(sample) }}</span>
            <button
              type="button"
              title="删除样本"
              aria-label="删除样本"
              :disabled="locked"
              @click="removeSample(sample.id)"
            >
              <Trash2 :size="15" />
            </button>
          </div>

          <label class="sample-card-label-field">
            <span>数字标签</span>
            <AppSelect
              :model-value="sample.label"
              :options="digitOptions"
              :label="`修改样本 ${shortId(sample)} 的数字标签`"
              :disabled="locked"
              mono
              @update:model-value="emit('update', sample.id, { label: Number($event) })"
            />
          </label>

          <SegmentedControl
            class="sample-card-split"
            :model-value="sample.split"
            :options="sampleSplitOptions"
            label="样本所属数据集"
            :disabled="locked"
            @update:model-value="emit('update', sample.id, { split: $event as DatasetSplit })"
          />

          <dl class="sample-card-metrics">
            <div><dt>有效像素</dt><dd>{{ sample.indices.length }}</dd></div>
            <div><dt>输入能量</dt><dd>{{ sampleEnergy(sample).toFixed(1) }}%</dd></div>
          </dl>
        </div>
      </article>
    </section>

    <section v-else class="sample-empty-state" aria-live="polite">
      <span><Database :size="24" /></span>
      <h2>{{ samples.length ? "没有符合筛选条件的样本" : "还没有自定义样本" }}</h2>
      <button v-if="!samples.length" class="tool-button" type="button" @click="emit('back')">
        <PencilLine :size="16" />
        返回手写输入
      </button>
    </section>
  </main>
</template>
