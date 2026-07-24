<script setup lang="ts">
export type SegmentedControlValue = string | number;

export interface SegmentedControlOption {
  value: SegmentedControlValue;
  label: string;
  title?: string;
  disabled?: boolean;
  testId?: string;
}

withDefaults(
  defineProps<{
    options: SegmentedControlOption[];
    label: string;
    disabled?: boolean;
  }>(),
  {
    disabled: false,
  },
);

const model = defineModel<SegmentedControlValue>({ required: true });
</script>

<template>
  <div
    class="segmented-control"
    role="group"
    :aria-label="label"
    :style="{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }"
  >
    <button
      v-for="option in options"
      :key="option.value"
      type="button"
      :class="{ active: model === option.value }"
      :title="option.title"
      :aria-pressed="model === option.value"
      :disabled="disabled || option.disabled"
      :data-testid="option.testId"
      @click="model = option.value"
    >
      {{ option.label }}
    </button>
  </div>
</template>
