<script setup lang="ts">
import { Check, ChevronDown } from "@lucide/vue";
import {
  type CSSProperties,
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  useId,
  watch,
} from "vue";

export type AppSelectValue = string | number;

export interface AppSelectOption {
  value: AppSelectValue;
  label: string;
  description?: string;
}

const props = withDefaults(
  defineProps<{
    modelValue: AppSelectValue;
    options: AppSelectOption[];
    label: string;
    disabled?: boolean;
    placement?: "auto" | "top" | "bottom";
    mono?: boolean;
  }>(),
  {
    disabled: false,
    placement: "auto",
    mono: false,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: AppSelectValue];
}>();

const root = ref<HTMLElement | null>(null);
const trigger = ref<HTMLButtonElement | null>(null);
const popup = ref<HTMLElement | null>(null);
const open = ref(false);
const activeIndex = ref(0);
const resolvedPlacement = ref<"top" | "bottom">("bottom");
const popupStyle = ref<CSSProperties>({});
const menuId = `app-select-${useId()}`;

const selectedIndex = computed(() =>
  props.options.findIndex((option) => option.value === props.modelValue),
);
const selectedOption = computed(
  () => props.options[selectedIndex.value] ?? props.options[0],
);

function optionButtons() {
  return popup.value?.querySelectorAll<HTMLButtonElement>(".app-select-option") ?? [];
}

function focusOption(index: number) {
  if (!props.options.length) return;
  activeIndex.value = (index + props.options.length) % props.options.length;
  nextTick(() => optionButtons()[activeIndex.value]?.focus());
}

function openMenu(focus = false, index = selectedIndex.value) {
  if (props.disabled || props.options.length === 0) return;
  activeIndex.value = Math.max(0, index);
  open.value = true;
  nextTick(() => {
    updatePosition();
    if (focus) focusOption(activeIndex.value);
  });
}

function closeMenu(restoreFocus = false) {
  open.value = false;
  if (restoreFocus) nextTick(() => trigger.value?.focus());
}

function toggleMenu() {
  if (open.value) closeMenu();
  else openMenu();
}

function choose(option: AppSelectOption) {
  emit("update:modelValue", option.value);
  closeMenu(true);
}

function handleTriggerKeydown(event: KeyboardEvent) {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    openMenu(true, selectedIndex.value);
  } else if (event.key === "Home") {
    event.preventDefault();
    openMenu(true, 0);
  } else if (event.key === "End") {
    event.preventDefault();
    openMenu(true, props.options.length - 1);
  } else if (event.key === "Escape") {
    closeMenu();
  }
}

function handleOptionKeydown(event: KeyboardEvent, index: number) {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    focusOption(index + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    focusOption(index - 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    focusOption(0);
  } else if (event.key === "End") {
    event.preventDefault();
    focusOption(props.options.length - 1);
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeMenu(true);
  } else if (event.key === "Tab") {
    closeMenu();
  } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const query = event.key.toLocaleLowerCase();
    const next = props.options.findIndex(
      (option, optionIndex) =>
        optionIndex > index && option.label.toLocaleLowerCase().startsWith(query),
    );
    const wrapped = next >= 0
      ? next
      : props.options.findIndex((option) => option.label.toLocaleLowerCase().startsWith(query));
    if (wrapped >= 0) {
      event.preventDefault();
      focusOption(wrapped);
    }
  }
}

function handleDocumentPointerDown(event: PointerEvent) {
  const target = event.target as Node;
  if (!root.value?.contains(target) && !popup.value?.contains(target)) closeMenu();
}

function updatePosition() {
  if (!open.value || !trigger.value) return;
  const rect = trigger.value.getBoundingClientRect();
  const viewportPadding = 12;
  const gap = 6;
  const width = Math.min(
    Math.max(rect.width, 176),
    Math.max(176, window.innerWidth - viewportPadding * 2),
  );
  const estimatedHeight = Math.min(
    248,
    props.options.reduce((height, option) => height + (option.description ? 48 : 36), 10),
  );
  const availableAbove = rect.top - viewportPadding - gap;
  const availableBelow = window.innerHeight - rect.bottom - viewportPadding - gap;
  const placeAbove =
    props.placement === "top" ||
    (props.placement === "auto" && availableBelow < estimatedHeight && availableAbove > availableBelow);
  const left = Math.max(
    viewportPadding,
    Math.min(rect.left, window.innerWidth - width - viewportPadding),
  );
  resolvedPlacement.value = placeAbove ? "top" : "bottom";
  popupStyle.value = placeAbove
    ? {
        left: `${left}px`,
        bottom: `${window.innerHeight - rect.top + gap}px`,
        width: `${width}px`,
        maxHeight: `${Math.max(96, availableAbove)}px`,
      }
    : {
        top: `${rect.bottom + gap}px`,
        left: `${left}px`,
        width: `${width}px`,
        maxHeight: `${Math.max(96, availableBelow)}px`,
      };
}

watch(
  () => props.disabled,
  (disabled) => {
    if (disabled) closeMenu();
  },
);

watch(selectedIndex, (index) => {
  if (!open.value || index < 0) return;
  activeIndex.value = index;
});

watch(
  () => props.options.length,
  () => nextTick(updatePosition),
);

onMounted(() => {
  document.addEventListener("pointerdown", handleDocumentPointerDown);
  window.addEventListener("resize", updatePosition);
  window.addEventListener("scroll", updatePosition, true);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", handleDocumentPointerDown);
  window.removeEventListener("resize", updatePosition);
  window.removeEventListener("scroll", updatePosition, true);
});
</script>

<template>
  <div
    ref="root"
    class="app-select"
    :class="{ open, disabled, mono }"
  >
    <button
      ref="trigger"
      class="app-select-trigger"
      type="button"
      aria-haspopup="listbox"
      :aria-controls="menuId"
      :aria-expanded="open"
      :aria-label="label"
      :disabled="disabled"
      @click="toggleMenu"
      @keydown="handleTriggerKeydown"
    >
      <span>{{ selectedOption?.label ?? "请选择" }}</span>
      <ChevronDown :size="13" aria-hidden="true" />
    </button>

  </div>

  <Teleport to="body">
    <Transition name="app-select-menu">
      <div
        v-if="open"
        :id="menuId"
        ref="popup"
        class="app-select-popup"
        :class="[`placement-${resolvedPlacement}`, { mono }]"
        :style="popupStyle"
        role="listbox"
        :aria-label="label"
      >
        <button
          v-for="(option, index) in options"
          :key="String(option.value)"
          class="app-select-option"
          :class="{ selected: option.value === modelValue }"
          type="button"
          role="option"
          :aria-selected="option.value === modelValue"
          :tabindex="index === activeIndex ? 0 : -1"
          @click="choose(option)"
          @keydown="handleOptionKeydown($event, index)"
        >
          <span>
            <b>{{ option.label }}</b>
            <small v-if="option.description">{{ option.description }}</small>
          </span>
          <Check v-if="option.value === modelValue" :size="13" aria-hidden="true" />
        </button>
      </div>
    </Transition>
  </Teleport>
</template>
