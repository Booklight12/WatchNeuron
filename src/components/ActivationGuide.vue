<script setup lang="ts">
import { computed } from "vue";
import type { ActivationKind } from "../types";
import { activateScalar } from "../lib/activations";
import AppSelect, { type AppSelectOption } from "./AppSelect.vue";

export type ActivationGuideKind = ActivationKind;

interface GuideDefinition {
  label: string;
  shortLabel: string;
  formula: string;
  summary: string;
  range: string;
  use: string;
  graphNote: string;
  yDomain: [number, number];
}

const props = defineProps<{ selected: ActivationGuideKind }>();
const emit = defineEmits<{ select: [kind: ActivationGuideKind] }>();

const guides: Record<ActivationGuideKind, GuideDefinition> = {
  relu: {
    label: "ReLU",
    shortLabel: "ReLU",
    formula: "f(x) = max(0, x)",
    summary: "负输入归零，正输入直接通过。计算快，并能产生稀疏激活。",
    range: "[0, +∞)",
    use: "多数隐藏层的默认选择",
    graphNote: "负半轴为 0，正半轴斜率为 1。",
    yDomain: [-1, 4],
  },
  leakyRelu: {
    label: "Leaky ReLU",
    shortLabel: "Leaky",
    formula: "f(x) = max(0.08x, x)",
    summary: "负输入保留 0.08 倍的小梯度，降低神经元长期停止更新的风险。",
    range: "(-∞, +∞)",
    use: "需要保留负区间梯度的隐藏层",
    graphNote: "负半轴仍有 0.08 的微小斜率。",
    yDomain: [-1, 4],
  },
  elu: {
    label: "ELU",
    shortLabel: "ELU",
    formula: "f(x) = x 或 eˣ - 1",
    summary: "正区间保持线性，负区间平滑趋近 -1，能减少零梯度并让平均激活更接近零。",
    range: "(-1, +∞)",
    use: "希望负区间平滑且非零的隐藏层",
    graphNote: "负半轴平滑饱和到 -1，原点处连续。",
    yDomain: [-1.2, 4],
  },
  selu: {
    label: "SELU",
    shortLabel: "SELU",
    formula: "f(x) = λ·ELUα(x)",
    summary: "ELU 的缩放版本，在合适初始化和网络条件下可推动激活均值与方差自动稳定。",
    range: "(-1.76, +∞)",
    use: "采用 LeCun 初始化的自归一化网络",
    graphNote: "正半轴斜率略大于 1，负半轴下限约 -1.76。",
    yDomain: [-2, 4.5],
  },
  relu6: {
    label: "ReLU6",
    shortLabel: "ReLU6",
    formula: "f(x) = min(6, max(0, x))",
    summary: "把 ReLU 的最大输出限制为 6，降低大激活带来的数值范围，便于低精度计算。",
    range: "[0, 6]",
    use: "移动端或量化友好的网络",
    graphNote: "0 以下归零，0 到 6 线性，6 以上封顶。",
    yDomain: [-1, 6],
  },
  gelu: {
    label: "GELU",
    shortLabel: "GELU",
    formula: "f(x) ≈ 0.5x(1 + tanh(...))",
    summary: "按输入大小进行平滑门控，小负值不会被完全截断，是 Transformer 中常见的选择。",
    range: "约 (-0.17, +∞)",
    use: "Transformer 与平滑深层网络",
    graphNote: "零点附近平滑弯曲，正区间逐渐接近线性。",
    yDomain: [-1, 4],
  },
  swish: {
    label: "SiLU / Swish",
    shortLabel: "SiLU",
    formula: "f(x) = x·σ(x)",
    summary: "输入乘以自身的 Sigmoid 门，曲线平滑且允许少量负输出，也称 SiLU。",
    range: "约 (-0.28, +∞)",
    use: "现代卷积网络与平滑隐藏层",
    graphNote: "负区间有浅谷，正区间逐渐接近 x。",
    yDomain: [-1, 4],
  },
  mish: {
    label: "Mish",
    shortLabel: "Mish",
    formula: "f(x) = x·tanh(softplus(x))",
    summary: "使用 Softplus 和 Tanh 构成的平滑非单调激活，保留小幅负信号。",
    range: "约 (-0.31, +∞)",
    use: "需要平滑非单调激活的视觉网络",
    graphNote: "负区间形成浅谷，正区间近似线性。",
    yDomain: [-1, 4],
  },
  sigmoid: {
    label: "Sigmoid",
    shortLabel: "Sigmoid",
    formula: "f(x) = 1 / (1 + e⁻ˣ)",
    summary: "把任意输入平滑压缩到 0 和 1 之间，但两端饱和时梯度会变小。",
    range: "(0, 1)",
    use: "门控值或独立二分类输出",
    graphNote: "x = 0 时输出 0.5。",
    yDomain: [0, 1],
  },
  tanh: {
    label: "Tanh",
    shortLabel: "Tanh",
    formula: "f(x) = tanh(x)",
    summary: "输出以 0 为中心并压缩到 -1 和 1 之间，较适合带方向的信号。",
    range: "(-1, 1)",
    use: "需要正负对称激活的隐藏层",
    graphNote: "原点斜率最大，两端逐渐饱和。",
    yDomain: [-1, 1],
  },
  hardSigmoid: {
    label: "Hard Sigmoid",
    shortLabel: "Hard Sig.",
    formula: "f(x) = clip(0.2x + 0.5, 0, 1)",
    summary: "用分段直线近似 Sigmoid，省去指数计算，代价是在两个拐点处不可导。",
    range: "[0, 1]",
    use: "轻量门控与移动端模型",
    graphNote: "-2.5 到 2.5 线性，两侧分别固定为 0 和 1。",
    yDomain: [0, 1],
  },
  hardTanh: {
    label: "Hard Tanh",
    shortLabel: "Hard Tanh",
    formula: "f(x) = clip(x, -1, 1)",
    summary: "将 Tanh 简化为中间线性、两端截断的分段函数，计算量很低。",
    range: "[-1, 1]",
    use: "需要有界对称输出的轻量网络",
    graphNote: "-1 到 1 保持线性，区间外直接饱和。",
    yDomain: [-1, 1],
  },
  softplus: {
    label: "Softplus",
    shortLabel: "Softplus",
    formula: "f(x) = ln(1 + eˣ)",
    summary: "ReLU 的平滑近似，输出始终为正，并在所有位置保持可导。",
    range: "(0, +∞)",
    use: "需要平滑正值输出的隐藏层",
    graphNote: "负区间平滑趋近 0，正区间逐渐接近 x。",
    yDomain: [0, 4.5],
  },
  softsign: {
    label: "Softsign",
    shortLabel: "Softsign",
    formula: "f(x) = x / (1 + |x|)",
    summary: "与 Tanh 类似但饱和更缓慢，使用有理式而不是指数函数。",
    range: "(-1, 1)",
    use: "希望缓慢饱和的对称隐藏层",
    graphNote: "以原点为中心，两端缓慢趋近 -1 和 1。",
    yDomain: [-1, 1],
  },
  softmax: {
    label: "Softmax",
    shortLabel: "Softmax",
    formula: "pᵢ = eᶻⁱ / Σⱼeᶻʲ",
    summary: "联合比较整层分数并转换成总和为 1 的分布，可用于输出层或竞争式隐藏层。",
    range: "每项 (0, 1)，总和 = 1",
    use: "分类输出或竞争式隐藏层",
    graphNote: "图为二分类切片 p₁ = σ(z₁-z₂)。",
    yDomain: [0, 1],
  },
  linear: {
    label: "Linear",
    shortLabel: "Linear",
    formula: "f(x) = x",
    summary: "不改变输入的恒等激活，可保留完整数值范围，但叠加多层后仍等价于一个线性变换。",
    range: "(-∞, +∞)",
    use: "回归输出、基线实验或线性投影",
    graphNote: "穿过原点且斜率恒为 1。",
    yDomain: [-4, 4],
  },
};

const guideEntries = Object.entries(guides) as [ActivationGuideKind, GuideDefinition][];
const activationOptions: AppSelectOption[] = guideEntries.map(([value, guide]) => ({
  value,
  label: guide.label,
}));
const active = computed(() => guides[props.selected]);
const plot = { left: 28, right: 250, top: 12, bottom: 108 };
const xValues = [-4, -2, 0, 2, 4];

function activationValue(kind: ActivationGuideKind, input: number) {
  if (kind === "softmax") return 1 / (1 + Math.exp(-input));
  return activateScalar(input, kind);
}

function scaleX(value: number) {
  return plot.left + ((value + 4) / 8) * (plot.right - plot.left);
}

function scaleY(value: number) {
  const [minimum, maximum] = active.value.yDomain;
  return plot.bottom - ((value - minimum) / (maximum - minimum)) * (plot.bottom - plot.top);
}

const curvePoints = computed(() =>
  Array.from({ length: 97 }, (_, index) => {
    const input = -4 + (index / 96) * 8;
    return `${scaleX(input).toFixed(2)},${scaleY(activationValue(props.selected, input)).toFixed(2)}`;
  }).join(" "),
);

const yValues = computed(() => {
  const [minimum, maximum] = active.value.yDomain;
  return [minimum, (minimum + maximum) / 2, maximum];
});

const zeroAxisY = computed(() => {
  const [minimum, maximum] = active.value.yDomain;
  return minimum <= 0 && maximum >= 0 ? scaleY(0) : plot.bottom;
});

function axisLabel(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
</script>

<template>
  <section class="activation-guide" aria-labelledby="activation-guide-heading">
    <header>
      <div>
        <span class="eyebrow">FUNCTION GUIDE</span>
        <h3 id="activation-guide-heading">神经元类型图解</h3>
      </div>
      <code>{{ active.formula }}</code>
    </header>

    <label class="activation-guide-select">
      <span>激活函数</span>
      <AppSelect
        :model-value="selected"
        :options="activationOptions"
        label="选择神经元类型"
        mono
        @update:model-value="emit('select', $event as ActivationGuideKind)"
      />
    </label>

    <div class="activation-guide-body">
      <div class="activation-copy">
        <h4>{{ active.label }}</h4>
        <p>{{ active.summary }}</p>
        <dl>
          <div>
            <dt>输出范围</dt>
            <dd>{{ active.range }}</dd>
          </div>
          <div>
            <dt>适用位置</dt>
            <dd>{{ active.use }}</dd>
          </div>
        </dl>
      </div>

      <figure class="activation-figure">
        <svg
          viewBox="0 0 262 132"
          role="img"
          :aria-label="`${active.label} 函数曲线`"
        >
          <g class="function-grid">
            <line
              v-for="value in xValues"
              :key="`x-${value}`"
              :x1="scaleX(value)"
              :x2="scaleX(value)"
              :y1="plot.top"
              :y2="plot.bottom"
            />
            <line
              v-for="value in yValues"
              :key="`y-${value}`"
              :x1="plot.left"
              :x2="plot.right"
              :y1="scaleY(value)"
              :y2="scaleY(value)"
            />
          </g>
          <g class="function-axes">
            <line :x1="scaleX(0)" :x2="scaleX(0)" :y1="plot.top" :y2="plot.bottom" />
            <line :x1="plot.left" :x2="plot.right" :y1="zeroAxisY" :y2="zeroAxisY" />
          </g>
          <polyline class="function-curve" :points="curvePoints" />
          <circle
            class="function-origin"
            :cx="scaleX(0)"
            :cy="scaleY(activationValue(selected, 0))"
            r="2.8"
          />
          <g class="function-labels">
            <text v-for="value in [-4, 0, 4]" :key="`xl-${value}`" :x="scaleX(value)" y="124">
              {{ value }}
            </text>
            <text
              v-for="value in yValues"
              :key="`yl-${value}`"
              x="22"
              :y="scaleY(value) + 3"
              text-anchor="end"
            >
              {{ axisLabel(value) }}
            </text>
          </g>
        </svg>
        <figcaption>{{ active.graphNote }}</figcaption>
      </figure>
    </div>
  </section>
</template>
