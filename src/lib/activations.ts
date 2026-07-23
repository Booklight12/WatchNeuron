import type { ActivationKind } from "../types";

const SELU_ALPHA = 1.6732632423543772;
const SELU_SCALE = 1.0507009873554805;
const GELU_FACTOR = Math.sqrt(2 / Math.PI);

function sigmoid(value: number) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function softplus(value: number) {
  if (value > 20) return value;
  if (value < -20) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

export function activateScalar(value: number, kind: ActivationKind) {
  if (kind === "relu") return Math.max(0, value);
  if (kind === "leakyRelu") return value >= 0 ? value : value * 0.08;
  if (kind === "elu") return value >= 0 ? value : Math.expm1(value);
  if (kind === "selu") {
    return SELU_SCALE * (value >= 0 ? value : SELU_ALPHA * Math.expm1(value));
  }
  if (kind === "relu6") return Math.min(6, Math.max(0, value));
  if (kind === "gelu") {
    const curve = GELU_FACTOR * (value + 0.044715 * value * value * value);
    return 0.5 * value * (1 + Math.tanh(curve));
  }
  if (kind === "swish") return value * sigmoid(value);
  if (kind === "mish") return value * Math.tanh(softplus(value));
  if (kind === "sigmoid") return sigmoid(value);
  if (kind === "tanh") return Math.tanh(value);
  if (kind === "hardSigmoid") return Math.min(1, Math.max(0, value * 0.2 + 0.5));
  if (kind === "hardTanh") return Math.min(1, Math.max(-1, value));
  if (kind === "softplus") return softplus(value);
  if (kind === "softsign") return value / (1 + Math.abs(value));
  return value;
}

export function activationDerivative(
  input: number,
  output: number,
  kind: ActivationKind,
) {
  if (kind === "relu") return input > 0 ? 1 : 0;
  if (kind === "leakyRelu") return input >= 0 ? 1 : 0.08;
  if (kind === "elu") return input >= 0 ? 1 : output + 1;
  if (kind === "selu") {
    return input >= 0 ? SELU_SCALE : output + SELU_SCALE * SELU_ALPHA;
  }
  if (kind === "relu6") return input > 0 && input < 6 ? 1 : 0;
  if (kind === "gelu") {
    const squared = input * input;
    const curve = GELU_FACTOR * (input + 0.044715 * input * squared);
    const tangent = Math.tanh(curve);
    return (
      0.5 * (1 + tangent) +
      0.5 * input * (1 - tangent * tangent) * GELU_FACTOR * (1 + 0.134145 * squared)
    );
  }
  if (kind === "swish") {
    const gate = sigmoid(input);
    return gate + input * gate * (1 - gate);
  }
  if (kind === "mish") {
    const tangent = Math.tanh(softplus(input));
    return tangent + input * sigmoid(input) * (1 - tangent * tangent);
  }
  if (kind === "sigmoid") return output * (1 - output);
  if (kind === "tanh") return 1 - output * output;
  if (kind === "hardSigmoid") return input > -2.5 && input < 2.5 ? 0.2 : 0;
  if (kind === "hardTanh") return input > -1 && input < 1 ? 1 : 0;
  if (kind === "softplus") return sigmoid(input);
  if (kind === "softsign") {
    const divisor = 1 + Math.abs(input);
    return 1 / (divisor * divisor);
  }
  return 1;
}
