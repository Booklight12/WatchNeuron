struct F32Buffer { values: array<f32> }
struct U32Buffer { values: array<u32> }
struct Params { values: array<vec4<u32>, 4> }

@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> weights: F32Buffer;
@group(0) @binding(2) var<storage, read> biases: F32Buffer;
@group(0) @binding(3) var<storage, read> auxiliary: F32Buffer;
@group(0) @binding(4) var<storage, read_write> output: F32Buffer;
@group(0) @binding(5) var<storage, read_write> output2: F32Buffer;
@group(0) @binding(6) var<storage, read_write> output3: F32Buffer;
@group(0) @binding(7) var<storage, read_write> output4: F32Buffer;
@group(0) @binding(8) var<uniform> params: Params;

fn p(index: u32) -> u32 {
  return params.values[index / 4u][index % 4u];
}

fn pf(index: u32) -> f32 {
  return bitcast<f32>(p(index));
}

fn fast_exp(raw: f32) -> f32 {
  if (raw < -12.0) { return 0.000006; }
  let value = min(raw, 12.0);
  return bitcast<f32>(u32(12102203.0 * value + 1064866805.0));
}

fn fast_log(value: f32) -> f32 {
  var bits = bitcast<u32>(value);
  let exponent = i32((bits >> 23u) & 255u) - 127;
  bits = (bits & 0x007fffffu) | 0x3f800000u;
  let normalized = bitcast<f32>(bits);
  let ratio = (normalized - 1.0) / (normalized + 1.0);
  let squared = ratio * ratio;
  var series = ratio;
  var power = ratio * squared;
  series += power / 3.0;
  power *= squared;
  series += power / 5.0;
  power *= squared;
  series += power / 7.0;
  power *= squared;
  series += power / 9.0;
  return 2.0 * series + f32(exponent) * 0.69314718056;
}

fn math_exp(value: f32) -> f32 {
  if (p(15u) == 0u) { return fast_exp(value); }
  return exp(value);
}

fn math_log(value: f32) -> f32 {
  if (p(15u) == 0u) { return fast_log(value); }
  return log(value);
}

fn math_tanh(value: f32) -> f32 {
  if (p(15u) == 0u) {
    return 2.0 / (1.0 + fast_exp(-2.0 * value)) - 1.0;
  }
  if (value > 10.0) { return 1.0; }
  if (value < -10.0) { return -1.0; }
  let exponential = exp(2.0 * value);
  return (exponential - 1.0) / (exponential + 1.0);
}

fn math_softplus(value: f32) -> f32 {
  if (p(15u) == 0u) {
    if (value > 12.0) { return value; }
    if (value < -12.0) { return fast_exp(value); }
    return fast_log(1.0 + fast_exp(value));
  }
  if (value > 20.0) { return value; }
  if (value < -20.0) { return exp(value); }
  return log(1.0 + exp(value));
}

fn clamp_delta(value: f32) -> f32 {
  return clamp(value, -5.0, 5.0);
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + math_exp(-value));
}

fn activation(value: f32, kind: u32) -> f32 {
  if (kind == 0u) { return value; }
  if (kind == 1u) { return max(value, 0.0); }
  if (kind == 2u) { return select(value * 0.08, value, value >= 0.0); }
  if (kind == 3u) { return sigmoid(value); }
  if (kind == 4u) { return math_tanh(value); }
  if (kind == 6u) { return select(math_exp(value) - 1.0, value, value >= 0.0); }
  if (kind == 7u) { return 1.050700987 * select(1.673263242 * (math_exp(value) - 1.0), value, value >= 0.0); }
  if (kind == 8u) { return 0.5 * value * (1.0 + math_tanh(0.79788456 * (value + 0.044715 * value * value * value))); }
  if (kind == 9u) { return value * sigmoid(value); }
  if (kind == 10u) { return value * math_tanh(math_softplus(value)); }
  if (kind == 11u) { return math_softplus(value); }
  if (kind == 12u) { return value / (1.0 + abs(value)); }
  if (kind == 13u) { return clamp(0.2 * value + 0.5, 0.0, 1.0); }
  if (kind == 14u) { return clamp(value, -1.0, 1.0); }
  if (kind == 15u) { return clamp(value, 0.0, 6.0); }
  return value;
}

fn activation_derivative(value: f32, kind: u32) -> f32 {
  if (kind == 0u) { return 1.0; }
  if (kind == 1u) { return select(0.0, 1.0, value > 0.0); }
  if (kind == 2u) { return select(0.08, 1.0, value >= 0.0); }
  if (kind == 3u) {
    let activated = sigmoid(value);
    return activated * (1.0 - activated);
  }
  if (kind == 4u) {
    let activated = math_tanh(value);
    return 1.0 - activated * activated;
  }
  if (kind == 6u) { return select(math_exp(value), 1.0, value >= 0.0); }
  if (kind == 7u) { return 1.050700987 * select(1.673263242 * math_exp(value), 1.0, value >= 0.0); }
  if (kind == 8u) {
    let inner = 0.79788456 * (value + 0.044715 * value * value * value);
    let t = math_tanh(inner);
    return 0.5 * (1.0 + t) + 0.5 * value * (1.0 - t * t) * 0.79788456 * (1.0 + 0.134145 * value * value);
  }
  if (kind == 9u) {
    let s = sigmoid(value);
    return s + value * s * (1.0 - s);
  }
  if (kind == 10u) {
    let softplus = math_softplus(value);
    let t = math_tanh(softplus);
    return t + value * sigmoid(value) * (1.0 - t * t);
  }
  if (kind == 11u) { return sigmoid(value); }
  if (kind == 12u) {
    let denominator = 1.0 + abs(value);
    return 1.0 / (denominator * denominator);
  }
  if (kind == 13u) { return select(0.0, 0.2, value > -2.5 && value < 2.5); }
  if (kind == 14u) { return select(0.0, 1.0, value > -1.0 && value < 1.0); }
  if (kind == 15u) { return select(0.0, 1.0, value > 0.0 && value < 6.0); }
  return 1.0;
}

fn random01(seed: u32, layer: u32, index: u32) -> f32 {
  var value = seed ^ ((layer + 1u) * 0x9e3779b9u) ^ ((index + 1u) * 0x85ebca6bu);
  value = value ^ (value >> 16u);
  value = value * 0x7feb352du;
  value = value ^ (value >> 15u);
  value = value * 0x846ca68bu;
  value = value ^ (value >> 16u);
  return f32(value & 0x00ffffffu) / 16777216.0;
}

@compute @workgroup_size(64)
fn dense_forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let input_size = p(1u);
  let output_size = p(2u);
  let kind = p(3u);
  let index = id.x;
  if (index >= batch_size * output_size) { return; }
  let batch = index / output_size;
  let row = index % output_size;
  var sum = biases.values[row];
  for (var column = 0u; column < input_size; column++) {
    sum += weights.values[row * input_size + column] * input.values[batch * input_size + column];
  }
  output2.values[index] = sum;
  var value = activation(sum, kind);
  if (kind == 5u) {
    var maximum = -3.4e38;
    for (var item = 0u; item < output_size; item++) {
      var item_sum = biases.values[item];
      for (var column = 0u; column < input_size; column++) {
        item_sum += weights.values[item * input_size + column] * input.values[batch * input_size + column];
      }
      maximum = max(maximum, item_sum);
    }
    var total = 0.0;
    for (var item = 0u; item < output_size; item++) {
      var item_sum = biases.values[item];
      for (var column = 0u; column < input_size; column++) {
        item_sum += weights.values[item * input_size + column] * input.values[batch * input_size + column];
      }
      total += math_exp(item_sum - maximum);
    }
    value = math_exp(sum - maximum) / total;
  }
  var mask = 1.0;
  if (p(4u) != 0u) {
    let rate = bitcast<f32>(p(7u));
    if (rate > 0.0) {
      let keep = 1.0 - rate;
      mask = select(1.0 / keep, 0.0, random01(p(5u), p(6u) + batch, row) < rate);
    }
  }
  output3.values[index] = mask;
  output.values[index] = value * mask;
}

@compute @workgroup_size(64)
fn output_loss(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let output_size = p(1u);
  let head = p(2u);
  let index = id.x;
  if (index >= batch_size * output_size) { return; }
  let batch = index / output_size;
  let item = index % output_size;
  let label = bitcast<u32>(auxiliary.values[batch]);
  var probability = 0.0;
  if (head == 1u) {
    probability = sigmoid(input.values[index]);
  } else {
    var maximum = -3.4e38;
    for (var digit = 0u; digit < output_size; digit++) {
      maximum = max(maximum, input.values[batch * output_size + digit]);
    }
    var total = 0.0;
    for (var digit = 0u; digit < output_size; digit++) {
      total += math_exp(input.values[batch * output_size + digit] - maximum);
    }
    probability = math_exp(input.values[index] - maximum) / total;
  }
  output.values[index] = probability;
  let desired = select(0.0, 1.0, item == label);
  output2.values[index] = (probability - desired) / select(1.0, f32(output_size), head == 1u);
  if (item == 0u) {
    var loss = 0.0;
    if (head == 1u) {
      for (var digit = 0u; digit < output_size; digit++) {
        let digit_target = select(0.0, 1.0, digit == label);
        let digit_probability = clamp(sigmoid(input.values[batch * output_size + digit]), 0.000001, 0.999999);
        loss -= select(math_log(1.0 - digit_probability), math_log(digit_probability), digit_target > 0.5) / f32(output_size);
      }
    } else {
      var maximum = -3.4e38;
      for (var digit = 0u; digit < output_size; digit++) {
        maximum = max(maximum, input.values[batch * output_size + digit]);
      }
      var total = 0.0;
      for (var digit = 0u; digit < output_size; digit++) {
        total += math_exp(input.values[batch * output_size + digit] - maximum);
      }
      let label_probability = math_exp(input.values[batch * output_size + label] - maximum) / total;
      loss = -math_log(max(label_probability, 0.00000001));
    }
    output3.values[batch] = loss;
  }
}

@compute @workgroup_size(64)
fn dense_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let output_size = p(2u);
  let kind = p(3u);
  let index = id.x;
  if (index >= batch_size * output_size) { return; }
  let batch = index / output_size;
  var gradient = clamp_delta(input.values[index]);
  if (kind == 5u) {
    var maximum = -3.4e38;
    for (var item = 0u; item < output_size; item++) {
      maximum = max(maximum, weights.values[batch * output_size + item]);
    }
    var total = 0.0;
    for (var item = 0u; item < output_size; item++) {
      total += math_exp(weights.values[batch * output_size + item] - maximum);
    }
    var dot = 0.0;
    for (var item = 0u; item < output_size; item++) {
      let probability = math_exp(weights.values[batch * output_size + item] - maximum) / total;
      dot += clamp_delta(input.values[batch * output_size + item]) * probability;
    }
    let probability = math_exp(weights.values[index] - maximum) / total;
    gradient = probability * (gradient - dot);
  } else {
    gradient *= biases.values[index] * activation_derivative(weights.values[index], kind);
  }
  output.values[index] = gradient;
}

@compute @workgroup_size(64)
fn dense_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let input_size = p(1u);
  let output_size = p(2u);
  let index = id.x;
  if (index >= batch_size * input_size) { return; }
  let batch = index / input_size;
  let column = index % input_size;
  var sum = 0.0;
  for (var row = 0u; row < output_size; row++) {
    sum += input.values[batch * output_size + row] * weights.values[row * input_size + column];
  }
  output.values[index] = clamp_delta(sum);
}

@compute @workgroup_size(64)
fn dense_parameter_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let input_size = p(1u);
  let output_size = p(2u);
  let index = id.x;
  if (index < input_size * output_size) {
    let row = index / input_size;
    let column = index % input_size;
    var sum = 0.0;
    for (var batch = 0u; batch < batch_size; batch++) {
      sum += input.values[batch * input_size + column] * auxiliary.values[batch * output_size + row];
    }
    output3.values[index] = sum;
  }
  if (index < output_size) {
    var sum = 0.0;
    for (var batch = 0u; batch < batch_size; batch++) {
      sum += auxiliary.values[batch * output_size + index];
    }
    output4.values[index] = sum;
  }
}

fn conv_output_width() -> u32 { return (p(1u) + p(7u) * 2u - p(5u)) / p(6u) + 1u; }
fn conv_output_height() -> u32 { return (p(2u) + p(7u) * 2u - p(5u)) / p(6u) + 1u; }

fn conv_sum(batch: u32, filter_index: u32, oy: u32, ox: u32) -> f32 {
  let iw = p(1u);
  let ih = p(2u);
  let channels = p(3u);
  let kernel = p(5u);
  let stride = p(6u);
  let padding = i32(p(7u));
  var sum = biases.values[filter_index];
  for (var channel = 0u; channel < channels; channel++) {
    for (var ky = 0u; ky < kernel; ky++) {
      let iy = i32(oy * stride + ky) - padding;
      if (iy < 0 || iy >= i32(ih)) { continue; }
      for (var kx = 0u; kx < kernel; kx++) {
        let ix = i32(ox * stride + kx) - padding;
        if (ix < 0 || ix >= i32(iw)) { continue; }
        let input_index = ((batch * channels + channel) * ih + u32(iy)) * iw + u32(ix);
        let weight_index = ((filter_index * channels + channel) * kernel + ky) * kernel + kx;
        sum += input.values[input_index] * weights.values[weight_index];
      }
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn conv_forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let ow = conv_output_width();
  let oh = conv_output_height();
  let sample_size = p(4u) * ow * oh;
  let index = id.x;
  if (index >= p(0u) * sample_size) { return; }
  let batch = index / sample_size;
  let local = index % sample_size;
  let filter_index = local / (ow * oh);
  let pixel = local % (ow * oh);
  let oy = pixel / ow;
  let ox = pixel % ow;
  let sum = conv_sum(batch, filter_index, oy, ox);
  output2.values[index] = sum;
  if (p(8u) == 5u) {
    var maximum = -3.4e38;
    for (var item = 0u; item < sample_size; item++) {
      let item_filter = item / (ow * oh);
      let item_pixel = item % (ow * oh);
      maximum = max(maximum, conv_sum(batch, item_filter, item_pixel / ow, item_pixel % ow));
    }
    var total = 0.0;
    for (var item = 0u; item < sample_size; item++) {
      let item_filter = item / (ow * oh);
      let item_pixel = item % (ow * oh);
      total += math_exp(conv_sum(batch, item_filter, item_pixel / ow, item_pixel % ow) - maximum);
    }
    output.values[index] = math_exp(sum - maximum) / total;
  } else {
    output.values[index] = activation(sum, p(8u));
  }
}

const CONV_TILE = 16u;
var<workgroup> conv_tile_input: array<f32, 256>;
var<workgroup> conv_tile_weight: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn conv_forward_tiled(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let ow = conv_output_width();
  let oh = conv_output_height();
  let spatial_size = ow * oh;
  let row_count = p(0u) * spatial_size;
  let reduction_size = p(3u) * p(5u) * p(5u);
  let row = workgroup_id.y * CONV_TILE + local_id.y;
  let filter_index = workgroup_id.x * CONV_TILE + local_id.x;
  var sum = 0.0;

  for (var base = 0u; base < reduction_size; base += CONV_TILE) {
    let input_k = base + local_id.x;
    let weight_k = base + local_id.y;
    var input_value = 0.0;
    if (row < row_count && input_k < reduction_size) {
      let batch = row / spatial_size;
      let pixel = row % spatial_size;
      let oy = pixel / ow;
      let ox = pixel % ow;
      let kernel_area = p(5u) * p(5u);
      let channel = input_k / kernel_area;
      let kernel_pixel = input_k % kernel_area;
      let ky = kernel_pixel / p(5u);
      let kx = kernel_pixel % p(5u);
      let iy = i32(oy * p(6u) + ky) - i32(p(7u));
      let ix = i32(ox * p(6u) + kx) - i32(p(7u));
      if (iy >= 0 && iy < i32(p(2u)) && ix >= 0 && ix < i32(p(1u))) {
        input_value = input.values[
          ((batch * p(3u) + channel) * p(2u) + u32(iy)) * p(1u) + u32(ix)
        ];
      }
    }
    conv_tile_input[local_id.y * CONV_TILE + local_id.x] = input_value;
    conv_tile_weight[local_id.y * CONV_TILE + local_id.x] = select(
      0.0,
      weights.values[filter_index * reduction_size + weight_k],
      filter_index < p(4u) && weight_k < reduction_size,
    );
    workgroupBarrier();
    for (var tile_k = 0u; tile_k < CONV_TILE; tile_k++) {
      sum += conv_tile_input[local_id.y * CONV_TILE + tile_k] *
        conv_tile_weight[tile_k * CONV_TILE + local_id.x];
    }
    workgroupBarrier();
  }

  if (row >= row_count || filter_index >= p(4u)) { return; }
  let batch = row / spatial_size;
  let pixel = row % spatial_size;
  let output_index = (batch * p(4u) + filter_index) * spatial_size + pixel;
  let value = sum + biases.values[filter_index];
  output2.values[output_index] = value;
  output.values[output_index] = activation(value, p(8u));
}

@compute @workgroup_size(64)
fn conv_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let ow = conv_output_width();
  let oh = conv_output_height();
  let sample_size = p(4u) * ow * oh;
  let index = id.x;
  if (index >= p(0u) * sample_size) { return; }
  let batch = index / sample_size;
  var gradient = clamp_delta(input.values[index]);
  if (p(8u) == 5u) {
    var maximum = -3.4e38;
    for (var item = 0u; item < sample_size; item++) {
      maximum = max(maximum, weights.values[batch * sample_size + item]);
    }
    var total = 0.0;
    for (var item = 0u; item < sample_size; item++) {
      total += math_exp(weights.values[batch * sample_size + item] - maximum);
    }
    var dot = 0.0;
    for (var item = 0u; item < sample_size; item++) {
      let probability = math_exp(weights.values[batch * sample_size + item] - maximum) / total;
      dot += clamp_delta(input.values[batch * sample_size + item]) * probability;
    }
    let probability = math_exp(weights.values[index] - maximum) / total;
    gradient = probability * (gradient - dot);
  } else {
    gradient *= activation_derivative(weights.values[index], p(8u));
  }
  output.values[index] = gradient;
}

@compute @workgroup_size(64)
fn conv_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let iw = p(1u);
  let ih = p(2u);
  let channels = p(3u);
  let filters = p(4u);
  let kernel = p(5u);
  let stride = p(6u);
  let padding = i32(p(7u));
  let ow = conv_output_width();
  let oh = conv_output_height();
  let input_sample_size = channels * iw * ih;
  let index = id.x;
  if (index >= batch_size * input_sample_size) { return; }
  let batch = index / input_sample_size;
  let local = index % input_sample_size;
  let channel = local / (iw * ih);
  let pixel = local % (iw * ih);
  let iy = pixel / iw;
  let ix = pixel % iw;
  var sum = 0.0;
  for (var filter_index = 0u; filter_index < filters; filter_index++) {
    for (var ky = 0u; ky < kernel; ky++) {
      let oy_numerator = i32(iy) + padding - i32(ky);
      if (oy_numerator < 0 || oy_numerator % i32(stride) != 0) { continue; }
      let oy = u32(oy_numerator / i32(stride));
      if (oy >= oh) { continue; }
      for (var kx = 0u; kx < kernel; kx++) {
        let ox_numerator = i32(ix) + padding - i32(kx);
        if (ox_numerator < 0 || ox_numerator % i32(stride) != 0) { continue; }
        let ox = u32(ox_numerator / i32(stride));
        if (ox >= ow) { continue; }
        let delta_index = ((batch * filters + filter_index) * oh + oy) * ow + ox;
        let weight_index = ((filter_index * channels + channel) * kernel + ky) * kernel + kx;
        sum += input.values[delta_index] * weights.values[weight_index];
      }
    }
  }
  output.values[index] = clamp_delta(sum);
}

@compute @workgroup_size(64)
fn conv_parameter_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let iw = p(1u);
  let ih = p(2u);
  let channels = p(3u);
  let filters = p(4u);
  let kernel = p(5u);
  let stride = p(6u);
  let padding = i32(p(7u));
  let ow = conv_output_width();
  let oh = conv_output_height();
  let weight_count = filters * channels * kernel * kernel;
  let index = id.x;
  if (index < weight_count) {
    let filter_index = index / (channels * kernel * kernel);
    let remainder = index % (channels * kernel * kernel);
    let channel = remainder / (kernel * kernel);
    let kernel_pixel = remainder % (kernel * kernel);
    let ky = kernel_pixel / kernel;
    let kx = kernel_pixel % kernel;
    var sum = 0.0;
    for (var batch = 0u; batch < batch_size; batch++) {
      for (var oy = 0u; oy < oh; oy++) {
        let iy = i32(oy * stride + ky) - padding;
        if (iy < 0 || iy >= i32(ih)) { continue; }
        for (var ox = 0u; ox < ow; ox++) {
          let ix = i32(ox * stride + kx) - padding;
          if (ix < 0 || ix >= i32(iw)) { continue; }
          let input_index = ((batch * channels + channel) * ih + u32(iy)) * iw + u32(ix);
          let delta_index = ((batch * filters + filter_index) * oh + oy) * ow + ox;
          sum += input.values[input_index] * auxiliary.values[delta_index];
        }
      }
    }
    output3.values[index] = sum;
  }
  if (index < filters) {
    var sum = 0.0;
    for (var batch = 0u; batch < batch_size; batch++) {
      for (var pixel = 0u; pixel < ow * oh; pixel++) {
        sum += auxiliary.values[(batch * filters + index) * ow * oh + pixel];
      }
    }
    output4.values[index] = sum;
  }
}

var<workgroup> conv_tile_delta: array<f32, 256>;
var<workgroup> conv_tile_patch: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn conv_weight_gradient_tiled(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let ow = conv_output_width();
  let oh = conv_output_height();
  let spatial_size = ow * oh;
  let sample_count = p(0u) * spatial_size;
  let reduction_size = p(3u) * p(5u) * p(5u);
  let kernel_column = workgroup_id.x * CONV_TILE + local_id.x;
  let filter_index = workgroup_id.y * CONV_TILE + local_id.y;
  var sum = 0.0;

  for (var base = 0u; base < sample_count; base += CONV_TILE) {
    let delta_sample = base + local_id.x;
    let patch_sample = base + local_id.y;
    conv_tile_delta[local_id.y * CONV_TILE + local_id.x] = select(
      0.0,
      auxiliary.values[
        (delta_sample / spatial_size * p(4u) + filter_index) * spatial_size +
        delta_sample % spatial_size
      ],
      filter_index < p(4u) && delta_sample < sample_count,
    );

    var patch_value = 0.0;
    if (patch_sample < sample_count && kernel_column < reduction_size) {
      let batch = patch_sample / spatial_size;
      let pixel = patch_sample % spatial_size;
      let oy = pixel / ow;
      let ox = pixel % ow;
      let kernel_area = p(5u) * p(5u);
      let channel = kernel_column / kernel_area;
      let kernel_pixel = kernel_column % kernel_area;
      let ky = kernel_pixel / p(5u);
      let kx = kernel_pixel % p(5u);
      let iy = i32(oy * p(6u) + ky) - i32(p(7u));
      let ix = i32(ox * p(6u) + kx) - i32(p(7u));
      if (iy >= 0 && iy < i32(p(2u)) && ix >= 0 && ix < i32(p(1u))) {
        patch_value = input.values[
          ((batch * p(3u) + channel) * p(2u) + u32(iy)) * p(1u) + u32(ix)
        ];
      }
    }
    conv_tile_patch[local_id.y * CONV_TILE + local_id.x] = patch_value;
    workgroupBarrier();
    for (var tile_k = 0u; tile_k < CONV_TILE; tile_k++) {
      sum += conv_tile_delta[local_id.y * CONV_TILE + tile_k] *
        conv_tile_patch[tile_k * CONV_TILE + local_id.x];
    }
    workgroupBarrier();
  }

  if (filter_index < p(4u) && kernel_column < reduction_size) {
    output3.values[filter_index * reduction_size + kernel_column] = sum;
  }
}

@compute @workgroup_size(64)
fn conv_bias_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let filter_index = id.x;
  if (filter_index >= p(4u)) { return; }
  let spatial_size = conv_output_width() * conv_output_height();
  var sum = 0.0;
  for (var batch = 0u; batch < p(0u); batch++) {
    for (var pixel = 0u; pixel < spatial_size; pixel++) {
      sum += auxiliary.values[
        (batch * p(4u) + filter_index) * spatial_size + pixel
      ];
    }
  }
  output4.values[filter_index] = sum;
}

@compute @workgroup_size(64)
fn pool_forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let iw = p(1u);
  let ih = p(2u);
  let channels = p(3u);
  let ow = p(4u);
  let oh = p(5u);
  let kernel = p(6u);
  let stride = p(7u);
  let padding = i32(p(8u));
  let kind = p(9u);
  let output_sample_size = channels * ow * oh;
  let index = id.x;
  if (index >= batch_size * output_sample_size) { return; }
  let batch = index / output_sample_size;
  let local = index % output_sample_size;
  let channel = local / (ow * oh);
  let pixel = local % (ow * oh);
  let oy = pixel / ow;
  let ox = pixel % ow;
  var value = select(0.0, -3.4e38, kind == 0u);
  var selected = 0xffffffffu;
  var count = 0u;
  for (var ky = 0u; ky < kernel; ky++) {
    let iy = select(i32(oy * stride + ky) - padding, i32(ky), kind == 2u);
    if (iy < 0 || iy >= i32(ih)) { continue; }
    for (var kx = 0u; kx < kernel; kx++) {
      let ix = select(i32(ox * stride + kx) - padding, i32(kx), kind == 2u);
      if (ix < 0 || ix >= i32(iw)) { continue; }
      let source = ((batch * channels + channel) * ih + u32(iy)) * iw + u32(ix);
      if (kind == 0u) {
        if (input.values[source] > value) {
          value = input.values[source];
          selected = source - batch * channels * iw * ih;
        }
      } else {
        value += input.values[source];
      }
      count++;
    }
  }
  output.values[index] = select(value / f32(max(count, 1u)), value, kind == 0u);
  output3.values[index] = bitcast<f32>(selected);
}

@compute @workgroup_size(64)
fn pool_backward(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_size = p(0u);
  let iw = p(1u);
  let ih = p(2u);
  let channels = p(3u);
  let ow = p(4u);
  let oh = p(5u);
  let kernel = p(6u);
  let stride = p(7u);
  let padding = i32(p(8u));
  let kind = p(9u);
  let input_sample_size = channels * iw * ih;
  let output_sample_size = channels * ow * oh;
  let index = id.x;
  if (index < batch_size * output_sample_size) {
    output2.values[index] = clamp_delta(input.values[index]);
  }
  if (index >= batch_size * input_sample_size) { return; }
  let batch = index / input_sample_size;
  let local = index % input_sample_size;
  let channel = local / (iw * ih);
  let pixel = local % (iw * ih);
  let iy = pixel / iw;
  let ix = pixel % iw;
  var sum = 0.0;
  for (var oy = 0u; oy < oh; oy++) {
    for (var ox = 0u; ox < ow; ox++) {
      let output_index = (batch * channels + channel) * ow * oh + oy * ow + ox;
      if (kind == 0u) {
        if (bitcast<u32>(auxiliary.values[output_index]) == local) {
          sum += input.values[output_index];
        }
      } else {
        var inside = false;
        var count = 0u;
        for (var ky = 0u; ky < kernel; ky++) {
          let source_y = select(i32(oy * stride + ky) - padding, i32(ky), kind == 2u);
          if (source_y < 0 || source_y >= i32(ih)) { continue; }
          for (var kx = 0u; kx < kernel; kx++) {
            let source_x = select(i32(ox * stride + kx) - padding, i32(kx), kind == 2u);
            if (source_x < 0 || source_x >= i32(iw)) { continue; }
            count++;
            if (u32(source_y) == iy && u32(source_x) == ix) { inside = true; }
          }
        }
        if (inside) { sum += input.values[output_index] / f32(max(count, 1u)); }
      }
    }
  }
  output.values[index] = clamp_delta(sum);
}

@compute @workgroup_size(64)
fn optimizer_update(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let length = p(0u);
  if (index >= length) { return; }

  let optimizer_kind = p(1u);
  let learning_rate = pf(2u);
  let momentum = pf(3u);
  let decay = pf(4u);
  let beta1 = pf(5u);
  let beta2 = pf(6u);
  let epsilon = pf(7u);
  let beta1_correction = pf(8u);
  let beta2_correction = pf(9u);
  let gradient = output2.values[index] * pf(10u);
  let weight_decay = pf(11u);

  var parameter = output.values[index];
  if (weight_decay > 0.0) {
    parameter *= 1.0 - learning_rate * weight_decay;
  }
  if (optimizer_kind == 0u) {
    parameter -= learning_rate * gradient;
  } else if (optimizer_kind == 1u) {
    let velocity = momentum * output3.values[index] + gradient;
    output3.values[index] = velocity;
    parameter -= learning_rate * velocity;
  } else if (optimizer_kind == 2u) {
    let first_moment = beta1 * output3.values[index] + (1.0 - beta1) * gradient;
    let second_moment = beta2 * output4.values[index] + (1.0 - beta2) * gradient * gradient;
    output3.values[index] = first_moment;
    output4.values[index] = second_moment;
    parameter -= learning_rate *
      (first_moment / beta1_correction) /
      (sqrt(second_moment / beta2_correction) + epsilon);
  } else if (optimizer_kind == 3u) {
    let mean_square = decay * output4.values[index] + (1.0 - decay) * gradient * gradient;
    output4.values[index] = mean_square;
    parameter -= learning_rate * gradient / (sqrt(mean_square) + epsilon);
  } else if (optimizer_kind == 4u) {
    let accumulated_square = output4.values[index] + gradient * gradient;
    output4.values[index] = accumulated_square;
    parameter -= learning_rate * gradient / (sqrt(accumulated_square) + epsilon);
  } else {
    parameter -= learning_rate * gradient;
  }
  output.values[index] = parameter;
  output2.values[index] = 0.0;
}
