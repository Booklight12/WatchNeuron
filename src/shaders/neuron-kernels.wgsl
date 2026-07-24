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

fn clamp_delta(value: f32) -> f32 {
  return clamp(value, -1e4, 1e4);
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

fn activation(value: f32, kind: u32) -> f32 {
  if (kind == 0u) { return value; }
  if (kind == 1u) { return max(value, 0.0); }
  if (kind == 2u) { return select(value * 0.08, value, value >= 0.0); }
  if (kind == 3u) { return sigmoid(value); }
  if (kind == 4u) { return tanh(value); }
  if (kind == 6u) { return select(exp(value) - 1.0, value, value >= 0.0); }
  if (kind == 7u) { return 1.050700987 * select(1.673263242 * (exp(value) - 1.0), value, value >= 0.0); }
  if (kind == 8u) { return 0.5 * value * (1.0 + tanh(0.79788456 * (value + 0.044715 * value * value * value))); }
  if (kind == 9u) { return value * sigmoid(value); }
  if (kind == 10u) { return value * tanh(log(1.0 + exp(value))); }
  if (kind == 11u) { return log(1.0 + exp(value)); }
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
    let activated = tanh(value);
    return 1.0 - activated * activated;
  }
  if (kind == 6u) { return select(exp(value), 1.0, value >= 0.0); }
  if (kind == 7u) { return 1.050700987 * select(1.673263242 * exp(value), 1.0, value >= 0.0); }
  if (kind == 8u) {
    let inner = 0.79788456 * (value + 0.044715 * value * value * value);
    let t = tanh(inner);
    return 0.5 * (1.0 + t) + 0.5 * value * (1.0 - t * t) * 0.79788456 * (1.0 + 0.134145 * value * value);
  }
  if (kind == 9u) {
    let s = sigmoid(value);
    return s + value * s * (1.0 - s);
  }
  if (kind == 10u) {
    let softplus = log(1.0 + exp(value));
    let t = tanh(softplus);
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
      total += exp(item_sum - maximum);
    }
    value = exp(sum - maximum) / total;
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
      total += exp(input.values[batch * output_size + digit] - maximum);
    }
    probability = exp(input.values[index] - maximum) / total;
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
        loss -= select(log(1.0 - digit_probability), log(digit_probability), digit_target > 0.5) / f32(output_size);
      }
    } else {
      var maximum = -3.4e38;
      for (var digit = 0u; digit < output_size; digit++) {
        maximum = max(maximum, input.values[batch * output_size + digit]);
      }
      var total = 0.0;
      for (var digit = 0u; digit < output_size; digit++) {
        total += exp(input.values[batch * output_size + digit] - maximum);
      }
      let label_probability = exp(input.values[batch * output_size + label] - maximum) / total;
      loss = -log(max(label_probability, 0.00000001));
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
      total += exp(weights.values[batch * output_size + item] - maximum);
    }
    var dot = 0.0;
    for (var item = 0u; item < output_size; item++) {
      let probability = exp(weights.values[batch * output_size + item] - maximum) / total;
      dot += clamp_delta(input.values[batch * output_size + item]) * probability;
    }
    let probability = exp(weights.values[index] - maximum) / total;
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
      total += exp(conv_sum(batch, item_filter, item_pixel / ow, item_pixel % ow) - maximum);
    }
    output.values[index] = exp(sum - maximum) / total;
  } else {
    output.values[index] = activation(sum, p(8u));
  }
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
      total += exp(weights.values[batch * sample_size + item] - maximum);
    }
    var dot = 0.0;
    for (var item = 0u; item < sample_size; item++) {
      let probability = exp(weights.values[batch * sample_size + item] - maximum) / total;
      dot += clamp_delta(input.values[batch * sample_size + item]) * probability;
    }
    let probability = exp(weights.values[index] - maximum) / total;
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
