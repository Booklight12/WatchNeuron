const builtin = @import("builtin");

var selected_math_mode: i32 = 0;

pub export fn set_math_mode(mode: i32) void {
    selected_math_mode = if (mode == 0) 0 else 1;
}

pub export fn math_mode() i32 {
    return selected_math_mode;
}

fn fastExp(raw: f32) f32 {
    var value = raw;
    if (value < -12.0) return 0.000006;
    if (value > 12.0) value = 12.0;
    const bits: u32 = @intFromFloat(12102203.0 * value + 1064866805.0);
    return @bitCast(bits);
}

fn absolute(value: f32) f32 {
    return if (value < 0.0) -value else value;
}

fn fastTanh(value: f32) f32 {
    const sigmoid = 1.0 / (1.0 + fastExp(-2.0 * value));
    return sigmoid * 2.0 - 1.0;
}

fn fastLog(value: f32) f32 {
    var bits: u32 = @bitCast(value);
    const exponent: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127;
    bits = (bits & 0x007fffff) | 0x3f800000;
    const normalized: f32 = @bitCast(bits);
    const ratio = (normalized - 1.0) / (normalized + 1.0);
    const squared = ratio * ratio;
    var series = ratio;
    var power = ratio;
    power *= squared;
    series += power / 3.0;
    power *= squared;
    series += power / 5.0;
    power *= squared;
    series += power / 7.0;
    power *= squared;
    series += power / 9.0;
    return 2.0 * series + @as(f32, @floatFromInt(exponent)) * 0.69314718056;
}

fn fastSoftplus(value: f32) f32 {
    if (value > 12.0) return value;
    if (value < -12.0) return fastExp(value);
    return fastLog(1.0 + fastExp(value));
}

fn mathExp(value: f32) f32 {
    return if (selected_math_mode == 0) fastExp(value) else @exp(value);
}

fn mathLog(value: f32) f32 {
    return if (selected_math_mode == 0) fastLog(value) else @log(value);
}

fn mathTanh(value: f32) f32 {
    if (selected_math_mode == 0) return fastTanh(value);
    if (value > 10.0) return 1.0;
    if (value < -10.0) return -1.0;
    const exponential = @exp(2.0 * value);
    return (exponential - 1.0) / (exponential + 1.0);
}

fn mathSoftplus(value: f32) f32 {
    if (selected_math_mode == 0) return fastSoftplus(value);
    if (value > 20.0) return value;
    if (value < -20.0) return @exp(value);
    return @log(1.0 + @exp(value));
}

fn clampDelta(value: f32) f32 {
    if (value < -5.0) return -5.0;
    if (value > 5.0) return 5.0;
    return value;
}

fn activateScalar(value: f32, kind: i32) f32 {
    return switch (kind) {
        1 => if (value > 0.0) value else 0.0,
        2 => if (value > 0.0) value else value * 0.08,
        3 => 1.0 / (1.0 + mathExp(-value)),
        4 => mathTanh(value),
        6 => if (value >= 0.0) value else mathExp(value) - 1.0,
        7 => 1.05070098736 * (if (value >= 0.0) value else 1.67326324235 * (mathExp(value) - 1.0)),
        8 => blk: {
            const cubic = value * value * value;
            const curve = 0.7978845608 * (value + 0.044715 * cubic);
            break :blk 0.5 * value * (1.0 + mathTanh(curve));
        },
        9 => blk: {
            const gate = 1.0 / (1.0 + mathExp(-value));
            break :blk value * gate;
        },
        10 => value * mathTanh(mathSoftplus(value)),
        11 => mathSoftplus(value),
        12 => value / (1.0 + absolute(value)),
        13 => blk: {
            var result = value * 0.2 + 0.5;
            if (result < 0.0) result = 0.0;
            if (result > 1.0) result = 1.0;
            break :blk result;
        },
        14 => if (value < -1.0) -1.0 else if (value > 1.0) 1.0 else value,
        15 => if (value < 0.0) 0.0 else if (value > 6.0) 6.0 else value,
        else => value,
    };
}

fn activateBuffer(values: [*]f32, length: usize, kind: i32) void {
    if (kind == 5) {
        if (length == 0) return;
        var maximum = values[0];
        var index: usize = 1;
        while (index < length) : (index += 1) {
            if (values[index] > maximum) maximum = values[index];
        }
        var total: f32 = 0.0;
        index = 0;
        while (index < length) : (index += 1) {
            values[index] = mathExp(values[index] - maximum);
            total += values[index];
        }
        if (total > 0.0) {
            index = 0;
            while (index < length) : (index += 1) values[index] /= total;
        }
        return;
    }
    var index: usize = 0;
    while (index < length) : (index += 1) {
        values[index] = activateScalar(values[index], kind);
    }
}

fn activationDerivative(input: f32, output: f32, kind: i32) f32 {
    return switch (kind) {
        1 => if (input > 0.0) 1.0 else 0.0,
        2 => if (input >= 0.0) 1.0 else 0.08,
        3 => output * (1.0 - output),
        4 => 1.0 - output * output,
        6 => if (input >= 0.0) 1.0 else output + 1.0,
        7 => if (input >= 0.0) 1.05070098736 else output + 1.05070098736 * 1.67326324235,
        8 => blk: {
            const squared = input * input;
            const curve = 0.7978845608 * (input + 0.044715 * input * squared);
            const tangent = mathTanh(curve);
            break :blk 0.5 * (1.0 + tangent) + 0.5 * input * (1.0 - tangent * tangent) * 0.7978845608 * (1.0 + 0.134145 * squared);
        },
        9 => blk: {
            const gate = 1.0 / (1.0 + mathExp(-input));
            break :blk gate + input * gate * (1.0 - gate);
        },
        10 => blk: {
            const tangent = mathTanh(mathSoftplus(input));
            const gate = 1.0 / (1.0 + mathExp(-input));
            break :blk tangent + input * gate * (1.0 - tangent * tangent);
        },
        11 => 1.0 / (1.0 + mathExp(-input)),
        12 => blk: {
            const divisor = 1.0 + absolute(input);
            break :blk 1.0 / (divisor * divisor);
        },
        13 => if (input > -2.5 and input < 2.5) 0.2 else 0.0,
        14 => if (input > -1.0 and input < 1.0) 1.0 else 0.0,
        15 => if (input > 0.0 and input < 6.0) 1.0 else 0.0,
        else => 1.0,
    };
}

fn applyActivationDerivative(gradient: [*]f32, output: [*]f32, preactivation: [*]f32, length: usize, kind: i32) void {
    if (kind == 5) {
        var dot: f32 = 0.0;
        var index: usize = 0;
        while (index < length) : (index += 1) dot += gradient[index] * output[index];
        index = 0;
        while (index < length) : (index += 1) {
            gradient[index] = output[index] * (gradient[index] - dot);
        }
        return;
    }
    var index: usize = 0;
    while (index < length) : (index += 1) {
        gradient[index] *= activationDerivative(preactivation[index], output[index], kind);
    }
}

fn applyTrainingActivationDerivative(
    gradient: [*]f32,
    preactivation: [*]f32,
    dropout_mask: [*]f32,
    length: usize,
    kind: i32,
) void {
    if (kind == 5) {
        var maximum = preactivation[0];
        var index: usize = 1;
        while (index < length) : (index += 1) {
            if (preactivation[index] > maximum) maximum = preactivation[index];
        }
        var total: f32 = 0.0;
        index = 0;
        while (index < length) : (index += 1) total += mathExp(preactivation[index] - maximum);
        var dot: f32 = 0.0;
        index = 0;
        while (index < length) : (index += 1) {
            const probability = mathExp(preactivation[index] - maximum) / total;
            gradient[index] *= dropout_mask[index];
            dot += gradient[index] * probability;
        }
        index = 0;
        while (index < length) : (index += 1) {
            const probability = mathExp(preactivation[index] - maximum) / total;
            gradient[index] = probability * (gradient[index] - dot);
        }
        return;
    }
    var index: usize = 0;
    while (index < length) : (index += 1) {
        const activated = activateScalar(preactivation[index], kind);
        gradient[index] *= dropout_mask[index] * activationDerivative(preactivation[index], activated, kind);
    }
}

fn dropoutRandom(seed: u32, layer: usize, index: usize) f32 {
    var value = seed ^ (@as(u32, @intCast(layer + 1)) *% 0x9e3779b9) ^ (@as(u32, @intCast(index + 1)) *% 0x85ebca6b);
    value ^= value >> 16;
    value *%= 0x7feb352d;
    value ^= value >> 15;
    value *%= 0x846ca68b;
    value ^= value >> 16;
    return @as(f32, @floatFromInt(value & 0x00ffffff)) / 16777216.0;
}

fn applyDropout(values: [*]f32, mask: [*]f32, length: usize, rate: f32, seed: u32, layer: usize) void {
    if (rate <= 0.0) {
        var index: usize = 0;
        while (index < length) : (index += 1) mask[index] = 1.0;
        return;
    }
    const keep = 1.0 - rate;
    const scale = 1.0 / keep;
    var index: usize = 0;
    while (index < length) : (index += 1) {
        const multiplier: f32 = if (dropoutRandom(seed, layer, index) < rate) 0.0 else scale;
        mask[index] = multiplier;
        values[index] *= multiplier;
    }
}

fn f32Ptr(address: u32) [*]f32 {
    return @ptrFromInt(address);
}

fn u16Ptr(address: u32) [*]u16 {
    return @ptrFromInt(address);
}

fn u32Ptr(address: u32) [*]u32 {
    return @ptrFromInt(address);
}

fn i32Ptr(address: u32) [*]i32 {
    return @ptrFromInt(address);
}

pub export fn simd_enabled() i32 {
    return if (builtin.cpu.has(.wasm, .simd128)) 1 else 0;
}

const F32x4 = @Vector(4, f32);

fn loadF32x4(values: [*]f32, index: usize) F32x4 {
    return @as(*align(1) const F32x4, @ptrCast(values + index)).*;
}

fn storeF32x4(values: [*]f32, index: usize, vector: F32x4) void {
    @as(*align(1) F32x4, @ptrCast(values + index)).* = vector;
}

fn dotProduct(left: [*]f32, right: [*]f32, length: usize) f32 {
    var lanes: F32x4 = @splat(0.0);
    var index: usize = 0;
    while (index + 4 <= length) : (index += 4) {
        lanes += loadF32x4(left, index) * loadF32x4(right, index);
    }
    var sum: f32 = @reduce(.Add, lanes);
    while (index < length) : (index += 1) sum += left[index] * right[index];
    return sum;
}

fn fillZero(values: [*]f32, length: usize) void {
    const zero: F32x4 = @splat(0.0);
    var index: usize = 0;
    while (index + 4 <= length) : (index += 4) storeF32x4(values, index, zero);
    while (index < length) : (index += 1) values[index] = 0.0;
}

fn poolingOutputExtent(input_size: usize, kernel_size: usize, stride: usize, padding: isize) usize {
    const padded_size = @as(isize, @intCast(input_size)) + padding * 2;
    const signed_kernel = @as(isize, @intCast(kernel_size));
    if (padded_size < signed_kernel) return 1;
    return @intCast(@divTrunc(padded_size - signed_kernel, @as(isize, @intCast(stride))) + 1);
}

fn accumulateScaled(target: [*]f32, source: [*]f32, length: usize, scale: f32) void {
    const scale_lanes: F32x4 = @splat(scale);
    var index: usize = 0;
    while (index + 4 <= length) : (index += 4) {
        storeF32x4(target, index, loadF32x4(target, index) + loadF32x4(source, index) * scale_lanes);
    }
    while (index < length) : (index += 1) target[index] += source[index] * scale;
}

pub export fn matvec(input_ptr: u32, weights_ptr: u32, bias_ptr: u32, output_ptr: u32, input_size_raw: i32, output_size_raw: i32) void {
    const input_size: usize = @intCast(input_size_raw);
    const output_size: usize = @intCast(output_size_raw);
    const input = f32Ptr(input_ptr);
    const weights = f32Ptr(weights_ptr);
    const bias = f32Ptr(bias_ptr);
    const output = f32Ptr(output_ptr);
    var row: usize = 0;
    while (row < output_size) : (row += 1) {
        const offset = row * input_size;
        output[row] = bias[row] + dotProduct(weights + offset, input, input_size);
    }
}

pub export fn activate(values_ptr: u32, length_raw: i32, kind: i32) void {
    activateBuffer(f32Ptr(values_ptr), @intCast(length_raw), kind);
}

pub export fn conv2d_forward(
    input_ptr: u32,
    weights_ptr: u32,
    biases_ptr: u32,
    output_ptr: u32,
    preactivation_ptr: u32,
    input_width_raw: i32,
    input_height_raw: i32,
    input_channels_raw: i32,
    filters_raw: i32,
    kernel_size_raw: i32,
    stride_raw: i32,
    padding_raw: i32,
    activation_kind: i32,
) void {
    const input_width: usize = @intCast(input_width_raw);
    const input_height: usize = @intCast(input_height_raw);
    const input_channels: usize = @intCast(input_channels_raw);
    const filters: usize = @intCast(filters_raw);
    const kernel_size: usize = @intCast(kernel_size_raw);
    const stride: usize = @intCast(stride_raw);
    const padding: isize = @intCast(padding_raw);
    const output_width = (input_width + @as(usize, @intCast(padding * 2)) - kernel_size) / stride + 1;
    const output_height = (input_height + @as(usize, @intCast(padding * 2)) - kernel_size) / stride + 1;
    const input = f32Ptr(input_ptr);
    const weights = f32Ptr(weights_ptr);
    const biases = f32Ptr(biases_ptr);
    const output = f32Ptr(output_ptr);
    const preactivation = f32Ptr(preactivation_ptr);

    var filter: usize = 0;
    while (filter < filters) : (filter += 1) {
        var output_y: usize = 0;
        while (output_y < output_height) : (output_y += 1) {
            var output_x: usize = 0;
            while (output_x < output_width) : (output_x += 1) {
                var sum = biases[filter];
                var channel: usize = 0;
                while (channel < input_channels) : (channel += 1) {
                    var kernel_y: usize = 0;
                    while (kernel_y < kernel_size) : (kernel_y += 1) {
                        const input_y = @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                        if (input_y < 0 or input_y >= @as(isize, @intCast(input_height))) continue;
                        var kernel_x: usize = 0;
                        while (kernel_x < kernel_size) : (kernel_x += 1) {
                            const input_x = @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                            if (input_x < 0 or input_x >= @as(isize, @intCast(input_width))) continue;
                            const input_index = channel * input_width * input_height + @as(usize, @intCast(input_y)) * input_width + @as(usize, @intCast(input_x));
                            const weight_index = ((filter * input_channels + channel) * kernel_size + kernel_y) * kernel_size + kernel_x;
                            sum += input[input_index] * weights[weight_index];
                        }
                    }
                }
                const output_index = filter * output_width * output_height + output_y * output_width + output_x;
                preactivation[output_index] = sum;
                output[output_index] = sum;
            }
        }
    }
    activateBuffer(output, output_width * output_height * filters, activation_kind);
}

pub export fn pool2d_forward(
    input_ptr: u32,
    output_ptr: u32,
    index_ptr: u32,
    input_width_raw: i32,
    input_height_raw: i32,
    channels_raw: i32,
    kernel_size_raw: i32,
    stride_raw: i32,
    padding_raw: i32,
    pooling_kind: i32,
) void {
    const input_width: usize = @intCast(input_width_raw);
    const input_height: usize = @intCast(input_height_raw);
    const channels: usize = @intCast(channels_raw);
    const kernel_size: usize = if (pooling_kind == 2) @max(input_width, input_height) else @intCast(@max(kernel_size_raw, 1));
    const stride: usize = if (pooling_kind == 2) 1 else @intCast(@max(stride_raw, 1));
    const padding: isize = if (pooling_kind == 2) 0 else @intCast(@max(padding_raw, 0));
    const output_width: usize = if (pooling_kind == 2) 1 else poolingOutputExtent(input_width, kernel_size, stride, padding);
    const output_height: usize = if (pooling_kind == 2) 1 else poolingOutputExtent(input_height, kernel_size, stride, padding);
    const input = f32Ptr(input_ptr);
    const output = f32Ptr(output_ptr);
    const indices = u32Ptr(index_ptr);

    var channel: usize = 0;
    while (channel < channels) : (channel += 1) {
        var output_y: usize = 0;
        while (output_y < output_height) : (output_y += 1) {
            var output_x: usize = 0;
            while (output_x < output_width) : (output_x += 1) {
                var value: f32 = if (pooling_kind == 0) -3.4028235e38 else 0.0;
                var selected: u32 = 0xffffffff;
                var count: usize = 0;
                var kernel_y: usize = 0;
                while (kernel_y < kernel_size) : (kernel_y += 1) {
                    const input_y: isize = if (pooling_kind == 2) @intCast(kernel_y) else @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                    if (input_y < 0 or input_y >= @as(isize, @intCast(input_height))) continue;
                    var kernel_x: usize = 0;
                    while (kernel_x < kernel_size) : (kernel_x += 1) {
                        const input_x: isize = if (pooling_kind == 2) @intCast(kernel_x) else @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                        if (input_x < 0 or input_x >= @as(isize, @intCast(input_width))) continue;
                        const source_index = channel * input_width * input_height + @as(usize, @intCast(input_y)) * input_width + @as(usize, @intCast(input_x));
                        if (pooling_kind == 0) {
                            if (input[source_index] > value) {
                                value = input[source_index];
                                selected = @intCast(source_index);
                            }
                        } else value += input[source_index];
                        count += 1;
                    }
                }
                const output_index = channel * output_width * output_height + output_y * output_width + output_x;
                output[output_index] = if (pooling_kind == 0)
                    (if (count > 0) value else 0.0)
                else
                    value / @as(f32, @floatFromInt(@max(count, 1)));
                indices[output_index] = selected;
            }
        }
    }
}

pub export fn pool2d_backward(
    output_gradient_ptr: u32,
    input_gradient_ptr: u32,
    index_ptr: u32,
    input_width_raw: i32,
    input_height_raw: i32,
    channels_raw: i32,
    kernel_size_raw: i32,
    stride_raw: i32,
    padding_raw: i32,
    pooling_kind: i32,
) void {
    const input_width: usize = @intCast(input_width_raw);
    const input_height: usize = @intCast(input_height_raw);
    const channels: usize = @intCast(channels_raw);
    const kernel_size: usize = if (pooling_kind == 2) @max(input_width, input_height) else @intCast(@max(kernel_size_raw, 1));
    const stride: usize = if (pooling_kind == 2) 1 else @intCast(@max(stride_raw, 1));
    const padding: isize = if (pooling_kind == 2) 0 else @intCast(@max(padding_raw, 0));
    const output_width: usize = if (pooling_kind == 2) 1 else poolingOutputExtent(input_width, kernel_size, stride, padding);
    const output_height: usize = if (pooling_kind == 2) 1 else poolingOutputExtent(input_height, kernel_size, stride, padding);
    const output_gradient = f32Ptr(output_gradient_ptr);
    const input_gradient = f32Ptr(input_gradient_ptr);
    const indices = u32Ptr(index_ptr);
    fillZero(input_gradient, input_width * input_height * channels);

    var channel: usize = 0;
    while (channel < channels) : (channel += 1) {
        var output_y: usize = 0;
        while (output_y < output_height) : (output_y += 1) {
            var output_x: usize = 0;
            while (output_x < output_width) : (output_x += 1) {
                const output_index = channel * output_width * output_height + output_y * output_width + output_x;
                const gradient = clampDelta(output_gradient[output_index]);
                if (pooling_kind == 0) {
                    if (indices[output_index] != 0xffffffff) {
                        input_gradient[indices[output_index]] += gradient;
                    }
                    continue;
                }
                var count: usize = 0;
                var kernel_y: usize = 0;
                while (kernel_y < kernel_size) : (kernel_y += 1) {
                    const input_y: isize = if (pooling_kind == 2) @intCast(kernel_y) else @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                    if (input_y < 0 or input_y >= @as(isize, @intCast(input_height))) continue;
                    var kernel_x: usize = 0;
                    while (kernel_x < kernel_size) : (kernel_x += 1) {
                        const input_x: isize = if (pooling_kind == 2) @intCast(kernel_x) else @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                        if (input_x >= 0 and input_x < @as(isize, @intCast(input_width))) count += 1;
                    }
                }
                const share = gradient / @as(f32, @floatFromInt(@max(count, 1)));
                kernel_y = 0;
                while (kernel_y < kernel_size) : (kernel_y += 1) {
                    const input_y: isize = if (pooling_kind == 2) @intCast(kernel_y) else @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                    if (input_y < 0 or input_y >= @as(isize, @intCast(input_height))) continue;
                    var kernel_x: usize = 0;
                    while (kernel_x < kernel_size) : (kernel_x += 1) {
                        const input_x: isize = if (pooling_kind == 2) @intCast(kernel_x) else @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                        if (input_x < 0 or input_x >= @as(isize, @intCast(input_width))) continue;
                        const source_index = channel * input_width * input_height + @as(usize, @intCast(input_y)) * input_width + @as(usize, @intCast(input_x));
                        input_gradient[source_index] += share;
                    }
                }
            }
        }
    }
}

fn forwardSparseInternal(
    layer_count: usize,
    sample_indices: [*]u16,
    sample_values: [*]f32,
    active_count: usize,
    input_sizes: [*]i32,
    output_sizes: [*]i32,
    activation_kinds: [*]i32,
    weight_pointers: [*]u32,
    bias_pointers: [*]u32,
    activation_pointers: [*]u32,
    preactivation_pointers: [*]u32,
    first_input_dense: i32,
    finalize_softmax: bool,
) void {
    var layer_index: usize = 0;
    while (layer_index < layer_count) : (layer_index += 1) {
        const input_size: usize = @intCast(input_sizes[layer_index]);
        const output_size: usize = @intCast(output_sizes[layer_index]);
        const weights = f32Ptr(weight_pointers[layer_index]);
        const biases = f32Ptr(bias_pointers[layer_index]);
        const outputs = f32Ptr(activation_pointers[layer_index]);
        const preactivations = f32Ptr(preactivation_pointers[layer_index]);
        var output: usize = 0;
        while (output < output_size) : (output += 1) {
            var sum = biases[output];
            const offset = output * input_size;
            if (layer_index == 0) {
                if (first_input_dense != 0) {
                    sum += dotProduct(weights + offset, sample_values, input_size);
                } else {
                    var pixel: usize = 0;
                    while (pixel < active_count) : (pixel += 1) {
                        sum += weights[offset + sample_indices[pixel]] * sample_values[pixel];
                    }
                }
            } else {
                const previous = f32Ptr(activation_pointers[layer_index - 1]);
                sum += dotProduct(weights + offset, previous, input_size);
            }
            preactivations[output] = sum;
            outputs[output] = sum;
        }
        activateBuffer(outputs, output_size, activation_kinds[layer_index]);
    }
    if (finalize_softmax) {
        const last = layer_count - 1;
        activateBuffer(f32Ptr(activation_pointers[last]), @intCast(output_sizes[last]), 5);
    }
}

fn forwardSparseTrainingInternal(
    layer_count: usize,
    sample_indices: [*]u16,
    sample_values: [*]f32,
    active_count: usize,
    input_sizes: [*]i32,
    output_sizes: [*]i32,
    activation_kinds: [*]i32,
    weight_pointers: [*]u32,
    bias_pointers: [*]u32,
    activation_pointers: [*]u32,
    preactivation_pointers: [*]u32,
    dropout_rates: [*]f32,
    dropout_mask_pointers: [*]u32,
    dropout_seed: u32,
    first_input_dense: i32,
    finalize_softmax: bool,
) void {
    var layer_index: usize = 0;
    while (layer_index < layer_count) : (layer_index += 1) {
        const input_size: usize = @intCast(input_sizes[layer_index]);
        const output_size: usize = @intCast(output_sizes[layer_index]);
        const weights = f32Ptr(weight_pointers[layer_index]);
        const biases = f32Ptr(bias_pointers[layer_index]);
        const outputs = f32Ptr(activation_pointers[layer_index]);
        const preactivations = f32Ptr(preactivation_pointers[layer_index]);
        var output: usize = 0;
        while (output < output_size) : (output += 1) {
            var sum = biases[output];
            const offset = output * input_size;
            if (layer_index == 0) {
                if (first_input_dense != 0) {
                    sum += dotProduct(weights + offset, sample_values, input_size);
                } else {
                    var input: usize = 0;
                    while (input < active_count) : (input += 1) {
                        sum += weights[offset + sample_indices[input]] * sample_values[input];
                    }
                }
            } else {
                const previous = f32Ptr(activation_pointers[layer_index - 1]);
                sum += dotProduct(weights + offset, previous, input_size);
            }
            preactivations[output] = sum;
            outputs[output] = sum;
        }
        activateBuffer(outputs, output_size, activation_kinds[layer_index]);
        if (layer_index + 1 < layer_count or !finalize_softmax) {
            applyDropout(
                outputs,
                f32Ptr(dropout_mask_pointers[layer_index]),
                output_size,
                dropout_rates[layer_index],
                dropout_seed,
                layer_index,
            );
        }
    }
    if (finalize_softmax) {
        const last = layer_count - 1;
        activateBuffer(f32Ptr(activation_pointers[last]), @intCast(output_sizes[last]), 5);
    }
}

pub export fn forward_sparse(
    layer_count_raw: i32,
    sample_indices_ptr: u32,
    sample_values_ptr: u32,
    active_count_raw: i32,
    input_sizes_ptr: u32,
    output_sizes_ptr: u32,
    activation_kinds_ptr: u32,
    weight_pointers_ptr: u32,
    bias_pointers_ptr: u32,
    activation_pointers_ptr: u32,
    preactivation_pointers_ptr: u32,
    first_input_dense: i32,
) void {
    forwardSparseInternal(
        @intCast(layer_count_raw),
        u16Ptr(sample_indices_ptr),
        f32Ptr(sample_values_ptr),
        @intCast(active_count_raw),
        i32Ptr(input_sizes_ptr),
        i32Ptr(output_sizes_ptr),
        i32Ptr(activation_kinds_ptr),
        u32Ptr(weight_pointers_ptr),
        u32Ptr(bias_pointers_ptr),
        u32Ptr(activation_pointers_ptr),
        u32Ptr(preactivation_pointers_ptr),
        first_input_dense,
        true,
    );
}

pub export fn forward_dense_block(
    layer_count_raw: i32,
    sample_indices_ptr: u32,
    sample_values_ptr: u32,
    active_count_raw: i32,
    input_sizes_ptr: u32,
    output_sizes_ptr: u32,
    activation_kinds_ptr: u32,
    weight_pointers_ptr: u32,
    bias_pointers_ptr: u32,
    activation_pointers_ptr: u32,
    preactivation_pointers_ptr: u32,
    first_input_dense: i32,
) void {
    forwardSparseInternal(
        @intCast(layer_count_raw),
        u16Ptr(sample_indices_ptr),
        f32Ptr(sample_values_ptr),
        @intCast(active_count_raw),
        i32Ptr(input_sizes_ptr),
        i32Ptr(output_sizes_ptr),
        i32Ptr(activation_kinds_ptr),
        u32Ptr(weight_pointers_ptr),
        u32Ptr(bias_pointers_ptr),
        u32Ptr(activation_pointers_ptr),
        u32Ptr(preactivation_pointers_ptr),
        first_input_dense,
        false,
    );
}

pub export fn forward_dense_training(
    layer_count_raw: i32,
    sample_indices_ptr: u32,
    sample_values_ptr: u32,
    active_count_raw: i32,
    input_sizes_ptr: u32,
    output_sizes_ptr: u32,
    activation_kinds_ptr: u32,
    weight_pointers_ptr: u32,
    bias_pointers_ptr: u32,
    activation_pointers_ptr: u32,
    preactivation_pointers_ptr: u32,
    dropout_rates_ptr: u32,
    dropout_mask_pointers_ptr: u32,
    dropout_seed: u32,
    first_input_dense: i32,
) void {
    forwardSparseTrainingInternal(
        @intCast(layer_count_raw),
        u16Ptr(sample_indices_ptr),
        f32Ptr(sample_values_ptr),
        @intCast(active_count_raw),
        i32Ptr(input_sizes_ptr),
        i32Ptr(output_sizes_ptr),
        i32Ptr(activation_kinds_ptr),
        u32Ptr(weight_pointers_ptr),
        u32Ptr(bias_pointers_ptr),
        u32Ptr(activation_pointers_ptr),
        u32Ptr(preactivation_pointers_ptr),
        f32Ptr(dropout_rates_ptr),
        u32Ptr(dropout_mask_pointers_ptr),
        dropout_seed,
        first_input_dense,
        false,
    );
}

fn updateParameter(
    parameters: [*]f32,
    index: usize,
    gradient: f32,
    learning_rate: f32,
    optimizer_kind: i32,
    momentum: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    first: [*]f32,
    second: [*]f32,
) void {
    if (optimizer_kind == 0) {
        parameters[index] -= learning_rate * gradient;
        return;
    }
    if (optimizer_kind == 1) {
        const velocity = momentum * first[index] + gradient;
        first[index] = velocity;
        parameters[index] -= learning_rate * velocity;
        return;
    }
    if (optimizer_kind == 2) {
        const first_moment = beta1 * first[index] + (1.0 - beta1) * gradient;
        const second_moment = beta2 * second[index] + (1.0 - beta2) * gradient * gradient;
        first[index] = first_moment;
        second[index] = second_moment;
        const corrected_first = first_moment / beta1_correction;
        const corrected_second = second_moment / beta2_correction;
        parameters[index] -= learning_rate * corrected_first / (@sqrt(corrected_second) + epsilon);
        return;
    }
    if (optimizer_kind == 3) {
        const mean_square = decay * second[index] + (1.0 - decay) * gradient * gradient;
        second[index] = mean_square;
        parameters[index] -= learning_rate * gradient / (@sqrt(mean_square) + epsilon);
        return;
    }
    if (optimizer_kind == 4) {
        const accumulated_square = second[index] + gradient * gradient;
        second[index] = accumulated_square;
        parameters[index] -= learning_rate * gradient / (@sqrt(accumulated_square) + epsilon);
        return;
    }
    parameters[index] -= learning_rate * gradient;
}

fn updateParameterF32x4(
    parameters: [*]f32,
    index: usize,
    gradient: F32x4,
    learning_rate: f32,
    optimizer_kind: i32,
    momentum: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    first: [*]f32,
    second: [*]f32,
) void {
    const learning_rate_lanes: F32x4 = @splat(learning_rate);
    var parameter_lanes = loadF32x4(parameters, index);
    if (optimizer_kind == 0) {
        storeF32x4(parameters, index, parameter_lanes - learning_rate_lanes * gradient);
        return;
    }
    if (optimizer_kind == 1) {
        const momentum_lanes: F32x4 = @splat(momentum);
        const velocity = momentum_lanes * loadF32x4(first, index) + gradient;
        storeF32x4(first, index, velocity);
        storeF32x4(parameters, index, parameter_lanes - learning_rate_lanes * velocity);
        return;
    }
    const epsilon_lanes: F32x4 = @splat(epsilon);
    if (optimizer_kind == 2) {
        const beta1_lanes: F32x4 = @splat(beta1);
        const beta2_lanes: F32x4 = @splat(beta2);
        const one_minus_beta1: F32x4 = @splat(1.0 - beta1);
        const one_minus_beta2: F32x4 = @splat(1.0 - beta2);
        const first_moment = beta1_lanes * loadF32x4(first, index) + one_minus_beta1 * gradient;
        const second_moment = beta2_lanes * loadF32x4(second, index) + one_minus_beta2 * gradient * gradient;
        storeF32x4(first, index, first_moment);
        storeF32x4(second, index, second_moment);
        const first_correction: F32x4 = @splat(beta1_correction);
        const second_correction: F32x4 = @splat(beta2_correction);
        const corrected_first = first_moment / first_correction;
        const corrected_second = second_moment / second_correction;
        parameter_lanes -= learning_rate_lanes * corrected_first / (@sqrt(corrected_second) + epsilon_lanes);
        storeF32x4(parameters, index, parameter_lanes);
        return;
    }
    if (optimizer_kind == 3) {
        const decay_lanes: F32x4 = @splat(decay);
        const one_minus_decay: F32x4 = @splat(1.0 - decay);
        const mean_square = decay_lanes * loadF32x4(second, index) + one_minus_decay * gradient * gradient;
        storeF32x4(second, index, mean_square);
        parameter_lanes -= learning_rate_lanes * gradient / (@sqrt(mean_square) + epsilon_lanes);
        storeF32x4(parameters, index, parameter_lanes);
        return;
    }
    if (optimizer_kind == 4) {
        const accumulated_square = loadF32x4(second, index) + gradient * gradient;
        storeF32x4(second, index, accumulated_square);
        parameter_lanes -= learning_rate_lanes * gradient / (@sqrt(accumulated_square) + epsilon_lanes);
        storeF32x4(parameters, index, parameter_lanes);
        return;
    }
    storeF32x4(parameters, index, parameter_lanes - learning_rate_lanes * gradient);
}

pub export fn apply_optimizer(
    parameters_ptr: u32,
    gradients_ptr: u32,
    first_ptr: u32,
    second_ptr: u32,
    length_raw: i32,
    optimizer_kind: i32,
    learning_rate: f32,
    momentum: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    gradient_scale: f32,
    weight_decay: f32,
) void {
    const parameters = f32Ptr(parameters_ptr);
    const gradients = f32Ptr(gradients_ptr);
    const first = f32Ptr(first_ptr);
    const second = f32Ptr(second_ptr);
    const length: usize = @intCast(length_raw);
    const shrink = 1.0 - learning_rate * weight_decay;
    const scale_lanes: F32x4 = @splat(gradient_scale);
    const shrink_lanes: F32x4 = @splat(shrink);
    var index: usize = 0;
    while (index + 4 <= length) : (index += 4) {
        if (weight_decay > 0.0) storeF32x4(parameters, index, loadF32x4(parameters, index) * shrink_lanes);
        updateParameterF32x4(parameters, index, loadF32x4(gradients, index) * scale_lanes, learning_rate, optimizer_kind, momentum, decay, beta1, beta2, epsilon, beta1_correction, beta2_correction, first, second);
        storeF32x4(gradients, index, @splat(0.0));
    }
    while (index < length) : (index += 1) {
        if (weight_decay > 0.0) parameters[index] *= shrink;
        updateParameter(parameters, index, gradients[index] * gradient_scale, learning_rate, optimizer_kind, momentum, decay, beta1, beta2, epsilon, beta1_correction, beta2_correction, first, second);
        gradients[index] = 0.0;
    }
}

pub export fn conv2d_train(
    input_ptr: u32,
    weights_ptr: u32,
    biases_ptr: u32,
    preactivation_ptr: u32,
    output_gradient_ptr: u32,
    input_gradient_ptr: u32,
    delta_ptr: u32,
    weight_gradient_ptr: u32,
    bias_gradient_ptr: u32,
    input_width_raw: i32,
    input_height_raw: i32,
    input_channels_raw: i32,
    filters_raw: i32,
    kernel_size_raw: i32,
    stride_raw: i32,
    padding_raw: i32,
    activation_kind: i32,
    trainable_raw: i32,
) void {
    const input_width: usize = @intCast(input_width_raw);
    const input_height: usize = @intCast(input_height_raw);
    const input_channels: usize = @intCast(input_channels_raw);
    const filters: usize = @intCast(filters_raw);
    const kernel_size: usize = @intCast(kernel_size_raw);
    const stride: usize = @intCast(stride_raw);
    const padding: isize = @intCast(padding_raw);
    const trainable = trainable_raw != 0;
    const output_width = (input_width + @as(usize, @intCast(padding * 2)) - kernel_size) / stride + 1;
    const output_height = (input_height + @as(usize, @intCast(padding * 2)) - kernel_size) / stride + 1;
    const input = f32Ptr(input_ptr);
    const weights = f32Ptr(weights_ptr);
    _ = biases_ptr;
    const preactivation = f32Ptr(preactivation_ptr);
    const output_gradient = f32Ptr(output_gradient_ptr);
    const input_gradient = f32Ptr(input_gradient_ptr);
    const delta = f32Ptr(delta_ptr);
    const weight_gradients = f32Ptr(weight_gradient_ptr);
    const bias_gradients = f32Ptr(bias_gradient_ptr);

    var input_index: usize = 0;
    while (input_index < input_width * input_height * input_channels) : (input_index += 1) input_gradient[input_index] = 0.0;

    const convolution_output_size = output_width * output_height * filters;
    if (activation_kind == 5) {
        var maximum = preactivation[0];
        var output_index: usize = 1;
        while (output_index < convolution_output_size) : (output_index += 1) {
            if (preactivation[output_index] > maximum) maximum = preactivation[output_index];
        }
        var total: f32 = 0.0;
        output_index = 0;
        while (output_index < convolution_output_size) : (output_index += 1) total += mathExp(preactivation[output_index] - maximum);
        var dot: f32 = 0.0;
        output_index = 0;
        while (output_index < convolution_output_size) : (output_index += 1) {
            const probability = mathExp(preactivation[output_index] - maximum) / total;
            dot += clampDelta(output_gradient[output_index]) * probability;
        }
        output_index = 0;
        while (output_index < convolution_output_size) : (output_index += 1) {
            const probability = mathExp(preactivation[output_index] - maximum) / total;
            delta[output_index] = probability * (clampDelta(output_gradient[output_index]) - dot);
        }
    }

    var filter: usize = 0;
    while (filter < filters) : (filter += 1) {
        var bias_gradient: f32 = 0.0;
        var output_y: usize = 0;
        while (output_y < output_height) : (output_y += 1) {
            var output_x: usize = 0;
            while (output_x < output_width) : (output_x += 1) {
                const output_index = filter * output_width * output_height + output_y * output_width + output_x;
                const activated = activateScalar(preactivation[output_index], activation_kind);
                const gradient = if (activation_kind == 5)
                    delta[output_index]
                else
                    clampDelta(output_gradient[output_index]) * activationDerivative(preactivation[output_index], activated, activation_kind);
                if (activation_kind != 5) delta[output_index] = gradient;
                if (trainable) bias_gradient += gradient;
                var channel: usize = 0;
                while (channel < input_channels) : (channel += 1) {
                    var kernel_y: usize = 0;
                    while (kernel_y < kernel_size) : (kernel_y += 1) {
                        const input_y = @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                        if (input_y < 0 or input_y >= @as(isize, @intCast(input_height))) continue;
                        var kernel_x: usize = 0;
                        while (kernel_x < kernel_size) : (kernel_x += 1) {
                            const input_x = @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                            if (input_x < 0 or input_x >= @as(isize, @intCast(input_width))) continue;
                            const source_index = channel * input_width * input_height + @as(usize, @intCast(input_y)) * input_width + @as(usize, @intCast(input_x));
                            const weight_index = ((filter * input_channels + channel) * kernel_size + kernel_y) * kernel_size + kernel_x;
                            input_gradient[source_index] += weights[weight_index] * gradient;
                        }
                    }
                }
            }
        }

        if (!trainable) continue;

        var channel: usize = 0;
        while (channel < input_channels) : (channel += 1) {
            var kernel_y: usize = 0;
            while (kernel_y < kernel_size) : (kernel_y += 1) {
                var kernel_x: usize = 0;
                while (kernel_x < kernel_size) : (kernel_x += 1) {
                    var kernel_gradient: f32 = 0.0;
                    output_y = 0;
                    while (output_y < output_height) : (output_y += 1) {
                        var output_x: usize = 0;
                        while (output_x < output_width) : (output_x += 1) {
                            const input_y = @as(isize, @intCast(output_y * stride + kernel_y)) - padding;
                            const input_x = @as(isize, @intCast(output_x * stride + kernel_x)) - padding;
                            if (input_y < 0 or input_y >= @as(isize, @intCast(input_height)) or input_x < 0 or input_x >= @as(isize, @intCast(input_width))) continue;
                            const source_index = channel * input_width * input_height + @as(usize, @intCast(input_y)) * input_width + @as(usize, @intCast(input_x));
                            const output_index = filter * output_width * output_height + output_y * output_width + output_x;
                            kernel_gradient += delta[output_index] * input[source_index];
                        }
                    }
                    const weight_index = ((filter * input_channels + channel) * kernel_size + kernel_y) * kernel_size + kernel_x;
                    weight_gradients[weight_index] += kernel_gradient;
                }
            }
        }
        bias_gradients[filter] += bias_gradient;
    }
}

fn updateDenseChain(
    layer_count: usize,
    sample_indices: [*]u16,
    sample_values: [*]f32,
    active_count: usize,
    input_sizes: [*]i32,
    output_sizes: [*]i32,
    activation_kinds: [*]i32,
    weight_pointers: [*]u32,
    bias_pointers: [*]u32,
    activation_pointers: [*]u32,
    preactivation_pointers: [*]u32,
    delta_pointers: [*]u32,
    weight_gradient_pointers: [*]u32,
    bias_gradient_pointers: [*]u32,
    dropout_mask_pointers: [*]u32,
    capture_input_gradient: i32,
    input_gradient_ptr: u32,
    first_input_dense: i32,
) void {
    var layer_cursor = layer_count;
    while (layer_cursor > 0) {
        layer_cursor -= 1;
        const input_size: usize = @intCast(input_sizes[layer_cursor]);
        const output_size: usize = @intCast(output_sizes[layer_cursor]);
        const weights = f32Ptr(weight_pointers[layer_cursor]);
        _ = bias_pointers;
        const delta = f32Ptr(delta_pointers[layer_cursor]);
        const weight_gradients = f32Ptr(weight_gradient_pointers[layer_cursor]);
        const bias_gradients = f32Ptr(bias_gradient_pointers[layer_cursor]);

        if (layer_cursor > 0) {
            const previous_delta = f32Ptr(delta_pointers[layer_cursor - 1]);
            fillZero(previous_delta, input_size);
            var output: usize = 0;
            while (output < output_size) : (output += 1) {
                const output_gradient = clampDelta(delta[output]);
                const offset = output * input_size;
                accumulateScaled(previous_delta, weights + offset, input_size, output_gradient);
            }
            applyTrainingActivationDerivative(previous_delta, f32Ptr(preactivation_pointers[layer_cursor - 1]), f32Ptr(dropout_mask_pointers[layer_cursor - 1]), input_size, activation_kinds[layer_cursor - 1]);
        } else if (capture_input_gradient != 0) {
            const input_gradient = f32Ptr(input_gradient_ptr);
            fillZero(input_gradient, input_size);
            var output: usize = 0;
            while (output < output_size) : (output += 1) {
                const output_gradient = clampDelta(delta[output]);
                const offset = output * input_size;
                accumulateScaled(input_gradient, weights + offset, input_size, output_gradient);
            }
        }

        const previous = if (layer_cursor > 0) f32Ptr(activation_pointers[layer_cursor - 1]) else f32Ptr(0);
        var output: usize = 0;
        while (output < output_size) : (output += 1) {
            const output_gradient = clampDelta(delta[output]);
            const offset = output * input_size;
            if (layer_cursor == 0) {
                if (first_input_dense != 0) {
                    var input: usize = 0;
                    const gradient_scale: F32x4 = @splat(output_gradient);
                    while (input + 4 <= input_size) : (input += 4) {
                        const weight_index = offset + input;
                        storeF32x4(weight_gradients, weight_index, loadF32x4(weight_gradients, weight_index) + loadF32x4(sample_values, input) * gradient_scale);
                    }
                    while (input < input_size) : (input += 1) {
                        const weight_index = offset + input;
                        weight_gradients[weight_index] += output_gradient * sample_values[input];
                    }
                } else {
                    var pixel: usize = 0;
                    while (pixel < active_count) : (pixel += 1) {
                        const weight_index = offset + sample_indices[pixel];
                        weight_gradients[weight_index] += output_gradient * sample_values[pixel];
                    }
                }
            } else {
                var input: usize = 0;
                const gradient_scale: F32x4 = @splat(output_gradient);
                while (input + 4 <= input_size) : (input += 4) {
                    const weight_index = offset + input;
                    storeF32x4(weight_gradients, weight_index, loadF32x4(weight_gradients, weight_index) + loadF32x4(previous, input) * gradient_scale);
                }
                while (input < input_size) : (input += 1) {
                    const weight_index = offset + input;
                    weight_gradients[weight_index] += output_gradient * previous[input];
                }
            }
            bias_gradients[output] += output_gradient;
        }
    }
}

pub export fn train_sample(
    layer_count_raw: i32,
    sample_indices_ptr: u32,
    sample_values_ptr: u32,
    active_count_raw: i32,
    label_raw: i32,
    input_sizes_ptr: u32,
    output_sizes_ptr: u32,
    activation_kinds_ptr: u32,
    weight_pointers_ptr: u32,
    bias_pointers_ptr: u32,
    activation_pointers_ptr: u32,
    preactivation_pointers_ptr: u32,
    delta_pointers_ptr: u32,
    weight_gradient_pointers_ptr: u32,
    bias_gradient_pointers_ptr: u32,
    output_head_kind: i32,
    capture_input_gradient: i32,
    input_gradient_ptr: u32,
    dropout_rates_ptr: u32,
    dropout_mask_pointers_ptr: u32,
    dropout_seed: u32,
    first_input_dense: i32,
) f32 {
    const layer_count: usize = @intCast(layer_count_raw);
    const active_count: usize = @intCast(active_count_raw);
    const label: usize = @intCast(label_raw);
    const sample_indices = u16Ptr(sample_indices_ptr);
    const sample_values = f32Ptr(sample_values_ptr);
    const input_sizes = i32Ptr(input_sizes_ptr);
    const output_sizes = i32Ptr(output_sizes_ptr);
    const activation_kinds = i32Ptr(activation_kinds_ptr);
    const weight_pointers = u32Ptr(weight_pointers_ptr);
    const bias_pointers = u32Ptr(bias_pointers_ptr);
    const activation_pointers = u32Ptr(activation_pointers_ptr);
    const preactivation_pointers = u32Ptr(preactivation_pointers_ptr);
    const delta_pointers = u32Ptr(delta_pointers_ptr);
    const weight_gradient_pointers = u32Ptr(weight_gradient_pointers_ptr);
    const bias_gradient_pointers = u32Ptr(bias_gradient_pointers_ptr);
    const dropout_rates = f32Ptr(dropout_rates_ptr);
    const dropout_mask_pointers = u32Ptr(dropout_mask_pointers_ptr);

    forwardSparseTrainingInternal(layer_count, sample_indices, sample_values, active_count, input_sizes, output_sizes, activation_kinds, weight_pointers, bias_pointers, activation_pointers, preactivation_pointers, dropout_rates, dropout_mask_pointers, dropout_seed, first_input_dense, false);

    const last = layer_count - 1;
    const probabilities = f32Ptr(activation_pointers[last]);
    const output_delta = f32Ptr(delta_pointers[last]);
    const output_count: usize = @intCast(output_sizes[last]);
    activateBuffer(probabilities, output_count, if (output_head_kind == 1) 3 else 5);
    var index: usize = 0;
    var loss: f32 = 0.0;
    while (index < output_count) : (index += 1) {
        const target: f32 = if (index == label) 1.0 else 0.0;
        output_delta[index] = probabilities[index] - target;
        if (output_head_kind == 1) {
            var probability = probabilities[index];
            if (probability < 0.00000001) probability = 0.00000001;
            if (probability > 0.99999999) probability = 0.99999999;
            loss -= (target * mathLog(probability) + (1.0 - target) * mathLog(1.0 - probability)) / @as(f32, @floatFromInt(output_count));
            output_delta[index] /= @as(f32, @floatFromInt(output_count));
        }
    }
    if (output_head_kind != 1) {
        var label_probability = probabilities[label];
        if (label_probability < 0.00000001) label_probability = 0.00000001;
        loss = -mathLog(label_probability);
    }

    updateDenseChain(layer_count, sample_indices, sample_values, active_count, input_sizes, output_sizes, activation_kinds, weight_pointers, bias_pointers, activation_pointers, preactivation_pointers, delta_pointers, weight_gradient_pointers, bias_gradient_pointers, dropout_mask_pointers, capture_input_gradient, input_gradient_ptr, first_input_dense);
    return loss;
}

pub export fn train_dense_from_gradient(
    layer_count_raw: i32,
    sample_indices_ptr: u32,
    sample_values_ptr: u32,
    active_count_raw: i32,
    output_gradient_ptr: u32,
    input_sizes_ptr: u32,
    output_sizes_ptr: u32,
    activation_kinds_ptr: u32,
    weight_pointers_ptr: u32,
    bias_pointers_ptr: u32,
    activation_pointers_ptr: u32,
    preactivation_pointers_ptr: u32,
    delta_pointers_ptr: u32,
    weight_gradient_pointers_ptr: u32,
    bias_gradient_pointers_ptr: u32,
    capture_input_gradient: i32,
    input_gradient_ptr: u32,
    dropout_mask_pointers_ptr: u32,
    first_input_dense: i32,
) void {
    const layer_count: usize = @intCast(layer_count_raw);
    const sample_indices = u16Ptr(sample_indices_ptr);
    const sample_values = f32Ptr(sample_values_ptr);
    const active_count: usize = @intCast(active_count_raw);
    const input_sizes = i32Ptr(input_sizes_ptr);
    const output_sizes = i32Ptr(output_sizes_ptr);
    const activation_kinds = i32Ptr(activation_kinds_ptr);
    const weight_pointers = u32Ptr(weight_pointers_ptr);
    const bias_pointers = u32Ptr(bias_pointers_ptr);
    const activation_pointers = u32Ptr(activation_pointers_ptr);
    const preactivation_pointers = u32Ptr(preactivation_pointers_ptr);
    const delta_pointers = u32Ptr(delta_pointers_ptr);
    const weight_gradient_pointers = u32Ptr(weight_gradient_pointers_ptr);
    const bias_gradient_pointers = u32Ptr(bias_gradient_pointers_ptr);
    const dropout_mask_pointers = u32Ptr(dropout_mask_pointers_ptr);
    const last = layer_count - 1;
    const output_count: usize = @intCast(output_sizes[last]);
    const output_gradient = f32Ptr(output_gradient_ptr);
    const output_delta = f32Ptr(delta_pointers[last]);
    var index: usize = 0;
    while (index < output_count) : (index += 1) {
        output_delta[index] = clampDelta(output_gradient[index]);
    }
    applyTrainingActivationDerivative(
        output_delta,
        f32Ptr(preactivation_pointers[last]),
        f32Ptr(dropout_mask_pointers[last]),
        output_count,
        activation_kinds[last],
    );
    updateDenseChain(layer_count, sample_indices, sample_values, active_count, input_sizes, output_sizes, activation_kinds, weight_pointers, bias_pointers, activation_pointers, preactivation_pointers, delta_pointers, weight_gradient_pointers, bias_gradient_pointers, dropout_mask_pointers, capture_input_gradient, input_gradient_ptr, first_input_dense);
}
