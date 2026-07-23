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

fn clampDelta(value: f32) f32 {
    if (value < -5.0) return -5.0;
    if (value > 5.0) return 5.0;
    return value;
}

fn activateScalar(value: f32, kind: i32) f32 {
    return switch (kind) {
        1 => if (value > 0.0) value else 0.0,
        2 => if (value > 0.0) value else value * 0.08,
        3 => 1.0 / (1.0 + fastExp(-value)),
        4 => fastTanh(value),
        6 => if (value >= 0.0) value else fastExp(value) - 1.0,
        7 => 1.05070098736 * (if (value >= 0.0) value else 1.67326324235 * (fastExp(value) - 1.0)),
        8 => blk: {
            const cubic = value * value * value;
            const curve = 0.7978845608 * (value + 0.044715 * cubic);
            break :blk 0.5 * value * (1.0 + fastTanh(curve));
        },
        9 => blk: {
            const gate = 1.0 / (1.0 + fastExp(-value));
            break :blk value * gate;
        },
        10 => value * fastTanh(fastSoftplus(value)),
        11 => fastSoftplus(value),
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
            values[index] = fastExp(values[index] - maximum);
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
            const tangent = fastTanh(curve);
            break :blk 0.5 * (1.0 + tangent) + 0.5 * input * (1.0 - tangent * tangent) * 0.7978845608 * (1.0 + 0.134145 * squared);
        },
        9 => blk: {
            const gate = 1.0 / (1.0 + fastExp(-input));
            break :blk gate + input * gate * (1.0 - gate);
        },
        10 => blk: {
            const tangent = fastTanh(fastSoftplus(input));
            const gate = 1.0 / (1.0 + fastExp(-input));
            break :blk tangent + input * gate * (1.0 - tangent * tangent);
        },
        11 => 1.0 / (1.0 + fastExp(-input)),
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

pub export fn matvec(input_ptr: u32, weights_ptr: u32, bias_ptr: u32, output_ptr: u32, input_size_raw: i32, output_size_raw: i32) void {
    const input_size: usize = @intCast(input_size_raw);
    const output_size: usize = @intCast(output_size_raw);
    const input = f32Ptr(input_ptr);
    const weights = f32Ptr(weights_ptr);
    const bias = f32Ptr(bias_ptr);
    const output = f32Ptr(output_ptr);
    var row: usize = 0;
    while (row < output_size) : (row += 1) {
        var sum = bias[row];
        const offset = row * input_size;
        var column: usize = 0;
        while (column < input_size) : (column += 1) {
            sum += weights[offset + column] * input[column];
        }
        output[row] = sum;
    }
}

pub export fn activate(values_ptr: u32, length_raw: i32, kind: i32) void {
    activateBuffer(f32Ptr(values_ptr), @intCast(length_raw), kind);
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
                var pixel: usize = 0;
                while (pixel < active_count) : (pixel += 1) {
                    sum += weights[offset + sample_indices[pixel]] * sample_values[pixel];
                }
            } else {
                const previous = f32Ptr(activation_pointers[layer_index - 1]);
                var input: usize = 0;
                while (input < input_size) : (input += 1) {
                    sum += weights[offset + input] * previous[input];
                }
            }
            preactivations[output] = sum;
            outputs[output] = sum;
        }
        activateBuffer(outputs, output_size, activation_kinds[layer_index]);
    }
    const last = layer_count - 1;
    activateBuffer(f32Ptr(activation_pointers[last]), @intCast(output_sizes[last]), 5);
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
    weight_first_pointers_ptr: u32,
    bias_first_pointers_ptr: u32,
    weight_second_pointers_ptr: u32,
    bias_second_pointers_ptr: u32,
    optimizer_kind: i32,
    learning_rate: f32,
    momentum: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    capture_input_gradient: i32,
    input_gradient_ptr: u32,
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
    const weight_first_pointers = u32Ptr(weight_first_pointers_ptr);
    const bias_first_pointers = u32Ptr(bias_first_pointers_ptr);
    const weight_second_pointers = u32Ptr(weight_second_pointers_ptr);
    const bias_second_pointers = u32Ptr(bias_second_pointers_ptr);

    forwardSparseInternal(layer_count, sample_indices, sample_values, active_count, input_sizes, output_sizes, activation_kinds, weight_pointers, bias_pointers, activation_pointers, preactivation_pointers);

    const last = layer_count - 1;
    const probabilities = f32Ptr(activation_pointers[last]);
    const output_delta = f32Ptr(delta_pointers[last]);
    const output_count: usize = @intCast(output_sizes[last]);
    var index: usize = 0;
    while (index < output_count) : (index += 1) {
        output_delta[index] = probabilities[index] - (if (index == label) @as(f32, 1.0) else 0.0);
    }
    var label_probability = probabilities[label];
    if (label_probability < 0.00000001) label_probability = 0.00000001;
    const loss = -fastLog(label_probability);

    var layer_cursor = layer_count;
    while (layer_cursor > 0) {
        layer_cursor -= 1;
        const input_size: usize = @intCast(input_sizes[layer_cursor]);
        const output_size: usize = @intCast(output_sizes[layer_cursor]);
        const weights = f32Ptr(weight_pointers[layer_cursor]);
        const biases = f32Ptr(bias_pointers[layer_cursor]);
        const delta = f32Ptr(delta_pointers[layer_cursor]);
        const weight_first = f32Ptr(weight_first_pointers[layer_cursor]);
        const bias_first = f32Ptr(bias_first_pointers[layer_cursor]);
        const weight_second = f32Ptr(weight_second_pointers[layer_cursor]);
        const bias_second = f32Ptr(bias_second_pointers[layer_cursor]);

        if (layer_cursor > 0) {
            const previous = f32Ptr(activation_pointers[layer_cursor - 1]);
            const previous_delta = f32Ptr(delta_pointers[layer_cursor - 1]);
            var input: usize = 0;
            while (input < input_size) : (input += 1) previous_delta[input] = 0.0;
            var output: usize = 0;
            while (output < output_size) : (output += 1) {
                const output_gradient = clampDelta(delta[output]);
                const offset = output * input_size;
                input = 0;
                while (input < input_size) : (input += 1) {
                    previous_delta[input] += weights[offset + input] * output_gradient;
                }
            }
            applyActivationDerivative(previous_delta, previous, f32Ptr(preactivation_pointers[layer_cursor - 1]), input_size, activation_kinds[layer_cursor - 1]);
        } else if (capture_input_gradient != 0) {
            const input_gradient = f32Ptr(input_gradient_ptr);
            var input: usize = 0;
            while (input < input_size) : (input += 1) input_gradient[input] = 0.0;
            var output: usize = 0;
            while (output < output_size) : (output += 1) {
                const output_gradient = clampDelta(delta[output]);
                const offset = output * input_size;
                input = 0;
                while (input < input_size) : (input += 1) {
                    input_gradient[input] += weights[offset + input] * output_gradient;
                }
            }
        }

        const previous = if (layer_cursor > 0) f32Ptr(activation_pointers[layer_cursor - 1]) else f32Ptr(0);
        var output: usize = 0;
        while (output < output_size) : (output += 1) {
            const output_gradient = clampDelta(delta[output]);
            const offset = output * input_size;
            if (layer_cursor == 0) {
                var pixel: usize = 0;
                while (pixel < active_count) : (pixel += 1) {
                    const weight_index = offset + sample_indices[pixel];
                    updateParameter(weights, weight_index, output_gradient * sample_values[pixel], learning_rate, optimizer_kind, momentum, decay, beta1, beta2, epsilon, beta1_correction, beta2_correction, weight_first, weight_second);
                }
            } else {
                var input: usize = 0;
                while (input < input_size) : (input += 1) {
                    const weight_index = offset + input;
                    updateParameter(weights, weight_index, output_gradient * previous[input], learning_rate, optimizer_kind, momentum, decay, beta1, beta2, epsilon, beta1_correction, beta2_correction, weight_first, weight_second);
                }
            }
            updateParameter(biases, output, output_gradient, learning_rate, optimizer_kind, momentum, decay, beta1, beta2, epsilon, beta1_correction, beta2_correction, bias_first, bias_second);
        }
    }
    return loss;
}
