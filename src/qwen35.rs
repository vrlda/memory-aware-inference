//! Exact reference state transitions for Qwen3.5/3.6 Gated DeltaNet layers.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Error(pub String);

pub type Result<T> = std::result::Result<T, Qwen35Error>;

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Apply the per-value-head RMSNorm followed by the SiLU gate used by
/// Qwen3.5/3.6 Gated DeltaNet.
pub fn gated_rms_norm(
    input: &[f32],
    gate: &[f32],
    weight: &[f32],
    heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if heads == 0
        || head_dim == 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.len() != heads * head_dim
        || gate.len() != input.len()
        || weight.len() != head_dim
    {
        return Err(Qwen35Error("Gated RMSNorm dimensions are invalid".into()));
    }
    let mut output = vec![0.0_f32; input.len()];
    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let inverse = (input[start..end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / head_dim as f32
            + epsilon)
            .sqrt()
            .recip();
        for index in start..end {
            output[index] = input[index] * inverse * weight[index - start] * silu(gate[index]);
        }
    }
    Ok(output)
}

/// Apply Qwen3.5's head-local RMSNorm. Its learned parameter is a centered
/// residual scale, so the multiplier is `(1 + weight)` rather than `weight`.
pub fn rms_norm_heads(
    input: &[f32],
    weight: &[f32],
    heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if heads == 0
        || head_dim == 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.len() != heads * head_dim
        || weight.len() != head_dim
    {
        return Err(Qwen35Error("head RMSNorm dimensions are invalid".into()));
    }
    let mut output = vec![0.0_f32; input.len()];
    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let inverse = (input[start..end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / head_dim as f32
            + epsilon)
            .sqrt()
            .recip();
        for index in start..end {
            output[index] = input[index] * inverse * (1.0 + weight[index - start]);
        }
    }
    Ok(output)
}

/// Apply one causal depthwise convolution update and SiLU activation.
///
/// `weights` is row-major `[channel][kernel]`; `state` stores the previous
/// `kernel - 1` values for each channel in chronological order. The state is
/// updated in place so callers can retain it across autoregressive tokens.
pub fn causal_conv1d_step(
    input: &[f32],
    state: &mut [f32],
    weights: &[f32],
    kernel_size: usize,
) -> Result<Vec<f32>> {
    if input.is_empty() || kernel_size < 2 {
        return Err(Qwen35Error(
            "causal convolution dimensions must be non-zero".into(),
        ));
    }
    let expected_state = input
        .len()
        .checked_mul(kernel_size - 1)
        .ok_or_else(|| Qwen35Error("causal convolution state length overflows".into()))?;
    let expected_weights = input
        .len()
        .checked_mul(kernel_size)
        .ok_or_else(|| Qwen35Error("causal convolution weight length overflows".into()))?;
    if state.len() != expected_state || weights.len() != expected_weights {
        return Err(Qwen35Error(
            "causal convolution input, state, or weight shapes do not match".into(),
        ));
    }
    let mut output = vec![0.0_f32; input.len()];
    for channel in 0..input.len() {
        let state_base = channel * (kernel_size - 1);
        let weight_base = channel * kernel_size;
        let mut value = input[channel] * weights[weight_base + kernel_size - 1];
        for tap in 0..kernel_size - 1 {
            value += state[state_base + tap] * weights[weight_base + tap];
        }
        output[channel] = value / (1.0 + (-value).exp());
        state.copy_within(state_base + 1..state_base + kernel_size - 1, state_base);
        state[state_base + kernel_size - 2] = input[channel];
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatedDeltaState {
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    values: Vec<f32>,
}

impl GatedDeltaState {
    pub fn zeros(heads: usize, key_dim: usize, value_dim: usize) -> Result<Self> {
        if heads == 0 || key_dim == 0 || value_dim == 0 {
            return Err(Qwen35Error(
                "Gated DeltaNet state dimensions must be non-zero".into(),
            ));
        }
        let length = heads
            .checked_mul(key_dim)
            .and_then(|length| length.checked_mul(value_dim))
            .ok_or_else(|| Qwen35Error("Gated DeltaNet state length overflows".into()))?;
        Ok(Self {
            heads,
            key_dim,
            value_dim,
            values: vec![0.0; length],
        })
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        gate: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>> {
        let expected_query_key = self.heads * self.key_dim;
        let expected_value = self.heads * self.value_dim;
        if query.len() != expected_query_key
            || key.len() != expected_query_key
            || value.len() != expected_value
            || gate.len() != self.heads
            || beta.len() != self.heads
        {
            return Err(Qwen35Error(
                "Gated DeltaNet step shapes do not match state".into(),
            ));
        }
        let mut output = vec![0.0_f32; expected_value];
        let mut memory = vec![0.0_f32; self.value_dim];
        let mut delta = vec![0.0_f32; self.value_dim];
        for head in 0..self.heads {
            let state_base = head * self.key_dim * self.value_dim;
            let key_base = head * self.key_dim;
            let value_base = head * self.value_dim;
            let decay = gate[head].exp();
            for key_index in 0..self.key_dim {
                let row_base = state_base + key_index * self.value_dim;
                for value_index in 0..self.value_dim {
                    self.values[row_base + value_index] *= decay;
                }
            }
            memory.fill(0.0);
            for key_index in 0..self.key_dim {
                let row_base = state_base + key_index * self.value_dim;
                let key_value = key[key_base + key_index];
                for (value_index, memory_value) in memory.iter_mut().enumerate() {
                    *memory_value += self.values[row_base + value_index] * key_value;
                }
            }
            for (value_index, delta_value) in delta.iter_mut().enumerate() {
                *delta_value = (value[value_base + value_index] - memory[value_index]) * beta[head];
            }
            for key_index in 0..self.key_dim {
                let row_base = state_base + key_index * self.value_dim;
                let key_value = key[key_base + key_index];
                for (value_index, delta_value) in delta.iter().enumerate() {
                    self.values[row_base + value_index] += key_value * delta_value;
                }
            }
            for value_index in 0..self.value_dim {
                let mut result = 0.0_f32;
                for key_index in 0..self.key_dim {
                    result += self.values[state_base + key_index * self.value_dim + value_index]
                        * query[key_base + key_index];
                }
                output[value_base + value_index] = result;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_rms_norm_matches_reference_equation() {
        let input = [1.0_f32, 2.0, -1.0, 3.0];
        let gate = [0.0_f32, 1.0, -1.0, 0.5];
        let weight = [1.0_f32, 0.5];
        let output =
            gated_rms_norm(&input, &gate, &weight, 2, 2, 1.0e-6).expect("gated RMSNorm should run");
        let inverse0 = (2.5_f32 + 1.0e-6).sqrt().recip();
        let inverse1 = (5.0_f32 + 1.0e-6).sqrt().recip();
        assert!(output[0].abs() < 1.0e-6);
        assert!((output[1] - input[1] * inverse0 * 0.5 * silu(1.0)).abs() < 1.0e-6);
        assert!((output[2] - input[2] * inverse1 * silu(-1.0)).abs() < 1.0e-6);
    }

    #[test]
    fn head_rms_norm_uses_centered_qwen_scale() {
        let input = [1.0_f32, 2.0, -1.0, 3.0];
        let weight = [0.25_f32, -0.5];
        let output =
            rms_norm_heads(&input, &weight, 2, 2, 1.0e-6).expect("head RMSNorm should run");
        let inverse0 = (2.5_f32 + 1.0e-6).sqrt().recip();
        let inverse1 = (5.0_f32 + 1.0e-6).sqrt().recip();
        assert!((output[0] - input[0] * inverse0 * 1.25).abs() < 1.0e-6);
        assert!((output[1] - input[1] * inverse0 * 0.5).abs() < 1.0e-6);
        assert!((output[2] - input[2] * inverse1 * 1.25).abs() < 1.0e-6);
    }

    #[test]
    fn causal_conv_step_updates_state_and_applies_silu() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [5.0, 6.0];
        let weights = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 1.0, 2.0];
        let output = causal_conv1d_step(&input, &mut state, &weights, 4)
            .expect("causal convolution should run");
        let first: f32 = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 5.0 * 4.0;
        let second: f32 = -4.0 + 5.0 * 0.5 + 6.0 + 6.0 * 2.0;
        assert!((output[0] - first / (1.0 + (-first).exp())).abs() < 1.0e-6);
        assert!((output[1] - second / (1.0 + (-second).exp())).abs() < 1.0e-6);
        assert_eq!(state, vec![2.0, 3.0, 5.0, 5.0, 6.0, 6.0]);
    }

    #[test]
    fn recurrent_step_updates_and_reuses_state() {
        let mut state = GatedDeltaState::zeros(1, 2, 1).expect("valid state");
        let first = state
            .step(&[1.0, 0.0], &[1.0, 0.0], &[2.0], &[0.0], &[0.5])
            .expect("first step should run");
        assert_eq!(first, vec![1.0]);
        assert_eq!(state.as_slice(), &[1.0, 0.0]);

        let second = state
            .step(&[0.0, 1.0], &[0.0, 1.0], &[4.0], &[0.5_f32.ln()], &[1.0])
            .expect("second step should run");
        assert!((second[0] - 4.0).abs() < 1.0e-6);
        assert!((state.as_slice()[0] - 0.5).abs() < 1.0e-6);
        assert!((state.as_slice()[1] - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_inconsistent_recurrent_shapes() {
        let mut state = GatedDeltaState::zeros(2, 2, 2).expect("valid state");
        assert!(state
            .step(&[0.0], &[0.0; 4], &[0.0; 4], &[0.0; 2], &[0.0; 2])
            .is_err());
    }
}
