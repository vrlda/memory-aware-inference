//! One-token Qwen3.5/3.6 GGUF decoder-block execution helpers.
//!
//! This module deliberately keeps state initialization explicit and bounded:
//! it evaluates a block from an empty cache for correctness and profiling,
//! while the persistent multi-layer state planner is developed separately.

#[cfg(target_os = "macos")]
use crate::metal::{AttentionDecodeInput, MetalContext};
#[cfg(target_os = "macos")]
use crate::metal::{PrivateStagedQwenLayer, QuantWeight};
#[cfg(target_os = "macos")]
use crate::model::StagedQwenLayer;
#[cfg(target_os = "macos")]
use std::collections::{BTreeSet, HashMap};

#[cfg(target_os = "macos")]
struct MappedStagedQwenLayer {
    backing: StagedQwenLayer,
    buffer: metal::Buffer,
    ranges: std::collections::BTreeMap<String, (usize, usize)>,
}

#[cfg(target_os = "macos")]
enum StagedWeights {
    /// A layer whose tensors are all already resident in `active`.
    ///
    /// Keeping this as an explicit staged result lets the execution pipeline
    /// preserve its fixed look-ahead structure without attempting to create a
    /// Metal buffer for an empty packed payload.
    Empty,
    Mapped(MappedStagedQwenLayer),
    Private(PrivateStagedQwenLayer),
}

#[cfg(target_os = "macos")]
impl StagedWeights {
    fn get(&self, name: &str) -> Option<&[u8]> {
        match self {
            Self::Empty => None,
            Self::Mapped(layer) => layer.backing.get(name),
            Self::Private(_) => None,
        }
    }

    fn quant_weight(&self, name: &str, ggml_type: u32) -> Option<QuantWeight> {
        match self {
            Self::Empty => None,
            Self::Mapped(layer) => {
                let (start, end) = layer.ranges.get(name).copied()?;
                Some(QuantWeight {
                    buffer: layer.buffer.clone(),
                    offset: start as u64,
                    bytes: (end - start) as u64,
                    ggml_type,
                    mapped: true,
                })
            }
            Self::Private(layer) => layer.quant_weight(name, ggml_type),
        }
    }
}

/// Persistent one-layer decode state. Gated DeltaNet keeps its convolution and
/// recurrent matrices; full attention keeps a bounded KV cache.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub enum Qwen35LayerState {
    GatedDelta {
        conv_state: Vec<f32>,
        recurrent_state: Vec<f32>,
    },
    FullAttention {
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        cached_tokens: usize,
        capacity_tokens: usize,
    },
}

#[cfg(target_os = "macos")]
impl Qwen35LayerState {
    pub fn new(
        store: &GgufModelStore,
        layer: usize,
        capacity_tokens: usize,
    ) -> Result<Self, String> {
        if capacity_tokens == 0 {
            return Err("layer state capacity must be non-zero".into());
        }
        let config = store.qwen35_config().map_err(|error| error.to_string())?;
        let kind = store
            .qwen35_layer_kind(layer)
            .map_err(|error| error.to_string())?;
        match kind {
            GgufQwen35LayerKind::GatedDeltaNet => Ok(Self::GatedDelta {
                conv_state: vec![
                    0.0_f32;
                    config.ssm_projection_size() * (config.ssm_conv_kernel - 1)
                ],
                recurrent_state: vec![
                    0.0_f32;
                    config.ssm_value_heads()
                        * config.ssm_value_dim()
                        * config.ssm_value_dim()
                ],
            }),
            GgufQwen35LayerKind::FullAttention => {
                let kv_width = config.num_key_value_heads * config.head_dim;
                Ok(Self::FullAttention {
                    key_cache: vec![0.0; kv_width * capacity_tokens],
                    value_cache: vec![0.0; kv_width * capacity_tokens],
                    cached_tokens: 0,
                    capacity_tokens,
                })
            }
        }
    }

    pub fn cached_tokens(&self) -> usize {
        match self {
            Self::GatedDelta { .. } => 0,
            Self::FullAttention { cached_tokens, .. } => *cached_tokens,
        }
    }

    pub fn state_bytes(&self) -> usize {
        match self {
            Self::GatedDelta {
                conv_state,
                recurrent_state,
            } => (conv_state.len() + recurrent_state.len()) * std::mem::size_of::<f32>(),
            Self::FullAttention {
                key_cache,
                value_cache,
                ..
            } => (key_cache.len() + value_cache.len()) * std::mem::size_of::<f32>(),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::GatedDelta {
                conv_state,
                recurrent_state,
            } => {
                conv_state.fill(0.0);
                recurrent_state.fill(0.0);
            }
            Self::FullAttention {
                key_cache,
                value_cache,
                cached_tokens,
                ..
            } => {
                key_cache.fill(0.0);
                value_cache.fill(0.0);
                *cached_tokens = 0;
            }
        }
    }
}
#[cfg(target_os = "macos")]
use crate::model::{GgufModelStore, GgufQwen35LayerKind, GgufTensorView};
#[cfg(target_os = "macos")]
use std::thread;

#[cfg(target_os = "macos")]
fn f32_values<'a>(tensor: &GgufTensorView<'a>) -> Result<&'a [f32], String> {
    if tensor.info.ggml_type != 0 {
        return Err(format!(
            "tensor {} is not F32 (type {})",
            tensor.info.name, tensor.info.ggml_type
        ));
    }
    // SAFETY: GGUF F32 payloads are aligned by the parser's tensor alignment.
    let (prefix, values, suffix) = unsafe { tensor.bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(format!(
            "tensor {} F32 payload is not aligned",
            tensor.info.name
        ));
    }
    let expected = tensor
        .info
        .shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| format!("tensor {} element count overflows", tensor.info.name))?;
    if values.len() != expected {
        return Err(format!(
            "tensor {} has {} F32 values; expected {expected}",
            tensor.info.name,
            values.len()
        ));
    }
    Ok(values)
}

#[cfg(target_os = "macos")]
fn gguf_matvec_with_retained(
    context: &MetalContext,
    retained: &HashMap<String, QuantWeight>,
    store: &GgufModelStore,
    staged: Option<&StagedWeights>,
    tensor: &GgufTensorView<'_>,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let rows = tensor
        .info
        .shape
        .get(1)
        .copied()
        .ok_or_else(|| format!("tensor {} is not a matrix", tensor.info.name))?;
    if let Some(weight) = retained.get(&tensor.info.name) {
        context.gguf_quant_matvec_weight(weight, rows, tensor.info.shape[0], input)
    } else {
        if let Some(weight) = staged
            .and_then(|weights| weights.quant_weight(&tensor.info.name, tensor.info.ggml_type))
        {
            return context.gguf_quant_matvec_weight(&weight, rows, tensor.info.shape[0], input);
        }
        #[cfg(target_os = "macos")]
        if tensor.info.ggml_type != 0 {
            if let Some(bytes) = staged.and_then(|weights| weights.get(&tensor.info.name)) {
                return context.gguf_quant_matvec_bytes(
                    tensor.info.ggml_type,
                    bytes,
                    rows,
                    tensor.info.shape[0],
                    input,
                );
            }
        }
        #[cfg(target_os = "macos")]
        if staged.is_none()
            && std::env::var_os("SI_STAGE_GGUF").is_some()
            && tensor.info.ggml_type != 0
        {
            let staged_bytes = store
                .stage_tensor_payload(tensor)
                .map_err(|error| error.to_string())?;
            return context.gguf_quant_matvec_bytes(
                tensor.info.ggml_type,
                &staged_bytes,
                rows,
                tensor.info.shape[0],
                input,
            );
        }
        context.gguf_matvec_tensor_rows(tensor, 0, rows, input)
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn gguf_matmul_many_with_retained(
    context: &MetalContext,
    retained: &HashMap<String, QuantWeight>,
    store: &GgufModelStore,
    staged: Option<&StagedWeights>,
    tensor: &GgufTensorView<'_>,
    batch: usize,
    inputs: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let rows = tensor
        .info
        .shape
        .get(1)
        .copied()
        .ok_or_else(|| format!("tensor {} is not a matrix", tensor.info.name))?;
    let columns = tensor
        .info
        .shape
        .first()
        .copied()
        .ok_or_else(|| format!("tensor {} is not a matrix", tensor.info.name))?;
    if inputs.len() != batch.saturating_mul(columns) {
        return Err(format!(
            "{} batched input length is invalid",
            tensor.info.name
        ));
    }
    if let Some(weight) = retained.get(&tensor.info.name) {
        return context.gguf_quant_matmul_many_weight(weight, rows, columns, batch, inputs);
    }
    if tensor.info.ggml_type == 0 {
        return Err(format!(
            "{} batched F32 matvec is not implemented",
            tensor.info.name
        ));
    }
    if let Some(weight) =
        staged.and_then(|weights| weights.quant_weight(&tensor.info.name, tensor.info.ggml_type))
    {
        return context.gguf_quant_matmul_many_weight(&weight, rows, columns, batch, inputs);
    }
    if let Some(bytes) = staged.and_then(|weights| weights.get(&tensor.info.name)) {
        return context.gguf_quant_matmul_many_bytes(
            tensor.info.ggml_type,
            bytes,
            rows,
            columns,
            batch,
            inputs,
        );
    }
    if staged.is_none() && std::env::var_os("SI_STAGE_GGUF").is_some() {
        let bytes = store
            .stage_tensor_payload(tensor)
            .map_err(|error| error.to_string())?;
        return context.gguf_quant_matmul_many_bytes(
            tensor.info.ggml_type,
            &bytes,
            rows,
            columns,
            batch,
            inputs,
        );
    }
    context.gguf_quant_matmul_many_bytes(
        tensor.info.ggml_type,
        tensor.bytes,
        rows,
        columns,
        batch,
        inputs,
    )
}

#[cfg(target_os = "macos")]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(target_os = "macos")]
fn l2_normalize_heads(values: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let mut normalized = values.to_vec();
    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let inverse = (values[start..end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            + 1.0e-6)
            .sqrt()
            .recip();
        for value in &mut normalized[start..end] {
            *value *= inverse;
        }
    }
    normalized
}

#[cfg(target_os = "macos")]
fn repeat_heads(
    values: &[f32],
    source_heads: usize,
    target_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(target_heads % source_heads, 0);
    let repeat = target_heads / source_heads;
    let mut output = vec![0.0_f32; target_heads * head_dim];
    for head in 0..target_heads {
        let source = (head / repeat) * head_dim;
        output[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&values[source..source + head_dim]);
    }
    output
}

#[cfg(target_os = "macos")]
fn partial_rope(
    context: &MetalContext,
    input: &[f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) -> Result<Vec<f32>, String> {
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.len() != heads * head_dim
    {
        return Err("partial RoPE dimensions are invalid".into());
    }
    let mut rotary = vec![0.0_f32; heads * rotary_dim];
    for head in 0..heads {
        rotary[head * rotary_dim..(head + 1) * rotary_dim]
            .copy_from_slice(&input[head * head_dim..head * head_dim + rotary_dim]);
    }
    let rotated = context.rope(&rotary, heads, rotary_dim, position, theta)?;
    let mut output = input.to_vec();
    for head in 0..heads {
        output[head * head_dim..head * head_dim + rotary_dim]
            .copy_from_slice(&rotated[head * rotary_dim..(head + 1) * rotary_dim]);
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn gdn_token_mixer(
    context: &MetalContext,
    store: &GgufModelStore,
    retained: &HashMap<String, QuantWeight>,
    staged: Option<&StagedWeights>,
    layer: usize,
    hidden: &[f32],
    state: &mut Qwen35LayerState,
) -> Result<Vec<f32>, String> {
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    let prefix = format!("blk.{layer}");
    let qkv = store
        .tensor(&format!("{prefix}.attn_qkv.weight"))
        .map_err(|error| error.to_string())?;
    let z_projection = store
        .tensor(&format!("{prefix}.attn_gate.weight"))
        .map_err(|error| error.to_string())?;
    let alpha_projection = store
        .tensor(&format!("{prefix}.ssm_alpha.weight"))
        .map_err(|error| error.to_string())?;
    let beta_projection = store
        .tensor(&format!("{prefix}.ssm_beta.weight"))
        .map_err(|error| error.to_string())?;
    let convolution = store
        .tensor(&format!("{prefix}.ssm_conv1d.weight"))
        .map_err(|error| error.to_string())?;
    let a_log = store
        .tensor(&format!("{prefix}.ssm_a"))
        .map_err(|error| error.to_string())?;
    let dt_bias = store
        .tensor(&format!("{prefix}.ssm_dt.bias"))
        .map_err(|error| error.to_string())?;
    let norm = store
        .tensor(&format!("{prefix}.ssm_norm.weight"))
        .map_err(|error| error.to_string())?;
    let output_projection = store
        .tensor(&format!("{prefix}.ssm_out.weight"))
        .map_err(|error| error.to_string())?;
    let (convolution_weights, a_log, dt_bias, norm) = (
        f32_values(&convolution)?,
        f32_values(&a_log)?,
        f32_values(&dt_bias)?,
        f32_values(&norm)?,
    );
    let (qkv_values, z) = if let (Some(qkv_weight), Some(z_weight)) = (
        retained.get(&qkv.info.name),
        retained.get(&z_projection.info.name),
    ) {
        let outputs = context.gguf_quant_matvec_many_weights(
            &[
                (qkv_weight, qkv.info.shape[1], qkv.info.shape[0]),
                (
                    z_weight,
                    z_projection.info.shape[1],
                    z_projection.info.shape[0],
                ),
            ],
            hidden,
        )?;
        let mut outputs = outputs.into_iter();
        (
            outputs
                .next()
                .ok_or("retained grouped Qwen3.6 QKV output is missing")?,
            outputs
                .next()
                .ok_or("retained grouped Qwen3.6 gate output is missing")?,
        )
    } else if retained.contains_key(&qkv.info.name)
        || retained.contains_key(&z_projection.info.name)
    {
        (
            gguf_matvec_with_retained(context, retained, store, staged, &qkv, hidden)?,
            gguf_matvec_with_retained(context, retained, store, staged, &z_projection, hidden)?,
        )
    } else if std::env::var("SI_GROUP_STAGED").ok().as_deref() == Some("1") && staged.is_some() {
        let staged = staged.expect("staged projection group should be present");
        if let (Some(qkv_bytes), Some(z_bytes)) = (
            staged.get(&qkv.info.name),
            staged.get(&z_projection.info.name),
        ) {
            let outputs = context.gguf_quant_matvec_many_bytes(
                &[
                    (
                        qkv.info.ggml_type,
                        qkv_bytes,
                        qkv.info.shape[1],
                        qkv.info.shape[0],
                    ),
                    (
                        z_projection.info.ggml_type,
                        z_bytes,
                        z_projection.info.shape[1],
                        z_projection.info.shape[0],
                    ),
                ],
                hidden,
            )?;
            let mut outputs = outputs.into_iter();
            (
                outputs
                    .next()
                    .ok_or("grouped staged Qwen3.6 QKV output is missing")?,
                outputs
                    .next()
                    .ok_or("grouped staged Qwen3.6 gate output is missing")?,
            )
        } else {
            (
                gguf_matvec_with_retained(context, retained, store, Some(staged), &qkv, hidden)?,
                gguf_matvec_with_retained(
                    context,
                    retained,
                    store,
                    Some(staged),
                    &z_projection,
                    hidden,
                )?,
            )
        }
    } else {
        let outputs = context.gguf_quant_matvec_many_tensors(&[&qkv, &z_projection], hidden)?;
        let mut outputs = outputs.into_iter();
        (
            outputs
                .next()
                .ok_or("grouped Qwen3.6 QKV output is missing")?,
            outputs
                .next()
                .ok_or("grouped Qwen3.6 gate output is missing")?,
        )
    };
    let (alpha, beta_logits) = {
        let outputs =
            context.f32_matvec_many_gguf_tensors(&[&alpha_projection, &beta_projection], hidden)?;
        let mut outputs = outputs.into_iter();
        (
            outputs
                .next()
                .ok_or("grouped Qwen3.6 alpha output is missing")?,
            outputs
                .next()
                .ok_or("grouped Qwen3.6 beta output is missing")?,
        )
    };
    let Qwen35LayerState::GatedDelta {
        conv_state,
        recurrent_state,
    } = state
    else {
        return Err("Gated DeltaNet layer received full-attention state".into());
    };
    let (mixed_qkv, updated_conv_state) = context.causal_conv1d_step(
        &qkv_values,
        conv_state,
        convolution_weights,
        config.ssm_projection_size(),
        config.ssm_conv_kernel,
    )?;
    *conv_state = updated_conv_state;
    let key_heads = config.ssm_key_heads();
    let value_heads = config.ssm_value_heads();
    let head_dim = config.ssm_value_dim();
    let key_width = key_heads * head_dim;
    let value_width = value_heads * head_dim;
    if mixed_qkv.len() != key_width * 2 + value_width {
        return Err("Gated DeltaNet projection width does not match metadata".into());
    }
    let query = repeat_heads(
        &l2_normalize_heads(&mixed_qkv[..key_width], key_heads, head_dim),
        key_heads,
        value_heads,
        head_dim,
    )
    .into_iter()
    .map(|value| value / (head_dim as f32).sqrt())
    .collect::<Vec<_>>();
    let key = repeat_heads(
        &l2_normalize_heads(&mixed_qkv[key_width..key_width * 2], key_heads, head_dim),
        key_heads,
        value_heads,
        head_dim,
    );
    let value = &mixed_qkv[key_width * 2..];
    let beta = beta_logits
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();
    let gate = alpha
        .iter()
        .zip(a_log.iter().zip(dt_bias))
        .map(|(value, (a_log, dt_bias))| -a_log.exp() * softplus(value + dt_bias))
        .collect::<Vec<_>>();
    let (core, updated_recurrent_state) = context.gated_delta_step(
        &query,
        &key,
        value,
        &gate,
        &beta,
        recurrent_state,
        value_heads,
        head_dim,
        head_dim,
    )?;
    *recurrent_state = updated_recurrent_state;
    let normalized =
        context.rms_norm_gated(&core, &z, norm, value_heads, head_dim, config.rms_norm_eps)?;
    gguf_matvec_with_retained(
        context,
        retained,
        store,
        staged,
        &output_projection,
        &normalized,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn attention_token_mixer(
    context: &MetalContext,
    store: &GgufModelStore,
    retained: &HashMap<String, QuantWeight>,
    staged: Option<&StagedWeights>,
    layer: usize,
    position: usize,
    hidden: &[f32],
    state: &mut Qwen35LayerState,
) -> Result<Vec<f32>, String> {
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    let prefix = format!("blk.{layer}");
    let q_projection = store
        .tensor(&format!("{prefix}.attn_q.weight"))
        .map_err(|error| error.to_string())?;
    let k_projection = store
        .tensor(&format!("{prefix}.attn_k.weight"))
        .map_err(|error| error.to_string())?;
    let v_projection = store
        .tensor(&format!("{prefix}.attn_v.weight"))
        .map_err(|error| error.to_string())?;
    let output_projection = store
        .tensor(&format!("{prefix}.attn_output.weight"))
        .map_err(|error| error.to_string())?;
    let q_norm = store
        .tensor(&format!("{prefix}.attn_q_norm.weight"))
        .map_err(|error| error.to_string())?;
    let k_norm = store
        .tensor(&format!("{prefix}.attn_k_norm.weight"))
        .map_err(|error| error.to_string())?;
    let q_norm = f32_values(&q_norm)?;
    let k_norm = f32_values(&k_norm)?;
    let (q_and_gate, key, value) = if let (Some(q_weight), Some(k_weight), Some(v_weight)) = (
        retained.get(&q_projection.info.name),
        retained.get(&k_projection.info.name),
        retained.get(&v_projection.info.name),
    ) {
        let outputs = context.gguf_quant_matvec_many_weights(
            &[
                (
                    q_weight,
                    q_projection.info.shape[1],
                    q_projection.info.shape[0],
                ),
                (
                    k_weight,
                    k_projection.info.shape[1],
                    k_projection.info.shape[0],
                ),
                (
                    v_weight,
                    v_projection.info.shape[1],
                    v_projection.info.shape[0],
                ),
            ],
            hidden,
        )?;
        let mut outputs = outputs.into_iter();
        (
            outputs
                .next()
                .ok_or("retained grouped Qwen3.6 query output is missing")?,
            outputs
                .next()
                .ok_or("retained grouped Qwen3.6 key output is missing")?,
            outputs
                .next()
                .ok_or("retained grouped Qwen3.6 value output is missing")?,
        )
    } else if retained.contains_key(&q_projection.info.name)
        || retained.contains_key(&k_projection.info.name)
        || retained.contains_key(&v_projection.info.name)
    {
        (
            gguf_matvec_with_retained(context, retained, store, staged, &q_projection, hidden)?,
            gguf_matvec_with_retained(context, retained, store, staged, &k_projection, hidden)?,
            gguf_matvec_with_retained(context, retained, store, staged, &v_projection, hidden)?,
        )
    } else if std::env::var("SI_GROUP_STAGED").ok().as_deref() == Some("1") && staged.is_some() {
        let staged = staged.expect("staged projection group should be present");
        if let (Some(q_bytes), Some(k_bytes), Some(v_bytes)) = (
            staged.get(&q_projection.info.name),
            staged.get(&k_projection.info.name),
            staged.get(&v_projection.info.name),
        ) {
            let outputs = context.gguf_quant_matvec_many_bytes(
                &[
                    (
                        q_projection.info.ggml_type,
                        q_bytes,
                        q_projection.info.shape[1],
                        q_projection.info.shape[0],
                    ),
                    (
                        k_projection.info.ggml_type,
                        k_bytes,
                        k_projection.info.shape[1],
                        k_projection.info.shape[0],
                    ),
                    (
                        v_projection.info.ggml_type,
                        v_bytes,
                        v_projection.info.shape[1],
                        v_projection.info.shape[0],
                    ),
                ],
                hidden,
            )?;
            let mut outputs = outputs.into_iter();
            (
                outputs
                    .next()
                    .ok_or("grouped staged Qwen3.6 query output is missing")?,
                outputs
                    .next()
                    .ok_or("grouped staged Qwen3.6 key output is missing")?,
                outputs
                    .next()
                    .ok_or("grouped staged Qwen3.6 value output is missing")?,
            )
        } else {
            (
                gguf_matvec_with_retained(
                    context,
                    retained,
                    store,
                    Some(staged),
                    &q_projection,
                    hidden,
                )?,
                gguf_matvec_with_retained(
                    context,
                    retained,
                    store,
                    Some(staged),
                    &k_projection,
                    hidden,
                )?,
                gguf_matvec_with_retained(
                    context,
                    retained,
                    store,
                    Some(staged),
                    &v_projection,
                    hidden,
                )?,
            )
        }
    } else {
        let outputs = context.gguf_quant_matvec_many_tensors(
            &[&q_projection, &k_projection, &v_projection],
            hidden,
        )?;
        let mut outputs = outputs.into_iter();
        (
            outputs
                .next()
                .ok_or("grouped Qwen3.6 query output is missing")?,
            outputs
                .next()
                .ok_or("grouped Qwen3.6 key output is missing")?,
            outputs
                .next()
                .ok_or("grouped Qwen3.6 value output is missing")?,
        )
    };
    let query_heads = config.num_attention_heads;
    let key_value_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let query_width = query_heads * head_dim;
    let kv_width = key_value_heads * head_dim;
    if q_and_gate.len() != query_width * 2 || key.len() != kv_width || value.len() != kv_width {
        return Err("full-attention projection shapes do not match metadata".into());
    }
    let query = context.rms_norm_heads(
        &q_and_gate[..query_width],
        q_norm,
        query_heads,
        head_dim,
        config.rms_norm_eps,
    )?;
    let key =
        context.rms_norm_heads(&key, k_norm, key_value_heads, head_dim, config.rms_norm_eps)?;
    let rotary_dim = store
        .metadata_u32("qwen35.rope.dimension_count")
        .ok_or("GGUF model is missing qwen35.rope.dimension_count")? as usize;
    let query = partial_rope(
        context,
        &query,
        query_heads,
        head_dim,
        rotary_dim,
        position,
        config.rope_theta,
    )?;
    let key = partial_rope(
        context,
        &key,
        key_value_heads,
        head_dim,
        rotary_dim,
        position,
        config.rope_theta,
    )?;
    let Qwen35LayerState::FullAttention {
        key_cache,
        value_cache,
        cached_tokens,
        capacity_tokens,
    } = state
    else {
        return Err("full-attention layer received Gated DeltaNet state".into());
    };
    if *cached_tokens >= *capacity_tokens {
        return Err("full-attention KV cache capacity exhausted".into());
    }
    let attended = context.attention_decode(AttentionDecodeInput {
        query: &query,
        key_cache,
        value_cache,
        new_keys: &key,
        new_values: &value,
        query_heads,
        key_value_heads,
        head_dim,
        cached_tokens: *cached_tokens,
        cache_capacity_tokens: *capacity_tokens,
    })?;
    let cache_offset = *cached_tokens * kv_width;
    key_cache[cache_offset..cache_offset + kv_width].copy_from_slice(&key);
    value_cache[cache_offset..cache_offset + kv_width].copy_from_slice(&value);
    *cached_tokens += 1;
    let gate = &q_and_gate[query_width..];
    let gated = attended
        .into_iter()
        .zip(gate)
        .map(|(value, gate)| value / (1.0 + (-gate).exp()))
        .collect::<Vec<_>>();
    gguf_matvec_with_retained(context, retained, store, staged, &output_projection, &gated)
}

#[cfg(target_os = "macos")]
fn flatten_batch(values: &[Vec<f32>]) -> Vec<f32> {
    values
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect()
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn qwen35_decoder_block_many(
    context: &MetalContext,
    store: &GgufModelStore,
    retained: &HashMap<String, QuantWeight>,
    staged: Option<&StagedWeights>,
    layer: usize,
    positions: &[usize],
    hidden: &[Vec<f32>],
    state: &mut Qwen35LayerState,
) -> Result<Vec<Vec<f32>>, String> {
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    let batch = hidden.len();
    if batch == 0 || batch != positions.len() || batch > 8 {
        return Err("Qwen3.6 batched block size is invalid".into());
    }
    if hidden.iter().any(|value| value.len() != config.hidden_size) {
        return Err("Qwen3.6 batched hidden size is invalid".into());
    }
    let prefix = format!("blk.{layer}");
    let input_norm = f32_values(
        &store
            .tensor(&format!("{prefix}.attn_norm.weight"))
            .map_err(|error| error.to_string())?,
    )?;
    let normalized_inputs = hidden
        .iter()
        .map(|value| {
            context.rms_norm_heads(
                value,
                input_norm,
                1,
                config.hidden_size,
                config.rms_norm_eps,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let normalized_flat = flatten_batch(&normalized_inputs);
    let mixer = match store
        .qwen35_layer_kind(layer)
        .map_err(|error| error.to_string())?
    {
        GgufQwen35LayerKind::GatedDeltaNet => {
            let qkv = store
                .tensor(&format!("{prefix}.attn_qkv.weight"))
                .map_err(|error| error.to_string())?;
            let z_projection = store
                .tensor(&format!("{prefix}.attn_gate.weight"))
                .map_err(|error| error.to_string())?;
            let alpha_projection = store
                .tensor(&format!("{prefix}.ssm_alpha.weight"))
                .map_err(|error| error.to_string())?;
            let beta_projection = store
                .tensor(&format!("{prefix}.ssm_beta.weight"))
                .map_err(|error| error.to_string())?;
            let convolution = store
                .tensor(&format!("{prefix}.ssm_conv1d.weight"))
                .map_err(|error| error.to_string())?;
            let a_log = f32_values(
                &store
                    .tensor(&format!("{prefix}.ssm_a"))
                    .map_err(|error| error.to_string())?,
            )?;
            let dt_bias = f32_values(
                &store
                    .tensor(&format!("{prefix}.ssm_dt.bias"))
                    .map_err(|error| error.to_string())?,
            )?;
            let norm = f32_values(
                &store
                    .tensor(&format!("{prefix}.ssm_norm.weight"))
                    .map_err(|error| error.to_string())?,
            )?;
            let output_projection = store
                .tensor(&format!("{prefix}.ssm_out.weight"))
                .map_err(|error| error.to_string())?;
            let qkv_values = gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &qkv,
                batch,
                &normalized_flat,
            )?;
            let z_values = gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &z_projection,
                batch,
                &normalized_flat,
            )?;
            let alpha_values = normalized_inputs
                .iter()
                .map(|value| {
                    context.f32_matvec_tensor_rows(
                        &alpha_projection,
                        0,
                        alpha_projection.info.shape[1],
                        value,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let beta_values = normalized_inputs
                .iter()
                .map(|value| {
                    context.f32_matvec_tensor_rows(
                        &beta_projection,
                        0,
                        beta_projection.info.shape[1],
                        value,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let Qwen35LayerState::GatedDelta {
                conv_state,
                recurrent_state,
            } = state
            else {
                return Err("batched Gated DeltaNet state kind is invalid".into());
            };
            let convolution_weights = f32_values(&convolution)?;
            let mut core_values = Vec::with_capacity(batch);
            let key_heads = config.ssm_key_heads();
            let value_heads = config.ssm_value_heads();
            let head_dim = config.ssm_value_dim();
            let key_width = key_heads * head_dim;
            for candidate in 0..batch {
                let (mixed_qkv, updated_conv_state) = context.causal_conv1d_step(
                    &qkv_values[candidate],
                    conv_state,
                    convolution_weights,
                    config.ssm_projection_size(),
                    config.ssm_conv_kernel,
                )?;
                *conv_state = updated_conv_state;
                let query = repeat_heads(
                    &l2_normalize_heads(&mixed_qkv[..key_width], key_heads, head_dim),
                    key_heads,
                    value_heads,
                    head_dim,
                )
                .into_iter()
                .map(|value| value / (head_dim as f32).sqrt())
                .collect::<Vec<_>>();
                let key = repeat_heads(
                    &l2_normalize_heads(&mixed_qkv[key_width..key_width * 2], key_heads, head_dim),
                    key_heads,
                    value_heads,
                    head_dim,
                );
                let value = &mixed_qkv[key_width * 2..];
                let beta = beta_values[candidate]
                    .iter()
                    .map(|value| 1.0 / (1.0 + (-value).exp()))
                    .collect::<Vec<_>>();
                let gate = alpha_values[candidate]
                    .iter()
                    .zip(a_log.iter().zip(dt_bias.iter()))
                    .map(|(value, (a_log, dt_bias))| -a_log.exp() * softplus(value + dt_bias))
                    .collect::<Vec<_>>();
                let (core, updated_recurrent_state) = context.gated_delta_step(
                    &query,
                    &key,
                    value,
                    &gate,
                    &beta,
                    recurrent_state,
                    value_heads,
                    head_dim,
                    head_dim,
                )?;
                *recurrent_state = updated_recurrent_state;
                let normalized = context.rms_norm_gated(
                    &core,
                    &z_values[candidate],
                    norm,
                    value_heads,
                    head_dim,
                    config.rms_norm_eps,
                )?;
                core_values.push(normalized);
            }
            gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &output_projection,
                batch,
                &flatten_batch(&core_values),
            )?
        }
        GgufQwen35LayerKind::FullAttention => {
            let q_projection = store
                .tensor(&format!("{prefix}.attn_q.weight"))
                .map_err(|error| error.to_string())?;
            let k_projection = store
                .tensor(&format!("{prefix}.attn_k.weight"))
                .map_err(|error| error.to_string())?;
            let v_projection = store
                .tensor(&format!("{prefix}.attn_v.weight"))
                .map_err(|error| error.to_string())?;
            let output_projection = store
                .tensor(&format!("{prefix}.attn_output.weight"))
                .map_err(|error| error.to_string())?;
            let q_norm = f32_values(
                &store
                    .tensor(&format!("{prefix}.attn_q_norm.weight"))
                    .map_err(|error| error.to_string())?,
            )?;
            let k_norm = f32_values(
                &store
                    .tensor(&format!("{prefix}.attn_k_norm.weight"))
                    .map_err(|error| error.to_string())?,
            )?;
            let q_values = gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &q_projection,
                batch,
                &normalized_flat,
            )?;
            let k_values = gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &k_projection,
                batch,
                &normalized_flat,
            )?;
            let v_values = gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &v_projection,
                batch,
                &normalized_flat,
            )?;
            let Qwen35LayerState::FullAttention {
                key_cache,
                value_cache,
                cached_tokens,
                capacity_tokens,
            } = state
            else {
                return Err("batched full-attention state kind is invalid".into());
            };
            let query_heads = config.num_attention_heads;
            let key_value_heads = config.num_key_value_heads;
            let head_dim = config.head_dim;
            let query_width = query_heads * head_dim;
            let kv_width = key_value_heads * head_dim;
            let rotary_dim = store
                .metadata_u32("qwen35.rope.dimension_count")
                .ok_or("GGUF model is missing qwen35.rope.dimension_count")?
                as usize;
            let mut attended_values = Vec::with_capacity(batch);
            for candidate in 0..batch {
                let query = context.rms_norm_heads(
                    &q_values[candidate][..query_width],
                    q_norm,
                    query_heads,
                    head_dim,
                    config.rms_norm_eps,
                )?;
                let key = context.rms_norm_heads(
                    &k_values[candidate],
                    k_norm,
                    key_value_heads,
                    head_dim,
                    config.rms_norm_eps,
                )?;
                let query = partial_rope(
                    context,
                    &query,
                    query_heads,
                    head_dim,
                    rotary_dim,
                    positions[candidate],
                    config.rope_theta,
                )?;
                let key = partial_rope(
                    context,
                    &key,
                    key_value_heads,
                    head_dim,
                    rotary_dim,
                    positions[candidate],
                    config.rope_theta,
                )?;
                let value = &v_values[candidate];
                if *cached_tokens >= *capacity_tokens {
                    return Err("batched full-attention KV cache capacity exhausted".into());
                }
                let attended = context.attention_decode(AttentionDecodeInput {
                    query: &query,
                    key_cache,
                    value_cache,
                    new_keys: &key,
                    new_values: value,
                    query_heads,
                    key_value_heads,
                    head_dim,
                    cached_tokens: *cached_tokens,
                    cache_capacity_tokens: *capacity_tokens,
                })?;
                let cache_offset = *cached_tokens * kv_width;
                key_cache[cache_offset..cache_offset + kv_width].copy_from_slice(&key);
                value_cache[cache_offset..cache_offset + kv_width].copy_from_slice(value);
                *cached_tokens += 1;
                let gate = &q_values[candidate][query_width..];
                attended_values.push(
                    attended
                        .into_iter()
                        .zip(gate)
                        .map(|(value, gate)| value / (1.0 + (-gate).exp()))
                        .collect::<Vec<_>>(),
                );
            }
            gguf_matmul_many_with_retained(
                context,
                retained,
                store,
                staged,
                &output_projection,
                batch,
                &flatten_batch(&attended_values),
            )?
        }
    };
    let mut residual = hidden
        .iter()
        .zip(mixer)
        .map(|(input, mixer)| {
            input
                .iter()
                .zip(mixer)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let post_norm = f32_values(
        &store
            .tensor(&format!("{prefix}.post_attention_norm.weight"))
            .map_err(|error| error.to_string())?,
    )?;
    let normalized = residual
        .iter()
        .map(|value| {
            context.rms_norm_heads(value, post_norm, 1, config.hidden_size, config.rms_norm_eps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let gate = store
        .tensor(&format!("{prefix}.ffn_gate.weight"))
        .map_err(|error| error.to_string())?;
    let up = store
        .tensor(&format!("{prefix}.ffn_up.weight"))
        .map_err(|error| error.to_string())?;
    let down = store
        .tensor(&format!("{prefix}.ffn_down.weight"))
        .map_err(|error| error.to_string())?;
    let gate_values = gguf_matmul_many_with_retained(
        context,
        retained,
        store,
        staged,
        &gate,
        batch,
        &flatten_batch(&normalized),
    )?;
    let up_values = gguf_matmul_many_with_retained(
        context,
        retained,
        store,
        staged,
        &up,
        batch,
        &flatten_batch(&normalized),
    )?;
    let fused = gate_values
        .into_iter()
        .zip(up_values)
        .map(|(gate, up)| {
            gate.into_iter()
                .zip(up)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mlp = gguf_matmul_many_with_retained(
        context,
        retained,
        store,
        staged,
        &down,
        batch,
        &flatten_batch(&fused),
    )?;
    for (residual, mlp) in residual.iter_mut().zip(mlp) {
        for (value, mlp_value) in residual.iter_mut().zip(mlp) {
            *value += mlp_value;
        }
    }
    Ok(residual)
}

/// Evaluate one Qwen3.6 decoder block using caller-owned recurrent/KV state.
#[cfg(target_os = "macos")]
pub fn qwen35_decoder_block_stateful(
    context: &MetalContext,
    store: &GgufModelStore,
    layer: usize,
    position: usize,
    hidden: &[f32],
    state: &mut Qwen35LayerState,
) -> Result<Vec<f32>, String> {
    qwen35_decoder_block_stateful_with_retained(
        context,
        store,
        &HashMap::new(),
        layer,
        position,
        hidden,
        state,
    )
}

#[cfg(target_os = "macos")]
pub fn qwen35_decoder_block_stateful_with_retained(
    context: &MetalContext,
    store: &GgufModelStore,
    retained: &HashMap<String, QuantWeight>,
    layer: usize,
    position: usize,
    hidden: &[f32],
    state: &mut Qwen35LayerState,
) -> Result<Vec<f32>, String> {
    qwen35_decoder_block_stateful_with_retained_and_staged(
        context, store, retained, None, layer, position, hidden, state,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn qwen35_decoder_block_stateful_with_retained_and_staged(
    context: &MetalContext,
    store: &GgufModelStore,
    retained: &HashMap<String, QuantWeight>,
    staged: Option<&StagedWeights>,
    layer: usize,
    position: usize,
    hidden: &[f32],
    state: &mut Qwen35LayerState,
) -> Result<Vec<f32>, String> {
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    if hidden.len() != config.hidden_size || layer >= config.num_hidden_layers {
        return Err("decoder block hidden size or layer index is invalid".into());
    }
    let prefix = format!("blk.{layer}");
    let input_norm = store
        .tensor(&format!("{prefix}.attn_norm.weight"))
        .map_err(|error| error.to_string())?;
    let input_norm = f32_values(&input_norm)?;
    let normalized = context.rms_norm_heads(
        hidden,
        input_norm,
        1,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    let mixer = match store
        .qwen35_layer_kind(layer)
        .map_err(|error| error.to_string())?
    {
        GgufQwen35LayerKind::GatedDeltaNet => {
            gdn_token_mixer(context, store, retained, staged, layer, &normalized, state)?
        }
        GgufQwen35LayerKind::FullAttention => attention_token_mixer(
            context,
            store,
            retained,
            staged,
            layer,
            position,
            &normalized,
            state,
        )?,
    };
    let mut residual = hidden.to_vec();
    for (value, mixer_value) in residual.iter_mut().zip(mixer) {
        *value += mixer_value;
    }
    let post_norm = store
        .tensor(&format!("{prefix}.post_attention_norm.weight"))
        .map_err(|error| error.to_string())?;
    let post_norm = f32_values(&post_norm)?;
    let normalized = context.rms_norm_heads(
        &residual,
        post_norm,
        1,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    let gate = store
        .tensor(&format!("{prefix}.ffn_gate.weight"))
        .map_err(|error| error.to_string())?;
    let up = store
        .tensor(&format!("{prefix}.ffn_up.weight"))
        .map_err(|error| error.to_string())?;
    let down = store
        .tensor(&format!("{prefix}.ffn_down.weight"))
        .map_err(|error| error.to_string())?;
    if gate.info.ggml_type != crate::quant::GGML_TYPE_Q4_K
        || up.info.ggml_type != crate::quant::GGML_TYPE_Q4_K
        || gate.info.shape != up.info.shape
        || gate.info.shape.len() != 2
    {
        return Err("decoder MLP gate/up tensors are not matching Q4_K matrices".into());
    }
    let columns = gate.info.shape[0];
    let rows = gate.info.shape[1];
    let staged_gate_owned = if staged.is_none()
        && std::env::var_os("SI_STAGE_GGUF").is_some()
        && gate.info.ggml_type != 0
    {
        Some(
            store
                .stage_tensor_payload(&gate)
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let staged_up_owned = if staged.is_none()
        && std::env::var_os("SI_STAGE_GGUF").is_some()
        && up.info.ggml_type != 0
    {
        Some(
            store
                .stage_tensor_payload(&up)
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let gate_bytes = staged
        .and_then(|weights| weights.get(&gate.info.name))
        .or(staged_gate_owned.as_deref())
        .unwrap_or(gate.bytes);
    let up_bytes = staged
        .and_then(|weights| weights.get(&up.info.name))
        .or(staged_up_owned.as_deref())
        .unwrap_or(up.bytes);
    let gate_weight = retained.get(&gate.info.name).cloned().or_else(|| {
        staged.and_then(|weights| weights.quant_weight(&gate.info.name, gate.info.ggml_type))
    });
    let up_weight = retained.get(&up.info.name).cloned().or_else(|| {
        staged.and_then(|weights| weights.quant_weight(&up.info.name, up.info.ggml_type))
    });
    let down_weight = retained.get(&down.info.name).cloned().or_else(|| {
        staged.and_then(|weights| weights.quant_weight(&down.info.name, down.info.ggml_type))
    });
    let mlp = if std::env::var("SI_FUSED_MLP").ok().as_deref() == Some("1")
        && down.info.shape.len() == 2
        && down.info.shape[0] == rows
        && down.info.shape[1] > 0
    {
        match (
            gate_weight.as_ref(),
            up_weight.as_ref(),
            down_weight.as_ref(),
        ) {
            (Some(gate_weight), Some(up_weight), Some(down_weight)) => context
                .gguf_quant_fused_mlp_weights(
                    gate_weight,
                    up_weight,
                    down_weight,
                    rows,
                    columns,
                    down.info.shape[1],
                    &normalized,
                )?,
            _ => {
                let fused =
                    context.q4_k_fused_gate_up(gate_bytes, up_bytes, rows, columns, &normalized)?;
                gguf_matvec_with_retained(context, retained, store, staged, &down, &fused)?
            }
        }
    } else {
        let fused = match (gate_weight.as_ref(), up_weight.as_ref()) {
            (Some(gate_weight), Some(up_weight)) => context.q4_k_fused_gate_up_weights(
                gate_weight,
                up_weight,
                rows,
                columns,
                &normalized,
            )?,
            _ => context.q4_k_fused_gate_up(gate_bytes, up_bytes, rows, columns, &normalized)?,
        };
        gguf_matvec_with_retained(context, retained, store, staged, &down, &fused)?
    };
    for (value, mlp_value) in residual.iter_mut().zip(mlp) {
        *value += mlp_value;
    }
    Ok(residual)
}

/// Evaluate one Qwen3.6 decoder block from a freshly initialized state.
#[cfg(target_os = "macos")]
pub fn qwen35_decoder_block(
    context: &MetalContext,
    store: &GgufModelStore,
    layer: usize,
    position: usize,
    hidden: &[f32],
) -> Result<Vec<f32>, String> {
    let mut state = Qwen35LayerState::new(store, layer, 1)?;
    qwen35_decoder_block_stateful_with_retained(
        context,
        store,
        &HashMap::new(),
        layer,
        position,
        hidden,
        &mut state,
    )
}

/// Sequential GGUF runtime for the full Qwen3.6 hybrid stack.
///
/// The runtime owns one persistent state object per layer. It intentionally
/// keeps tensor bytes mmap-backed and only allocates recurrent/KV state plus
/// operation outputs, preserving the low-residency execution model.
#[cfg(target_os = "macos")]
pub struct Qwen35Runtime<'a> {
    context: &'a MetalContext,
    store: &'a GgufModelStore,
    states: Vec<Qwen35LayerState>,
    active: HashMap<String, QuantWeight>,
    prefetched_layer0: Option<StagedWeights>,
}

#[cfg(target_os = "macos")]
impl<'a> Qwen35Runtime<'a> {
    pub fn new(
        context: &'a MetalContext,
        store: &'a GgufModelStore,
        capacity_tokens: usize,
    ) -> Result<Self, String> {
        Self::new_with_retained_layers_and_head(context, store, capacity_tokens, 0, false)
    }

    pub fn new_with_retained_layers(
        context: &'a MetalContext,
        store: &'a GgufModelStore,
        capacity_tokens: usize,
        retained_layers: usize,
    ) -> Result<Self, String> {
        Self::new_with_retained_layers_and_head(
            context,
            store,
            capacity_tokens,
            retained_layers,
            false,
        )
    }

    pub fn new_with_retained_layers_and_head(
        context: &'a MetalContext,
        store: &'a GgufModelStore,
        capacity_tokens: usize,
        retained_layers: usize,
        retain_output_head: bool,
    ) -> Result<Self, String> {
        let config = store.qwen35_config().map_err(|error| error.to_string())?;
        let mut states = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            states.push(Qwen35LayerState::new(store, layer, capacity_tokens)?);
        }
        let mut active = HashMap::new();
        let retain_profiled_weights = retained_layers == 0;
        let retained_mlp_layers = if retain_profiled_weights {
            std::env::var("SI_RETAIN_MLP_LAYERS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let retained_mlp_stride = if retain_profiled_weights {
            std::env::var("SI_RETAIN_MLP_STRIDE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|stride| *stride > 0)
                .unwrap_or(0)
        } else {
            0
        };
        let retained_mlp_offset = if retain_profiled_weights {
            std::env::var("SI_RETAIN_MLP_OFFSET")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let retained_q6_layers = if retain_profiled_weights {
            std::env::var("SI_RETAIN_Q6_LAYERS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let retained_down_stride = if retain_profiled_weights {
            std::env::var("SI_RETAIN_DOWN_STRIDE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|stride| *stride > 0)
                .unwrap_or(0)
        } else {
            0
        };
        let retained_down_offset = if retain_profiled_weights {
            std::env::var("SI_RETAIN_DOWN_OFFSET")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let retained_attention_stride = if retain_profiled_weights {
            std::env::var("SI_RETAIN_ATTENTION_STRIDE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|stride| *stride > 0)
                .unwrap_or(0)
        } else {
            0
        };
        let retained_attention_offset = if retain_profiled_weights {
            std::env::var("SI_RETAIN_ATTENTION_OFFSET")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let retained_full_layers = if retain_profiled_weights {
            std::env::var("SI_RETAIN_FULL_LAYERS")
                .ok()
                .map(|value| parse_layer_set(&value))
                .unwrap_or_default()
        } else {
            BTreeSet::new()
        };
        let map_all_weights = std::env::var("SI_MAP_ALL_GGUF").ok().as_deref() == Some("1");
        if !map_all_weights {
            for name in store.tensors.keys() {
                let is_retained_layer =
                    (0..retained_layers).any(|layer| name.starts_with(&format!("blk.{layer}.")));
                let is_retained_mlp = retained_mlp_layers > 0
                    && (name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight"))
                    && (0..retained_mlp_layers)
                        .any(|layer| name.starts_with(&format!("blk.{layer}.")))
                    || retained_mlp_stride > 0
                        && (name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight"))
                        && (0..config.num_hidden_layers).any(|layer| {
                            retains_stride_layer(layer, retained_mlp_stride, retained_mlp_offset)
                                && name.starts_with(&format!("blk.{layer}."))
                        });
                let is_retained_q6 = retained_q6_layers > 0
                    && (0..retained_q6_layers)
                        .any(|layer| name.starts_with(&format!("blk.{layer}.")))
                    && store
                        .tensor(name)
                        .map(|tensor| tensor.info.ggml_type == crate::quant::GGML_TYPE_Q6_K)
                        .unwrap_or(false);
                let layer_for_tensor = name
                    .strip_prefix("blk.")
                    .and_then(|value| value.split('.').next())
                    .and_then(|value| value.parse::<usize>().ok());
                let is_retained_down = retained_down_stride > 0
                    && name.ends_with(".ffn_down.weight")
                    && layer_for_tensor.is_some_and(|layer| {
                        retains_stride_layer(layer, retained_down_stride, retained_down_offset)
                    });
                let is_retained_attention = retained_attention_stride > 0
                    && (name.ends_with(".attn_q.weight")
                        || name.ends_with(".attn_k.weight")
                        || name.ends_with(".attn_v.weight")
                        || name.ends_with(".attn_output.weight"))
                    && layer_for_tensor.is_some_and(|layer| {
                        retains_stride_layer(
                            layer,
                            retained_attention_stride,
                            retained_attention_offset,
                        )
                    });
                let is_retained_full = retained_full_layers
                    .iter()
                    .any(|layer| name.starts_with(&format!("blk.{layer}.")));
                if !is_retained_layer
                    && !is_retained_mlp
                    && !is_retained_q6
                    && !is_retained_down
                    && !is_retained_attention
                    && !is_retained_full
                {
                    continue;
                }
                let tensor = store.tensor(name).map_err(|error| error.to_string())?;
                if tensor.info.shape.len() != 2 || tensor.info.ggml_type == 0 {
                    continue;
                }
                let weight =
                    context.upload_quant_weight_private(tensor.bytes, tensor.info.ggml_type)?;
                active.insert(name.clone(), weight);
            }
        } else {
            for name in store.tensors.keys() {
                let tensor = store.tensor(name).map_err(|error| error.to_string())?;
                if tensor.info.shape.len() != 2 || tensor.info.ggml_type == 0 {
                    continue;
                }
                let weight =
                    context.bind_quant_weight_mapped(tensor.bytes, tensor.info.ggml_type)?;
                active.insert(name.clone(), weight);
            }
        }
        if retain_output_head {
            let tensor = store
                .tensor("output.weight")
                .map_err(|error| error.to_string())?;
            if tensor.info.shape.len() != 2 || tensor.info.ggml_type == 0 {
                return Err("GGUF output head is not a quantized matrix".into());
            }
            let weight =
                context.upload_quant_weight_private(tensor.bytes, tensor.info.ggml_type)?;
            active.insert(tensor.info.name.clone(), weight);
        }
        Ok(Self {
            context,
            store,
            states,
            active,
            prefetched_layer0: None,
        })
    }

    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(Qwen35LayerState::state_bytes).sum()
    }

    pub fn retained_weight_bytes(&self) -> u64 {
        self.active
            .values()
            .filter(|weight| !weight.mapped)
            .map(QuantWeight::byte_len)
            .sum()
    }

    pub fn layer_count(&self) -> usize {
        self.states.len()
    }

    pub fn config(&self) -> Result<crate::model::GgufQwen35Config, String> {
        self.store
            .qwen35_config()
            .map_err(|error| error.to_string())
    }

    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
        self.prefetched_layer0 = None;
    }

    pub fn snapshot_states(&self) -> Vec<Qwen35LayerState> {
        self.states.clone()
    }

    pub fn restore_states(&mut self, snapshot: &[Qwen35LayerState]) -> Result<(), String> {
        if snapshot.len() != self.states.len() {
            return Err("Qwen3.6 state snapshot depth does not match runtime".into());
        }
        self.states.clone_from_slice(snapshot);
        Ok(())
    }

    pub fn decode_hidden(&mut self, position: usize, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        if hidden.len() != config.hidden_size {
            return Err("runtime hidden state has the wrong length".into());
        }
        let context = self.context;
        let store = self.store;
        let profile_layers = std::env::var_os("SI_PROFILE_QWEN35").is_some();
        let prefetch_layers = std::env::var_os("SI_PREFETCH_GGUF").is_some();
        let stage_pipeline = std::env::var_os("SI_STAGE_PIPELINE").is_some();
        let stage_depth = if stage_pipeline {
            stage_pipeline_depth(std::env::var("SI_STAGE_DEPTH").ok().as_deref())
        } else {
            0
        };
        let retained_names = self.active.keys().cloned().collect::<BTreeSet<_>>();
        let initial_staged = if stage_pipeline {
            Some(match self.prefetched_layer0.take() {
                Some(staged) => staged,
                None => stage_qwen_layer_for_execution(context, store, 0, &retained_names)?,
            })
        } else {
            None
        };
        thread::scope(|scope| {
            let mut current = hidden.to_vec();
            let mut staged = initial_staged;
            let mut pending_next = None;
            let mut pending_next2 = None;
            let mut pending_next3 = None;
            for layer in 0..self.states.len() {
                if prefetch_layers && !stage_pipeline {
                    if let Some(next_layer) = layer
                        .checked_add(1)
                        .filter(|next| *next < self.states.len())
                    {
                        let _ = store.advise_qwen35_layer(next_layer);
                    }
                }
                let next_handle = if stage_pipeline {
                    match pending_next.take() {
                        Some(handle) => Some(handle),
                        None => {
                            let retained_names_ref = &retained_names;
                            layer
                                .checked_add(1)
                                .filter(|next| *next < self.states.len())
                                .map(|next_layer| {
                                    scope.spawn(move || {
                                        stage_qwen_layer_for_execution(
                                            context,
                                            store,
                                            next_layer,
                                            retained_names_ref,
                                        )
                                    })
                                })
                        }
                    }
                } else {
                    None
                };
                let next2_handle = if stage_pipeline && stage_depth >= 2 {
                    match pending_next2.take() {
                        Some(handle) => Some(handle),
                        None => {
                            let retained_names_ref = &retained_names;
                            layer
                                .checked_add(2)
                                .filter(|next| *next < self.states.len())
                                .map(|next_layer| {
                                    scope.spawn(move || {
                                        stage_qwen_layer_for_execution(
                                            context,
                                            store,
                                            next_layer,
                                            retained_names_ref,
                                        )
                                    })
                                })
                        }
                    }
                } else {
                    None
                };
                let next3_handle = if stage_pipeline && stage_depth >= 3 {
                    match pending_next3.take() {
                        Some(handle) => Some(handle),
                        None => {
                            let retained_names_ref = &retained_names;
                            layer
                                .checked_add(3)
                                .filter(|next| *next < self.states.len())
                                .map(|next_layer| {
                                    scope.spawn(move || {
                                        stage_qwen_layer_for_execution(
                                            context,
                                            store,
                                            next_layer,
                                            retained_names_ref,
                                        )
                                    })
                                })
                        }
                    }
                } else {
                    None
                };
                let next4_handle = if stage_pipeline && stage_depth >= 4 {
                    let retained_names_ref = &retained_names;
                    layer
                        .checked_add(4)
                        .filter(|next| *next < self.states.len())
                        .map(|next_layer| {
                            scope.spawn(move || {
                                stage_qwen_layer_for_execution(
                                    context,
                                    store,
                                    next_layer,
                                    retained_names_ref,
                                )
                            })
                        })
                } else {
                    None
                };
                let state = &mut self.states[layer];
                let layer_start = std::time::Instant::now();
                let staged_ref = staged.as_ref();
                current = metal::objc::rc::autoreleasepool(|| {
                    qwen35_decoder_block_stateful_with_retained_and_staged(
                        context,
                        store,
                        &self.active,
                        staged_ref,
                        layer,
                        position,
                        &current,
                        state,
                    )
                })?;
                if profile_layers {
                    eprintln!(
                        "si_qwen35_layer layer={} ms={:.3}",
                        layer,
                        layer_start.elapsed().as_secs_f64() * 1_000.0
                    );
                }
                staged = match next_handle {
                    Some(handle) => Some(
                        handle
                            .join()
                            .map_err(|_| "GGUF staging worker panicked".to_owned())??,
                    ),
                    None => None,
                };
                pending_next = next2_handle;
                pending_next2 = next3_handle;
                pending_next3 = next4_handle;
            }
            Ok(current)
        })
    }

    /// Run only a retained prefix of the target layers. This is an
    /// intentionally approximate self-drafter primitive: callers must restore
    /// the layer-state snapshot before exact target verification. No target
    /// weights are changed and the full decoder path remains untouched.
    pub fn decode_hidden_prefix_layers(
        &mut self,
        position: usize,
        hidden: &[f32],
        layer_count: usize,
    ) -> Result<Vec<f32>, String> {
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        if layer_count == 0 || layer_count > self.states.len() || hidden.len() != config.hidden_size
        {
            return Err("Qwen3.6 prefix layer count or hidden size is invalid".into());
        }
        let mut current = hidden.to_vec();
        for layer in 0..layer_count {
            let state = &mut self.states[layer];
            current = qwen35_decoder_block_stateful_with_retained(
                self.context,
                self.store,
                &self.active,
                layer,
                position,
                &current,
                state,
            )?;
        }
        Ok(current)
    }

    pub fn decode_hidden_prefix(
        &mut self,
        position: usize,
        hidden: &[f32],
        layers: usize,
    ) -> Result<Vec<f32>, String> {
        if layers == 0 || layers > self.states.len() {
            return Err("Qwen3.6 draft layer count is invalid".into());
        }
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        if hidden.len() != config.hidden_size {
            return Err("runtime hidden state has the wrong length".into());
        }
        let mut current = hidden.to_vec();
        for layer in 0..layers {
            let state = &mut self.states[layer];
            current = qwen35_decoder_block_stateful_with_retained(
                self.context,
                self.store,
                &self.active,
                layer,
                position,
                &current,
                state,
            )?;
        }
        Ok(current)
    }

    /// Execute an explicitly selected ordered subset of decoder layers. This
    /// is the lossless target-independent hook used by self-speculative
    /// experiments: the result is only a proposal, and exact verification
    /// still runs through every target layer.
    pub fn decode_hidden_layer_set(
        &mut self,
        position: usize,
        hidden: &[f32],
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        if layers.is_empty() || layers.iter().any(|layer| *layer >= self.states.len()) {
            return Err("Qwen3.6 layer-set draft is invalid".into());
        }
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        if hidden.len() != config.hidden_size {
            return Err("runtime hidden state has the wrong length".into());
        }
        let mut current = hidden.to_vec();
        for &layer in layers {
            let state = &mut self.states[layer];
            current = qwen35_decoder_block_stateful_with_retained(
                self.context,
                self.store,
                &self.active,
                layer,
                position,
                &current,
                state,
            )?;
        }
        Ok(current)
    }

    /// Evaluate a sequential candidate window while binding each streamed
    /// quantized matrix once for the whole window. This is the GGUF SI-004
    /// verifier primitive; callers still decide which prefix to commit.
    pub fn decode_tokens_many(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tokens.is_empty() || tokens.len() != positions.len() || tokens.len() > 8 {
            return Err("Qwen3.6 candidate window must contain 1..8 tokens".into());
        }
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        let mut current = tokens
            .iter()
            .map(|token| self.embed_token(*token))
            .collect::<Result<Vec<_>, _>>()?;
        let context = self.context;
        let store = self.store;
        let stage_pipeline = std::env::var_os("SI_STAGE_PIPELINE").is_some();
        let retained_names = self.active.keys().cloned().collect::<BTreeSet<_>>();
        thread::scope(|scope| {
            let mut staged = if stage_pipeline {
                Some(stage_qwen_layer_for_execution(
                    context,
                    store,
                    0,
                    &retained_names,
                )?)
            } else {
                None
            };
            for layer in 0..config.num_hidden_layers {
                let next_handle = if stage_pipeline {
                    let retained_names_ref = &retained_names;
                    layer
                        .checked_add(1)
                        .filter(|next| *next < config.num_hidden_layers)
                        .map(|next_layer| {
                            scope.spawn(move || {
                                stage_qwen_layer_for_execution(
                                    context,
                                    store,
                                    next_layer,
                                    retained_names_ref,
                                )
                            })
                        })
                } else {
                    None
                };
                let staged_ref = staged.as_ref();
                current = qwen35_decoder_block_many(
                    context,
                    store,
                    &self.active,
                    staged_ref,
                    layer,
                    positions,
                    &current,
                    &mut self.states[layer],
                )?;
                staged = match next_handle {
                    Some(handle) => Some(
                        handle
                            .join()
                            .map_err(|_| "GGUF batched staging worker panicked".to_owned())??,
                    ),
                    None => None,
                };
            }
            Ok(current)
        })
    }

    pub fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let tensor = self
            .store
            .tensor("token_embd.weight")
            .map_err(|error| error.to_string())?;
        self.context
            .q4_k_embedding_tensor(&tensor, token_id as usize)
            .map_err(|error| error.to_string())
    }

    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        let norm = self
            .store
            .tensor("output_norm.weight")
            .map_err(|error| error.to_string())?;
        let norm = f32_values(&norm)?;
        let normalized = self.context.rms_norm_heads(
            hidden,
            norm,
            1,
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let output = self
            .store
            .tensor("output.weight")
            .map_err(|error| error.to_string())?;
        let stage_pipeline = std::env::var_os("SI_STAGE_PIPELINE").is_some();
        let stage_output_prefetch = stage_output_prefetch_enabled(
            stage_pipeline,
            std::env::var("SI_STAGE_OUTPUT_PREFETCH").ok().as_deref(),
        );
        if !stage_output_prefetch {
            return gguf_matvec_with_retained(
                self.context,
                &self.active,
                self.store,
                None,
                &output,
                &normalized,
            );
        }

        let retained_names = self.active.keys().cloned().collect::<BTreeSet<_>>();
        let store = self.store;
        let context = self.context;
        let active = &self.active;
        let output_started = std::time::Instant::now();
        let (logits, prefetched) = thread::scope(|scope| {
            let handle = scope
                .spawn(move || stage_qwen_layer_for_execution(context, store, 0, &retained_names));
            let logits =
                gguf_matvec_with_retained(context, active, store, None, &output, &normalized);
            let prefetched = handle
                .join()
                .map_err(|_| "GGUF output-head staging worker panicked".to_owned())??;
            Ok::<_, String>((logits?, prefetched))
        })?;
        self.prefetched_layer0 = Some(prefetched);
        if std::env::var_os("SI_PROFILE_QWEN35").is_some() {
            eprintln!(
                "si_qwen35_output_head ms={:.3}",
                output_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        Ok(logits)
    }

    pub fn logits_many(&self, hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        if hidden.is_empty() || hidden.len() > 8 {
            return Err("Qwen3.6 batched logits size is invalid".into());
        }
        let config = self
            .store
            .qwen35_config()
            .map_err(|error| error.to_string())?;
        let norm = self
            .store
            .tensor("output_norm.weight")
            .map_err(|error| error.to_string())?;
        let norm = f32_values(&norm)?;
        let normalized = hidden
            .iter()
            .map(|value| {
                self.context
                    .rms_norm_heads(value, norm, 1, config.hidden_size, config.rms_norm_eps)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .store
            .tensor("output.weight")
            .map_err(|error| error.to_string())?;
        gguf_matmul_many_with_retained(
            self.context,
            &self.active,
            self.store,
            None,
            &output,
            hidden.len(),
            &flatten_batch(&normalized),
        )
    }

    pub fn decode_token(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        let embedding = self.embed_token(token_id)?;
        let hidden = self.decode_hidden(position, &embedding)?;
        let logits = self.logits(&hidden)?;
        let mut best_id = 0_u32;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in logits.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_id = u32::try_from(index).map_err(|_| "vocabulary index exceeds u32")?;
            }
        }
        Ok((best_id, logits))
    }

    pub fn decode_token_prefix(
        &mut self,
        token_id: u32,
        position: usize,
        layers: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        let embedding = self.embed_token(token_id)?;
        let hidden = self.decode_hidden_prefix(position, &embedding, layers)?;
        let logits = self.logits(&hidden)?;
        let mut best_id = 0_u32;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in logits.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_id = u32::try_from(index).map_err(|_| "vocabulary index exceeds u32")?;
            }
        }
        Ok((best_id, logits))
    }
}

#[cfg(target_os = "macos")]
fn retains_stride_layer(layer: usize, stride: usize, offset: usize) -> bool {
    stride != 0 && layer % stride == offset % stride
}

#[cfg(target_os = "macos")]
fn parse_layer_set(value: &str) -> BTreeSet<usize> {
    value
        .split(',')
        .filter_map(|item| item.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(target_os = "macos")]
fn stage_output_prefetch_enabled(stage_pipeline: bool, setting: Option<&str>) -> bool {
    stage_pipeline && setting != Some("0")
}

#[cfg(target_os = "macos")]
fn stage_pipeline_depth(setting: Option<&str>) -> usize {
    setting
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|depth| *depth > 0)
        .map_or(3, |depth| depth.min(4))
}

#[cfg(target_os = "macos")]
fn stage_qwen_layer_for_execution(
    context: &MetalContext,
    store: &GgufModelStore,
    layer: usize,
    retained_names: &BTreeSet<String>,
) -> Result<StagedWeights, String> {
    let staged = store
        .stage_qwen35_layer_packed(layer, retained_names)
        .map_err(|error| error.to_string())?;
    promote_staged_layer(context, staged)
}

#[cfg(target_os = "macos")]
fn promote_staged_layer(
    context: &MetalContext,
    staged: StagedQwenLayer,
) -> Result<StagedWeights, String> {
    if staged.packed_bytes().is_empty() {
        return Ok(StagedWeights::Empty);
    }
    if std::env::var("SI_STAGE_PRIVATE").ok().as_deref() == Some("1") {
        Ok(StagedWeights::Private(
            context.upload_staged_qwen_layer_private(&staged)?,
        ))
    } else {
        let buffer = context.bind_staged_qwen_layer_shared(&staged)?;
        let ranges = staged
            .ranges()
            .map(|(name, start, end)| (name.to_owned(), (start, end)))
            .collect();
        Ok(StagedWeights::Mapped(MappedStagedQwenLayer {
            backing: staged,
            buffer,
            ranges,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_layer_set, retains_stride_layer, stage_output_prefetch_enabled, stage_pipeline_depth,
    };

    #[test]
    fn stride_residency_supports_offset_without_changing_budget() {
        assert!(retains_stride_layer(0, 2, 0));
        assert!(!retains_stride_layer(1, 2, 0));
        assert!(!retains_stride_layer(0, 2, 1));
        assert!(retains_stride_layer(1, 2, 1));
        assert!(!retains_stride_layer(3, 0, 1));
    }

    #[test]
    fn output_prefetch_defaults_on_only_with_staging_pipeline() {
        assert!(!stage_output_prefetch_enabled(false, None));
        assert!(stage_output_prefetch_enabled(true, None));
        assert!(stage_output_prefetch_enabled(true, Some("1")));
        assert!(!stage_output_prefetch_enabled(true, Some("0")));
    }

    #[test]
    fn stage_depth_is_bounded_and_defaults_to_one() {
        assert_eq!(stage_pipeline_depth(None), 3);
        assert_eq!(stage_pipeline_depth(Some("2")), 2);
        assert_eq!(stage_pipeline_depth(Some("99")), 4);
        assert_eq!(stage_pipeline_depth(Some("0")), 3);
        assert_eq!(stage_pipeline_depth(Some("invalid")), 3);
    }

    #[test]
    fn full_layer_profile_parses_a_bounded_set() {
        assert_eq!(
            parse_layer_set("0, 8,invalid,8, 63"),
            [0_usize, 8, 63].into_iter().collect()
        );
    }
}
