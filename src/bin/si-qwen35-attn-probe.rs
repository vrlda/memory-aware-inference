use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use super_inference::model::{GgufModelStore, GgufQwen35LayerKind, GgufTensorView};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    layer: usize,
    position: usize,
    repetitions: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut layer = 3;
        let mut position = 0;
        let mut repetitions = 3;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--layer" => {
                    layer = value
                        .parse::<usize>()
                        .map_err(|_| "--layer must be a non-negative integer".to_owned())?;
                }
                "--position" => {
                    position = value
                        .parse::<usize>()
                        .map_err(|_| "--position must be a non-negative integer".to_owned())?;
                }
                "--repetitions" => {
                    repetitions = value
                        .parse::<usize>()
                        .map_err(|_| "--repetitions must be a positive integer".to_owned())?;
                    if repetitions == 0 {
                        return Err("--repetitions must be a positive integer".into());
                    }
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            layer,
            position,
            repetitions,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StageTimes {
    projections: Duration,
    normalization: Duration,
    attention: Duration,
    output: Duration,
}

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("error: Qwen3.6 attention probe requires macOS");
    ExitCode::from(1)
}

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
fn partial_rope(
    context: &super_inference::metal::MetalContext,
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
fn run_layer(
    context: &super_inference::metal::MetalContext,
    store: &GgufModelStore,
    layer: usize,
    position: usize,
    hidden: &[f32],
) -> Result<(Vec<f32>, StageTimes), String> {
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

    let projections_started = Instant::now();
    let q_and_gate =
        context.gguf_matvec_tensor_rows(&q_projection, 0, q_projection.info.shape[1], hidden)?;
    let key =
        context.gguf_matvec_tensor_rows(&k_projection, 0, k_projection.info.shape[1], hidden)?;
    let value =
        context.gguf_matvec_tensor_rows(&v_projection, 0, v_projection.info.shape[1], hidden)?;
    let projections = projections_started.elapsed();

    let query_heads = config.num_attention_heads;
    let key_value_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let query_width = query_heads * head_dim;
    let kv_width = key_value_heads * head_dim;
    if q_and_gate.len() != query_width * 2 || key.len() != kv_width || value.len() != kv_width {
        return Err("full-attention projection shapes do not match metadata".into());
    }

    let normalization_started = Instant::now();
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
    let normalization = normalization_started.elapsed();

    let attention_started = Instant::now();
    let cache_elements = kv_width;
    let key_cache = vec![0.0_f32; cache_elements];
    let value_cache = vec![0.0_f32; cache_elements];
    let attended = context.attention_decode(super_inference::metal::AttentionDecodeInput {
        query: &query,
        key_cache: &key_cache,
        value_cache: &value_cache,
        new_keys: &key,
        new_values: &value,
        query_heads,
        key_value_heads,
        head_dim,
        cached_tokens: 0,
        cache_capacity_tokens: 1,
    })?;
    let attention = attention_started.elapsed();

    let output_started = Instant::now();
    let gate = &q_and_gate[query_width..];
    let gated = attended
        .into_iter()
        .zip(gate)
        .map(|(value, gate)| value / (1.0 + (-gate).exp()))
        .collect::<Vec<_>>();
    let output = context.gguf_matvec_tensor_rows(
        &output_projection,
        0,
        output_projection.info.shape[1],
        &gated,
    )?;
    let output_time = output_started.elapsed();

    Ok((
        output,
        StageTimes {
            projections,
            normalization,
            attention,
            output: output_time,
        },
    ))
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = GgufModelStore::open(&args.model).map_err(|error| error.to_string())?;
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    if args.layer >= config.num_hidden_layers {
        return Err(format!(
            "--layer {} exceeds {} layers",
            args.layer, config.num_hidden_layers
        ));
    }
    if store
        .qwen35_layer_kind(args.layer)
        .map_err(|error| error.to_string())?
        != GgufQwen35LayerKind::FullAttention
    {
        return Err(format!("layer {} is a Gated DeltaNet layer", args.layer));
    }
    let hidden = (0..config.hidden_size)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let (warmup_output, _) = run_layer(&context, &store, args.layer, args.position, &hidden)?;
    context.reset_peaks();
    let started = Instant::now();
    let mut checksum = warmup_output.iter().copied().sum::<f32>();
    let mut stages = StageTimes::default();
    for _ in 0..args.repetitions {
        let (output, timing) = run_layer(&context, &store, args.layer, args.position, &hidden)?;
        checksum += output.iter().copied().sum::<f32>();
        stages.projections += timing.projections;
        stages.normalization += timing.normalization;
        stages.attention += timing.attention;
        stages.output += timing.output;
        std::hint::black_box(&output);
    }
    let repetitions = args.repetitions as f64;
    println!(
        "probe=qwen35_attention_layer layer={} position={} query_heads={} kv_heads={} head_dim={} repetitions={} total_ms={:.3} projections_ms={:.3} normalization_ms={:.3} attention_ms={:.3} output_ms={:.3} checksum={:.6} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.layer,
        args.position,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        args.repetitions,
        started.elapsed().as_secs_f64() * 1_000.0 / repetitions,
        stages.projections.as_secs_f64() * 1_000.0 / repetitions,
        stages.normalization.as_secs_f64() * 1_000.0 / repetitions,
        stages.attention.as_secs_f64() * 1_000.0 / repetitions,
        stages.output.as_secs_f64() * 1_000.0 / repetitions,
        checksum,
        context.peak_allocated_bytes(),
        context.peak_active_weight_bytes(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_attention_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "7".into(),
            "--position".into(),
            "11".into(),
            "--repetitions".into(),
            "5".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.layer, 7);
        assert_eq!(args.position, 11);
        assert_eq!(args.repetitions, 5);
    }

    #[test]
    fn rejects_missing_model_and_zero_repetitions() {
        assert_eq!(
            Args::parse(["--position".into(), "1".into()]).unwrap_err(),
            "--model is required"
        );
        assert!(Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--repetitions".into(),
            "0".into(),
        ])
        .is_err());
    }
}
