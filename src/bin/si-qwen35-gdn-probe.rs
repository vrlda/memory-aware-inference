use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use super_inference::model::{GgufModelStore, GgufTensorView};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    layer: usize,
    repetitions: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut layer = 0;
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
            repetitions,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StageTimes {
    projections: Duration,
    convolution: Duration,
    recurrent: Duration,
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
    eprintln!("error: Qwen3.6 Gated DeltaNet probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn f32_values<'a>(tensor: &'a GgufTensorView<'a>) -> Result<&'a [f32], String> {
    if tensor.info.ggml_type != 0 {
        return Err(format!(
            "tensor {} is not F32 (type {})",
            tensor.info.name, tensor.info.ggml_type
        ));
    }
    // SAFETY: GGUF F32 payloads are aligned by the parser's tensor data
    // alignment, and the shape-derived byte length is checked here.
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
fn run_layer(
    context: &super_inference::metal::MetalContext,
    store: &GgufModelStore,
    layer: usize,
    hidden: &[f32],
) -> Result<(Vec<f32>, StageTimes), String> {
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
    let convolution_weights = f32_values(&convolution)?;
    let a_log = f32_values(&a_log)?;
    let dt_bias = f32_values(&dt_bias)?;
    let norm = f32_values(&norm)?;

    let projection_started = Instant::now();
    let qkv_values = context.gguf_matvec_tensor_rows(&qkv, 0, qkv.info.shape[1], hidden)?;
    let z =
        context.gguf_matvec_tensor_rows(&z_projection, 0, z_projection.info.shape[1], hidden)?;
    let alpha = context.f32_matvec_tensor_rows(
        &alpha_projection,
        0,
        alpha_projection.info.shape[1],
        hidden,
    )?;
    let beta_logits = context.f32_matvec_tensor_rows(
        &beta_projection,
        0,
        beta_projection.info.shape[1],
        hidden,
    )?;
    let projections = projection_started.elapsed();

    let convolution_started = Instant::now();
    let conv_state = vec![0.0_f32; config.ssm_projection_size() * (config.ssm_conv_kernel - 1)];
    let (mixed_qkv, _) = context.causal_conv1d_step(
        &qkv_values,
        &conv_state,
        convolution_weights,
        config.ssm_projection_size(),
        config.ssm_conv_kernel,
    )?;
    let convolution_time = convolution_started.elapsed();

    let key_heads = config.ssm_key_heads();
    let value_heads = config.ssm_value_heads();
    let head_dim = config.ssm_value_dim();
    let key_width = key_heads * head_dim;
    let value_width = value_heads * head_dim;
    let raw_query = &mixed_qkv[..key_width];
    let raw_key = &mixed_qkv[key_width..key_width * 2];
    let value = &mixed_qkv[key_width * 2..key_width * 2 + value_width];
    let query = repeat_heads(
        &l2_normalize_heads(raw_query, key_heads, head_dim),
        key_heads,
        value_heads,
        head_dim,
    )
    .into_iter()
    .map(|value| value / (head_dim as f32).sqrt())
    .collect::<Vec<_>>();
    let key = repeat_heads(
        &l2_normalize_heads(raw_key, key_heads, head_dim),
        key_heads,
        value_heads,
        head_dim,
    );
    let beta = beta_logits
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();
    let gate = alpha
        .iter()
        .zip(a_log.iter().zip(dt_bias))
        .map(|(value, (a_log, dt_bias))| -a_log.exp() * softplus(value + dt_bias))
        .collect::<Vec<_>>();

    let recurrent_started = Instant::now();
    let recurrent_state = vec![0.0_f32; value_heads * head_dim * head_dim];
    let (core, _) = context.gated_delta_step(
        &query,
        &key,
        value,
        &gate,
        &beta,
        &recurrent_state,
        value_heads,
        head_dim,
        head_dim,
    )?;
    let recurrent = recurrent_started.elapsed();

    let output_started = Instant::now();
    let normalized =
        context.rms_norm_gated(&core, &z, norm, value_heads, head_dim, config.rms_norm_eps)?;
    let output = context.gguf_matvec_tensor_rows(
        &output_projection,
        0,
        output_projection.info.shape[1],
        &normalized,
    )?;
    let output_time = output_started.elapsed();

    Ok((
        output,
        StageTimes {
            projections,
            convolution: convolution_time,
            recurrent,
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
        != super_inference::model::GgufQwen35LayerKind::GatedDeltaNet
    {
        return Err(format!("layer {} is a full-attention layer", args.layer));
    }
    let hidden = (0..config.hidden_size)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let (warmup_output, _) = run_layer(&context, &store, args.layer, &hidden)?;
    context.reset_peaks();
    let started = Instant::now();
    let mut checksum = warmup_output.iter().copied().sum::<f32>();
    let mut stages = StageTimes::default();
    for _ in 0..args.repetitions {
        let (output, timing) = run_layer(&context, &store, args.layer, &hidden)?;
        checksum += output.iter().copied().sum::<f32>();
        stages.projections += timing.projections;
        stages.convolution += timing.convolution;
        stages.recurrent += timing.recurrent;
        stages.output += timing.output;
        std::hint::black_box(&output);
    }
    let repetitions = args.repetitions as f64;
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0 / repetitions;
    println!(
        "probe=qwen35_gdn_layer layer={} hidden={} qkv_width={} value_heads={} head_dim={} repetitions={} total_ms={:.3} projections_ms={:.3} convolution_ms={:.3} recurrent_ms={:.3} output_ms={:.3} checksum={:.6} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.layer,
        config.hidden_size,
        config.ssm_projection_size(),
        config.ssm_value_heads(),
        config.ssm_value_dim(),
        args.repetitions,
        elapsed,
        stages.projections.as_secs_f64() * 1_000.0 / repetitions,
        stages.convolution.as_secs_f64() * 1_000.0 / repetitions,
        stages.recurrent.as_secs_f64() * 1_000.0 / repetitions,
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
    fn parses_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "4".into(),
            "--repetitions".into(),
            "5".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.layer, 4);
        assert_eq!(args.repetitions, 5);
    }

    #[test]
    fn rejects_missing_model_and_zero_repetitions() {
        assert_eq!(
            Args::parse(["--layer".into(), "1".into()]).unwrap_err(),
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
