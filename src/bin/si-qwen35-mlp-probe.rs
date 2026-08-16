use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use super_inference::model::GgufModelStore;

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
    fused_gate_up: Duration,
    down: Duration,
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
    eprintln!("error: Qwen3.6 MLP probe requires macOS");
    ExitCode::from(1)
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
    let prefix = format!("blk.{}", args.layer);
    let gate = store
        .tensor(&format!("{prefix}.ffn_gate.weight"))
        .map_err(|error| error.to_string())?;
    let up = store
        .tensor(&format!("{prefix}.ffn_up.weight"))
        .map_err(|error| error.to_string())?;
    let down = store
        .tensor(&format!("{prefix}.ffn_down.weight"))
        .map_err(|error| error.to_string())?;
    if gate.info.ggml_type != super_inference::quant::GGML_TYPE_Q4_K
        || up.info.ggml_type != super_inference::quant::GGML_TYPE_Q4_K
        || gate.info.shape != up.info.shape
        || gate.info.shape.len() != 2
    {
        return Err("MLP gate/up tensors must be matching rank-2 Q4_K matrices".into());
    }
    let columns = gate.info.shape[0];
    let rows = gate.info.shape[1];
    let row_bytes = columns / super_inference::quant::Q4_K_BLOCK_ELEMENTS
        * super_inference::quant::Q4_K_BLOCK_BYTES;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or("MLP gate/up byte range overflows")?;
    let gate_bytes = gate
        .bytes
        .get(..expected_bytes)
        .ok_or("MLP gate tensor is shorter than its shape")?;
    let up_bytes = up
        .bytes
        .get(..expected_bytes)
        .ok_or("MLP up tensor is shorter than its shape")?;
    let hidden = (0..columns)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let run_once = |context: &super_inference::metal::MetalContext| -> Result<Vec<f32>, String> {
        let fused = context.q4_k_fused_gate_up(gate_bytes, up_bytes, rows, columns, &hidden)?;
        context.gguf_matvec_tensor_rows(&down, 0, down.info.shape[1], &fused)
    };
    let warmup = run_once(&context)?;
    context.reset_peaks();
    let started = Instant::now();
    let mut checksum = warmup.iter().copied().sum::<f32>();
    let mut stages = StageTimes::default();
    for _ in 0..args.repetitions {
        let fused_started = Instant::now();
        let fused = context.q4_k_fused_gate_up(gate_bytes, up_bytes, rows, columns, &hidden)?;
        stages.fused_gate_up += fused_started.elapsed();
        let down_started = Instant::now();
        let output = context.gguf_matvec_tensor_rows(&down, 0, down.info.shape[1], &fused)?;
        stages.down += down_started.elapsed();
        checksum += output.iter().copied().sum::<f32>();
        std::hint::black_box(&output);
    }
    let repetitions = args.repetitions as f64;
    println!(
        "probe=qwen35_mlp layer={} hidden={} intermediate={} repetitions={} total_ms={:.3} fused_gate_up_ms={:.3} down_ms={:.3} checksum={:.6} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.layer,
        columns,
        rows,
        args.repetitions,
        started.elapsed().as_secs_f64() * 1_000.0 / repetitions,
        stages.fused_gate_up.as_secs_f64() * 1_000.0 / repetitions,
        stages.down.as_secs_f64() * 1_000.0 / repetitions,
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
    fn parses_mlp_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "2".into(),
            "--repetitions".into(),
            "4".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.layer, 2);
        assert_eq!(args.repetitions, 4);
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
