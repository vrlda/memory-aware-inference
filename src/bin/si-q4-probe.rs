use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    tensor: Option<String>,
    repetitions: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut tensor = None;
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
                "--tensor" => tensor = Some(value.clone()),
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
            tensor,
            repetitions,
        })
    }
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
    eprintln!("error: Q4_K Metal probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let tensor_name = args.tensor.unwrap_or_else(|| {
        store
            .tensors
            .values()
            .find(|tensor| tensor.is_q4_k() && tensor.shape.len() == 2)
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default()
    });
    if tensor_name.is_empty() {
        return Err("GGUF model has no rank-2 Q4_K tensor".into());
    }
    let tensor = store
        .tensor(&tensor_name)
        .map_err(|error| error.to_string())?;
    if !matches!(
        tensor.info.ggml_type,
        super_inference::quant::GGML_TYPE_Q4_K
            | super_inference::quant::GGML_TYPE_Q5_K
            | super_inference::quant::GGML_TYPE_Q6_K
    ) || tensor.info.shape.len() != 2
    {
        return Err(format!(
            "tensor {tensor_name} is not a supported rank-2 K-quant matrix"
        ));
    }
    let columns = tensor.info.shape[0];
    let rows = tensor.info.shape[1];
    let input = (0..columns)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let dequantized = if std::env::var("SI_Q4_DEQUANT").ok().as_deref() == Some("1") {
        Some(match tensor.info.ggml_type {
            super_inference::quant::GGML_TYPE_Q4_K => store
                .dequantize_q4_k(&tensor_name)
                .map_err(|error| error.to_string())?,
            super_inference::quant::GGML_TYPE_Q5_K => store
                .dequantize_q5_k(&tensor_name)
                .map_err(|error| error.to_string())?,
            super_inference::quant::GGML_TYPE_Q6_K => store
                .dequantize_q6_k(&tensor_name)
                .map_err(|error| error.to_string())?,
            _ => return Err("unsupported dequantization type".into()),
        })
    } else {
        None
    };
    let private_weight = if std::env::var("SI_Q4_PRIVATE").ok().as_deref() == Some("1") {
        Some(context.upload_quant_weight_private(tensor.bytes, tensor.info.ggml_type)?)
    } else {
        None
    };
    let run_once = |context: &super_inference::metal::MetalContext| {
        if let Some(values) = dequantized.as_ref() {
            context.f32_matvec(values, rows, columns, &input)
        } else if let Some(weight) = private_weight.as_ref() {
            context.gguf_quant_matvec_weight(weight, rows, columns, &input)
        } else {
            context.gguf_quant_matvec_tensor_rows(&tensor, 0, rows, &input)
        }
    };
    let warmup = run_once(&context)?;
    let compare_max_abs = if std::env::var("SI_Q4_COMPARE_X").ok().as_deref() == Some("1") {
        std::env::set_var("SI_QK_ROWS4X128", "1");
        let x = run_once(&context)?;
        std::env::remove_var("SI_QK_ROWS4X128");
        Some(
            warmup
                .iter()
                .zip(x.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max),
        )
    } else {
        None
    };
    context.reset_peaks();
    let started = std::time::Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..args.repetitions {
        let output = run_once(&context)?;
        checksum += output.iter().copied().sum::<f32>();
        std::hint::black_box(&output);
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repetitions as f64;
    println!(
        "probe=gguf_k_metal tensor={} ggml_type={} rows={} columns={} weight_bytes={} warmup_checksum={:.6} warmup_argmax={} compare_x_max_abs={} checksum={:.6} average_ms={:.3} mapped_bytes={} peak_metal_bytes={} peak_active_weight_bytes={}",
        tensor_name,
        tensor.info.ggml_type,
        rows,
        columns,
        tensor.info.byte_len(),
        warmup.iter().copied().sum::<f32>(),
        warmup
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(0),
        compare_max_abs.map_or_else(|| "none".to_owned(), |value| format!("{value:.6}")),
        checksum,
        elapsed_ms,
        tensor.info.byte_len(),
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
            "--tensor".into(),
            "blk.0.ffn_gate.weight".into(),
            "--repetitions".into(),
            "5".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.tensor.as_deref(), Some("blk.0.ffn_gate.weight"));
        assert_eq!(args.repetitions, 5);
    }

    #[test]
    fn rejects_missing_model_and_zero_repetitions() {
        assert_eq!(
            Args::parse(["--repetitions".into(), "1".into()]).unwrap_err(),
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
