use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use super_inference::model::GgufModelStore;

#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    tensor: String,
    batch: usize,
    repetitions: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut tensor = "blk.0.attn_qkv.weight".to_owned();
        let mut batch = 4;
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
                "--tensor" => tensor = value.clone(),
                "--batch" => batch = value.parse().map_err(|_| "--batch is invalid")?,
                "--repetitions" => {
                    repetitions = value.parse().map_err(|_| "--repetitions is invalid")?
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(1..=8).contains(&batch) || repetitions == 0 {
            return Err("batch must be 1..8 and repetitions must be positive".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            tensor,
            batch,
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
    eprintln!("error: GGUF batch probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = GgufModelStore::open(&args.model).map_err(|error| error.to_string())?;
    let tensor = store
        .tensor(&args.tensor)
        .map_err(|error| error.to_string())?;
    if tensor.info.shape.len() != 2 {
        return Err("tensor must be rank-2".into());
    }
    let columns = tensor.info.shape[0];
    let rows = tensor.info.shape[1];
    let inputs = (0..args.batch * columns)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let _ = context.gguf_quant_matmul_many_bytes(
        tensor.info.ggml_type,
        tensor.bytes,
        rows,
        columns,
        args.batch,
        &inputs,
    )?;
    let started = Instant::now();
    let mut batched = Vec::new();
    for _ in 0..args.repetitions {
        batched = context.gguf_quant_matmul_many_bytes(
            tensor.info.ggml_type,
            tensor.bytes,
            rows,
            columns,
            args.batch,
            &inputs,
        )?;
    }
    let batch_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repetitions as f64;
    let started = Instant::now();
    let mut separate_checksum = 0.0_f32;
    for _ in 0..args.repetitions {
        for input in inputs.chunks_exact(columns) {
            let output = context.gguf_matvec_tensor_rows(&tensor, 0, rows, input)?;
            separate_checksum += output.iter().copied().sum::<f32>();
        }
    }
    let separate_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repetitions as f64;
    let private_weight =
        context.upload_quant_weight_private(tensor.bytes, tensor.info.ggml_type)?;
    let started = Instant::now();
    for _ in 0..args.repetitions {
        for input in inputs.chunks_exact(columns) {
            let _ = context.gguf_quant_matvec_weight(&private_weight, rows, columns, input)?;
        }
    }
    let private_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repetitions as f64;
    let batched_checksum = batched
        .iter()
        .flat_map(|output| output.iter())
        .copied()
        .sum::<f32>();
    let max_error = batched
        .iter()
        .zip(inputs.chunks_exact(columns))
        .map(|(batched, input)| {
            let separate = context
                .gguf_matvec_tensor_rows(&tensor, 0, rows, input)
                .expect("separate matvec should succeed");
            batched
                .iter()
                .zip(separate)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max)
        })
        .fold(0.0_f32, f32::max);
    println!(
        "probe=gguf_batch tensor={} type={} rows={} columns={} batch={} repetitions={} batched_ms={:.3} separate_ms={:.3} private_separate_ms={:.3} speedup={:.3} max_error={:.6} checksum={:.6} separate_checksum={:.6}",
        args.tensor,
        tensor.info.ggml_type,
        rows,
        columns,
        args.batch,
        args.repetitions,
        batch_ms,
        separate_ms,
        private_ms,
        separate_ms / batch_ms.max(f64::MIN_POSITIVE),
        max_error,
        batched_checksum,
        separate_checksum,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_defaults_and_batch() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--batch".into(),
            "8".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 8);
        assert_eq!(args.tensor, "blk.0.attn_qkv.weight");
    }
}
