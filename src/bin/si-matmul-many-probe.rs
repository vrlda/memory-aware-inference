use std::path::PathBuf;
use std::process::ExitCode;

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
    eprintln!("error: batched matmul probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::ModelStore::open(&args.model, args.verify_manifest)
        .map_err(|error| error.to_string())?;
    let tensor = store
        .tensor("model.layers.0.self_attn.q_proj.weight")
        .map_err(|error| error.to_string())?;
    if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
        return Err("q_proj.weight must be a rank-2 BF16 tensor".into());
    }
    let rows = tensor.info.shape[0];
    let columns = tensor.info.shape[1];
    let context = super_inference::metal::MetalContext::new()?;
    println!(
        "probe=device={} tensor=q_proj rows={rows} columns={columns} warmup={} repetitions={}",
        super_inference::metal::probe()?.name,
        args.warmup,
        args.repetitions,
    );

    for batch in [1_usize, 2, 4, 8] {
        let inputs = (0..batch * columns)
            .map(|index| ((index as f32) * 0.017).sin())
            .collect::<Vec<_>>();
        for _ in 0..args.warmup {
            let _ = run_separate(&context, &tensor, batch, &inputs)?;
            let _ = context.bf16_matmul_many_tensor(&tensor, batch, &inputs)?;
        }
        let mut separate_elapsed = std::time::Duration::ZERO;
        let mut batched_elapsed = std::time::Duration::ZERO;
        let mut separate = Vec::new();
        let mut batched = Vec::new();
        for _ in 0..args.repetitions {
            let started = std::time::Instant::now();
            separate = run_separate(&context, &tensor, batch, &inputs)?;
            separate_elapsed += started.elapsed();

            let started = std::time::Instant::now();
            batched = context.bf16_matmul_many_tensor(&tensor, batch, &inputs)?;
            batched_elapsed += started.elapsed();
        }
        let max_abs_diff = separate
            .iter()
            .flatten()
            .zip(batched.iter().flatten())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let separate_ms = separate_elapsed.as_secs_f64() * 1_000.0 / args.repetitions as f64;
        let batched_ms = batched_elapsed.as_secs_f64() * 1_000.0 / args.repetitions as f64;
        println!(
            "batch={batch} separate_ms={separate_ms:.3} batched_ms={batched_ms:.3} speedup={:.3} max_abs_diff={max_abs_diff:.6}",
            separate_ms / batched_ms,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_separate(
    context: &super_inference::metal::MetalContext,
    tensor: &super_inference::model::TensorView<'_>,
    batch: usize,
    inputs: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let columns = tensor.info.shape[1];
    (0..batch)
        .map(|candidate| {
            let start = candidate * columns;
            context.bf16_matvec_tensor(tensor, &inputs[start..start + columns])
        })
        .collect()
}

#[cfg(target_os = "macos")]
struct Args {
    model: PathBuf,
    verify_manifest: bool,
    warmup: usize,
    repetitions: usize,
}

#[cfg(target_os = "macos")]
impl Args {
    fn parse(mut values: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut verify_manifest = false;
        let mut warmup = 1;
        let mut repetitions = 5;
        while let Some(value) = values.next() {
            match value.as_str() {
                "--model" => {
                    model = Some(PathBuf::from(
                        values.next().ok_or("--model requires a path")?,
                    ));
                }
                "--verify-manifest" => verify_manifest = true,
                "--warmup" => {
                    warmup = values
                        .next()
                        .ok_or("--warmup requires an integer")?
                        .parse()
                        .map_err(|_| "--warmup requires a non-negative integer")?;
                }
                "--repetitions" => {
                    repetitions = values
                        .next()
                        .ok_or("--repetitions requires an integer")?
                        .parse()
                        .map_err(|_| "--repetitions requires a positive integer")?;
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: si-matmul-many-probe --model PATH [--verify-manifest] [--warmup N] [--repetitions N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        let model = model.ok_or("--model is required")?;
        if repetitions == 0 {
            return Err("--repetitions must be greater than zero".into());
        }
        Ok(Self {
            model,
            verify_manifest,
            warmup,
            repetitions,
        })
    }
}
