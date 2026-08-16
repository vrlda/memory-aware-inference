use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

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
        let mut layer = 0;
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
    eprintln!("error: Qwen3.6 block probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    if args.layer >= config.num_hidden_layers {
        return Err(format!(
            "--layer {} exceeds {} layers",
            args.layer, config.num_hidden_layers
        ));
    }
    let hidden = (0..config.hidden_size)
        .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
        .collect::<Vec<_>>();
    let context = super_inference::metal::MetalContext::new()?;
    let warmup = super_inference::qwen35_runtime::qwen35_decoder_block(
        &context,
        &store,
        args.layer,
        args.position,
        &hidden,
    )?;
    context.reset_peaks();
    let started = Instant::now();
    let mut checksum = warmup.iter().copied().sum::<f32>();
    for _ in 0..args.repetitions {
        let output = super_inference::qwen35_runtime::qwen35_decoder_block(
            &context,
            &store,
            args.layer,
            args.position,
            &hidden,
        )?;
        checksum += output.iter().copied().sum::<f32>();
        std::hint::black_box(&output);
    }
    let repetitions = args.repetitions as f64;
    println!(
        "probe=qwen35_decoder_block layer={} position={} hidden={} repetitions={} total_ms={:.3} checksum={:.6} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.layer,
        args.position,
        config.hidden_size,
        args.repetitions,
        started.elapsed().as_secs_f64() * 1_000.0 / repetitions,
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
    fn parses_block_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "3".into(),
            "--position".into(),
            "4".into(),
            "--repetitions".into(),
            "2".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.layer, 3);
        assert_eq!(args.position, 4);
        assert_eq!(args.repetitions, 2);
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
