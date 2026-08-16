use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    token: u32,
    tokens: usize,
    capacity: usize,
    output_head: bool,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut token = 0;
        let mut tokens = 1;
        let mut capacity = 8;
        let mut output_head = false;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            if flag == "--output-head" {
                output_head = true;
                index += 1;
                continue;
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--token" => {
                    token = value
                        .parse::<u32>()
                        .map_err(|_| "--token must be a u32".to_owned())?;
                }
                "--tokens" => {
                    tokens = value
                        .parse::<usize>()
                        .map_err(|_| "--tokens must be a positive integer".to_owned())?;
                }
                "--capacity" => {
                    capacity = value
                        .parse::<usize>()
                        .map_err(|_| "--capacity must be a positive integer".to_owned())?;
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if tokens == 0 || capacity == 0 {
            return Err("--tokens and --capacity must be positive".into());
        }
        if tokens > capacity {
            return Err("--tokens cannot exceed --capacity".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            token,
            tokens,
            capacity,
            output_head,
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
    eprintln!("error: Qwen3.6 runtime probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let context = super_inference::metal::MetalContext::new()?;
    let mut runtime =
        super_inference::qwen35_runtime::Qwen35Runtime::new(&context, &store, args.capacity)?;
    context.reset_peaks();
    let started = Instant::now();
    let mut current_token = args.token;
    let mut checksum = 0.0_f32;
    for position in 0..args.tokens {
        if args.output_head {
            let (next_token, logits) = runtime.decode_token(current_token, position)?;
            checksum += logits.iter().copied().sum::<f32>();
            current_token = next_token;
        } else {
            let embedding = runtime.embed_token(current_token)?;
            let hidden = runtime.decode_hidden(position, &embedding)?;
            checksum += hidden.iter().copied().sum::<f32>();
            current_token = (current_token + 1) % 248_320;
        }
    }
    let elapsed = started.elapsed();
    println!(
        "probe=qwen35_runtime tokens={} output_head={} final_token={} total_ms={:.3} tok_s={:.3} checksum={:.6} layers={} state_bytes={} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.tokens,
        args.output_head,
        current_token,
        elapsed.as_secs_f64() * 1_000.0,
        args.tokens as f64 / elapsed.as_secs_f64(),
        checksum,
        runtime.layer_count(),
        runtime.state_bytes(),
        context.peak_allocated_bytes(),
        context.peak_active_weight_bytes(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_runtime_probe_arguments_and_flag() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--token".into(),
            "42".into(),
            "--tokens".into(),
            "3".into(),
            "--capacity".into(),
            "4".into(),
            "--output-head".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.token, 42);
        assert_eq!(args.tokens, 3);
        assert_eq!(args.capacity, 4);
        assert!(args.output_head);
    }

    #[test]
    fn rejects_missing_model_zero_values_and_over_capacity() {
        assert_eq!(
            Args::parse(["--tokens".into(), "1".into()]).unwrap_err(),
            "--model is required"
        );
        assert!(Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--tokens".into(),
            "0".into(),
        ])
        .is_err());
        assert!(Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--tokens".into(),
            "3".into(),
            "--capacity".into(),
            "2".into(),
        ])
        .is_err());
    }
}
