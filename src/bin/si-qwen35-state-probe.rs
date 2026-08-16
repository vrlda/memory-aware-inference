use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    layer: usize,
    tokens: usize,
    capacity: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut layer = 0;
        let mut tokens = 4;
        let mut capacity = 8;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let parsed = match flag.as_str() {
                "--model" => {
                    model = Some(PathBuf::from(value));
                    None
                }
                "--layer" => Some(("layer", value.parse::<usize>())),
                "--tokens" => Some(("tokens", value.parse::<usize>())),
                "--capacity" => Some(("capacity", value.parse::<usize>())),
                _ => return Err(format!("unknown option: {flag}")),
            };
            if let Some((name, value)) = parsed {
                let value =
                    value.map_err(|_| format!("--{name} must be a non-negative integer"))?;
                match name {
                    "layer" => layer = value,
                    "tokens" => tokens = value,
                    "capacity" => capacity = value,
                    _ => unreachable!(),
                }
            }
            index += 2;
        }
        if tokens == 0 {
            return Err("--tokens must be positive".into());
        }
        if capacity == 0 {
            return Err("--capacity must be positive".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            layer,
            tokens,
            capacity,
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
    eprintln!("error: Qwen3.6 state probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    if args.tokens > args.capacity {
        return Err("--tokens cannot exceed --capacity for a bounded KV state".into());
    }
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    if args.layer >= config.num_hidden_layers {
        return Err(format!(
            "--layer {} exceeds {} layers",
            args.layer, config.num_hidden_layers
        ));
    }
    let mut state =
        super_inference::qwen35_runtime::Qwen35LayerState::new(&store, args.layer, args.capacity)?;
    let context = super_inference::metal::MetalContext::new()?;
    context.reset_peaks();
    let started = Instant::now();
    let mut checksum = 0.0_f32;
    for token in 0..args.tokens {
        let hidden = (0..config.hidden_size)
            .map(|index| ((index % 97) as f32 - 48.0) / 97.0 + token as f32 * 1.0e-3)
            .collect::<Vec<_>>();
        let output = super_inference::qwen35_runtime::qwen35_decoder_block_stateful(
            &context, &store, args.layer, token, &hidden, &mut state,
        )?;
        checksum += output.iter().copied().sum::<f32>();
        std::hint::black_box(&output);
    }
    println!(
        "probe=qwen35_state layer={} tokens={} capacity={} state_bytes={} cached_tokens={} total_ms={:.3} tok_s={:.3} checksum={:.6} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.layer,
        args.tokens,
        args.capacity,
        state.state_bytes(),
        state.cached_tokens(),
        started.elapsed().as_secs_f64() * 1_000.0,
        args.tokens as f64 / started.elapsed().as_secs_f64(),
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
    fn parses_state_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "3".into(),
            "--tokens".into(),
            "6".into(),
            "--capacity".into(),
            "8".into(),
        ])
        .expect("probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.layer, 3);
        assert_eq!(args.tokens, 6);
        assert_eq!(args.capacity, 8);
    }

    #[test]
    fn rejects_missing_model_zero_values_and_overflow() {
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
        .is_ok());
    }
}
