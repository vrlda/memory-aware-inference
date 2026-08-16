use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

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
                        .map_err(|_| "--repetitions must be positive".to_owned())?;
                    if repetitions == 0 {
                        return Err("--repetitions must be positive".into());
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
    eprintln!("error: Qwen3.6 stage probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let config = store.qwen35_config().map_err(|error| error.to_string())?;
    if args.layer >= config.num_hidden_layers {
        return Err("--layer exceeds model depth".into());
    }
    let excluded = std::collections::BTreeSet::new();
    let warmup = store
        .stage_qwen35_layer_packed(args.layer, &excluded)
        .map_err(|error| error.to_string())?;
    let mut total = 0_u128;
    let mut bytes = warmup.packed_bytes().len();
    for _ in 0..args.repetitions {
        let started = Instant::now();
        let staged = store
            .stage_qwen35_layer_packed(args.layer, &excluded)
            .map_err(|error| error.to_string())?;
        total += started.elapsed().as_nanos();
        bytes = staged.packed_bytes().len();
        std::hint::black_box(staged);
    }
    println!(
        "probe=qwen35_stage layer={} repetitions={} bytes={} average_ms={:.3}",
        args.layer,
        args.repetitions,
        bytes,
        total as f64 / args.repetitions as f64 / 1_000_000.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_stage_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--layer".into(),
            "4".into(),
            "--repetitions".into(),
            "2".into(),
        ])
        .expect("stage probe arguments should parse");
        assert_eq!(args.layer, 4);
        assert_eq!(args.repetitions, 2);
    }
}
