use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    token: u32,
    layers: usize,
    top_k: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut token = None;
        let mut layers = 8;
        let mut top_k = 8;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--token" => {
                    token = Some(
                        value
                            .parse::<u32>()
                            .map_err(|_| "--token must be a non-negative integer".to_owned())?,
                    )
                }
                "--layers" => {
                    layers = value
                        .parse::<usize>()
                        .map_err(|_| "--layers must be a positive integer".to_owned())?;
                }
                "--top-k" => {
                    top_k = value
                        .parse::<usize>()
                        .map_err(|_| "--top-k must be a positive integer".to_owned())?;
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if layers == 0 || top_k == 0 {
            return Err("--layers and --top-k must be positive".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            token: token.ok_or("--token is required")?,
            layers,
            top_k,
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
    eprintln!("error: Qwen3.6 prefix probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = super_inference::model::GgufModelStore::open(&args.model)
        .map_err(|error| error.to_string())?;
    let context = super_inference::metal::MetalContext::new()?;
    let mut model =
        super_inference::qwen35_runtime::Qwen35Runtime::new_with_retained_layers_and_head(
            &context,
            &store,
            8,
            args.layers,
            true,
        )?;
    let hidden = model.embed_token(args.token)?;

    let full_started = Instant::now();
    let full_hidden = model.decode_hidden(0, &hidden)?;
    let full_logits = model.logits(&full_hidden)?;
    let full_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    model.reset();

    let prefix_started = Instant::now();
    let prefix_hidden = model.decode_hidden_prefix_layers(0, &hidden, args.layers)?;
    let prefix_logits = model.logits(&prefix_hidden)?;
    let prefix_ms = prefix_started.elapsed().as_secs_f64() * 1_000.0;
    model.reset();

    let full_top1 = argmax(&full_logits);
    let prefix_top1 = argmax(&prefix_logits);
    let mut ranking = (0..prefix_logits.len()).collect::<Vec<_>>();
    ranking.sort_unstable_by(|left, right| {
        prefix_logits[*right]
            .partial_cmp(&prefix_logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_k = args.top_k.min(ranking.len());
    let prefix_top_k = &ranking[..top_k];
    let target_rank = prefix_top_k
        .iter()
        .position(|candidate| *candidate == full_top1)
        .map(|rank| rank + 1);
    println!(
        "probe=qwen35_prefix token={} layers={} full_ms={:.3} prefix_ms={:.3} full_top1={} prefix_top1={} top{}_target_rank={:?} top1_match={} peak_metal_bytes={} peak_active_weight_bytes={}",
        args.token,
        args.layers,
        full_ms,
        prefix_ms,
        full_top1,
        prefix_top1,
        top_k,
        target_rank,
        full_top1 == prefix_top1,
        context.peak_allocated_bytes(),
        context.peak_active_weight_bytes(),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_prefix_probe_arguments() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--token".into(),
            "548".into(),
            "--layers".into(),
            "8".into(),
            "--top-k".into(),
            "16".into(),
        ])
        .expect("prefix probe arguments should parse");
        assert_eq!(args.model.to_string_lossy(), "model.gguf");
        assert_eq!(args.token, 548);
        assert_eq!(args.layers, 8);
        assert_eq!(args.top_k, 16);
    }

    #[test]
    fn rejects_missing_token_and_zero_layers() {
        assert!(Args::parse(["--model".into(), "model.gguf".into()]).is_err());
        assert!(Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--token".into(),
            "1".into(),
            "--layers".into(),
            "0".into(),
        ])
        .is_err());
    }
}
