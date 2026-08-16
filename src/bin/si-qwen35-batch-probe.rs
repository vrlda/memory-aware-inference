use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use super_inference::model::GgufModelStore;
use super_inference::qwen35_runtime::Qwen35Runtime;

#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    batch: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut batch = 2;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--batch" => batch = value.parse().map_err(|_| "--batch is invalid")?,
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(2..=8).contains(&batch) {
            return Err("batch must be 2..8".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            batch,
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
    eprintln!("error: Qwen3.6 batched probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = GgufModelStore::open(&args.model).map_err(|error| error.to_string())?;
    let context = super_inference::metal::MetalContext::new()?;
    let mut batched = Qwen35Runtime::new(&context, &store, 32)?;
    let prompt = batched.embed_token(220)?;
    let _ = batched.decode_hidden(0, &prompt)?;
    let tokens = vec![220_u32; args.batch];
    let positions = (1..=args.batch).collect::<Vec<_>>();
    let started = Instant::now();
    let hidden = batched.decode_tokens_many(&tokens, &positions)?;
    let batched_logits = batched.logits_many(&hidden)?;
    let batched_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let mut sequential = Qwen35Runtime::new(&context, &store, 32)?;
    let prompt = sequential.embed_token(220)?;
    let _ = sequential.decode_hidden(0, &prompt)?;
    let started = Instant::now();
    let mut sequential_logits = Vec::with_capacity(args.batch);
    for (token, position) in tokens.iter().copied().zip(&positions) {
        let (_, logits) = sequential.decode_token(token, *position)?;
        sequential_logits.push(logits);
    }
    let sequential_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let max_error = batched_logits
        .iter()
        .zip(&sequential_logits)
        .flat_map(|(left, right)| left.iter().zip(right))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let batched_ids = batched_logits
        .iter()
        .map(|logits| argmax(logits))
        .collect::<Vec<_>>();
    let sequential_ids = sequential_logits
        .iter()
        .map(|logits| argmax(logits))
        .collect::<Vec<_>>();
    println!(
        "probe=qwen35_batch batch={} batched_ms={:.3} sequential_ms={:.3} speedup={:.3} max_logit_error={:.6} ids_match={} batched_ids={:?} sequential_ids={:?}",
        args.batch,
        batched_ms,
        sequential_ms,
        sequential_ms / batched_ms.max(f64::MIN_POSITIVE),
        max_error,
        batched_ids == sequential_ids,
        batched_ids,
        sequential_ids,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::Args;

    #[test]
    fn parses_batch() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--batch".into(),
            "4".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 4);
    }
}
