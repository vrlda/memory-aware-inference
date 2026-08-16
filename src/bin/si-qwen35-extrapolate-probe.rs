use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use super_inference::model::GgufModelStore;
use super_inference::qwen35_runtime::Qwen35Runtime;
use super_inference::tokenizer::QwenTokenizer;

#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    prompt: String,
    batch: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut prompt = "Hello".to_owned();
        let mut batch = 8;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--prompt" => prompt = value.clone(),
                "--batch" => batch = value.parse().map_err(|_| "--batch is invalid")?,
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(2..=8).contains(&batch) {
            return Err("batch must be between 2 and 8".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            prompt,
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
    eprintln!("error: Qwen3.6 extrapolation probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let store = GgufModelStore::open(&args.model).map_err(|error| error.to_string())?;
    let tokenizer = QwenTokenizer::from_gguf_model(&args.model)?;
    let prompt_tokens = tokenizer.encode(&args.prompt)?;
    if prompt_tokens.is_empty() {
        return Err("prompt produced no tokens".into());
    }
    let context = super_inference::metal::MetalContext::new()?;
    let mut target = Qwen35Runtime::new(&context, &store, 32)?;
    let mut previous_hidden = None;
    let mut current_hidden = Vec::new();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        let embedding = target.embed_token(token)?;
        let next_hidden = target.decode_hidden(position, &embedding)?;
        previous_hidden = (!current_hidden.is_empty()).then(|| current_hidden.clone());
        current_hidden = next_hidden;
    }
    let state_snapshot = target.snapshot_states();
    let target_logits = target.logits(&current_hidden)?;
    let first = argmax(&target_logits);
    let delta = previous_hidden
        .as_ref()
        .map(|previous| {
            current_hidden
                .iter()
                .zip(previous)
                .map(|(current, previous)| current - previous)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0.0; current_hidden.len()]);
    let draft_started = Instant::now();
    let mut predicted = current_hidden.clone();
    let mut candidates = Vec::with_capacity(args.batch);
    candidates.push(first);
    for _ in 1..args.batch {
        predicted
            .iter_mut()
            .zip(&delta)
            .for_each(|(value, delta)| *value += delta);
        candidates.push(argmax(&target.logits(&predicted)?));
    }
    let draft_ms = draft_started.elapsed().as_secs_f64() * 1_000.0;
    let positions = (prompt_tokens.len()..prompt_tokens.len() + args.batch).collect::<Vec<_>>();
    let verify_started = Instant::now();
    let hidden_many = target.decode_tokens_many(&candidates, &positions)?;
    let logits_many = target.logits_many(&hidden_many)?;
    let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;
    let mut expected = first;
    let mut accepted = 0;
    for (candidate, logits) in candidates.iter().zip(&logits_many) {
        if *candidate != expected {
            break;
        }
        accepted += 1;
        expected = argmax(logits);
    }
    target.restore_states(&state_snapshot)?;
    println!(
        "probe=qwen35_extrapolate prompt={:?} batch={} draft_ms={:.3} verify_ms={:.3} candidates={:?} accepted={} acceptance={:.3} target_traversals_per_accepted={:.3}",
        args.prompt,
        args.batch,
        draft_ms,
        verify_ms,
        candidates,
        accepted,
        accepted as f64 / args.batch as f64,
        1.0 / accepted.max(1) as f64,
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
    fn parses_extrapolation_options() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--prompt".into(),
            "x".into(),
            "--batch".into(),
            "4".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 4);
        assert_eq!(args.prompt, "x");
    }
}
