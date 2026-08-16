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
    iterations: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut prompt = "Hello".to_owned();
        let mut batch = 4;
        let mut iterations = 2;
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
                "--iterations" => {
                    iterations = value.parse().map_err(|_| "--iterations is invalid")?
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(2..=8).contains(&batch) || iterations == 0 {
            return Err("batch must be 2..8 and iterations must be positive".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            prompt,
            batch,
            iterations,
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
    eprintln!("error: Qwen3.6 Jacobi probe requires macOS");
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
    let mut target =
        Qwen35Runtime::new_with_retained_layers_and_head(&context, &store, 32, 0, true)?;
    let mut hidden = Vec::new();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        hidden = target.decode_hidden(position, &target.embed_token(token)?)?;
    }
    let previous_logits = target.logits(&hidden)?;
    let first = argmax(&previous_logits);
    let mut candidates = vec![first; args.batch];
    let positions = (prompt_tokens.len()..prompt_tokens.len() + args.batch).collect::<Vec<_>>();
    let mut total_ms = 0.0;
    let mut best_accepted = 0;
    let mut best_candidates = candidates.clone();
    for iteration in 0..args.iterations {
        candidates[0] = first;
        let snapshot = target.snapshot_states();
        let started = Instant::now();
        let hidden_many = target.decode_tokens_many(&candidates, &positions)?;
        let logits_many = target.logits_many(&hidden_many)?;
        total_ms += started.elapsed().as_secs_f64() * 1_000.0;
        let accepted = accepted_prefix(first, &candidates, &logits_many);
        if accepted > best_accepted {
            best_accepted = accepted;
            best_candidates.clone_from(&candidates);
        }
        let mut next = Vec::with_capacity(args.batch);
        next.push(first);
        for logits in logits_many.iter().take(args.batch.saturating_sub(1)) {
            next.push(argmax(logits));
        }
        candidates = next;
        target.restore_states(&snapshot)?;
        println!(
            "iteration={} accepted={} candidates={:?}",
            iteration + 1,
            accepted,
            best_candidates
        );
    }
    println!(
        "probe=qwen35_jacobi prompt={:?} batch={} iterations={} total_verify_ms={:.3} best_accepted={} best_candidates={:?} target_passes_per_accepted={:.3}",
        args.prompt,
        args.batch,
        args.iterations,
        total_ms,
        best_accepted,
        best_candidates,
        args.iterations as f64 / best_accepted.max(1) as f64,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn accepted_prefix(first: u32, candidates: &[u32], logits: &[Vec<f32>]) -> usize {
    let mut expected = first;
    let mut accepted = 0;
    for (candidate, candidate_logits) in candidates.iter().zip(logits) {
        if *candidate != expected {
            break;
        }
        accepted += 1;
        expected = argmax(candidate_logits);
    }
    accepted
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
    fn parses_jacobi_options() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--batch".into(),
            "8".into(),
            "--iterations".into(),
            "3".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 8);
        assert_eq!(args.iterations, 3);
    }
}
