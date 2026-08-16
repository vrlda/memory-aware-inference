use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use super_inference::model::GgufModelStore;
use super_inference::qwen35_runtime::Qwen35Runtime;
use super_inference::runtime::{MetalQwen3, WeightResidency};
use super_inference::tokenizer::QwenTokenizer;

#[derive(Debug, Clone)]
struct Args {
    target: PathBuf,
    draft: PathBuf,
    prompt: String,
    batch: usize,
    capacity: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut target = None;
        let mut draft = None;
        let mut prompt = "Hello".to_owned();
        let mut batch = 4;
        let mut capacity = 32;
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--target" => target = Some(PathBuf::from(value)),
                "--draft" => draft = Some(PathBuf::from(value)),
                "--prompt" => prompt = value.clone(),
                "--batch" => batch = value.parse().map_err(|_| "--batch is invalid")?,
                "--capacity" => capacity = value.parse().map_err(|_| "--capacity is invalid")?,
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(2..=8).contains(&batch) || capacity == 0 {
            return Err("batch must be 2..8 and capacity must be positive".into());
        }
        Ok(Self {
            target: target.ok_or("--target is required")?,
            draft: draft.ok_or("--draft is required")?,
            prompt,
            batch,
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
    eprintln!("error: external Qwen3.6 drafter probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let target_store = GgufModelStore::open(&args.target).map_err(|error| error.to_string())?;
    let tokenizer = QwenTokenizer::from_gguf_model(&args.target)?;
    let prompt_tokens = tokenizer.encode(&args.prompt)?;
    if prompt_tokens.is_empty() {
        return Err("prompt produced no tokens".into());
    }
    let target_context = super_inference::metal::MetalContext::new()?;
    let mut target = Qwen35Runtime::new_with_retained_layers_and_head(
        &target_context,
        &target_store,
        args.capacity,
        0,
        true,
    )?;
    let mut draft = MetalQwen3::from_model_dir_with_residency(
        &args.draft,
        false,
        args.capacity,
        WeightResidency::SharedResident,
    )?;

    let mut target_hidden = Vec::new();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        let embedding = target.embed_token(token)?;
        target_hidden = target.decode_hidden(position, &embedding)?;
    }
    let target_logits = target.logits(&target_hidden)?;
    let draft_vocab = draft.config().vocab_size;
    if draft_vocab > target_logits.len() {
        return Err("draft vocabulary exceeds target vocabulary".into());
    }
    let draft_prompt = prompt_tokens
        .iter()
        .copied()
        .find(|token| (*token as usize) >= draft_vocab);
    if let Some(token) = draft_prompt {
        return Err(format!(
            "prompt token {token} is outside draft vocabulary; choose a shared-token prompt"
        ));
    }
    let _ = draft.prefill(&prompt_tokens)?;

    let draft_started = Instant::now();
    let candidates = draft.draft_candidates(
        &target_logits[..draft_vocab],
        prompt_tokens.len(),
        args.batch,
    )?;
    let draft_ms = draft_started.elapsed().as_secs_f64() * 1_000.0;
    if candidates.first().copied() != Some(argmax(&target_logits)) {
        return Err(format!(
            "draft first token {} disagrees with target {}",
            candidates.first().copied().unwrap_or_default(),
            argmax(&target_logits)
        ));
    }

    let positions = (prompt_tokens.len()..prompt_tokens.len() + args.batch).collect::<Vec<_>>();
    let snapshot = target.snapshot_states();
    let verify_started = Instant::now();
    let hidden = target.decode_tokens_many(&candidates, &positions)?;
    let logits = target.logits_many(&hidden)?;
    let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;
    let mut expected = argmax(&target_logits);
    let mut accepted = 0;
    for (candidate, candidate_logits) in candidates.iter().zip(&logits) {
        if *candidate != expected {
            break;
        }
        accepted += 1;
        expected = argmax(candidate_logits);
    }
    target.restore_states(&snapshot)?;
    println!(
        "probe=qwen35_external_draft prompt={:?} batch={} draft_vocab={} draft_ms={:.3} verify_ms={:.3} candidates={:?} accepted={} acceptance={:.3} target_passes_per_accepted={:.3} exact_first=true",
        args.prompt,
        args.batch,
        draft_vocab,
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
    fn parses_external_draft_options() {
        let args = Args::parse([
            "--target".into(),
            "target.gguf".into(),
            "--draft".into(),
            "draft".into(),
            "--batch".into(),
            "8".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 8);
        assert_eq!(args.prompt, "Hello");
    }
}
