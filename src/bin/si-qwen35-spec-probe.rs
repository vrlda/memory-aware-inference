use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use super_inference::model::GgufModelStore;
use super_inference::qwen35_runtime::Qwen35Runtime;
use super_inference::tokenizer::QwenTokenizer;

#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    batch: usize,
    draft_layers: usize,
    draft_set: Option<Vec<usize>>,
    prompt: String,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut batch = 4;
        let mut draft_layers = 4;
        let mut draft_set = None;
        let mut prompt = "x".to_owned();
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
                "--draft-layers" => {
                    draft_layers = value.parse().map_err(|_| "--draft-layers is invalid")?
                }
                "--draft-set" => {
                    let layers = value
                        .split(',')
                        .map(|item| item.parse::<usize>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "--draft-set is invalid")?;
                    if layers.is_empty() {
                        return Err("--draft-set must not be empty".into());
                    }
                    draft_set = Some(layers);
                }
                "--prompt" => prompt = value.clone(),
                _ => return Err(format!("unknown option: {flag}")),
            }
            index += 2;
        }
        if !(2..=8).contains(&batch) || draft_layers == 0 {
            return Err("batch must be 2..8 and draft layers must be positive".into());
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            batch,
            draft_layers,
            draft_set,
            prompt,
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
    eprintln!("error: Qwen3.6 speculation probe requires macOS");
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
    if let Some(layers) = &args.draft_set {
        let layer_set = layers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        std::env::set_var("SI_RETAIN_FULL_LAYERS", layer_set);
    }
    let mut draft = Qwen35Runtime::new_with_retained_layers_and_head(
        &context,
        &store,
        32,
        if args.draft_set.is_some() {
            0
        } else {
            args.draft_layers
        },
        true,
    )?;
    let mut target_hidden = Vec::new();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        let target_embedding = target.embed_token(token)?;
        target_hidden = target.decode_hidden(position, &target_embedding)?;
        let draft_embedding = draft.embed_token(token)?;
        if let Some(layers) = &args.draft_set {
            let _ = draft.decode_hidden_layer_set(position, &draft_embedding, layers)?;
        } else {
            let _ = draft.decode_hidden_prefix(position, &draft_embedding, args.draft_layers)?;
        }
    }
    let target_logits = target.logits(&target_hidden)?;
    let mut candidates = Vec::with_capacity(args.batch);
    let draft_started = Instant::now();
    let first = argmax(&target_logits);
    candidates.push(first);
    let first_embedding = draft.embed_token(first)?;
    let first_hidden = if let Some(layers) = &args.draft_set {
        draft.decode_hidden_layer_set(prompt_tokens.len(), &first_embedding, layers)?
    } else {
        draft.decode_hidden_prefix(prompt_tokens.len(), &first_embedding, args.draft_layers)?
    };
    let first_logits = draft.logits(&first_hidden)?;
    let mut draft_logits = first_logits;
    for index in 1..args.batch {
        let token = argmax(&draft_logits);
        candidates.push(token);
        let embedding = draft.embed_token(token)?;
        let hidden = if let Some(layers) = &args.draft_set {
            draft.decode_hidden_layer_set(prompt_tokens.len() + index, &embedding, layers)?
        } else {
            draft.decode_hidden_prefix(
                prompt_tokens.len() + index,
                &embedding,
                args.draft_layers,
            )?
        };
        draft_logits = draft.logits(&hidden)?;
    }
    let draft_ms = draft_started.elapsed().as_secs_f64() * 1_000.0;
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
        "probe=qwen35_spec prompt={:?} batch={} draft_layers={} draft_set={:?} draft_ms={:.3} verify_ms={:.3} candidates={:?} accepted={} acceptance={:.3} target_traversals_per_accepted={:.3}",
        args.prompt,
        args.batch,
        args.draft_layers,
        args.draft_set,
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
    fn parses_speculation_options() {
        let args = Args::parse([
            "--model".into(),
            "model.gguf".into(),
            "--batch".into(),
            "4".into(),
            "--draft-layers".into(),
            "6".into(),
        ])
        .expect("arguments should parse");
        assert_eq!(args.batch, 4);
        assert_eq!(args.draft_layers, 6);
    }
}
