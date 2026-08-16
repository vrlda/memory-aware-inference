//! Collect exact target labels paired with retained-prefix hidden states.
//! This is a training/coverage diagnostic; it never changes target weights or
//! accepts an unverified sidecar token.

use std::path::PathBuf;

#[cfg(target_os = "macos")]
use serde::Serialize;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::from(1)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("error: resident drafter probe requires macOS");
}

#[cfg(target_os = "macos")]
fn top_k_ids(logits: &[f32], limit: usize) -> Vec<usize> {
    let mut ranked = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
    ranked
        .into_iter()
        .take(limit)
        .map(|(index, _)| index)
        .collect()
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Serialize)]
struct TraceHeader {
    record_type: &'static str,
    model: String,
    model_revision: String,
    prompt: String,
    prompt_tokens: Vec<u32>,
    layers: usize,
    hidden_size: usize,
    target_hidden_size: usize,
    target_top_k: usize,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Serialize)]
struct TraceRecord {
    record_type: &'static str,
    position: usize,
    input_token: u32,
    target_top1: usize,
    target_top4: Vec<usize>,
    hidden: Vec<f32>,
    target_hidden: Vec<f32>,
}

#[cfg(target_os = "macos")]
struct Args {
    model: PathBuf,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    trace: Option<PathBuf>,
    layers: usize,
    tokens: usize,
    context: usize,
    verify_manifest: bool,
}

#[cfg(target_os = "macos")]
impl Args {
    fn parse(mut values: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut prompt = None;
        let mut prompt_file = None;
        let mut trace = None;
        let mut layers = 8;
        let mut tokens = 32;
        let mut context = 2048;
        let mut verify_manifest = false;
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(next_value(&mut values, &flag)?)),
                "--prompt" => prompt = Some(next_value(&mut values, &flag)?),
                "--prompt-file" => {
                    prompt_file = Some(PathBuf::from(next_value(&mut values, &flag)?))
                }
                "--trace" => trace = Some(PathBuf::from(next_value(&mut values, &flag)?)),
                "--layers" => layers = parse_positive(&mut values, &flag)?,
                "--tokens" => tokens = parse_positive(&mut values, &flag)?,
                "--context" => context = parse_positive(&mut values, &flag)?,
                "--verify-manifest" => verify_manifest = true,
                "-h" | "--help" => return Err(Self::usage().into()),
                unknown => return Err(format!("unknown option: {unknown}")),
            }
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            prompt,
            prompt_file,
            trace,
            layers,
            tokens,
            context,
            verify_manifest,
        })
    }

    fn usage() -> &'static str {
        "Usage: si-resident-drafter-probe --model PATH [--prompt TEXT | --prompt-file PATH] [--trace PATH] [--layers N] [--tokens N] [--context N] [--verify-manifest]"
    }
}

#[cfg(target_os = "macos")]
fn next_value(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[cfg(target_os = "macos")]
fn parse_positive(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let value = next_value(values, flag)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} requires a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{flag} requires a positive integer"));
    }
    Ok(parsed)
}

#[cfg(target_os = "macos")]
const DEFAULT_PROMPT: &str = "Explain why memory paging is useful for local model inference.";

#[cfg(target_os = "macos")]
fn load_prompts(args: &Args) -> Result<Vec<String>, String> {
    if args.prompt.is_some() && args.prompt_file.is_some() {
        return Err("--prompt and --prompt-file are mutually exclusive".into());
    }
    if let Some(path) = &args.prompt_file {
        let contents = std::fs::read_to_string(path)
            .map_err(|error| format!("read prompt file {}: {error}", path.display()))?;
        let prompts = contents
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if prompts.is_empty() {
            return Err(format!(
                "prompt file {} contains no non-empty lines",
                path.display()
            ));
        }
        return Ok(prompts);
    }
    Ok(vec![args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_owned())])
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let prompts = load_prompts(&args)?;
    let mut model = super_inference::runtime::MetalQwen3::from_model_dir(
        &args.model,
        args.verify_manifest,
        args.context,
    )?;
    if args.layers > model.config().num_hidden_layers {
        return Err(format!(
            "--layers {} exceeds model depth {}",
            args.layers,
            model.config().num_hidden_layers
        ));
    }
    model.set_retain_layers(args.layers)?;
    let mut trace_writer = args
        .trace
        .as_ref()
        .map(|path| {
            std::fs::File::create(path)
                .map(std::io::BufWriter::new)
                .map_err(|error| format!("create trace {}: {error}", path.display()))
        })
        .transpose()?;
    let mut records = 0_usize;
    let started = std::time::Instant::now();
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let prompt_tokens = model.tokenizer().encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(format!("prompt {prompt_index} tokenized to zero tokens"));
        }
        if prompt_tokens.len().saturating_add(args.tokens) > args.context {
            return Err(format!(
                "prompt {prompt_index} plus trace tokens exceeds context"
            ));
        }
        if let Some(writer) = trace_writer.as_mut() {
            let header = TraceHeader {
                record_type: "header",
                model: args.model.display().to_string(),
                model_revision: model.model_revision().to_owned(),
                prompt: prompt.clone(),
                prompt_tokens: prompt_tokens.clone(),
                layers: args.layers,
                hidden_size: model.config().hidden_size,
                target_hidden_size: model.config().hidden_size,
                target_top_k: 4,
            };
            serde_json::to_writer(&mut *writer, &header)
                .map_err(|error| format!("write trace header: {error}"))?;
            std::io::Write::write_all(writer, b"\n")
                .map_err(|error| format!("write trace header newline: {error}"))?;
        }
        let mut target_logits = model.prefill(&prompt_tokens)?;
        model.prepare_partial_draft(&prompt_tokens, args.layers)?;
        for index in 0..args.tokens {
            let input_token = top_k_ids(&target_logits, 1)
                .first()
                .copied()
                .ok_or("target logits contain no finite values")?
                as u32;
            let position = prompt_tokens.len() + index;
            let hidden = model.partial_draft_hidden(input_token as usize, position, args.layers)?;
            let (target_hidden, next_logits) =
                model.forward_token_with_hidden(input_token as usize, position)?;
            let target_top4 = top_k_ids(&next_logits, 4);
            let target_top1 = target_top4
                .first()
                .copied()
                .ok_or("target logits contain no finite values")?;
            if let Some(writer) = trace_writer.as_mut() {
                let record = TraceRecord {
                    record_type: "record",
                    position,
                    input_token,
                    target_top1,
                    target_top4,
                    hidden,
                    target_hidden,
                };
                serde_json::to_writer(&mut *writer, &record)
                    .map_err(|error| format!("write trace record: {error}"))?;
                std::io::Write::write_all(writer, b"\n")
                    .map_err(|error| format!("write trace record newline: {error}"))?;
            }
            target_logits = next_logits;
            records += 1;
        }
    }
    if let Some(writer) = trace_writer.as_mut() {
        std::io::Write::flush(writer).map_err(|error| format!("flush trace: {error}"))?;
    }
    println!(
        "resident_drafter_trace=model={} prompts={} layers={} records={} hidden_size={} labels=exact_target_top1_top4 elapsed_ms={:.3} trace={}",
        args.model.display(),
        prompts.len(),
        args.layers,
        records,
        model.config().hidden_size,
        started.elapsed().as_secs_f64() * 1_000.0,
        args.trace
            .as_ref()
            .map_or_else(|| "none".to_owned(), |path| path.display().to_string()),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn args_accept_prompt_file() {
        let args = super::Args::parse(
            [
                "--model",
                "models/qwen3-4b-base",
                "--prompt-file",
                "prompts.txt",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .expect("prompt-file arguments should parse");
        assert_eq!(args.prompt, None);
        assert_eq!(
            args.prompt_file.as_deref(),
            Some(std::path::Path::new("prompts.txt"))
        );
    }

    #[test]
    fn prompt_and_prompt_file_are_rejected_together() {
        let args = super::Args::parse(
            [
                "--model",
                "models/qwen3-4b-base",
                "--prompt",
                "one",
                "--prompt-file",
                "prompts.txt",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .expect("arguments should parse before semantic validation");
        assert!(super::load_prompts(&args)
            .expect_err("both prompt sources must be rejected")
            .contains("mutually exclusive"));
    }

    #[test]
    fn top_k_ids_returns_descending_finite_logits() {
        let logits = [f32::NAN, 1.0, 4.0, 2.0, f32::INFINITY, 3.0];
        assert_eq!(super::top_k_ids(&logits, 3), vec![2, 5, 3]);
    }
}
