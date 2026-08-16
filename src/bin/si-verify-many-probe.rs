use std::path::PathBuf;
use std::process::ExitCode;

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
    eprintln!("error: verify-many probe requires macOS");
    ExitCode::from(1)
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let mut model = super_inference::runtime::MetalQwen3::from_model_dir(
        &args.model,
        args.verify_manifest,
        args.context,
    )?;
    model.set_retain_layers(args.retain_layers)?;
    let prompt_tokens = model.tokenizer().encode(&args.prompt)?;
    if prompt_tokens.is_empty() {
        return Err("probe prompt tokenized to zero tokens".into());
    }
    println!(
        "probe=device={} prompt_tokens={} warmup={} repetitions={} contract=lossless-candidate-forward",
        model.device_info()?.name,
        prompt_tokens.len(),
        args.warmup,
        args.repetitions,
    );

    let previous_logits = model.prefill(&prompt_tokens)?;
    let first_candidate = argmax(&previous_logits) as u32;
    let verification =
        model.verify_many(&previous_logits, &[first_candidate, 0], prompt_tokens.len())?;
    let committed_position = prompt_tokens.len() + verification.accepted_tokens;
    if model.cached_tokens() != committed_position {
        return Err(format!(
            "verify_many committed {} cache tokens, expected {committed_position}",
            model.cached_tokens()
        ));
    }
    println!(
        "verify_many_smoke=accepted:{} next_token:{} committed_cache_tokens:{}",
        verification.accepted_tokens,
        verification.next_token,
        model.cached_tokens()
    );

    for batch in [1_usize, 2, 4, 8] {
        let candidates = (1..=batch).map(|token| token as u32).collect::<Vec<_>>();
        for _ in 0..args.warmup {
            let _ = model.prefill(&prompt_tokens)?;
            let _ = run_separate(&mut model, &candidates, prompt_tokens.len())?;
            let _ = model.prefill(&prompt_tokens)?;
            let _ = model.forward_tokens_many(&candidates, prompt_tokens.len())?;
        }
        let mut separate_elapsed = std::time::Duration::ZERO;
        let mut batched_elapsed = std::time::Duration::ZERO;
        let mut separate_logits = Vec::new();
        let mut batched_logits = Vec::new();
        for _ in 0..args.repetitions {
            model.prefill(&prompt_tokens)?;
            let started = std::time::Instant::now();
            separate_logits = run_separate(&mut model, &candidates, prompt_tokens.len())?;
            separate_elapsed += started.elapsed();

            model.prefill(&prompt_tokens)?;
            let started = std::time::Instant::now();
            batched_logits = model.forward_tokens_many(&candidates, prompt_tokens.len())?;
            batched_elapsed += started.elapsed();
        }
        let max_abs_diff = separate_logits
            .iter()
            .flatten()
            .zip(batched_logits.iter().flatten())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let greedy_ids_match = separate_logits
            .iter()
            .zip(&batched_logits)
            .all(|(separate, batched)| argmax(separate) == argmax(batched));
        let separate_ms = separate_elapsed.as_secs_f64() * 1_000.0 / args.repetitions as f64;
        let batched_ms = batched_elapsed.as_secs_f64() * 1_000.0 / args.repetitions as f64;
        println!(
            "batch={batch} separate_ms={separate_ms:.3} batched_ms={batched_ms:.3} speedup={:.3} separate_tok_s={:.3} batched_tok_s={:.3} max_abs_diff={max_abs_diff:.6} greedy_ids_match={greedy_ids_match}",
            separate_ms / batched_ms,
            batch as f64 * 1_000.0 / separate_ms,
            batch as f64 * 1_000.0 / batched_ms,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[cfg(target_os = "macos")]
fn run_separate(
    model: &mut super_inference::runtime::MetalQwen3,
    candidates: &[u32],
    position: usize,
) -> Result<Vec<Vec<f32>>, String> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, token)| model.forward_token(*token as usize, position + index))
        .collect()
}

#[cfg(target_os = "macos")]
struct Args {
    model: PathBuf,
    prompt: String,
    context: usize,
    verify_manifest: bool,
    retain_layers: usize,
    warmup: usize,
    repetitions: usize,
}

#[cfg(target_os = "macos")]
impl Args {
    fn parse(mut values: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut prompt = "Hello".to_owned();
        let mut context = 2048;
        let mut verify_manifest = false;
        let mut retain_layers = 0;
        let mut warmup = 0;
        let mut repetitions = 1;
        while let Some(value) = values.next() {
            match value.as_str() {
                "--model" => {
                    model = Some(PathBuf::from(
                        values.next().ok_or("--model requires a path")?,
                    ));
                }
                "--prompt" => prompt = values.next().ok_or("--prompt requires text")?,
                "--context" => {
                    context = values
                        .next()
                        .ok_or("--context requires an integer")?
                        .parse()
                        .map_err(|_| "--context requires a positive integer")?;
                }
                "--verify-manifest" => verify_manifest = true,
                "--retain-layers" => {
                    retain_layers = values
                        .next()
                        .ok_or("--retain-layers requires an integer")?
                        .parse()
                        .map_err(|_| "--retain-layers requires a positive integer")?;
                }
                "--warmup" => {
                    warmup = values
                        .next()
                        .ok_or("--warmup requires an integer")?
                        .parse()
                        .map_err(|_| "--warmup requires a non-negative integer")?;
                }
                "--repetitions" => {
                    repetitions = values
                        .next()
                        .ok_or("--repetitions requires an integer")?
                        .parse()
                        .map_err(|_| "--repetitions requires a positive integer")?;
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: si-verify-many-probe --model PATH [--prompt TEXT] [--context N] [--retain-layers N] [--verify-manifest] [--warmup N] [--repetitions N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        let model = model.ok_or("--model is required")?;
        if context == 0 {
            return Err("--context must be greater than zero".into());
        }
        if repetitions == 0 {
            return Err("--repetitions must be greater than zero".into());
        }
        Ok(Self {
            model,
            prompt,
            context,
            verify_manifest,
            retain_layers,
            warmup,
            repetitions,
        })
    }
}
