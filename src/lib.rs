//! Measurement primitives for Super-Inference experiments.
//!
//! The first backend is deliberately deterministic. It validates reporting and
//! comparison plumbing before a runtime backend is introduced.

pub mod cache;
pub mod metal;
pub mod model;
pub mod planner;
pub mod quality;
pub mod quant;
pub mod qwen35;
pub mod qwen35_runtime;
pub mod runtime;
pub mod telemetry;
pub mod tokenizer;

use std::fmt::Write as _;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub model_path: String,
    pub backend: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub max_context: usize,
    pub chunk_rows: Option<usize>,
    pub retain_output_head: bool,
    pub retain_layers: usize,
    pub quality_fixture: Option<String>,
    pub warmup: u32,
    pub repetitions: u32,
    pub verify_manifest: bool,
    pub output_format: OutputFormat,
    pub expected_output: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    pub model_path: String,
    pub backend: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub warmup_runs: u32,
    pub prefill: Duration,
    pub decode: Duration,
    pub peak_vram_mib: u64,
    pub peak_ram_mib: u64,
    pub output_match: Option<bool>,
    pub generated_output: String,
}

impl RunReport {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.decode)
    }

    pub fn total_tokens_per_second(&self) -> f64 {
        rate(
            self.prompt_tokens + self.generated_tokens,
            self.prefill + self.decode,
        )
    }

    pub fn as_text(&self) -> String {
        format!(
            "model={}\nbackend={}\nprompt_tokens={}\ngenerated_tokens={}\nwarmup_runs={}\nprefill_ms={:.3}\ndecode_ms={:.3}\nprefill_tok_s={:.3}\ndecode_tok_s={:.3}\ntotal_tok_s={:.3}\npeak_vram_mib={}\npeak_ram_mib={}\noutput_match={}\ngenerated_output={}",
            self.model_path,
            self.backend,
            self.prompt_tokens,
            self.generated_tokens,
            self.warmup_runs,
            self.prefill.as_secs_f64() * 1_000.0,
            self.decode.as_secs_f64() * 1_000.0,
            self.prefill_tokens_per_second(),
            self.decode_tokens_per_second(),
            self.total_tokens_per_second(),
            self.peak_vram_mib,
            self.peak_ram_mib,
            option_bool(self.output_match),
            self.generated_output,
        )
    }

    pub fn as_json(&self) -> String {
        format!(
            concat!(
                "{{\"model_path\":\"{}\",\"backend\":\"{}\",",
                "\"prompt_tokens\":{},\"generated_tokens\":{},\"warmup_runs\":{},",
                "\"prefill_ms\":{:.3},\"decode_ms\":{:.3},",
                "\"prefill_tok_s\":{:.3},\"decode_tok_s\":{:.3},\"total_tok_s\":{:.3},",
                "\"peak_vram_mib\":{},\"peak_ram_mib\":{},\"output_match\":{},",
                "\"generated_output\":\"{}\"}}"
            ),
            json_escape(&self.model_path),
            json_escape(&self.backend),
            self.prompt_tokens,
            self.generated_tokens,
            self.warmup_runs,
            self.prefill.as_secs_f64() * 1_000.0,
            self.decode.as_secs_f64() * 1_000.0,
            self.prefill_tokens_per_second(),
            self.decode_tokens_per_second(),
            self.total_tokens_per_second(),
            self.peak_vram_mib,
            self.peak_ram_mib,
            option_bool(self.output_match),
            json_escape(&self.generated_output),
        )
    }
}

pub fn run_mock(config: &Config) -> RunReport {
    let prompt_tokens = token_count(&config.prompt);
    let generated_output = (0..config.max_tokens)
        .map(|index| format!("mock-{index}"))
        .collect::<Vec<_>>()
        .join(" ");

    for _ in 0..config.warmup {
        std::hint::black_box(token_count(&config.prompt));
    }

    let prefill_start = Instant::now();
    let prefill_checksum = config
        .prompt
        .as_bytes()
        .iter()
        .fold(0_u64, |sum, byte| sum + u64::from(*byte));
    std::hint::black_box(prefill_checksum);
    let prefill = prefill_start.elapsed();

    let decode_start = Instant::now();
    for token in generated_output.split_whitespace() {
        std::hint::black_box(token);
    }
    let decode = decode_start.elapsed();

    RunReport {
        model_path: config.model_path.clone(),
        backend: config.backend.clone(),
        prompt_tokens,
        generated_tokens: config.max_tokens,
        warmup_runs: config.warmup,
        prefill,
        decode,
        // Deterministic placeholders only. A real backend must replace these
        // estimates with allocator/device telemetry.
        peak_vram_mib: 64 + u64::from(config.max_tokens) / 8,
        peak_ram_mib: 32 + config.model_path.len() as u64 / 64,
        output_match: config
            .expected_output
            .as_ref()
            .map(|expected| expected == &generated_output),
        generated_output,
    }
}

fn token_count(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

fn rate(tokens: u32, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        f64::from(tokens) / duration.as_secs_f64()
    }
}

fn option_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    }
}

pub(crate) fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => {
                write!(escaped, "\\u{:04x}", c as u32).expect("writing to String cannot fail")
            }
            c => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_handle_zero_duration() {
        assert_eq!(rate(12, Duration::ZERO), 0.0);
    }

    #[test]
    fn json_report_escapes_strings_and_has_expected_fields() {
        let report = RunReport {
            model_path: "model\\\"x".into(),
            backend: "mock".into(),
            prompt_tokens: 2,
            generated_tokens: 3,
            warmup_runs: 1,
            prefill: Duration::from_millis(2),
            decode: Duration::from_millis(3),
            peak_vram_mib: 64,
            peak_ram_mib: 32,
            output_match: Some(true),
            generated_output: "a\nb".into(),
        };
        let json = report.as_json();
        assert!(json.contains("\"backend\":\"mock\""));
        assert!(json.contains("model\\\\\\\"x"));
        assert!(json.contains("a\\nb"));
    }

    #[test]
    fn mock_compares_expected_output() {
        let mut config = test_config();
        config.max_tokens = 2;
        config.expected_output = Some("mock-0 mock-1".into());
        assert_eq!(run_mock(&config).output_match, Some(true));
    }

    fn test_config() -> Config {
        Config {
            model_path: "model.gguf".into(),
            backend: "mock".into(),
            prompt: "hello world".into(),
            max_tokens: 1,
            max_context: 2048,
            chunk_rows: None,
            retain_output_head: false,
            retain_layers: 0,
            quality_fixture: None,
            warmup: 0,
            repetitions: 1,
            verify_manifest: false,
            output_format: OutputFormat::Text,
            expected_output: None,
        }
    }
}
