use std::env;
use std::process::ExitCode;

use super_inference::{run_mock, Config, OutputFormat};

const USAGE: &str = "Usage: si-bench --model PATH --backend mock|metal-streaming|metal-resident --prompt TEXT [options]\n\nOptions:\n  --max-tokens N          Generated tokens (default: 32)\n  --context N             Maximum context capacity (default: 2048)\n  --chunk-rows N          Stream matvec weights in row chunks\n  --retain-output-head   Keep the LM head in a private Metal buffer\n  --retain-layers N       Keep the first N transformer layers in Metal\n  --quality-fixture PATH  Run the versioned capability suite (slow, opt-in)\n  --warmup N              Warmup runs (default: 1)\n  --repetitions N         Measured repetitions (default: 3)\n  --verify-manifest       Verify pinned model file digests\n  --output text|json      Report format (default: text)\n  --expected-output TEXT  Compare generated text\n  -h, --help              Show this help";

fn main() -> ExitCode {
    match parse_args(env::args().skip(1).collect()) {
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(config)) => match config.backend.as_str() {
            "mock" => {
                let report = run_mock(&config);
                match config.output_format {
                    OutputFormat::Text => println!("{}", report.as_text()),
                    OutputFormat::Json => println!("{}", report.as_json()),
                }
                if report.output_match == Some(false) {
                    ExitCode::from(2)
                } else {
                    ExitCode::SUCCESS
                }
            }
            "metal-streaming" | "metal-resident" => run_metal_resident(&config),
            _ => unreachable!("backend is validated by parse_args"),
        },
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            ExitCode::from(1)
        }
    }
}

fn parse_args(arguments: Vec<String>) -> Result<Option<Config>, String> {
    if arguments.is_empty()
        || arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
    {
        return Ok(None);
    }
    let mut model_path = None;
    let mut backend = None;
    let mut prompt = None;
    let mut max_tokens = 32;
    let mut max_context = 2048;
    let mut chunk_rows = None;
    let mut retain_output_head = false;
    let mut retain_layers = 0;
    let mut quality_fixture = None;
    let mut warmup = 1;
    let mut repetitions = 3;
    let mut verify_manifest = false;
    let mut output_format = OutputFormat::Text;
    let mut expected_output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        if flag == "--verify-manifest" {
            verify_manifest = true;
            index += 1;
            continue;
        }
        if flag == "--retain-output-head" {
            retain_output_head = true;
            index += 1;
            continue;
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model_path = Some(value.clone()),
            "--backend" => backend = Some(value.clone()),
            "--prompt" => prompt = Some(value.clone()),
            "--max-tokens" => max_tokens = positive_u32(flag, value)?,
            "--context" => max_context = positive_usize(flag, value)?,
            "--chunk-rows" => chunk_rows = Some(positive_usize(flag, value)?),
            "--retain-layers" => retain_layers = positive_usize(flag, value)?,
            "--quality-fixture" => quality_fixture = Some(value.clone()),
            "--warmup" => {
                warmup = value
                    .parse()
                    .map_err(|_| format!("{flag} must be a non-negative integer"))?
            }
            "--repetitions" => repetitions = positive_u32(flag, value)?,
            "--output" => {
                output_format = match value.as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    _ => return Err("--output must be text or json".into()),
                }
            }
            "--expected-output" => expected_output = Some(value.clone()),
            _ => return Err(format!("unknown option: {flag}")),
        }
        index += 2;
    }
    let backend = backend.ok_or("--backend is required")?;
    if backend != "mock" && backend != "metal-streaming" && backend != "metal-resident" {
        return Err(format!(
            "unsupported backend: {backend}; available: mock, metal-streaming, metal-resident"
        ));
    }
    Ok(Some(Config {
        model_path: model_path.ok_or("--model is required")?,
        backend,
        prompt: prompt.ok_or("--prompt is required")?,
        max_tokens,
        max_context,
        chunk_rows,
        retain_output_head,
        retain_layers,
        quality_fixture,
        warmup,
        repetitions,
        verify_manifest,
        output_format,
        expected_output,
    }))
}

fn positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    match value.parse::<u32>() {
        Ok(number) if number > 0 => Ok(number),
        _ => Err(format!("{flag} must be a positive integer")),
    }
}

fn positive_usize(flag: &str, value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(number) if number > 0 => Ok(number),
        _ => Err(format!("{flag} must be a positive integer")),
    }
}

fn run_metal_resident(config: &Config) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        let result = match super_inference::model::detect_model_format(&config.model_path) {
            Ok(super_inference::model::ModelFormat::Gguf) => {
                super_inference::runtime::run_gguf_resident(config)
            }
            Ok(super_inference::model::ModelFormat::Safetensors) => {
                super_inference::runtime::run_resident(config)
            }
            Err(error) => Err(error.to_string()),
        };
        match result {
            Ok(report) => {
                match config.output_format {
                    OutputFormat::Text => println!("{}", report.as_text()),
                    OutputFormat::Json => println!("{}", report.as_json()),
                }
                if report.output_match == Some(false) {
                    ExitCode::from(2)
                } else {
                    ExitCode::SUCCESS
                }
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::from(1)
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = config;
        eprintln!("error: metal-resident backend requires macOS");
        ExitCode::from(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_required_arguments() {
        assert_eq!(
            parse_args(vec!["--backend".into(), "mock".into()]).unwrap_err(),
            "--model is required"
        );
    }

    #[test]
    fn parses_json_configuration() {
        let config = parse_args(vec![
            "--model".into(),
            "m".into(),
            "--backend".into(),
            "mock".into(),
            "--prompt".into(),
            "hi".into(),
            "--output".into(),
            "json".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(config.output_format, OutputFormat::Json);
        assert_eq!(config.max_context, 2048);
        assert_eq!(config.chunk_rows, None);
        assert!(!config.retain_output_head);
        assert_eq!(config.retain_layers, 0);
        assert_eq!(config.repetitions, 3);
    }

    #[test]
    fn parses_streaming_chunk_cap() {
        let config = parse_args(vec![
            "--model".into(),
            "m".into(),
            "--backend".into(),
            "metal-streaming".into(),
            "--prompt".into(),
            "hi".into(),
            "--chunk-rows".into(),
            "16".into(),
        ])
        .expect("arguments should parse")
        .expect("configuration should exist");
        assert_eq!(config.chunk_rows, Some(16));
    }

    #[test]
    fn parses_output_head_residency_flag() {
        let config = parse_args(vec![
            "--model".into(),
            "m".into(),
            "--backend".into(),
            "metal-streaming".into(),
            "--prompt".into(),
            "hi".into(),
            "--retain-output-head".into(),
        ])
        .expect("arguments should parse")
        .expect("configuration should exist");
        assert!(config.retain_output_head);
    }

    #[test]
    fn parses_retained_layer_count() {
        let config = parse_args(vec![
            "--model".into(),
            "m".into(),
            "--backend".into(),
            "metal-streaming".into(),
            "--prompt".into(),
            "hi".into(),
            "--retain-layers".into(),
            "4".into(),
        ])
        .expect("arguments should parse")
        .expect("configuration should exist");
        assert_eq!(config.retain_layers, 4);
    }

    #[test]
    fn parses_repetition_count() {
        let config = parse_args(vec![
            "--model".into(),
            "m".into(),
            "--backend".into(),
            "mock".into(),
            "--prompt".into(),
            "hi".into(),
            "--repetitions".into(),
            "5".into(),
        ])
        .expect("arguments should parse")
        .expect("configuration should exist");
        assert_eq!(config.repetitions, 5);
    }
}
