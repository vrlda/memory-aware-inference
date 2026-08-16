use std::env;
use std::process::ExitCode;

use super_inference::model::pack_model_execution_order;

const USAGE: &str = "Usage: si-pack-model --model PATH --output PATH [--verify-manifest]";

fn main() -> ExitCode {
    match parse_args(env::args().skip(1).collect()) {
        Ok(Some((model, output, verify_manifest))) => {
            match pack_model_execution_order(model, verify_manifest, output) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::from(1)
                }
            }
        }
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            ExitCode::from(1)
        }
    }
}

fn parse_args(arguments: Vec<String>) -> Result<Option<(String, String, bool)>, String> {
    if arguments.is_empty() || arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(None);
    }
    let mut model = None;
    let mut output = None;
    let mut verify_manifest = false;
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--verify-manifest" {
            verify_manifest = true;
            index += 1;
            continue;
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {}", arguments[index]))?;
        match arguments[index].as_str() {
            "--model" => model = Some(value.clone()),
            "--output" => output = Some(value.clone()),
            flag => return Err(format!("unknown option: {flag}")),
        }
        index += 2;
    }
    Ok(Some((
        model.ok_or("--model is required")?,
        output.ok_or("--output is required")?,
        verify_manifest,
    )))
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    #[test]
    fn parses_pack_arguments() {
        assert_eq!(
            parse_args(vec![
                "--model".into(),
                "model".into(),
                "--output".into(),
                "packed".into(),
                "--verify-manifest".into(),
            ])
            .expect("arguments should parse"),
            Some(("model".into(), "packed".into(), true))
        );
    }
}
