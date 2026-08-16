use std::env;
use std::process::ExitCode;

use super_inference::model::{GgufModelStore, GgufValue};

fn main() -> ExitCode {
    let Some(model) = parse_model(env::args().skip(1).collect()) else {
        eprintln!("Usage: si-gguf-inspect --model PATH");
        return ExitCode::from(2);
    };
    match GgufModelStore::open(&model) {
        Ok(store) => {
            println!("model={model}");
            println!("tensors={}", store.tensors.len());
            for tensor in store
                .tensors
                .values()
                .filter(|tensor| tensor.is_q4_k())
                .take(12)
            {
                println!(
                    "q4_k_tensor={} shape={:?} bytes={} elements={}",
                    tensor.name,
                    tensor.shape,
                    tensor.byte_len(),
                    tensor.element_count().unwrap_or(0)
                );
            }
            let mut type_counts = std::collections::BTreeMap::<u32, usize>::new();
            for tensor in store.tensors.values() {
                *type_counts.entry(tensor.ggml_type).or_default() += 1;
            }
            println!("tensor_type_counts={type_counts:?}");
            for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
                if let Some(tensor) = store.tensors.get(name) {
                    println!(
                        "special_tensor={} type={} shape={:?} bytes={}",
                        name,
                        tensor.ggml_type,
                        tensor.shape,
                        tensor.byte_len()
                    );
                }
            }
            for tensor in store
                .tensors
                .values()
                .filter(|tensor| !tensor.is_q4_k())
                .take(24)
            {
                println!(
                    "non_q4_tensor={} type={} shape={:?} bytes={}",
                    tensor.name,
                    tensor.ggml_type,
                    tensor.shape,
                    tensor.byte_len()
                );
            }
            for (key, value) in &store.metadata {
                println!("metadata.{key}={}", summarize(value));
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn parse_model(arguments: Vec<String>) -> Option<String> {
    let mut model = None;
    let mut index = 0;
    while index < arguments.len() {
        if arguments.get(index).map(String::as_str) != Some("--model") {
            return None;
        }
        model = arguments.get(index + 1).cloned();
        index += 2;
    }
    model
}

fn summarize(value: &GgufValue) -> String {
    match value {
        GgufValue::Array(values) => {
            let preview = values
                .iter()
                .take(3)
                .map(summarize)
                .collect::<Vec<_>>()
                .join(", ");
            format!("array(len={}, first=[{}])", values.len(), preview)
        }
        GgufValue::String(value) if value.len() > 160 => {
            let prefix = value.chars().take(160).collect::<String>();
            format!("string(len={}, prefix={prefix:?})", value.len())
        }
        GgufValue::String(value) => format!("string({value:?})"),
        other => format!("{other:?}"),
    }
}
