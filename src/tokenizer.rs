//! Qwen tokenizer wrapper.

use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct QwenTokenizer {
    path: PathBuf,
    tokenizer: Tokenizer,
}

impl QwenTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>, revision: &str) -> Result<Self, String> {
        let path = model_dir.as_ref().join(revision).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|error| format!("load tokenizer {}: {error}", path.display()))?;
        Ok(Self { path, tokenizer })
    }

    pub fn from_gguf_model(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let model_path = model_path.as_ref();
        let path = model_path
            .parent()
            .ok_or_else(|| {
                format!(
                    "GGUF model {} has no parent directory",
                    model_path.display()
                )
            })?
            .join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path).map_err(|error| {
            format!(
                "load GGUF tokenizer sidecar {} (place the matching tokenizer.json beside the GGUF): {error}",
                path.display()
            )
        })?;
        Ok(Self { path, tokenizer })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| format!("encode prompt: {error}"))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|error| format!("decode tokens: {error}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn loads_real_qwen_tokenizer_when_requested() {
        let Ok(model_dir) = std::env::var("SI_MODEL_DIR") else {
            return;
        };
        let revision = "906bfd4b4dc7f14ee4320094d8b41684abff8539";
        let tokenizer = super::QwenTokenizer::from_model_dir(model_dir, revision)
            .expect("Qwen tokenizer should load");
        let tokens = tokenizer.encode("Hello, Super-Inference!").expect("encode");
        assert!(!tokens.is_empty());
        assert!(tokenizer.vocab_size() > 100_000);
        assert!(!tokenizer.decode(&tokens).expect("decode").is_empty());
    }

    #[test]
    fn loads_real_gguf_tokenizer_when_requested() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let tokenizer = super::QwenTokenizer::from_gguf_model(&model_path)
            .expect("GGUF tokenizer sidecar should load");
        let tokens = tokenizer.encode("Hello, Super-Inference!").expect("encode");
        assert!(!tokens.is_empty());
        assert_eq!(tokenizer.vocab_size(), 248070);
        assert!(!tokenizer.decode(&tokens).expect("decode").is_empty());
    }
}
