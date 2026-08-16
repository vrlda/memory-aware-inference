//! File-backed model metadata and Safetensors access.
//!
//! This module deliberately stops before tensor execution. It validates the
//! immutable artifact and exposes borrowed tensor bytes so Metal can upload
//! only the ranges requested by a later scheduler.

use memmap2::{Mmap, MmapOptions};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
#[cfg(target_os = "macos")]
use std::ops::Range;
#[cfg(target_os = "macos")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "macos")]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::thread::{self, JoinHandle};

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Safetensors,
    Gguf,
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Detect the immutable model container without loading any weight payload.
/// Safetensors models are represented by a directory with `manifest.json`;
/// GGUF models are single files with the GGUF magic prefix.
pub fn detect_model_format(path: impl AsRef<Path>) -> Result<ModelFormat> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)
        .map_err(|error| ModelError(format!("stat model {}: {error}", path.display())))?;
    if metadata.is_dir() {
        let manifest_path = path.join("manifest.json");
        let manifest = std::fs::read(&manifest_path).map_err(|error| {
            ModelError(format!(
                "read model manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let value: serde_json::Value = serde_json::from_slice(&manifest)?;
        if value.get("format").and_then(serde_json::Value::as_str) == Some("safetensors")
            && value.get("dtype").and_then(serde_json::Value::as_str) == Some("bfloat16")
        {
            return Ok(ModelFormat::Safetensors);
        }
        return Err(ModelError(
            "model directory is not a BF16 Safetensors artifact".into(),
        ));
    }

    let mut file = File::open(path)
        .map_err(|error| ModelError(format!("open model {}: {error}", path.display())))?;
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)?;
    if &magic == GGUF_MAGIC {
        Ok(ModelFormat::Gguf)
    } else {
        Err(ModelError(format!(
            "unsupported model container {}; expected a Safetensors directory or GGUF file",
            path.display()
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(pub String);

impl Display for ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ModelError {}

impl From<std::io::Error> for ModelError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ModelError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub torch_dtype: String,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
}

/// Rank-1 and projection tensors in the order touched by one Qwen3 layer.
/// Keeping this order shared by execution and prefetching makes read-ahead
/// follow actual accesses instead of metadata or lexical tensor order.
pub const QWEN3_LAYER_TENSOR_SUFFIXES: &[&str] = &[
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

pub fn qwen3_layer_tensor_names(layer: usize) -> Vec<String> {
    let prefix = format!("model.layers.{layer}");
    QWEN3_LAYER_TENSOR_SUFFIXES
        .iter()
        .map(|suffix| format!("{prefix}.{suffix}"))
        .collect()
}

pub fn qwen3_execution_tensor_names(config: &ModelConfig) -> Vec<String> {
    let mut names =
        Vec::with_capacity(2 + config.num_hidden_layers * QWEN3_LAYER_TENSOR_SUFFIXES.len());
    names.push("model.embed_tokens.weight".into());
    for layer in 0..config.num_hidden_layers {
        names.extend(qwen3_layer_tensor_names(layer));
    }
    names.push("model.norm.weight".into());
    names
}

impl ModelConfig {
    pub fn validate_qwen3(&self) -> Result<()> {
        if self.model_type != "qwen3" {
            return Err(ModelError(format!(
                "unsupported model_type {}; expected qwen3",
                self.model_type
            )));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err(ModelError("model config contains zero dimensions".into()));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ModelError(
                "attention heads must be divisible by key/value heads".into(),
            ));
        }
        if self.torch_dtype != "bfloat16" {
            return Err(ModelError(format!(
                "unsupported torch_dtype {}; expected bfloat16",
                self.torch_dtype
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct IndexMetadata {
    pub total_size: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct RawIndex {
    pub metadata: IndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIndex {
    pub metadata: IndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

impl ModelIndex {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let raw: RawIndex = read_json(path.as_ref())?;
        if raw.weight_map.is_empty() {
            return Err(ModelError("Safetensors index has no tensors".into()));
        }
        Ok(Self {
            metadata: raw.metadata,
            weight_map: raw.weight_map,
        })
    }

    pub fn shards(&self) -> BTreeSet<&str> {
        self.weight_map.values().map(String::as_str).collect()
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ManifestFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub model: String,
    pub revision: String,
    pub format: String,
    pub dtype: String,
    pub total_weight_bytes: u64,
    pub files: Vec<ManifestFile>,
}

impl ModelManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let manifest: Self = read_json(path.as_ref())?;
        if manifest.format != "safetensors" || manifest.dtype != "bfloat16" {
            return Err(ModelError("manifest must describe BF16 Safetensors".into()));
        }
        if manifest.revision.is_empty() || manifest.files.is_empty() {
            return Err(ModelError("manifest is missing revision or files".into()));
        }
        Ok(manifest)
    }

    pub fn validate_files(&self, model_dir: impl AsRef<Path>) -> Result<()> {
        let model_dir = model_dir.as_ref();
        for entry in &self.files {
            let path = model_dir.join(&entry.path);
            let metadata = std::fs::metadata(&path).map_err(|error| {
                ModelError(format!("manifest file {}: {error}", path.display()))
            })?;
            if metadata.len() != entry.size_bytes {
                return Err(ModelError(format!(
                    "size mismatch for {}: expected {}, got {}",
                    path.display(),
                    entry.size_bytes,
                    metadata.len()
                )));
            }
            let digest = sha256_file(&path)?;
            if digest != entry.sha256 {
                return Err(ModelError(format!(
                    "SHA-256 mismatch for {}: expected {}, got {}",
                    path.display(),
                    entry.sha256,
                    digest
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

impl TensorInfo {
    pub fn byte_len(&self) -> usize {
        self.data_end - self.data_start
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufQwen35Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_time_step_rank: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQwen35LayerKind {
    GatedDeltaNet,
    FullAttention,
}

impl GgufQwen35Config {
    pub fn from_metadata(metadata: &BTreeMap<String, GgufValue>) -> Result<Self> {
        let model_type = required_string(metadata, "general.architecture")?;
        if model_type != "qwen35" {
            return Err(ModelError(format!(
                "unsupported GGUF architecture {model_type}; expected qwen35"
            )));
        }
        let config = Self {
            model_type,
            hidden_size: required_usize(metadata, "qwen35.embedding_length")?,
            intermediate_size: required_usize(metadata, "qwen35.feed_forward_length")?,
            num_hidden_layers: required_usize(metadata, "qwen35.block_count")?,
            num_attention_heads: required_usize(metadata, "qwen35.attention.head_count")?,
            num_key_value_heads: required_usize(metadata, "qwen35.attention.head_count_kv")?,
            head_dim: required_usize(metadata, "qwen35.attention.key_length")?,
            value_dim: required_usize(metadata, "qwen35.attention.value_length")?,
            vocab_size: required_array_len(metadata, "tokenizer.ggml.tokens")?,
            context_length: required_usize(metadata, "qwen35.context_length")?,
            rms_norm_eps: required_f32(metadata, "qwen35.attention.layer_norm_rms_epsilon")?,
            rope_theta: required_f32(metadata, "qwen35.rope.freq_base")?,
            eos_token_id: required_u32(metadata, "tokenizer.ggml.eos_token_id")?,
            full_attention_interval: required_usize(metadata, "qwen35.full_attention_interval")?,
            ssm_inner_size: required_usize(metadata, "qwen35.ssm.inner_size")?,
            ssm_state_size: required_usize(metadata, "qwen35.ssm.state_size")?,
            ssm_group_count: required_usize(metadata, "qwen35.ssm.group_count")?,
            ssm_conv_kernel: required_usize(metadata, "qwen35.ssm.conv_kernel")?,
            ssm_time_step_rank: required_usize(metadata, "qwen35.ssm.time_step_rank")?,
        };
        if config.hidden_size == 0
            || config.intermediate_size == 0
            || config.num_hidden_layers == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
            || config.head_dim == 0
            || config.value_dim == 0
            || config.vocab_size == 0
            || config.context_length == 0
            || config.full_attention_interval == 0
            || config.ssm_inner_size == 0
            || config.ssm_state_size == 0
            || config.ssm_group_count == 0
            || config.ssm_conv_kernel == 0
            || config.ssm_time_step_rank == 0
            || !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.0
            || !config.rope_theta.is_finite()
            || config.rope_theta <= 1.0
        {
            return Err(ModelError(
                "Qwen3.6 GGUF config contains invalid dimensions".into(),
            ));
        }
        if !config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads)
        {
            return Err(ModelError(
                "Qwen3.6 attention heads must be divisible by key/value heads".into(),
            ));
        }
        if !config.ssm_inner_size.is_multiple_of(config.ssm_state_size) {
            return Err(ModelError(
                "Qwen3.6 SSM inner size must be divisible by state size".into(),
            ));
        }
        Ok(config)
    }

    /// Number of key/query heads in the Gated DeltaNet projection.
    pub fn ssm_key_heads(&self) -> usize {
        self.ssm_group_count
    }

    /// Number of value heads in the Gated DeltaNet projection.
    pub fn ssm_value_heads(&self) -> usize {
        self.ssm_inner_size / self.ssm_state_size
    }

    /// Per-head key/query dimension.
    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_state_size
    }

    /// Per-head value dimension.
    pub fn ssm_value_dim(&self) -> usize {
        self.ssm_state_size
    }

    /// Width of the fused QKV projection before channel-wise convolution.
    pub fn ssm_projection_size(&self) -> usize {
        (self.ssm_key_heads() * 2 + self.ssm_value_heads()) * self.ssm_state_size
    }
}

fn required_string(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<String> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) => Ok(value.clone()),
        Some(value) => Err(ModelError(format!(
            "GGUF metadata {key} has unexpected value {value:?}"
        ))),
        None => Err(ModelError(format!("GGUF metadata is missing {key}"))),
    }
}

fn required_u32(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<u32> {
    let value = match metadata.get(key) {
        Some(GgufValue::U8(value)) => u32::from(*value),
        Some(GgufValue::U16(value)) => u32::from(*value),
        Some(GgufValue::U32(value)) => *value,
        Some(GgufValue::U64(value)) => u32::try_from(*value)
            .map_err(|_| ModelError(format!("GGUF metadata {key} overflows u32")))?,
        Some(value) => {
            return Err(ModelError(format!(
                "GGUF metadata {key} has unexpected value {value:?}"
            )))
        }
        None => return Err(ModelError(format!("GGUF metadata is missing {key}"))),
    };
    Ok(value)
}

fn required_usize(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<usize> {
    usize::try_from(required_u32(metadata, key)?)
        .map_err(|_| ModelError(format!("GGUF metadata {key} overflows platform usize")))
}

fn required_f32(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<f32> {
    match metadata.get(key) {
        Some(GgufValue::F32(value)) => Ok(*value),
        Some(GgufValue::F64(value)) => Ok(*value as f32),
        Some(value) => Err(ModelError(format!(
            "GGUF metadata {key} has unexpected value {value:?}"
        ))),
        None => Err(ModelError(format!("GGUF metadata is missing {key}"))),
    }
}

fn required_array_len(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<usize> {
    match metadata.get(key) {
        Some(GgufValue::Array(values)) => Ok(values.len()),
        Some(value) => Err(ModelError(format!(
            "GGUF metadata {key} has unexpected value {value:?}"
        ))),
        None => Err(ModelError(format!("GGUF metadata is missing {key}"))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
    pub data_start: usize,
    pub data_end: usize,
}

impl GgufTensorInfo {
    pub fn byte_len(&self) -> usize {
        self.data_end - self.data_start
    }

    pub fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1_usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                ModelError(format!("GGUF tensor {} element count overflows", self.name))
            })
        })
    }

    pub fn is_q4_k(&self) -> bool {
        self.ggml_type == crate::quant::GGML_TYPE_Q4_K
    }

    pub fn is_q5_k(&self) -> bool {
        self.ggml_type == crate::quant::GGML_TYPE_Q5_K
    }

    pub fn is_q6_k(&self) -> bool {
        self.ggml_type == crate::quant::GGML_TYPE_Q6_K
    }

    pub fn expected_payload_bytes(&self) -> Result<Option<usize>> {
        let elements = self.element_count()?;
        let bytes = match self.ggml_type {
            0 => elements.checked_mul(4),      // F32
            1 | 30 => elements.checked_mul(2), // F16 / BF16
            crate::quant::GGML_TYPE_Q4_K => {
                if !elements.is_multiple_of(crate::quant::Q4_K_BLOCK_ELEMENTS) {
                    return Err(ModelError(format!(
                        "GGUF Q4_K tensor {} has {} elements; expected a multiple of 256",
                        self.name, elements
                    )));
                }
                elements
                    .checked_div(crate::quant::Q4_K_BLOCK_ELEMENTS)
                    .and_then(|blocks| blocks.checked_mul(crate::quant::Q4_K_BLOCK_BYTES))
            }
            crate::quant::GGML_TYPE_Q5_K => {
                if !elements.is_multiple_of(crate::quant::Q5_K_BLOCK_ELEMENTS) {
                    return Err(ModelError(format!(
                        "GGUF Q5_K tensor {} has {} elements; expected a multiple of 256",
                        self.name, elements
                    )));
                }
                elements
                    .checked_div(crate::quant::Q5_K_BLOCK_ELEMENTS)
                    .and_then(|blocks| blocks.checked_mul(crate::quant::Q5_K_BLOCK_BYTES))
            }
            crate::quant::GGML_TYPE_Q6_K => {
                if !elements.is_multiple_of(crate::quant::Q6_K_BLOCK_ELEMENTS) {
                    return Err(ModelError(format!(
                        "GGUF Q6_K tensor {} has {} elements; expected a multiple of 256",
                        self.name, elements
                    )));
                }
                elements
                    .checked_div(crate::quant::Q6_K_BLOCK_ELEMENTS)
                    .and_then(|blocks| blocks.checked_mul(crate::quant::Q6_K_BLOCK_BYTES))
            }
            _ => None,
        };
        Ok(bytes)
    }
}

#[derive(Debug)]
struct ParsedGguf {
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
}

#[derive(Debug)]
pub struct GgufModelStore {
    pub model_path: PathBuf,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: BTreeMap<String, GgufTensorInfo>,
    resident_tensors: BTreeMap<String, Vec<u8>>,
    file: Arc<File>,
    bytes: Mmap,
}

#[derive(Debug)]
pub struct GgufTensorView<'a> {
    pub info: &'a GgufTensorInfo,
    pub bytes: &'a [u8],
}

/// Packed anonymous bytes for one staged Qwen3.6 layer. Tensor ranges point
/// into shared backing storage, avoiding one allocation and one read buffer
/// per tensor while keeping staged layers independently evictable.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct StagedQwenLayer {
    bytes: Vec<u8>,
    ranges: BTreeMap<String, Range<usize>>,
}

#[cfg(target_os = "macos")]
impl StagedQwenLayer {
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.ranges
            .get(name)
            .and_then(|range| self.bytes.get(range.clone()))
    }

    pub fn packed_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn ranges(&self) -> impl Iterator<Item = (&str, usize, usize)> {
        self.ranges
            .iter()
            .map(|(name, range)| (name.as_str(), range.start, range.end))
    }
}

impl GgufModelStore {
    pub fn open(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let file = Arc::new(File::open(&model_path).map_err(|error| {
            ModelError(format!("open GGUF model {}: {error}", model_path.display()))
        })?);
        // Host staging reads one tensor at a time into anonymous memory. Keep
        // those reads file-cacheable by default: repeated token traversals
        // are much faster from RAM than from SSD. F_NOCACHE remains an
        // explicit diagnostic for machines where cache pressure wins.
        #[cfg(target_os = "macos")]
        if stage_nocache_enabled(
            std::env::var_os("SI_STAGE_GGUF").is_some()
                || std::env::var_os("SI_STAGE_PIPELINE").is_some(),
            std::env::var("SI_STAGE_NOCACHE").ok().as_deref(),
        ) {
            // F_NOCACHE is advisory; a failure should not make the model
            // unusable because the read path remains correct without it.
            unsafe {
                let _ = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
            }
        }
        // SAFETY: the file remains open through `file`, and this mapping is
        // read-only for the lifetime of the store.
        let bytes = unsafe { MmapOptions::new().map(file.as_ref()) }?;
        let parsed = parse_gguf(&bytes)?;
        let tensors = parsed
            .tensors
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect();
        Ok(Self {
            model_path,
            metadata: parsed.metadata,
            tensors,
            resident_tensors: BTreeMap::new(),
            file,
            bytes,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<GgufTensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError(format!("unknown GGUF tensor {name}")))?;
        if let Some(bytes) = self.resident_tensors.get(name) {
            Ok(GgufTensorView { info, bytes })
        } else {
            Ok(GgufTensorView {
                info,
                bytes: &self.bytes[info.data_start..info.data_end],
            })
        }
    }

    /// Keep the immutable payloads for the first `count` decoder layers in an
    /// anonymous RAM cache. This is the GGUF equivalent of SI-001's bounded
    /// `--retain-layers` lease: the on-disk GGUF remains untouched, while the
    /// Metal no-copy path avoids repeatedly faulting those hot layers through
    /// a file-backed mapping.
    pub fn retain_prefix_layers(&mut self, count: usize) -> Result<u64> {
        if count == 0 {
            return Ok(0);
        }
        let prefixes = (0..count)
            .map(|layer| format!("blk.{layer}."))
            .collect::<Vec<_>>();
        let names = self
            .tensors
            .keys()
            .filter(|name| prefixes.iter().any(|prefix| name.starts_with(prefix)))
            .cloned()
            .collect::<Vec<_>>();
        if names.is_empty() {
            return Err(ModelError(format!(
                "GGUF has no tensors for retained layer prefix 0..{count}"
            )));
        }
        let mut bytes = 0_u64;
        for name in names {
            if self.resident_tensors.contains_key(&name) {
                continue;
            }
            let info = self
                .tensors
                .get(&name)
                .ok_or_else(|| ModelError(format!("unknown GGUF tensor {name}")))?;
            let payload = self.bytes[info.data_start..info.data_end].to_vec();
            bytes = bytes.saturating_add(payload.len() as u64);
            self.resident_tensors.insert(name, payload);
        }
        Ok(bytes)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_tensors
            .values()
            .map(|bytes| bytes.len() as u64)
            .sum()
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(value)) => Some(*value),
            Some(GgufValue::U64(value)) => u32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn qwen35_config(&self) -> Result<GgufQwen35Config> {
        GgufQwen35Config::from_metadata(&self.metadata)
    }

    pub fn qwen35_layer_kind(&self, layer: usize) -> Result<GgufQwen35LayerKind> {
        let config = self.qwen35_config()?;
        if layer >= config.num_hidden_layers {
            return Err(ModelError(format!(
                "Qwen3.6 layer {layer} exceeds {} layers",
                config.num_hidden_layers
            )));
        }
        let prefix = format!("blk.{layer}");
        let has_delta = self.tensors.contains_key(&format!("{prefix}.ssm_a"));
        let has_attention = self
            .tensors
            .contains_key(&format!("{prefix}.attn_q.weight"));
        if has_delta == has_attention {
            return Err(ModelError(format!(
                "Qwen3.6 layer {layer} does not have exactly one execution kind"
            )));
        }
        let kind = if has_delta {
            GgufQwen35LayerKind::GatedDeltaNet
        } else {
            GgufQwen35LayerKind::FullAttention
        };
        let interval_kind = if (layer + 1).is_multiple_of(config.full_attention_interval) {
            GgufQwen35LayerKind::FullAttention
        } else {
            GgufQwen35LayerKind::GatedDeltaNet
        };
        if kind != interval_kind {
            return Err(ModelError(format!(
                "Qwen3.6 layer {layer} tensor kind {kind:?} disagrees with interval {}",
                config.full_attention_interval
            )));
        }
        Ok(kind)
    }

    pub fn qwen35_layer_kinds(&self) -> Result<Vec<GgufQwen35LayerKind>> {
        let config = self.qwen35_config()?;
        (0..config.num_hidden_layers)
            .map(|layer| self.qwen35_layer_kind(layer))
            .collect()
    }

    pub fn dequantize_q4_k(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if !tensor.info.is_q4_k() {
            return Err(ModelError(format!(
                "GGUF tensor {name} is not Q4_K (type {})",
                tensor.info.ggml_type
            )));
        }
        let elements = tensor.info.element_count()?;
        let expected_bytes = tensor
            .info
            .expected_payload_bytes()?
            .ok_or_else(|| ModelError(format!("GGUF tensor {name} has unknown payload size")))?;
        if tensor.bytes.len() < expected_bytes {
            return Err(ModelError(format!(
                "GGUF tensor {name} has {} bytes; expected at least {expected_bytes}",
                tensor.bytes.len()
            )));
        }
        crate::quant::dequantize_q4_k(&tensor.bytes[..expected_bytes], elements)
            .map_err(|error| ModelError(error.0))
    }

    pub fn dequantize_q5_k(&self, name: &str) -> Result<Vec<f32>> {
        self.dequantize_quantized(name, crate::quant::GGML_TYPE_Q5_K, |bytes, elements| {
            crate::quant::dequantize_q5_k(bytes, elements)
        })
    }

    pub fn dequantize_q6_k(&self, name: &str) -> Result<Vec<f32>> {
        self.dequantize_quantized(name, crate::quant::GGML_TYPE_Q6_K, |bytes, elements| {
            crate::quant::dequantize_q6_k(bytes, elements)
        })
    }

    fn dequantize_quantized<F>(&self, name: &str, ggml_type: u32, decode: F) -> Result<Vec<f32>>
    where
        F: FnOnce(&[u8], usize) -> crate::quant::Result<Vec<f32>>,
    {
        let tensor = self.tensor(name)?;
        if tensor.info.ggml_type != ggml_type {
            return Err(ModelError(format!(
                "GGUF tensor {name} has type {}; expected {ggml_type}",
                tensor.info.ggml_type
            )));
        }
        let elements = tensor.info.element_count()?;
        let expected_bytes = tensor
            .info
            .expected_payload_bytes()?
            .ok_or_else(|| ModelError(format!("GGUF tensor {name} has unknown payload size")))?;
        if tensor.bytes.len() < expected_bytes {
            return Err(ModelError(format!(
                "GGUF tensor {name} has {} bytes; expected at least {expected_bytes}",
                tensor.bytes.len()
            )));
        }
        decode(&tensor.bytes[..expected_bytes], elements).map_err(|error| ModelError(error.0))
    }

    pub fn mapped_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn backing_file(&self) -> &Arc<File> {
        &self.file
    }

    /// Copy one GGUF tensor payload into bounded anonymous host memory.
    ///
    /// This is intentionally opt-in.  The default zero-copy mmap path is
    /// useful when the complete model fits in the machine's working set; on a
    /// 19 GB unified-memory machine a 16.8 GB model does not leave room for
    /// Metal, so staging prevents the OS from thrashing the mapped file.
    #[cfg(target_os = "macos")]
    pub fn stage_tensor_payload(&self, tensor: &GgufTensorView<'_>) -> Result<Vec<u8>> {
        let length = tensor
            .info
            .data_end
            .checked_sub(tensor.info.data_start)
            .ok_or_else(|| ModelError(format!("tensor {} range is invalid", tensor.info.name)))?;
        let offset = u64::try_from(tensor.info.data_start)
            .map_err(|_| ModelError(format!("tensor {} offset overflows", tensor.info.name)))?;
        let mut bytes = vec![0_u8; length];
        self.file
            .read_exact_at(&mut bytes, offset)
            .map_err(|error| {
                ModelError(format!("read staged tensor {}: {error}", tensor.info.name))
            })?;
        Ok(bytes)
    }

    /// Stage all payloads belonging to one Qwen3.6 decoder layer.  The map is
    /// intentionally per-layer so a pipeline can hold only the current and
    /// next layer while their reads and GPU execution overlap.
    #[cfg(target_os = "macos")]
    pub fn stage_qwen35_layer(&self, layer: usize) -> Result<BTreeMap<String, Vec<u8>>> {
        self.stage_qwen35_layer_except(layer, &BTreeSet::new())
    }

    /// Variant of [`stage_qwen35_layer`] that omits weights already retained
    /// in private Metal storage, avoiding redundant disk reads in the staged
    /// pipeline.
    #[cfg(target_os = "macos")]
    pub fn stage_qwen35_layer_except(
        &self,
        layer: usize,
        excluded: &BTreeSet<String>,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        let prefix = format!("blk.{layer}.");
        let mut staged = BTreeMap::new();
        for info in self
            .tensors
            .values()
            .filter(|info| info.name.starts_with(&prefix))
        {
            if excluded.contains(&info.name) {
                continue;
            }
            let tensor = GgufTensorView {
                info,
                bytes: &self.bytes[info.data_start..info.data_end],
            };
            staged.insert(info.name.clone(), self.stage_tensor_payload(&tensor)?);
        }
        Ok(staged)
    }

    /// Stage one layer into packed anonymous runs. Nearby tensor payloads are
    /// read together; large gaps (typically retained tensors) stay excluded,
    /// so private-resident weights are not duplicated in the staging buffer.
    #[cfg(target_os = "macos")]
    pub fn stage_qwen35_layer_packed(
        &self,
        layer: usize,
        excluded: &BTreeSet<String>,
    ) -> Result<StagedQwenLayer> {
        let prefix = format!("blk.{layer}.");
        let mut infos = self
            .tensors
            .values()
            .filter(|info| {
                info.name.starts_with(&prefix)
                    && !excluded.contains(&info.name)
                    && !self.resident_tensors.contains_key(&info.name)
                    && info.ggml_type != 0
            })
            .collect::<Vec<_>>();
        infos.sort_by_key(|info| info.data_start);
        let mut runs = Vec::<(usize, usize)>::new();
        for info in &infos {
            match runs.last_mut() {
                Some((_, end)) if info.data_start <= end.saturating_add(4096) => {
                    *end = (*end).max(info.data_end);
                }
                _ => runs.push((info.data_start, info.data_end)),
            }
        }
        let backing_len = runs.iter().try_fold(0_usize, |total, (start, end)| {
            total
                .checked_add(end.saturating_sub(*start))
                .ok_or_else(|| ModelError("packed staged layer size overflows".into()))
        })?;
        let mut bytes = vec![0_u8; backing_len];
        let mut ranges = BTreeMap::new();
        let mut backing_offset = 0_usize;
        for (run_start, run_end) in runs {
            let run_len = run_end.saturating_sub(run_start);
            let run_end_offset = backing_offset
                .checked_add(run_len)
                .ok_or_else(|| ModelError("packed staged layer range overflows".into()))?;
            if std::env::var_os("SI_STAGE_MMAP_COPY").is_some() {
                bytes[backing_offset..run_end_offset]
                    .copy_from_slice(&self.bytes[run_start..run_end]);
            } else {
                self.file
                    .read_exact_at(
                        &mut bytes[backing_offset..run_end_offset],
                        u64::try_from(run_start)
                            .map_err(|_| ModelError("staged layer offset overflows".into()))?,
                    )
                    .map_err(|error| ModelError(format!("read packed staged layer: {error}")))?;
            }
            for info in &infos {
                if info.data_start >= run_start && info.data_end <= run_end {
                    let start = backing_offset + (info.data_start - run_start);
                    let end = start + (info.data_end - info.data_start);
                    ranges.insert(info.name.clone(), start..end);
                }
            }
            backing_offset = run_end_offset;
        }
        Ok(StagedQwenLayer { bytes, ranges })
    }

    /// Ask macOS to prefetch one decoder layer's mapped payloads.  This is a
    /// bounded hint: it never creates a second model representation and only
    /// touches the next layer selected by the scheduler.
    #[cfg(target_os = "macos")]
    pub fn advise_qwen35_layer(&self, layer: usize) -> Result<usize> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(ModelError("macOS page size is invalid".into()));
        }
        let page_size = page_size as usize;
        let prefix = format!("blk.{layer}.");
        let mut advised = 0_usize;
        for info in self
            .tensors
            .values()
            .filter(|info| info.name.starts_with(&prefix))
        {
            let start = self.bytes.as_ptr() as usize + info.data_start;
            let end = start
                .checked_add(info.data_end.saturating_sub(info.data_start))
                .ok_or_else(|| ModelError(format!("tensor {} address overflows", info.name)))?;
            let aligned_start = start & !(page_size - 1);
            let aligned_end = (end + page_size - 1) & !(page_size - 1);
            let length = aligned_end.saturating_sub(aligned_start);
            if length == 0 {
                continue;
            }
            // SAFETY: the range is inside the read-only GGUF mmap and the
            // advisory call does not dereference or mutate the mapping.
            unsafe {
                let _ = libc::madvise(
                    aligned_start as *mut libc::c_void,
                    length,
                    libc::MADV_WILLNEED,
                );
            }
            advised = advised.saturating_add(length);
        }
        Ok(advised)
    }
}

#[cfg(target_os = "macos")]
fn stage_nocache_enabled(staging: bool, setting: Option<&str>) -> bool {
    staging && setting == Some("1")
}

fn parse_gguf(bytes: &[u8]) -> Result<ParsedGguf> {
    let mut reader = GgufReader { bytes, offset: 0 };
    if reader.read_bytes(4)? != GGUF_MAGIC {
        return Err(ModelError("GGUF header has invalid magic".into()));
    }
    let version = reader.read_u32()?;
    if version != 2 && version != 3 {
        return Err(ModelError(format!("unsupported GGUF version {version}")));
    }
    let tensor_count = usize::try_from(reader.read_u64()?)
        .map_err(|_| ModelError("GGUF tensor count overflows platform size".into()))?;
    let metadata_count = usize::try_from(reader.read_u64()?)
        .map_err(|_| ModelError("GGUF metadata count overflows platform size".into()))?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = reader.read_u32()?;
        let value = reader.read_value(value_type)?;
        if metadata.insert(key, value).is_some() {
            return Err(ModelError("GGUF metadata contains duplicate keys".into()));
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let dimensions = usize::try_from(reader.read_u32()?)
            .map_err(|_| ModelError(format!("GGUF tensor {name} rank overflows")))?;
        let mut shape = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            shape.push(
                usize::try_from(reader.read_u64()?)
                    .map_err(|_| ModelError(format!("GGUF tensor {name} dimension overflows")))?,
            );
        }
        let ggml_type = reader.read_u32()?;
        let offset = reader.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            shape,
            ggml_type,
            offset,
            data_start: 0,
            data_end: 0,
        });
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(value)) => usize::try_from(*value).ok(),
        Some(GgufValue::U64(value)) => usize::try_from(*value).ok(),
        _ => None,
    }
    .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ModelError(
            "GGUF alignment must be a non-zero power of two".into(),
        ));
    }
    let data_base = align_up(reader.offset, alignment)?;

    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|index| tensors[*index].offset);
    for position in 0..order.len() {
        let index = order[position];
        let start_offset = usize::try_from(tensors[index].offset).map_err(|_| {
            ModelError(format!(
                "GGUF tensor {} offset overflows",
                tensors[index].name
            ))
        })?;
        let data_start = data_base.checked_add(start_offset).ok_or_else(|| {
            ModelError(format!(
                "GGUF tensor {} start overflows",
                tensors[index].name
            ))
        })?;
        let data_end = if let Some(next_index) = order.get(position + 1) {
            data_base
                .checked_add(usize::try_from(tensors[*next_index].offset).map_err(|_| {
                    ModelError(format!(
                        "GGUF tensor {} offset overflows",
                        tensors[*next_index].name
                    ))
                })?)
                .ok_or_else(|| {
                    ModelError(format!("GGUF tensor {} end overflows", tensors[index].name))
                })?
        } else {
            bytes.len()
        };
        if data_start > data_end || data_end > bytes.len() {
            return Err(ModelError(format!(
                "GGUF tensor {} exceeds file size",
                tensors[index].name
            )));
        }
        tensors[index].data_start = data_start;
        tensors[index].data_end = data_end;
    }

    Ok(ParsedGguf { metadata, tensors })
}

struct GgufReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> GgufReader<'a> {
    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| ModelError("GGUF offset overflows".into()))?;
        if end > self.bytes.len() {
            return Err(ModelError("GGUF header is truncated".into()));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.read_bytes(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut value = [0_u8; 4];
        value.copy_from_slice(self.read_bytes(4)?);
        Ok(u32::from_le_bytes(value))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.read_bytes(8)?);
        Ok(u64::from_le_bytes(value))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| ModelError("GGUF string length overflows".into()))?;
        let bytes = self.read_bytes(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| ModelError(format!("GGUF string is not UTF-8: {error}")))
    }

    fn read_value(&mut self, value_type: u32) -> Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let element_type = self.read_u32()?;
                let count = usize::try_from(self.read_u64()?)
                    .map_err(|_| ModelError("GGUF array length overflows".into()))?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_value(element_type)?);
                }
                Ok(GgufValue::Array(values))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            _ => Err(ModelError(format!(
                "unsupported GGUF metadata type {value_type}"
            ))),
        }
    }
}

#[derive(Debug)]
struct MappedShard {
    file: Arc<File>,
    bytes: Mmap,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct PrefetchRequest {
    file: Arc<File>,
    offset: libc::off_t,
    count: libc::c_int,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct Prefetcher {
    sender: Option<SyncSender<PrefetchRequest>>,
    worker: Option<JoinHandle<()>>,
}

#[cfg(target_os = "macos")]
impl Prefetcher {
    fn new() -> Self {
        let (sender, receiver) = sync_channel(2);
        let worker = thread::spawn(move || prefetch_worker(receiver));
        Self {
            sender: Some(sender),
            worker: Some(worker),
        }
    }

    fn submit(&self, file: &Arc<File>, offset: libc::off_t, count: libc::c_int) -> Result<()> {
        let Some(sender) = &self.sender else {
            return Err(ModelError("prefetch worker is stopped".into()));
        };
        match sender.try_send(PrefetchRequest {
            file: Arc::clone(file),
            offset,
            count,
        }) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                Err(ModelError("prefetch worker disconnected".into()))
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(target_os = "macos")]
fn prefetch_worker(receiver: Receiver<PrefetchRequest>) {
    while let Ok(request) = receiver.recv() {
        // SAFETY: request holds an open shard file and a valid advisory range;
        // F_RDADVISE only schedules cache read-ahead and does not mutate data.
        unsafe {
            let _ = libc::fcntl(
                request.file.as_raw_fd(),
                libc::F_RDADVISE,
                &libc::radvisory {
                    ra_offset: request.offset,
                    ra_count: request.count,
                } as *const libc::radvisory,
            );
        }
    }
}

#[derive(Debug)]
pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
    /// Complete read-only mapping that contains `bytes`. Metal can bind this
    /// page-aligned backing range once and address the tensor with an offset.
    pub backing: &'a [u8],
}

/// File range used by the optional host staging worker. The descriptor owns
/// the file handle so a worker can read without borrowing the model store.
#[derive(Debug, Clone)]
pub struct TensorStageDescriptor {
    pub name: String,
    pub file: Arc<File>,
    pub offset: u64,
    pub length: usize,
}

const PACKED_MAGIC: &[u8; 8] = b"SIPACK01";
const PACKED_ALIGNMENT: usize = 4096;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PackedTensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    offset: u64,
    length: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PackedHeader {
    format: String,
    model_revision: String,
    source_weight_bytes: u64,
    tensors: BTreeMap<String, PackedTensorMetadata>,
}

#[derive(Debug)]
struct PackedMapping {
    file: Arc<File>,
    bytes: Mmap,
}

#[derive(Debug)]
pub struct ModelStore {
    pub model_dir: PathBuf,
    pub config: ModelConfig,
    pub index: ModelIndex,
    pub manifest: ModelManifest,
    shards: BTreeMap<String, MappedShard>,
    tensors: BTreeMap<String, TensorInfo>,
    packed: Option<PackedMapping>,
    #[cfg(target_os = "macos")]
    prefetcher: Prefetcher,
}

impl ModelStore {
    pub fn open(model_dir: impl AsRef<Path>, verify_manifest: bool) -> Result<Self> {
        let packed_cache = std::env::var_os("SI_PACKED_CACHE").map(PathBuf::from);
        Self::open_with_packed_cache(model_dir, verify_manifest, packed_cache.as_deref())
    }

    pub fn open_with_packed_cache(
        model_dir: impl AsRef<Path>,
        verify_manifest: bool,
        packed_cache: Option<&Path>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let manifest = ModelManifest::load(model_dir.join("manifest.json"))?;
        if verify_manifest {
            manifest.validate_files(&model_dir)?;
        }
        let revision_dir = model_dir.join(&manifest.revision);
        let config: ModelConfig = read_json(&revision_dir.join("config.json"))?;
        config.validate_qwen3()?;
        let index = ModelIndex::load(revision_dir.join("model.safetensors.index.json"))?;
        if index.metadata.total_size != manifest.total_weight_bytes {
            return Err(ModelError(format!(
                "weight size mismatch: index {}, manifest {}",
                index.metadata.total_size, manifest.total_weight_bytes
            )));
        }
        if let Some(packed_cache) = packed_cache {
            return Self::open_packed(model_dir, config, index, manifest, packed_cache);
        }

        let mut shards = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for shard_name in index.shards() {
            let path = revision_dir.join(shard_name);
            let file =
                Arc::new(File::open(&path).map_err(|error| {
                    ModelError(format!("open shard {}: {error}", path.display()))
                })?);
            // SAFETY: file remains open for the lifetime of the mapping and the
            // mapping is read-only; callers only receive shared byte slices.
            let bytes = unsafe { MmapOptions::new().map(file.as_ref()) }?;
            let shard_tensors = parse_shard_header(shard_name, &bytes)?;
            for tensor in shard_tensors {
                if tensors.insert(tensor.name.clone(), tensor).is_some() {
                    return Err(ModelError("duplicate tensor name across shards".into()));
                }
            }
            shards.insert(shard_name.to_owned(), MappedShard { file, bytes });
        }

        for (name, shard) in &index.weight_map {
            let Some(tensor) = tensors.get(name) else {
                return Err(ModelError(format!(
                    "index tensor {name} missing from shard {shard}"
                )));
            };
            if tensor.shard != *shard {
                return Err(ModelError(format!(
                    "tensor {name} is in {}, index points to {shard}",
                    tensor.shard
                )));
            }
        }

        Ok(Self {
            model_dir,
            config,
            index,
            manifest,
            shards,
            tensors,
            packed: None,
            #[cfg(target_os = "macos")]
            prefetcher: Prefetcher::new(),
        })
    }

    fn open_packed(
        model_dir: PathBuf,
        config: ModelConfig,
        index: ModelIndex,
        manifest: ModelManifest,
        packed_path: &Path,
    ) -> Result<Self> {
        let file = Arc::new(File::open(packed_path).map_err(|error| {
            ModelError(format!(
                "open packed cache {}: {error}",
                packed_path.display()
            ))
        })?);
        // SAFETY: file remains open through the Arc held by PackedMapping and
        // the mapping is read-only.
        let bytes = unsafe { MmapOptions::new().map(file.as_ref()) }?;
        let (header, data_base) = parse_packed_header(&bytes)?;
        if header.format != "si-packed-bf16-v1" {
            return Err(ModelError(format!(
                "unsupported packed cache format {}",
                header.format
            )));
        }
        if header.model_revision != manifest.revision {
            return Err(ModelError(format!(
                "packed cache revision {} does not match {}",
                header.model_revision, manifest.revision
            )));
        }
        if header.source_weight_bytes != manifest.total_weight_bytes {
            return Err(ModelError(format!(
                "packed cache source size {} does not match {}",
                header.source_weight_bytes, manifest.total_weight_bytes
            )));
        }
        if header.tensors.len() != index.weight_map.len()
            || header
                .tensors
                .keys()
                .any(|name| !index.weight_map.contains_key(name))
        {
            return Err(ModelError(
                "packed cache tensor index does not match model".into(),
            ));
        }

        let mut tensors = BTreeMap::new();
        for (name, metadata) in header.tensors {
            let start =
                data_base
                    .checked_add(usize::try_from(metadata.offset).map_err(|_| {
                        ModelError(format!("packed tensor {name} offset overflows"))
                    })?)
                    .ok_or_else(|| ModelError(format!("packed tensor {name} offset overflows")))?;
            let length = usize::try_from(metadata.length)
                .map_err(|_| ModelError(format!("packed tensor {name} length overflows")))?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| ModelError(format!("packed tensor {name} end overflows")))?;
            if end > bytes.len() {
                return Err(ModelError(format!(
                    "packed tensor {name} exceeds cache size"
                )));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shard: "packed".into(),
                    dtype: metadata.dtype,
                    shape: metadata.shape,
                    data_start: start,
                    data_end: end,
                },
            );
        }

        Ok(Self {
            model_dir,
            config,
            index,
            manifest,
            shards: BTreeMap::new(),
            tensors,
            packed: Some(PackedMapping { file, bytes }),
            #[cfg(target_os = "macos")]
            prefetcher: Prefetcher::new(),
        })
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError(format!("unknown tensor {name}")))?;
        if let Some(packed) = &self.packed {
            return Ok(TensorView {
                info,
                bytes: &packed.bytes[info.data_start..info.data_end],
                backing: &packed.bytes,
            });
        }
        let shard = self
            .shards
            .get(&info.shard)
            .ok_or_else(|| ModelError(format!("missing mapped shard {}", info.shard)))?;
        Ok(TensorView {
            info,
            bytes: &shard.bytes[info.data_start..info.data_end],
            backing: &shard.bytes,
        })
    }

    pub fn tensor_stage_descriptor(&self, name: &str) -> Result<TensorStageDescriptor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError(format!("unknown tensor {name}")))?;
        let file = if let Some(packed) = &self.packed {
            Arc::clone(&packed.file)
        } else {
            Arc::clone(
                &self
                    .shards
                    .get(&info.shard)
                    .ok_or_else(|| ModelError(format!("missing mapped shard {}", info.shard)))?
                    .file,
            )
        };
        Ok(TensorStageDescriptor {
            name: info.name.clone(),
            file,
            offset: u64::try_from(info.data_start)
                .map_err(|_| ModelError(format!("tensor {name} offset overflows")))?,
            length: info.byte_len(),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn mapped_bytes(&self) -> usize {
        self.packed.as_ref().map_or_else(
            || self.shards.values().map(|shard| shard.bytes.len()).sum(),
            |packed| packed.bytes.len(),
        )
    }

    /// Ask macOS to read a bounded prefix of one tensor into its file cache.
    /// `F_RDADVISE` schedules read-ahead without faulting pages into this
    /// process's RSS or allocating a duplicate weight buffer.
    #[cfg(target_os = "macos")]
    pub fn advise_tensor_prefix(&self, name: &str, max_bytes: usize) -> Result<usize> {
        if max_bytes == 0 {
            return Ok(0);
        }
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError(format!("unknown tensor {name}")))?;
        let length = tensor.byte_len().min(max_bytes);
        if length == 0 {
            return Ok(0);
        }
        #[cfg(target_os = "macos")]
        {
            let count = libc::c_int::try_from(length).map_err(|_| {
                ModelError(format!("read-ahead length for {name} exceeds macOS limits"))
            })?;
            let offset = libc::off_t::try_from(tensor.data_start).map_err(|_| {
                ModelError(format!("read-ahead offset for {name} exceeds macOS limits"))
            })?;
            let file = if let Some(packed) = &self.packed {
                &packed.file
            } else {
                &self
                    .shards
                    .get(&tensor.shard)
                    .ok_or_else(|| ModelError(format!("missing mapped shard {}", tensor.shard)))?
                    .file
            };
            self.prefetcher.submit(file, offset, count)?;
        }
        Ok(length)
    }

    /// Hint the next bounded execution window in model-access order. The byte
    /// budget prevents read-ahead from turning into an unbounded resident copy.
    #[cfg(target_os = "macos")]
    pub fn advise_layer_prefix(&self, layer: usize, byte_budget: usize) -> Result<usize> {
        let mut advised = 0_usize;
        for name in qwen3_layer_tensor_names(layer) {
            if advised >= byte_budget {
                break;
            }
            advised = advised.saturating_add(
                self.advise_tensor_prefix(&name, byte_budget.saturating_sub(advised))?,
            );
        }
        Ok(advised)
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        return Err(ModelError("packed cache alignment must be non-zero".into()));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| ModelError("packed cache alignment overflows".into()))
    }
}

fn parse_packed_header(bytes: &[u8]) -> Result<(PackedHeader, usize)> {
    if bytes.len() < PACKED_MAGIC.len() + std::mem::size_of::<u64>()
        || &bytes[..PACKED_MAGIC.len()] != PACKED_MAGIC
    {
        return Err(ModelError("packed cache has an invalid header".into()));
    }
    let mut length_bytes = [0_u8; 8];
    length_bytes.copy_from_slice(&bytes[PACKED_MAGIC.len()..PACKED_MAGIC.len() + 8]);
    let header_length = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| ModelError("packed cache header length overflows".into()))?;
    let header_start = PACKED_MAGIC.len() + 8;
    let header_end = header_start
        .checked_add(header_length)
        .ok_or_else(|| ModelError("packed cache header length overflows".into()))?;
    if header_end > bytes.len() {
        return Err(ModelError("packed cache header is truncated".into()));
    }
    let header = serde_json::from_slice(&bytes[header_start..header_end])?;
    let data_base = align_up(header_end, PACKED_ALIGNMENT)?;
    if data_base > bytes.len() {
        return Err(ModelError(
            "packed cache data base exceeds file size".into(),
        ));
    }
    Ok((header, data_base))
}

pub fn pack_model_execution_order(
    model_dir: impl AsRef<Path>,
    verify_manifest: bool,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let store = ModelStore::open_with_packed_cache(model_dir, verify_manifest, None)?;
    let names = qwen3_execution_tensor_names(&store.config);
    if names.len() != store.tensor_count() {
        return Err(ModelError(format!(
            "execution order names {} do not cover {} tensors",
            names.len(),
            store.tensor_count()
        )));
    }

    let mut tensors = BTreeMap::new();
    let mut offset = 0_u64;
    for name in &names {
        let tensor = store.tensor(name)?;
        let length = u64::try_from(tensor.bytes.len())
            .map_err(|_| ModelError(format!("tensor {name} length overflows")))?;
        tensors.insert(
            name.clone(),
            PackedTensorMetadata {
                dtype: tensor.info.dtype.clone(),
                shape: tensor.info.shape.clone(),
                offset,
                length,
            },
        );
        offset = offset
            .checked_add(length)
            .ok_or_else(|| ModelError("packed tensor data length overflows".into()))?;
    }
    let header = PackedHeader {
        format: "si-packed-bf16-v1".into(),
        model_revision: store.manifest.revision.clone(),
        source_weight_bytes: store.manifest.total_weight_bytes,
        tensors,
    };
    let header_bytes = serde_json::to_vec(&header)?;
    let data_base = align_up(
        PACKED_MAGIC.len() + 8 + header_bytes.len(),
        PACKED_ALIGNMENT,
    )?;
    let data_length = usize::try_from(offset)
        .map_err(|_| ModelError("packed data length overflows platform size".into()))?;
    let file_length = data_base
        .checked_add(data_length)
        .ok_or_else(|| ModelError("packed file length overflows".into()))?;
    let output_path = output_path.as_ref();
    let mut output = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(output_path)
        .map_err(|error| {
            ModelError(format!(
                "open packed output {}: {error}",
                output_path.display()
            ))
        })?;
    output.set_len(u64::try_from(file_length).map_err(|_| {
        ModelError(format!(
            "packed file length for {} overflows",
            output_path.display()
        ))
    })?)?;
    output.write_all(PACKED_MAGIC)?;
    output.write_all(
        &u64::try_from(header_bytes.len())
            .map_err(|_| ModelError("packed header length overflows".into()))?
            .to_le_bytes(),
    )?;
    output.write_all(&header_bytes)?;
    let padding = data_base - (PACKED_MAGIC.len() + 8 + header_bytes.len());
    if padding > 0 {
        output.write_all(&vec![0_u8; padding])?;
    }
    for name in names {
        output.write_all(store.tensor(&name)?.bytes)?;
    }
    output.flush()?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: Vec<u64>,
}

fn parse_shard_header(shard: &str, bytes: &[u8]) -> Result<Vec<TensorInfo>> {
    if bytes.len() < 8 {
        return Err(ModelError(format!(
            "shard {shard} is smaller than header prefix"
        )));
    }
    let mut length_bytes = [0_u8; 8];
    length_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(length_bytes) as usize;
    let header_end = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| ModelError("Safetensors header length overflow".into()))?;
    if header_end > bytes.len() {
        return Err(ModelError(format!("shard {shard} has truncated header")));
    }
    let values: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..header_end])?;
    let mut tensors = Vec::new();
    for (name, value) in values {
        if name == "__metadata__" {
            continue;
        }
        let raw: RawTensorInfo = serde_json::from_value(value)
            .map_err(|error| ModelError(format!("tensor {name} metadata: {error}")))?;
        if raw.data_offsets.len() != 2 || raw.data_offsets[0] > raw.data_offsets[1] {
            return Err(ModelError(format!(
                "tensor {name} has invalid data offsets"
            )));
        }
        let start = header_end
            .checked_add(raw.data_offsets[0] as usize)
            .ok_or_else(|| ModelError(format!("tensor {name} offset overflow")))?;
        let end = header_end
            .checked_add(raw.data_offsets[1] as usize)
            .ok_or_else(|| ModelError(format!("tensor {name} offset overflow")))?;
        if end > bytes.len() {
            return Err(ModelError(format!("tensor {name} exceeds shard size")));
        }
        tensors.push(TensorInfo {
            name,
            shard: shard.to_owned(),
            dtype: raw.dtype,
            shape: raw.shape,
            data_start: start,
            data_end: end,
        });
    }
    if tensors.is_empty() {
        return Err(ModelError(format!("shard {shard} has no tensors")));
    }
    Ok(tensors)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path)
        .map_err(|error| ModelError(format!("read {}: {error}", path.display())))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn detects_safetensors_directories_and_gguf_files() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp fixture");
        fs::write(
            root.join("manifest.json"),
            br#"{"model":"test","revision":"r","format":"safetensors","dtype":"bfloat16","total_weight_bytes":0,"files":[]}"#,
        )
        .expect("write manifest fixture");
        let gguf = root.join("tiny.gguf");
        fs::write(&gguf, b"GGUF").expect("write GGUF fixture");

        assert_eq!(
            detect_model_format(&root).expect("directory format"),
            ModelFormat::Safetensors
        );
        assert_eq!(
            detect_model_format(&gguf).expect("GGUF format"),
            ModelFormat::Gguf
        );
        fs::remove_dir_all(root).expect("remove temp fixture");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stage_nocache_is_enabled_by_default_and_can_be_disabled() {
        assert!(!stage_nocache_enabled(false, None));
        assert!(!stage_nocache_enabled(true, None));
        assert!(stage_nocache_enabled(true, Some("1")));
        assert!(!stage_nocache_enabled(true, Some("0")));
    }

    #[test]
    fn parses_gguf_metadata_and_tensor_index() {
        let bytes = tiny_gguf_fixture();
        let parsed = parse_gguf(&bytes).expect("valid GGUF fixture");

        assert_eq!(
            parsed.metadata["general.architecture"],
            GgufValue::String("qwen35".into())
        );
        assert_eq!(parsed.metadata["general.alignment"], GgufValue::U32(32));
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(parsed.tensors[0].shape, vec![4, 2]);
        assert_eq!(parsed.tensors[0].ggml_type, 0);
        assert_eq!(parsed.tensors[0].data_start % 32, 0);
    }

    #[test]
    fn derives_qwen35_execution_config_from_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufValue::String("qwen35".into()),
        );
        metadata.insert("qwen35.embedding_length".into(), GgufValue::U32(5120));
        metadata.insert("qwen35.feed_forward_length".into(), GgufValue::U32(17408));
        metadata.insert("qwen35.block_count".into(), GgufValue::U32(64));
        metadata.insert("qwen35.attention.head_count".into(), GgufValue::U32(24));
        metadata.insert("qwen35.attention.head_count_kv".into(), GgufValue::U32(4));
        metadata.insert("qwen35.attention.key_length".into(), GgufValue::U32(256));
        metadata.insert("qwen35.attention.value_length".into(), GgufValue::U32(256));
        metadata.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1.0e-6),
        );
        metadata.insert("qwen35.rope.freq_base".into(), GgufValue::F32(10_000_000.0));
        metadata.insert("qwen35.context_length".into(), GgufValue::U32(262144));
        metadata.insert("qwen35.full_attention_interval".into(), GgufValue::U32(4));
        metadata.insert("qwen35.ssm.inner_size".into(), GgufValue::U32(6144));
        metadata.insert("qwen35.ssm.state_size".into(), GgufValue::U32(128));
        metadata.insert("qwen35.ssm.group_count".into(), GgufValue::U32(16));
        metadata.insert("qwen35.ssm.conv_kernel".into(), GgufValue::U32(4));
        metadata.insert("qwen35.ssm.time_step_rank".into(), GgufValue::U32(48));
        metadata.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(248046));
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("a".into()); 248320]),
        );

        let config = GgufQwen35Config::from_metadata(&metadata).expect("valid Qwen3.6 metadata");

        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.intermediate_size, 17408);
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.full_attention_interval, 4);
        assert_eq!(config.ssm_inner_size, 6144);
    }

    #[test]
    fn rejects_invalid_gguf_magic_and_truncation() {
        assert!(parse_gguf(b"NOPE").is_err());
        let mut bytes = tiny_gguf_fixture();
        bytes.truncate(20);
        assert!(parse_gguf(&bytes).is_err());
    }

    fn tiny_gguf_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_gguf_string(&mut bytes, "qwen35");
        push_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        push_gguf_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(align_up(bytes.len(), 32).expect("fixture alignment"), 0);
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        bytes
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_safetensors_header_and_borrows_tensor_bytes() {
        let header = br#"{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"pt"}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let tensors = parse_shard_header("tiny.safetensors", &bytes).expect("valid header");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].dtype, "BF16");
        assert_eq!(tensors[0].shape, vec![2]);
        assert_eq!(tensors[0].byte_len(), 4);
        assert_eq!(
            &bytes[tensors[0].data_start..tensors[0].data_end],
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn rejects_tensor_data_outside_shard() {
        let header = br#"{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,5]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let error = parse_shard_header("tiny.safetensors", &bytes).unwrap_err();
        assert!(error.0.contains("exceeds shard size"));
    }

    #[test]
    fn validates_qwen3_config_dimensions_and_dtype() {
        let config = ModelConfig {
            model_type: "qwen3".into(),
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            torch_dtype: "bfloat16".into(),
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            eos_token_id: 151643,
            tie_word_embeddings: true,
        };
        config.validate_qwen3().expect("Qwen3 config is valid");
    }

    #[test]
    fn qwen3_layer_tensor_names_follow_execution_order() {
        let names = qwen3_layer_tensor_names(3);

        assert_eq!(
            names,
            vec![
                "model.layers.3.input_layernorm.weight",
                "model.layers.3.self_attn.q_proj.weight",
                "model.layers.3.self_attn.k_proj.weight",
                "model.layers.3.self_attn.v_proj.weight",
                "model.layers.3.self_attn.q_norm.weight",
                "model.layers.3.self_attn.k_norm.weight",
                "model.layers.3.self_attn.o_proj.weight",
                "model.layers.3.post_attention_layernorm.weight",
                "model.layers.3.mlp.gate_proj.weight",
                "model.layers.3.mlp.up_proj.weight",
                "model.layers.3.mlp.down_proj.weight",
            ]
        );
    }

    #[test]
    fn qwen3_execution_tensor_names_cover_embeddings_layers_and_norm() {
        let config = ModelConfig {
            model_type: "qwen3".into(),
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            vocab_size: 32,
            torch_dtype: "bfloat16".into(),
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            eos_token_id: 1,
            tie_word_embeddings: true,
        };

        let names = qwen3_execution_tensor_names(&config);

        assert_eq!(names.len(), 24);
        assert_eq!(names.first().unwrap(), "model.embed_tokens.weight");
        assert_eq!(names.last().unwrap(), "model.norm.weight");
        assert_eq!(names[1], "model.layers.0.input_layernorm.weight");
        assert_eq!(names[12], "model.layers.1.input_layernorm.weight");
    }

    #[test]
    fn packed_header_round_trips_with_aligned_data_base() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.norm.weight".into(),
            PackedTensorMetadata {
                dtype: "BF16".into(),
                shape: vec![8],
                offset: 0,
                length: 16,
            },
        );
        let header = PackedHeader {
            format: "si-packed-bf16-v1".into(),
            model_revision: "revision".into(),
            source_weight_bytes: 16,
            tensors,
        };
        let header_bytes = serde_json::to_vec(&header).expect("header should serialize");
        let data_base = align_up(
            PACKED_MAGIC.len() + 8 + header_bytes.len(),
            PACKED_ALIGNMENT,
        )
        .expect("alignment should fit");
        let mut bytes = Vec::with_capacity(data_base + 16);
        bytes.extend_from_slice(PACKED_MAGIC);
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.resize(data_base + 16, 0);

        let (parsed, parsed_base) = parse_packed_header(&bytes).expect("header should parse");

        assert_eq!(parsed.format, "si-packed-bf16-v1");
        assert_eq!(parsed_base, data_base);
        assert_eq!(parsed.tensors["model.norm.weight"].length, 16);
    }

    #[test]
    fn validates_manifest_size_and_digest() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp fixture");
        let file = root.join("tiny.bin");
        fs::write(&file, b"fixture").expect("write temp fixture");
        let manifest = ModelManifest {
            model: "test".into(),
            revision: "revision".into(),
            format: "safetensors".into(),
            dtype: "bfloat16".into(),
            total_weight_bytes: 0,
            files: vec![ManifestFile {
                path: "tiny.bin".into(),
                size_bytes: 7,
                sha256: "8df5c3f0a8cc6f0e4a9fbb4d8a7b6c5d4e3f2a19080706050403020100ffeedd".into(),
            }],
        };
        let mut manifest = manifest;
        manifest
            .validate_files(&root)
            .expect_err("wrong digest must fail");
        manifest.files[0].sha256 = sha256_file(&file).expect("hash fixture");
        manifest
            .validate_files(&root)
            .expect("manifest matches fixture");
        fs::remove_dir_all(root).expect("remove temp fixture");
    }

    #[test]
    fn opens_real_model_when_si_model_dir_is_set() {
        let Ok(model_dir) = std::env::var("SI_MODEL_DIR") else {
            return;
        };
        let store = ModelStore::open(model_dir, false).expect("pinned model should parse");
        assert_eq!(store.config.model_type, "qwen3");
        assert_eq!(store.config.num_hidden_layers, 36);
        assert_eq!(store.tensor_count(), store.index.weight_map.len());
        assert_eq!(store.index.metadata.total_size, 8_045_591_552);
    }

    #[test]
    fn opens_real_gguf_when_si_gguf_model_is_set() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let store = GgufModelStore::open(model_path).expect("GGUF model should parse");
        assert_eq!(
            store.metadata_string("general.architecture"),
            Some("qwen35")
        );
        assert!(!store.tensors.is_empty());
        let config = store.qwen35_config().expect("Qwen3.6 config should parse");
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.ssm_key_heads(), 16);
        assert_eq!(config.ssm_value_heads(), 48);
        assert_eq!(config.ssm_key_dim(), 128);
        assert_eq!(config.ssm_projection_size(), 10_240);
        let layer_kinds = store
            .qwen35_layer_kinds()
            .expect("Qwen3.6 layer kinds should validate");
        assert_eq!(layer_kinds.len(), 64);
        assert_eq!(layer_kinds[0], GgufQwen35LayerKind::GatedDeltaNet);
        assert_eq!(layer_kinds[3], GgufQwen35LayerKind::FullAttention);
        assert_eq!(layer_kinds[63], GgufQwen35LayerKind::FullAttention);
        assert!(store.tensor("output.weight").is_ok() || store.tensor("token_embd.weight").is_ok());
        let q4 = store
            .tensors
            .values()
            .find(|tensor| tensor.is_q4_k())
            .expect("GGUF model should contain Q4_K tensors");
        let view = store.tensor(&q4.name).expect("Q4_K tensor should open");
        let values = crate::quant::dequantize_q4_k(
            &view.bytes[..crate::quant::Q4_K_BLOCK_BYTES],
            crate::quant::Q4_K_BLOCK_ELEMENTS,
        )
        .expect("first Q4_K block should decode");
        assert!(values.iter().all(|value| value.is_finite()));
        for (ggml_type, block_bytes, decode) in [
            (
                crate::quant::GGML_TYPE_Q5_K,
                crate::quant::Q5_K_BLOCK_BYTES,
                crate::quant::dequantize_q5_k as fn(&[u8], usize) -> crate::quant::Result<Vec<f32>>,
            ),
            (
                crate::quant::GGML_TYPE_Q6_K,
                crate::quant::Q6_K_BLOCK_BYTES,
                crate::quant::dequantize_q6_k as fn(&[u8], usize) -> crate::quant::Result<Vec<f32>>,
            ),
        ] {
            let tensor = store
                .tensors
                .values()
                .find(|tensor| tensor.ggml_type == ggml_type)
                .expect("GGUF model should contain mixed K quant tensors");
            let view = store
                .tensor(&tensor.name)
                .expect("mixed K tensor should open");
            let decoded =
                decode(&view.bytes[..block_bytes], 256).expect("first mixed K block should decode");
            assert!(decoded.iter().all(|value| value.is_finite()));
        }
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("super-inference-model-test-{nanos}"))
    }
}
