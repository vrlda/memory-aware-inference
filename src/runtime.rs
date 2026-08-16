//! First end-to-end Qwen3 execution path.
//!
//! Profiles share one model-level execution order while varying weight leases:
//! streaming binds immutable Safetensors mappings per operation, and resident
//! retains private GPU buffers. The memory planner can replace the lease path
//! without changing the model-level execution order.

#[cfg(target_os = "macos")]
use crate::cache::KvCache;
#[cfg(target_os = "macos")]
use crate::metal::{
    Bf16Weight, ChainedAttentionBufferRequest, ChainedAttentionConfig,
    ChainedAttentionTensorRequest, MetalContext, MetalDeviceInfo, PendingMatvec,
};
#[cfg(target_os = "macos")]
use crate::model::{
    qwen3_layer_tensor_names, GgufModelStore, ModelConfig, ModelStore, TensorStageDescriptor,
    TensorView, QWEN3_LAYER_TENSOR_SUFFIXES,
};
#[cfg(target_os = "macos")]
use crate::planner::{plan_sequential_layers, LayerPlan, LayerTensor, PlannerTrace};
#[cfg(target_os = "macos")]
use crate::quality::{QualitySuite, QualitySummary};
#[cfg(target_os = "macos")]
use crate::qwen35_runtime::Qwen35Runtime;
#[cfg(target_os = "macos")]
use crate::telemetry::{sample_resources, ProcessMemorySampler};
#[cfg(target_os = "macos")]
use crate::tokenizer::QwenTokenizer;
#[cfg(target_os = "macos")]
use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "macos")]
use std::sync::mpsc::{channel, sync_channel, Receiver, SyncSender};
#[cfg(target_os = "macos")]
use std::thread::{self, JoinHandle};

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightResidency {
    Streaming,
    Resident,
    SharedResident,
}

/// Opt-in read-ahead window for the next streamed layer. The hint affects
/// file-backed pages only; it does not create a second weight representation.
/// It stays disabled by default until a platform-specific sweep proves that
/// its page-cache behavior fits the memory budget.
const DEFAULT_STREAM_PREFETCH_MIB: usize = 0;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedProjectionGroup {
    Qkv,
    GateUp,
}

#[cfg(target_os = "macos")]
fn fused_projection_group(names: &[String]) -> Option<FusedProjectionGroup> {
    match names {
        [q, k, v]
            if q.ends_with("self_attn.q_proj.weight")
                && k.ends_with("self_attn.k_proj.weight")
                && v.ends_with("self_attn.v_proj.weight") =>
        {
            Some(FusedProjectionGroup::Qkv)
        }
        [gate, up]
            if gate.ends_with("mlp.gate_proj.weight") && up.ends_with("mlp.up_proj.weight") =>
        {
            Some(FusedProjectionGroup::GateUp)
        }
        _ => None,
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct StageRequest {
    names: Vec<String>,
    descriptors: Vec<TensorStageDescriptor>,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct StagedProjectionGroup {
    names: Vec<String>,
    tensors: Vec<Vec<u8>>,
}

#[cfg(target_os = "macos")]
struct TreeForward {
    logits: Vec<Vec<Vec<f32>>>,
    caches: Vec<Vec<KvCache>>,
}

#[cfg(target_os = "macos")]
struct WeightStager {
    sender: Option<SyncSender<StageRequest>>,
    receiver: Receiver<Result<StagedProjectionGroup, String>>,
    worker: Option<JoinHandle<()>>,
    queued: bool,
}

#[cfg(target_os = "macos")]
impl WeightStager {
    fn new() -> Self {
        let (sender, requests) = sync_channel(1);
        let (results, receiver) = channel();
        let worker = thread::spawn(move || {
            while let Ok(request) = requests.recv() {
                let result = read_stage_request(request);
                if results.send(result).is_err() {
                    break;
                }
            }
        });
        Self {
            sender: Some(sender),
            receiver,
            worker: Some(worker),
            queued: false,
        }
    }

    fn prefetch(
        &mut self,
        store: &ModelStore,
        names: &[String],
        byte_budget: usize,
    ) -> Result<bool, String> {
        if self.queued || byte_budget == 0 || fused_projection_group(names).is_none() {
            return Ok(false);
        }
        let descriptors = names
            .iter()
            .map(|name| store.tensor_stage_descriptor(name).map_err(|error| error.0))
            .collect::<Result<Vec<_>, _>>()?;
        let total_bytes = descriptors.iter().try_fold(0_usize, |total, descriptor| {
            total
                .checked_add(descriptor.length)
                .ok_or_else(|| "staged projection byte length overflows".to_owned())
        })?;
        if total_bytes == 0 || total_bytes > byte_budget {
            return Ok(false);
        }
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| "weight staging worker is stopped".to_owned())?;
        sender
            .send(StageRequest {
                names: names.to_vec(),
                descriptors,
            })
            .map_err(|_| "weight staging worker disconnected".to_owned())?;
        self.queued = true;
        Ok(true)
    }

    fn take(&mut self, names: &[String]) -> Result<Option<StagedProjectionGroup>, String> {
        if !self.queued {
            return Ok(None);
        }
        let result = self
            .receiver
            .recv()
            .map_err(|_| "weight staging worker disconnected".to_owned())?;
        self.queued = false;
        let staged = result?;
        if staged.names != names {
            return Err("weight staging result order does not match execution order".into());
        }
        Ok(Some(staged))
    }
}

#[cfg(target_os = "macos")]
impl Drop for WeightStager {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(target_os = "macos")]
fn read_stage_request(request: StageRequest) -> Result<StagedProjectionGroup, String> {
    let mut tensors = Vec::with_capacity(request.descriptors.len());
    for descriptor in request.descriptors {
        let mut bytes = vec![0_u8; descriptor.length];
        descriptor
            .file
            .read_exact_at(&mut bytes, descriptor.offset)
            .map_err(|error| format!("stage {}: {error}", descriptor.name))?;
        tensors.push(bytes);
    }
    Ok(StagedProjectionGroup {
        names: request.names,
        tensors,
    })
}

#[cfg(target_os = "macos")]
struct ResidentMatrix {
    rows: usize,
    columns: usize,
    weight: Bf16Weight,
}

#[cfg(target_os = "macos")]
struct ResidentWeights {
    matrices: HashMap<String, ResidentMatrix>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct OutputHeadIndex {
    block_rows: usize,
    rows: usize,
    columns: usize,
    centroids: Vec<f32>,
    radii: Vec<f32>,
}

#[cfg(target_os = "macos")]
struct ResidentSidecar {
    hidden_size: usize,
    rank: usize,
    vocab_size: usize,
    input_mean: Vec<f32>,
    input_to_latent: Vec<f32>,
    vocab_projection: Vec<f32>,
    vocab_bias: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl ResidentSidecar {
    const MAGIC: &'static [u8; 8] = b"SISCAR01";

    fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read resident sidecar {}: {error}", path.display()))?;
        if bytes.len() < Self::MAGIC.len() + 16 || &bytes[..Self::MAGIC.len()] != Self::MAGIC {
            return Err("resident sidecar has an invalid header".into());
        }
        let mut cursor = Self::MAGIC.len();
        let hidden_size = read_u32(&bytes, &mut cursor)? as usize;
        let rank = read_u32(&bytes, &mut cursor)? as usize;
        let vocab_size = read_u32(&bytes, &mut cursor)? as usize;
        let version = read_u32(&bytes, &mut cursor)?;
        if version != 1 || hidden_size == 0 || rank == 0 || vocab_size == 0 {
            return Err("resident sidecar dimensions or version are invalid".into());
        }
        let input_to_latent_len = hidden_size
            .checked_mul(rank)
            .ok_or("resident sidecar input map dimensions overflow")?;
        let vocab_projection_len = vocab_size
            .checked_mul(rank)
            .ok_or("resident sidecar vocabulary map dimensions overflow")?;
        let input_mean = read_f32_vec(&bytes, &mut cursor, hidden_size)?;
        let input_to_latent = read_f32_vec(&bytes, &mut cursor, input_to_latent_len)?;
        let vocab_projection = read_f32_vec(&bytes, &mut cursor, vocab_projection_len)?;
        let vocab_bias = read_f32_vec(&bytes, &mut cursor, vocab_size)?;
        if cursor != bytes.len() {
            return Err("resident sidecar contains trailing bytes".into());
        }
        Ok(Self {
            hidden_size,
            rank,
            vocab_size,
            input_mean,
            input_to_latent,
            vocab_projection,
            vocab_bias,
        })
    }

    fn propose_scored(&self, hidden: &[f32], limit: usize) -> Result<Vec<(u32, f32)>, String> {
        if hidden.len() != self.hidden_size || limit == 0 || limit > self.vocab_size {
            return Err("resident sidecar proposal dimensions are invalid".into());
        }
        let mut latent = vec![0.0_f32; self.rank];
        for (index, value) in hidden.iter().enumerate() {
            let centered = *value - self.input_mean[index];
            let map = &self.input_to_latent[index * self.rank..][..self.rank];
            for (output, coefficient) in latent.iter_mut().zip(map) {
                *output += centered * coefficient;
            }
        }
        let mut best = Vec::with_capacity(limit);
        for row in 0..self.vocab_size {
            let projection = &self.vocab_projection[row * self.rank..][..self.rank];
            let score = self.vocab_bias[row]
                + latent
                    .iter()
                    .zip(projection)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
            let insertion = best
                .iter()
                .position(|(_, current)| score > *current)
                .unwrap_or(best.len());
            if insertion < limit {
                best.insert(insertion, (row as u32, score));
                if best.len() > limit {
                    best.pop();
                }
            }
        }
        Ok(best)
    }

    fn propose(&self, hidden: &[f32], limit: usize) -> Result<Vec<u32>, String> {
        Ok(self
            .propose_scored(hidden, limit)?
            .into_iter()
            .map(|(token, _)| token)
            .collect())
    }
}

#[cfg(target_os = "macos")]
fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let end = cursor
        .checked_add(4)
        .ok_or("sidecar header offset overflow")?;
    let raw = bytes
        .get(*cursor..end)
        .ok_or("resident sidecar header is truncated")?;
    *cursor = end;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

#[cfg(target_os = "macos")]
fn read_f32_vec(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let byte_count = count
        .checked_mul(4)
        .ok_or("resident sidecar tensor size overflow")?;
    let end = cursor
        .checked_add(byte_count)
        .ok_or("resident sidecar tensor offset overflow")?;
    let raw = bytes
        .get(*cursor..end)
        .ok_or("resident sidecar tensor is truncated")?;
    *cursor = end;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(target_os = "macos")]
impl OutputHeadIndex {
    const BLOCK_ROWS: usize = 256;

    fn build(tensor: &TensorView<'_>) -> Result<Self, String> {
        if tensor.info.dtype != "BF16"
            || tensor.info.shape.len() != 2
            || tensor.info.shape[0] == 0
            || tensor.info.shape[1] == 0
        {
            return Err("exact output-head index requires a non-empty rank-2 BF16 tensor".into());
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("output-head index dimensions overflow")?;
        if tensor.bytes.len() != expected_bytes {
            return Err("output-head index tensor bytes do not match its shape".into());
        }
        let block_count = rows.div_ceil(Self::BLOCK_ROWS);
        let mut centroids = Vec::with_capacity(block_count * columns);
        let mut radii = Vec::with_capacity(block_count);
        let row_bytes = columns * 2;
        for block in 0..block_count {
            let row_start = block * Self::BLOCK_ROWS;
            let row_count = Self::BLOCK_ROWS.min(rows - row_start);
            let mut centroid = vec![0.0_f32; columns];
            for row in 0..row_count {
                let bytes = &tensor.bytes[(row_start + row) * row_bytes..][..row_bytes];
                for (column, value) in bytes.chunks_exact(2).enumerate() {
                    centroid[column] += bf16_to_f32(value);
                }
            }
            let inverse_count = 1.0 / row_count as f32;
            for value in &mut centroid {
                *value *= inverse_count;
            }
            let mut radius = 0.0_f32;
            for row in 0..row_count {
                let bytes = &tensor.bytes[(row_start + row) * row_bytes..][..row_bytes];
                let distance = bytes
                    .chunks_exact(2)
                    .enumerate()
                    .map(|(column, value)| {
                        let difference = bf16_to_f32(value) - centroid[column];
                        difference * difference
                    })
                    .sum::<f32>()
                    .sqrt();
                radius = radius.max(distance);
            }
            centroids.extend(centroid);
            radii.push(radius);
        }
        Ok(Self {
            block_rows: Self::BLOCK_ROWS,
            rows,
            columns,
            centroids,
            radii,
        })
    }

    fn search(
        &self,
        context: &MetalContext,
        tensor: &TensorView<'_>,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.columns
            || tensor.info.shape != [self.rows, self.columns]
            || self.radii.len() * self.block_rows < self.rows
        {
            return Err("exact output-head search dimensions are invalid".into());
        }
        let input_norm = input
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        let mut order = (0..self.radii.len()).collect::<Vec<_>>();
        let mut bounds = vec![0.0_f64; self.radii.len()];
        for (block, bound) in bounds.iter_mut().enumerate() {
            let centroid = &self.centroids[block * self.columns..][..self.columns];
            let dot = centroid
                .iter()
                .zip(input)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
            *bound = dot + f64::from(self.radii[block]) * input_norm;
        }
        order.sort_unstable_by(|left, right| bounds[*right].total_cmp(&bounds[*left]));

        let mut logits = vec![f32::NEG_INFINITY; self.rows];
        let mut best = f32::NEG_INFINITY;
        let mut evaluated_blocks = 0_usize;
        let mut evaluated_rows = 0_usize;
        for block in order {
            // The small margin covers centroid/radius rounding and the
            // target kernel's FP32 accumulation order. Blocks are skipped
            // only when their conservative bound is clearly below best.
            if best.is_finite() && bounds[block] + 1.0e-2 < f64::from(best) {
                continue;
            }
            let row_start = block * self.block_rows;
            let row_count = self.block_rows.min(self.rows - row_start);
            evaluated_blocks += 1;
            evaluated_rows += row_count;
            let values = context.bf16_matvec_tensor_rows(tensor, input, row_start, row_count)?;
            for (offset, value) in values.into_iter().enumerate() {
                logits[row_start + offset] = value;
                if value > best {
                    best = value;
                }
            }
        }
        if std::env::var_os("SI_PROFILE_HEAD").is_some() {
            eprintln!(
                "si_head blocks_evaluated={} blocks_total={} rows_evaluated={} rows_total={}",
                evaluated_blocks,
                self.radii.len(),
                evaluated_rows,
                self.rows,
            );
        }
        Ok(logits)
    }
}

#[cfg(target_os = "macos")]
pub struct MetalQwen3 {
    store: ModelStore,
    tokenizer: QwenTokenizer,
    context: MetalContext,
    caches: Vec<KvCache>,
    candidate_cache_pool: Option<Vec<KvCache>>,
    draft_caches: Option<Vec<KvCache>>,
    max_context: usize,
    resident_weights: Option<ResidentWeights>,
    hot_weights: Option<ResidentWeights>,
    retained_layers: Option<ResidentWeights>,
    exact_head_index: Option<OutputHeadIndex>,
    resident_sidecar: Option<ResidentSidecar>,
    stream_chunk_rows: Option<usize>,
    stream_prefetch_bytes: usize,
    batch_streaming_projections: bool,
    fused_streaming_projections: bool,
    async_metal: bool,
    chain_attention: bool,
    stage_byte_budget: usize,
    stager: Option<WeightStager>,
    staged: Option<StagedProjectionGroup>,
    layer_limit: Option<usize>,
}

#[cfg(target_os = "macos")]
impl MetalQwen3 {
    pub fn from_model_dir(
        model_dir: impl AsRef<std::path::Path>,
        verify_manifest: bool,
        max_context: usize,
    ) -> Result<Self, String> {
        Self::from_model_dir_with_residency(
            model_dir,
            verify_manifest,
            max_context,
            WeightResidency::Streaming,
        )
    }

    pub fn from_model_dir_with_residency(
        model_dir: impl AsRef<std::path::Path>,
        verify_manifest: bool,
        max_context: usize,
        residency: WeightResidency,
    ) -> Result<Self, String> {
        let store = ModelStore::open(model_dir, verify_manifest).map_err(|error| error.0)?;
        Self::from_store_with_residency(store, max_context, residency)
    }

    pub fn from_store(store: ModelStore, max_context: usize) -> Result<Self, String> {
        Self::from_store_with_residency(store, max_context, WeightResidency::Streaming)
    }

    pub fn from_store_with_residency(
        store: ModelStore,
        max_context: usize,
        residency: WeightResidency,
    ) -> Result<Self, String> {
        if max_context == 0 {
            return Err("maximum context must be non-zero".into());
        }
        let config = store.config.clone();
        validate_required_tensors(&store, &config)?;
        let tokenizer = QwenTokenizer::from_model_dir(&store.model_dir, &store.manifest.revision)?;
        let caches = (0..config.num_hidden_layers)
            .map(|_| {
                KvCache::new(config.num_key_value_heads, config.head_dim, max_context)
                    .map_err(|error| error.0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let context = MetalContext::new()?;
        let resident_weights = match residency {
            WeightResidency::Streaming => None,
            WeightResidency::Resident => {
                Some(load_resident_weights(&context, &store, &config, true)?)
            }
            WeightResidency::SharedResident => {
                Some(load_resident_weights(&context, &store, &config, false)?)
            }
        };
        let stream_prefetch_mib = std::env::var("SI_PREFETCH_MIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_STREAM_PREFETCH_MIB);
        let batch_streaming_projections = std::env::var_os("SI_BATCH_STREAMING").is_some();
        let stage_mib = std::env::var("SI_STAGE_MIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let stage_byte_budget = stage_mib.saturating_mul(1024 * 1024);
        let chain_attention =
            std::env::var_os("SI_CHAIN_ATTENTION").is_some() && stage_byte_budget == 0;
        let fused_streaming_projections = std::env::var_os("SI_FUSED_PROJECTIONS").is_some()
            || stage_byte_budget > 0
            || chain_attention;
        let async_metal = std::env::var_os("SI_ASYNC_METAL").is_some();
        Ok(Self {
            store,
            tokenizer,
            context,
            caches,
            candidate_cache_pool: None,
            draft_caches: None,
            max_context,
            resident_weights,
            hot_weights: None,
            retained_layers: None,
            exact_head_index: None,
            resident_sidecar: None,
            stream_chunk_rows: None,
            stream_prefetch_bytes: stream_prefetch_mib.saturating_mul(1024 * 1024),
            batch_streaming_projections,
            fused_streaming_projections,
            async_metal,
            chain_attention,
            stage_byte_budget,
            stager: (stage_byte_budget > 0).then(WeightStager::new),
            staged: None,
            layer_limit: None,
        })
    }

    pub fn device_info(&self) -> Result<MetalDeviceInfo, String> {
        crate::metal::probe()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.store.config
    }

    pub fn model_revision(&self) -> &str {
        &self.store.manifest.revision
    }

    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn cached_tokens(&self) -> usize {
        self.caches.first().map_or(0, KvCache::cached_tokens)
    }

    pub fn kv_cache_bytes(&self) -> u64 {
        self.caches.iter().map(KvCache::active_bytes).sum()
    }

    pub fn plan_layer_stream(&self, metal_budget_bytes: u64) -> Result<PlannerTrace, String> {
        let mut layers = Vec::with_capacity(self.store.config.num_hidden_layers);
        for layer in 0..self.store.config.num_hidden_layers {
            let mut tensors = Vec::new();
            for name in qwen3_layer_tensor_names(layer) {
                let tensor = self.tensor(&name)?;
                tensors.push(LayerTensor {
                    id: name,
                    bytes: tensor.info.byte_len() as u64,
                });
            }
            layers.push(LayerPlan {
                operation_id: format!("model.layers.{layer}"),
                tensors,
            });
        }
        plan_sequential_layers(&layers, metal_budget_bytes).map_err(|error| error.0)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
        if let Some(caches) = &mut self.draft_caches {
            for cache in caches {
                cache.clear();
            }
        }
    }

    fn take_candidate_cache_pool(&mut self) -> Result<Vec<KvCache>, String> {
        let config = self.store.config.clone();
        let mut pool = self.candidate_cache_pool.take().unwrap_or_else(|| {
            self.caches
                .iter()
                .map(|cache| {
                    KvCache::new(cache.key_value_heads(), cache.head_dim(), self.max_context)
                        .expect("validated model cache dimensions")
                })
                .collect()
        });
        if pool.len() != self.caches.len() {
            pool = (0..config.num_hidden_layers)
                .map(|_| {
                    KvCache::new(
                        config.num_key_value_heads,
                        config.head_dim,
                        self.max_context,
                    )
                    .map_err(|error| error.0)
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        for (destination, source) in pool.iter_mut().zip(&self.caches) {
            destination
                .copy_prefix_from(source)
                .map_err(|error| error.0)?;
        }
        Ok(pool)
    }

    fn recycle_candidate_caches(&mut self, mut caches: Vec<KvCache>, position: usize) {
        for cache in &mut caches {
            let _ = cache.truncate_to(position.min(cache.cached_tokens()));
        }
        self.candidate_cache_pool = Some(caches);
    }

    pub fn set_stream_chunk_rows(&mut self, rows: Option<usize>) -> Result<(), String> {
        if rows == Some(0) {
            return Err("streaming row chunk cap must be non-zero".into());
        }
        self.stream_chunk_rows = rows;
        Ok(())
    }

    pub fn set_retain_output_head(&mut self, retain: bool) -> Result<(), String> {
        if !retain {
            self.hot_weights = None;
            return Ok(());
        }
        if self.resident_weights.is_some() {
            return Err("output-head retention is only valid for streaming residency".into());
        }
        if self.hot_weights.is_none() {
            self.hot_weights = Some(load_output_head(&self.context, &self.store)?);
        }
        Ok(())
    }

    /// Build a lossless block index for exact greedy output-head search. The
    /// source BF16 rows remain untouched; centroids and radii are only safe
    /// upper-bound metadata used to skip blocks whose maximum possible logit
    /// cannot beat the current exact candidate.
    pub fn set_exact_head_search(&mut self, enabled: bool) -> Result<(), String> {
        if !enabled {
            self.exact_head_index = None;
            return Ok(());
        }
        let tensor = self.tensor("model.embed_tokens.weight")?;
        self.exact_head_index = Some(OutputHeadIndex::build(&tensor)?);
        Ok(())
    }

    pub fn set_retain_layers(&mut self, count: usize) -> Result<(), String> {
        if self.resident_weights.is_some() && count > 0 {
            return Err("layer retention is only valid for streaming residency".into());
        }
        if count == 0 {
            self.retained_layers = None;
            return Ok(());
        }
        if count > self.store.config.num_hidden_layers {
            return Err(format!(
                "retained layer count {count} exceeds model depth {}",
                self.store.config.num_hidden_layers
            ));
        }
        if self.retained_layers.is_none() {
            self.retained_layers = Some(load_retained_layers(&self.context, &self.store, count)?);
        }
        Ok(())
    }

    /// Load a proposal-only resident sidecar. It never participates in target
    /// verification; the untouched BF16 target remains the source of truth.
    pub fn set_resident_sidecar(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let sidecar = ResidentSidecar::load(path.as_ref())?;
        if sidecar.hidden_size != self.store.config.hidden_size {
            return Err(format!(
                "resident sidecar hidden size {} does not match model {}",
                sidecar.hidden_size, self.store.config.hidden_size
            ));
        }
        if sidecar.vocab_size != self.store.config.vocab_size {
            return Err(format!(
                "resident sidecar vocabulary size {} does not match model {}",
                sidecar.vocab_size, self.store.config.vocab_size
            ));
        }
        self.resident_sidecar = Some(sidecar);
        Ok(())
    }

    /// Run one causal token through every transformer block and return its
    /// final vocabulary logits. The current implementation is deliberately
    /// sequential so every tensor lifetime is visible to the future planner.
    pub fn forward_token(&mut self, token_id: usize, position: usize) -> Result<Vec<f32>, String> {
        self.forward_token_with_hidden(token_id, position)
            .map(|(_, logits)| logits)
    }

    /// Return both the final pre-norm hidden state and exact target logits in
    /// one traversal. This is a trace-only seam for training disposable
    /// resident-layer proposal machinery; normal generation still calls
    /// `forward_token` and receives only logits.
    pub fn forward_token_with_hidden(
        &mut self,
        token_id: usize,
        position: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let hidden = self.forward_token_hidden(token_id, position)?;
        let config = self.store.config.clone();
        let normalized = {
            let weight = self.tensor("model.norm.weight")?;
            self.context
                .rms_norm_bf16_tensor(&hidden, &weight, config.rms_norm_eps)?
        };
        let logits = self.matvec("model.embed_tokens.weight", &normalized)?;
        Ok((hidden, logits))
    }

    /// Run one causal token through the configured transformer prefix and
    /// return its final hidden state before the model norm and vocabulary head.
    /// This is the observation seam for disposable resident-layer drafters;
    /// it never changes target logits or target cache ownership.
    fn forward_token_hidden(
        &mut self,
        token_id: usize,
        position: usize,
    ) -> Result<Vec<f32>, String> {
        if position >= self.max_context {
            return Err("token position exceeds configured context capacity".into());
        }
        if self
            .caches
            .iter()
            .any(|cache| cache.cached_tokens() != position)
        {
            return Err("KV-cache position is inconsistent across layers".into());
        }
        if self.resident_weights.is_none() && self.stream_prefetch_bytes > 0 {
            self.store
                .advise_layer_prefix(0, self.stream_prefetch_bytes)
                .map_err(|error| error.0)?;
        }
        let config = self.store.config.clone();
        let layer_count = self.layer_limit.unwrap_or(config.num_hidden_layers);
        if layer_count == 0 || layer_count > config.num_hidden_layers {
            return Err("forward layer limit is outside model depth".into());
        }
        let mut hidden = self.embedding(token_id)?;
        for layer in 0..layer_count {
            if self.resident_weights.is_none()
                && self.stream_prefetch_bytes > 0
                && layer + 1 < config.num_hidden_layers
            {
                self.store
                    .advise_layer_prefix(layer + 1, self.stream_prefetch_bytes)
                    .map_err(|error| error.0)?;
            }
            let prefix = format!("model.layers.{layer}");
            let qkv_names = vec![
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
            ];
            let gate_up_names = vec![
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ];
            let residual = hidden.clone();
            let normalized = {
                let weight = self.tensor(&format!("{prefix}.input_layernorm.weight"))?;
                rms_norm_bf16_cpu(&hidden, weight.bytes, 1, hidden.len(), config.rms_norm_eps)?
            };
            if self.chain_attention {
                let chained = self.chained_attention_layer(layer, &prefix, &normalized)?;
                self.caches[layer]
                    .append_token(&chained.keys, &chained.values)
                    .map_err(|error| error.0)?;
                hidden = add_vectors(&chained.projected, &residual)?;
            } else {
                self.prefetch_stage(&qkv_names)?;
                self.wait_for_stage(&qkv_names)?;
                let pending_qkv = self.matvec_many_async(&qkv_names, &normalized)?;
                let mut qkv = if let Some(pending) = pending_qkv {
                    self.prefetch_stage(&gate_up_names)?;
                    pending.wait()?
                } else {
                    self.prefetch_stage(&gate_up_names)?;
                    self.matvec_many(&qkv_names, &normalized)?
                };
                let mut query = qkv.remove(0);
                let mut keys = qkv.remove(0);
                let values = qkv.remove(0);
                {
                    let weight = self.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?;
                    query = rms_norm_bf16_cpu(
                        &query,
                        weight.bytes,
                        config.num_attention_heads,
                        config.head_dim,
                        config.rms_norm_eps,
                    )?;
                }
                {
                    let weight = self.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?;
                    keys = rms_norm_bf16_cpu(
                        &keys,
                        weight.bytes,
                        config.num_key_value_heads,
                        config.head_dim,
                        config.rms_norm_eps,
                    )?;
                }
                query = rope_cpu(
                    &query,
                    config.num_attention_heads,
                    config.head_dim,
                    position,
                    config.rope_theta,
                )?;
                keys = rope_cpu(
                    &keys,
                    config.num_key_value_heads,
                    config.head_dim,
                    position,
                    config.rope_theta,
                )?;
                let attended = attention_decode_cpu(
                    &query,
                    &self.caches[layer],
                    &keys,
                    &values,
                    config.num_attention_heads,
                )?;
                self.caches[layer]
                    .append_token(&keys, &values)
                    .map_err(|error| error.0)?;
                let projected =
                    self.matvec(&format!("{prefix}.self_attn.o_proj.weight"), &attended)?;
                hidden = add_vectors(&projected, &residual)?;
            }

            let residual = hidden.clone();
            let normalized = {
                let weight = self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
                rms_norm_bf16_cpu(&hidden, weight.bytes, 1, hidden.len(), config.rms_norm_eps)?
            };
            let pending_gate_up = self.matvec_many_async(&gate_up_names, &normalized)?;
            let gate_up_async = pending_gate_up.is_some();
            let mut gate_up = if let Some(pending) = pending_gate_up {
                if layer + 1 < config.num_hidden_layers {
                    let next_prefix = format!("model.layers.{}", layer + 1);
                    let next_qkv_names = vec![
                        format!("{next_prefix}.self_attn.q_proj.weight"),
                        format!("{next_prefix}.self_attn.k_proj.weight"),
                        format!("{next_prefix}.self_attn.v_proj.weight"),
                    ];
                    self.prefetch_stage(&next_qkv_names)?;
                }
                pending.wait()?
            } else {
                self.matvec_many(&gate_up_names, &normalized)?
            };
            let gate = gate_up.remove(0);
            let up = gate_up.remove(0);
            let activated = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| silu(gate) * up)
                .collect::<Vec<_>>();
            if !gate_up_async && layer + 1 < config.num_hidden_layers {
                let next_prefix = format!("model.layers.{}", layer + 1);
                let next_qkv_names = vec![
                    format!("{next_prefix}.self_attn.q_proj.weight"),
                    format!("{next_prefix}.self_attn.k_proj.weight"),
                    format!("{next_prefix}.self_attn.v_proj.weight"),
                ];
                self.prefetch_stage(&next_qkv_names)?;
            }
            let projected = self.matvec(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = add_vectors(&projected, &residual)?;
        }
        Ok(hidden)
    }

    /// Prepare an optional same-model partial-layer drafter. The drafter uses
    /// only the first `layers` transformer blocks and maintains its own small
    /// KV state; target caches and target outputs remain untouched.
    pub fn prepare_partial_draft(
        &mut self,
        prompt_tokens: &[u32],
        layers: usize,
    ) -> Result<(), String> {
        if prompt_tokens.is_empty() {
            return Err("partial draft requires a non-empty prompt".into());
        }
        if layers == 0 || layers > self.store.config.num_hidden_layers {
            return Err("partial draft layer count is outside model depth".into());
        }
        if self.stage_byte_budget > 0 || self.stream_chunk_rows.is_some() {
            return Err("partial draft does not support staged or row-chunked weights".into());
        }
        let config = self.store.config.clone();
        let mut caches = (0..layers)
            .map(|_| {
                KvCache::new(
                    config.num_key_value_heads,
                    config.head_dim,
                    self.max_context,
                )
                .map_err(|error| error.0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (position, token) in prompt_tokens.iter().copied().enumerate() {
            self.forward_token_prefix_hidden(token as usize, position, &mut caches, layers)?;
        }
        self.draft_caches = Some(caches);
        Ok(())
    }

    /// Advance the prepared resident-layer drafter and return its hidden state
    /// before final normalization and the vocabulary head. The target cache
    /// remains untouched; only the drafter KV state advances.
    pub fn partial_draft_hidden(
        &mut self,
        token_id: usize,
        position: usize,
        layers: usize,
    ) -> Result<Vec<f32>, String> {
        if layers == 0 || layers > self.store.config.num_hidden_layers {
            return Err("partial draft layer count is outside model depth".into());
        }
        let mut caches = self
            .draft_caches
            .take()
            .ok_or("partial draft has not been prepared")?;
        if caches.len() != layers || caches.iter().any(|cache| cache.cached_tokens() != position) {
            self.draft_caches = Some(caches);
            return Err("partial draft KV position is inconsistent".into());
        }
        let result = self.forward_token_prefix_hidden(token_id, position, &mut caches, layers);
        self.draft_caches = Some(caches);
        result
    }

    /// Generate a candidate window with the partial-layer drafter. The first
    /// token is always the target greedy token; later tokens come from the
    /// drafter's sequential logits and are checked by exact target
    /// verification before they can be committed.
    pub fn partial_draft_candidates(
        &mut self,
        previous_logits: &[f32],
        position: usize,
        width: usize,
        layers: usize,
    ) -> Result<Vec<u32>, String> {
        if !(2..=8).contains(&width) {
            return Err("partial draft width must be between two and eight".into());
        }
        let mut caches = self
            .draft_caches
            .take()
            .ok_or("partial draft has not been prepared")?;
        if caches.len() != layers || caches.iter().any(|cache| cache.cached_tokens() != position) {
            self.draft_caches = Some(caches);
            return Err("partial draft KV position is inconsistent".into());
        }
        let result = (|| {
            let mut candidates = Vec::with_capacity(width);
            let mut logits = previous_logits.to_vec();
            for index in 0..width {
                let token = argmax(&logits)? as u32;
                candidates.push(token);
                // The final candidate is verified by the target but is never
                // used to produce another draft token. Avoid advancing the
                // retained-prefix KV cache for that dead state.
                if index + 1 < width {
                    logits = self.forward_token_prefix(
                        token as usize,
                        position + index,
                        &mut caches,
                        layers,
                    )?;
                }
            }
            Ok(candidates)
        })();
        self.draft_caches = Some(caches);
        result
    }

    /// Generate a sequential candidate window from the retained prefix and a
    /// disposable sidecar scorer. The first token is always the exact target
    /// greedy token; every later token is only a proposal and must be verified
    /// by the untouched target before it is committed.
    pub fn resident_sidecar_candidates(
        &mut self,
        previous_logits: &[f32],
        position: usize,
        width: usize,
        layers: usize,
        min_margin: Option<f32>,
    ) -> Result<Option<Vec<u32>>, String> {
        if !(2..=8).contains(&width) {
            return Err("resident sidecar width must be between two and eight".into());
        }
        if self.resident_sidecar.is_none() {
            return Err("resident sidecar has not been loaded".into());
        }
        let mut caches = self
            .draft_caches
            .take()
            .ok_or("partial draft has not been prepared")?;
        if caches.len() != layers || caches.iter().any(|cache| cache.cached_tokens() != position) {
            self.draft_caches = Some(caches);
            return Err("partial draft KV position is inconsistent".into());
        }
        let result = (|| {
            let first = argmax(previous_logits)? as u32;
            let mut candidates = Vec::with_capacity(width);
            candidates.push(first);
            let mut token = first;
            for offset in 0..width {
                if offset + 1 < width {
                    let hidden = self.forward_token_prefix_hidden(
                        token as usize,
                        position + offset,
                        &mut caches,
                        layers,
                    )?;
                    let scored = self
                        .resident_sidecar
                        .as_ref()
                        .ok_or("resident sidecar has not been loaded")?
                        .propose_scored(&hidden, if offset == 0 { 2 } else { 1 })?;
                    if offset == 0 {
                        if let Some(threshold) = min_margin {
                            let second = scored
                                .get(1)
                                .ok_or("resident sidecar returned fewer than two scores")?;
                            let margin = scored[0].1 - second.1;
                            if margin < threshold {
                                return Ok(None);
                            }
                        }
                    }
                    let next = scored
                        .first()
                        .map(|(token, _)| *token)
                        .ok_or("resident sidecar returned no proposal")?;
                    candidates.push(next);
                    token = next;
                }
            }
            Ok(Some(candidates))
        })();
        self.draft_caches = Some(caches);
        result
    }

    /// Build four two-token branches from one retained-prefix sidecar state.
    /// The target verifies all eight branch positions in one exact traversal;
    /// only the selected branch's accepted prefix is committed to the draft
    /// cache. This is opt-in because it is a proposal experiment, not a model
    /// transformation.
    pub fn resident_sidecar_tree_step(
        &mut self,
        previous_logits: &[f32],
        position: usize,
        layers: usize,
    ) -> Result<TreeVerification, String> {
        if self.resident_sidecar.is_none() {
            return Err("resident sidecar has not been loaded".into());
        }
        let mut base_caches = self
            .draft_caches
            .take()
            .ok_or("partial draft has not been prepared")?;
        if base_caches.len() != layers
            || base_caches
                .iter()
                .any(|cache| cache.cached_tokens() != position)
        {
            self.draft_caches = Some(base_caches);
            return Err("partial draft KV position is inconsistent".into());
        }
        let result = (|| {
            let first = argmax(previous_logits)? as u32;
            let hidden = self.forward_token_prefix_hidden(
                first as usize,
                position,
                &mut base_caches,
                layers,
            )?;
            let proposals = self
                .resident_sidecar
                .as_ref()
                .ok_or("resident sidecar has not been loaded")?
                .propose(&hidden, 4)?;
            let branches = proposals
                .iter()
                .copied()
                .map(|token| vec![first, token])
                .collect::<Vec<_>>();
            let mut branch_caches = Vec::with_capacity(branches.len());
            for branch in &branches {
                let mut caches = base_caches.clone();
                self.forward_token_prefix_hidden(
                    branch[1] as usize,
                    position + 1,
                    &mut caches,
                    layers,
                )?;
                branch_caches.push(caches);
            }
            let verification = self.verify_tree(previous_logits, &branches, position)?;
            let selected = branch_caches
                .get(verification.selected_branch)
                .ok_or("resident sidecar selected an invalid branch")?;
            let mut selected = selected.clone();
            selected.iter_mut().try_for_each(|cache| {
                cache
                    .truncate_to(position + verification.verification.accepted_tokens)
                    .map_err(|error| error.0)
            })?;
            self.draft_caches = Some(selected);
            Ok(verification)
        })();
        if self.draft_caches.is_none() {
            self.draft_caches = Some(base_caches);
        }
        result
    }

    pub fn truncate_partial_draft(&mut self, cached_tokens: usize) -> Result<(), String> {
        if let Some(caches) = &mut self.draft_caches {
            for cache in caches {
                cache.truncate_to(cached_tokens).map_err(|error| error.0)?;
            }
        }
        Ok(())
    }

    /// Truncate the target cache to an exact logical prefix. This is used by
    /// an external small-model drafter after the target rejects a suffix.
    pub fn truncate_cache_to(&mut self, cached_tokens: usize) -> Result<(), String> {
        for cache in &mut self.caches {
            cache.truncate_to(cached_tokens).map_err(|error| error.0)?;
        }
        Ok(())
    }

    /// Generate a sequential candidate window with this model acting as an
    /// optional drafter. Its cache is advanced through the whole window so a
    /// caller can truncate it to the target-accepted prefix after verification.
    pub fn draft_candidates(
        &mut self,
        previous_logits: &[f32],
        position: usize,
        width: usize,
    ) -> Result<Vec<u32>, String> {
        if !(2..=8).contains(&width) {
            return Err("draft width must be between two and eight".into());
        }
        if self.cached_tokens() != position {
            return Err("drafter KV position is inconsistent".into());
        }
        let mut candidates = Vec::with_capacity(width);
        let mut logits = previous_logits.to_vec();
        for index in 0..width {
            let token = argmax(&logits)? as u32;
            candidates.push(token);
            logits = self.forward_token(token as usize, position + index)?;
        }
        Ok(candidates)
    }

    fn forward_token_prefix(
        &mut self,
        token_id: usize,
        position: usize,
        caches: &mut Vec<KvCache>,
        layers: usize,
    ) -> Result<Vec<f32>, String> {
        if caches.len() != layers {
            return Err("partial draft cache count does not match layer count".into());
        }
        std::mem::swap(&mut self.caches, caches);
        let previous_limit = self.layer_limit;
        self.layer_limit = Some(layers);
        let result = self.forward_token(token_id, position);
        self.layer_limit = previous_limit;
        std::mem::swap(&mut self.caches, caches);
        result
    }

    fn forward_token_prefix_hidden(
        &mut self,
        token_id: usize,
        position: usize,
        caches: &mut Vec<KvCache>,
        layers: usize,
    ) -> Result<Vec<f32>, String> {
        if caches.len() != layers {
            return Err("partial draft cache count does not match layer count".into());
        }
        std::mem::swap(&mut self.caches, caches);
        let previous_limit = self.layer_limit;
        self.layer_limit = Some(layers);
        let result = self.forward_token_hidden(token_id, position);
        self.layer_limit = previous_limit;
        std::mem::swap(&mut self.caches, caches);
        result
    }

    /// Evaluate a consecutive candidate-token sequence against a snapshot of
    /// the current KV state. The real cache is untouched until verification
    /// commits an accepted prefix. Large projections use the exact batched
    /// BF16 kernel, so K=4/K=8 candidates share each streamed weight pass.
    pub fn forward_tokens_many(
        &mut self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let (logits, candidate_caches) = self.forward_tokens_many_internal(token_ids, position)?;
        self.recycle_candidate_caches(candidate_caches, position);
        Ok(logits)
    }

    fn forward_tokens_many_internal(
        &mut self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<KvCache>), String> {
        if token_ids.is_empty() {
            return Err("batched forward requires at least one token".into());
        }
        if token_ids.len() > 8 {
            return Err("batched forward supports at most eight tokens".into());
        }
        if position
            .checked_add(token_ids.len())
            .is_none_or(|end| end > self.max_context)
        {
            return Err("batched token positions exceed configured context capacity".into());
        }
        if self
            .caches
            .iter()
            .any(|cache| cache.cached_tokens() != position)
        {
            return Err("KV-cache position is inconsistent across layers".into());
        }
        if self.stream_chunk_rows.is_some() || self.stage_byte_budget > 0 {
            return Err("batched forward requires unchunked, unstaged projection weights".into());
        }

        let mut candidate_caches = self.take_candidate_cache_pool()?;
        let mut hidden = token_ids
            .iter()
            .map(|token_id| self.embedding(*token_id as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let config = self.store.config.clone();

        for (layer, candidate_cache) in candidate_caches.iter_mut().enumerate() {
            let prefix = format!("model.layers.{layer}");
            let residual = hidden.clone();
            let input_norm_bytes = {
                let tensor = self.tensor(&format!("{prefix}.input_layernorm.weight"))?;
                tensor.bytes.to_vec()
            };
            let normalized = residual
                .iter()
                .map(|vector| {
                    rms_norm_bf16_cpu(
                        vector,
                        &input_norm_bytes,
                        1,
                        vector.len(),
                        config.rms_norm_eps,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let qkv_names = [
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
            ];
            let qkv = self.matmul_many_fused(&qkv_names, &normalized)?;
            let q = qkv[0].clone();
            let k = qkv[1].clone();
            let v = qkv[2].clone();
            let q_norm_bytes = {
                let tensor = self.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?;
                tensor.bytes.to_vec()
            };
            let k_norm_bytes = {
                let tensor = self.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?;
                tensor.bytes.to_vec()
            };
            let mut attended = Vec::with_capacity(token_ids.len());
            for candidate in 0..token_ids.len() {
                let query = rms_norm_bf16_cpu(
                    &q[candidate],
                    &q_norm_bytes,
                    config.num_attention_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                )?;
                let keys = rms_norm_bf16_cpu(
                    &k[candidate],
                    &k_norm_bytes,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                )?;
                let candidate_position = position
                    .checked_add(candidate)
                    .ok_or("candidate token position overflows")?;
                let query = rope_cpu(
                    &query,
                    config.num_attention_heads,
                    config.head_dim,
                    candidate_position,
                    config.rope_theta,
                )?;
                let keys = rope_cpu(
                    &keys,
                    config.num_key_value_heads,
                    config.head_dim,
                    candidate_position,
                    config.rope_theta,
                )?;
                let output = attention_decode_cpu(
                    &query,
                    candidate_cache,
                    &keys,
                    &v[candidate],
                    config.num_attention_heads,
                )?;
                candidate_cache
                    .append_token(&keys, &v[candidate])
                    .map_err(|error| error.0)?;
                attended.push(output);
            }
            let projected =
                self.matmul_many(&format!("{prefix}.self_attn.o_proj.weight"), &attended)?;
            hidden = projected
                .into_iter()
                .zip(residual)
                .map(|(projected, residual)| add_vectors(&projected, &residual))
                .collect::<Result<Vec<_>, _>>()?;

            let residual = hidden.clone();
            let post_norm_bytes = {
                let tensor = self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
                tensor.bytes.to_vec()
            };
            let normalized = residual
                .iter()
                .map(|vector| {
                    rms_norm_bf16_cpu(
                        vector,
                        &post_norm_bytes,
                        1,
                        vector.len(),
                        config.rms_norm_eps,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let gate_up_names = [
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ];
            let gate_up = self.matmul_many_fused(&gate_up_names, &normalized)?;
            let gate = gate_up[0].clone();
            let up = gate_up[1].clone();
            let activated = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| {
                    gate.into_iter()
                        .zip(up)
                        .map(|(gate, up)| silu(gate) * up)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let projected =
                self.matmul_many(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = projected
                .into_iter()
                .zip(residual)
                .map(|(projected, residual)| add_vectors(&projected, &residual))
                .collect::<Result<Vec<_>, _>>()?;
        }

        let final_norm_bytes = {
            let tensor = self.tensor("model.norm.weight")?;
            tensor.bytes.to_vec()
        };
        let normalized = hidden
            .iter()
            .map(|vector| {
                rms_norm_bf16_cpu(
                    vector,
                    &final_norm_bytes,
                    1,
                    vector.len(),
                    config.rms_norm_eps,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let logits = self.matmul_many("model.embed_tokens.weight", &normalized)?;
        Ok((logits, candidate_caches))
    }

    /// Verify a candidate sequence against the target model's greedy path and
    /// commit only the accepted KV prefix. Rejected speculative suffixes are
    /// discarded without reallocating the fixed-capacity caches.
    pub fn verify_many(
        &mut self,
        previous_logits: &[f32],
        candidate_tokens: &[u32],
        position: usize,
    ) -> Result<Verification, String> {
        if candidate_tokens.is_empty() {
            return Err("verification requires at least one candidate token".into());
        }
        let expected_first = argmax(previous_logits)? as u32;
        if candidate_tokens[0] != expected_first {
            return Ok(Verification {
                accepted_tokens: 0,
                next_token: expected_first,
                target_logits: Vec::new(),
            });
        }
        let (target_logits, mut candidate_caches) =
            self.forward_tokens_many_internal(candidate_tokens, position)?;
        let verification = self.commit_verified_forward(
            previous_logits,
            candidate_tokens,
            position,
            target_logits,
            &mut candidate_caches,
        )?;
        self.recycle_candidate_caches(candidate_caches, position);
        Ok(verification)
    }

    /// Evaluate several independent candidate branches in one target-model
    /// traversal. The flattened candidate positions share every streamed
    /// projection weight; only their compact branch-local KV suffixes differ.
    /// This is an exact verifier: branch selection and cache commit are based
    /// on the original BF16 target logits, never on an approximation.
    pub fn verify_tree(
        &mut self,
        previous_logits: &[f32],
        branches: &[Vec<u32>],
        position: usize,
    ) -> Result<TreeVerification, String> {
        if branches.len() < 2 {
            return Err("tree verification requires at least two branches".into());
        }
        let depth = branches
            .first()
            .map(Vec::len)
            .ok_or("tree verification requires branches")?;
        if depth == 0 || branches.iter().any(|branch| branch.len() != depth) {
            return Err("tree branches must have equal non-zero depth".into());
        }
        if branches.len().saturating_mul(depth) > 8 {
            return Err("tree verification supports at most eight candidates".into());
        }
        let TreeForward {
            logits: target_logits,
            caches: branch_caches,
        } = self.forward_tree_many_internal(branches, position)?;
        self.commit_tree_forward(
            previous_logits,
            branches,
            position,
            target_logits,
            branch_caches,
        )
    }

    /// Run a bounded Jacobi loop over independent tree branches. Each
    /// iteration still performs one shared-weight traversal; only the small
    /// candidate tensors and compact branch KV suffixes are replaced.
    pub fn lookahead_tree_step(
        &mut self,
        previous_logits: &[f32],
        initial_branches: &[Vec<u32>],
        position: usize,
        iterations: usize,
    ) -> Result<TreeVerification, String> {
        if iterations == 0 {
            return Err("tree lookahead iterations must be non-zero".into());
        }
        let mut branches = initial_branches.to_vec();
        let mut final_branches = None;
        let mut final_logits = None;
        let mut final_caches = None;
        for _ in 0..iterations {
            let evaluated_branches = branches.clone();
            let TreeForward { logits, caches } =
                self.forward_tree_many_internal(&evaluated_branches, position)?;
            let next = evaluated_branches
                .iter()
                .zip(&logits)
                .map(|(candidates, target_logits)| {
                    update_lookahead_candidates(candidates, target_logits)
                })
                .collect::<Result<Vec<_>, _>>()?;
            final_branches = Some(evaluated_branches);
            final_logits = Some(logits);
            final_caches = Some(caches);
            if next == branches {
                break;
            }
            branches = next;
        }
        self.commit_tree_forward(
            previous_logits,
            &final_branches.ok_or("tree lookahead produced no candidates")?,
            position,
            final_logits.ok_or("tree lookahead produced no logits")?,
            final_caches.ok_or("tree lookahead produced no caches")?,
        )
    }

    fn commit_tree_forward(
        &mut self,
        previous_logits: &[f32],
        branches: &[Vec<u32>],
        position: usize,
        target_logits: Vec<Vec<Vec<f32>>>,
        mut branch_caches: Vec<Vec<KvCache>>,
    ) -> Result<TreeVerification, String> {
        let mut selected_branch = 0;
        let mut selected = None;
        for (branch, candidates) in branches.iter().enumerate() {
            let verification =
                assess_verification(previous_logits, candidates, target_logits[branch].clone())?;
            if selected.as_ref().is_none_or(|current: &Verification| {
                verification.accepted_tokens > current.accepted_tokens
            }) {
                selected_branch = branch;
                selected = Some(verification);
            }
        }
        let verification = selected.ok_or("tree verification selected no branch")?;
        if verification.accepted_tokens > 0 {
            let committed_tokens = position
                .checked_add(verification.accepted_tokens)
                .ok_or("verified tree cache position overflows")?;
            for layer_caches in &mut branch_caches {
                let cache = &mut layer_caches[selected_branch];
                cache
                    .truncate_to(committed_tokens)
                    .map_err(|error| error.0.clone())?;
            }
            for (target, layer_caches) in self.caches.iter_mut().zip(&branch_caches) {
                target
                    .copy_prefix_from(&layer_caches[selected_branch])
                    .map_err(|error| error.0)?;
            }
        }
        Ok(TreeVerification {
            candidates: branches.to_vec(),
            selected_branch,
            verification,
        })
    }

    fn forward_tree_many_internal(
        &mut self,
        branches: &[Vec<u32>],
        position: usize,
    ) -> Result<TreeForward, String> {
        let depth = branches
            .first()
            .map(Vec::len)
            .ok_or("tree verification requires branches")?;
        if branches.len() < 2 || depth == 0 || branches.iter().any(|branch| branch.len() != depth) {
            return Err("tree branches must have equal non-zero depth".into());
        }
        let batch = branches
            .len()
            .checked_mul(depth)
            .ok_or("tree candidate count overflows")?;
        if batch > 8 {
            return Err("tree verification supports at most eight candidates".into());
        }
        if position
            .checked_add(depth)
            .is_none_or(|end| end > self.max_context)
        {
            return Err("tree token positions exceed configured context capacity".into());
        }
        if self
            .caches
            .iter()
            .any(|cache| cache.cached_tokens() != position)
        {
            return Err("KV-cache position is inconsistent across layers".into());
        }
        if self.stream_chunk_rows.is_some() || self.stage_byte_budget > 0 {
            return Err("tree verification requires unchunked, unstaged projection weights".into());
        }

        let config = self.store.config.clone();
        let branch_capacity = position
            .checked_add(depth)
            .ok_or("tree cache capacity overflows")?;
        let mut branch_caches = self
            .caches
            .iter()
            .map(|source| {
                branches
                    .iter()
                    .map(|_| {
                        let mut cache = KvCache::new(
                            source.key_value_heads(),
                            source.head_dim(),
                            branch_capacity,
                        )
                        .map_err(|error| error.0.clone())?;
                        cache.copy_prefix_from(source).map_err(|error| error.0)?;
                        Ok(cache)
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        let token_ids = branches
            .iter()
            .flat_map(|branch| branch.iter().copied())
            .collect::<Vec<_>>();
        let mut hidden = token_ids
            .iter()
            .map(|token_id| self.embedding(*token_id as usize))
            .collect::<Result<Vec<_>, _>>()?;

        for (layer, layer_caches) in branch_caches.iter_mut().enumerate() {
            let prefix = format!("model.layers.{layer}");
            let residual = hidden.clone();
            let input_norm_bytes = self
                .tensor(&format!("{prefix}.input_layernorm.weight"))?
                .bytes
                .to_vec();
            let normalized = residual
                .iter()
                .map(|vector| {
                    rms_norm_bf16_cpu(
                        vector,
                        &input_norm_bytes,
                        1,
                        vector.len(),
                        config.rms_norm_eps,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let qkv_names = [
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
            ];
            let qkv = self.matmul_many_fused(&qkv_names, &normalized)?;
            let q_norm_bytes = self
                .tensor(&format!("{prefix}.self_attn.q_norm.weight"))?
                .bytes
                .to_vec();
            let k_norm_bytes = self
                .tensor(&format!("{prefix}.self_attn.k_norm.weight"))?
                .bytes
                .to_vec();
            let mut attended = vec![Vec::new(); batch];
            for (branch_index, _branch) in branches.iter().enumerate() {
                let cache = &mut layer_caches[branch_index];
                for candidate_index in 0..depth {
                    let flat_index = branch_index * depth + candidate_index;
                    let query = rms_norm_bf16_cpu(
                        &qkv[0][flat_index],
                        &q_norm_bytes,
                        config.num_attention_heads,
                        config.head_dim,
                        config.rms_norm_eps,
                    )?;
                    let keys = rms_norm_bf16_cpu(
                        &qkv[1][flat_index],
                        &k_norm_bytes,
                        config.num_key_value_heads,
                        config.head_dim,
                        config.rms_norm_eps,
                    )?;
                    let candidate_position = position
                        .checked_add(candidate_index)
                        .ok_or("tree candidate token position overflows")?;
                    let query = rope_cpu(
                        &query,
                        config.num_attention_heads,
                        config.head_dim,
                        candidate_position,
                        config.rope_theta,
                    )?;
                    let keys = rope_cpu(
                        &keys,
                        config.num_key_value_heads,
                        config.head_dim,
                        candidate_position,
                        config.rope_theta,
                    )?;
                    attended[branch_index * depth + candidate_index] = attention_decode_cpu(
                        &query,
                        cache,
                        &keys,
                        &qkv[2][flat_index],
                        config.num_attention_heads,
                    )?;
                    cache
                        .append_token(&keys, &qkv[2][flat_index])
                        .map_err(|error| error.0)?;
                }
            }
            let projected =
                self.matmul_many(&format!("{prefix}.self_attn.o_proj.weight"), &attended)?;
            hidden = projected
                .into_iter()
                .zip(residual)
                .map(|(projected, residual)| add_vectors(&projected, &residual))
                .collect::<Result<Vec<_>, _>>()?;

            let residual = hidden.clone();
            let post_norm_bytes = self
                .tensor(&format!("{prefix}.post_attention_layernorm.weight"))?
                .bytes
                .to_vec();
            let normalized = residual
                .iter()
                .map(|vector| {
                    rms_norm_bf16_cpu(
                        vector,
                        &post_norm_bytes,
                        1,
                        vector.len(),
                        config.rms_norm_eps,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let gate_up_names = [
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
            ];
            let gate_up = self.matmul_many_fused(&gate_up_names, &normalized)?;
            let activated = gate_up[0]
                .iter()
                .zip(&gate_up[1])
                .map(|(gate, up)| {
                    gate.iter()
                        .zip(up)
                        .map(|(gate, up)| silu(*gate) * *up)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let projected =
                self.matmul_many(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = projected
                .into_iter()
                .zip(residual)
                .map(|(projected, residual)| add_vectors(&projected, &residual))
                .collect::<Result<Vec<_>, _>>()?;
        }

        let final_norm_bytes = self.tensor("model.norm.weight")?.bytes.to_vec();
        let normalized = hidden
            .iter()
            .map(|vector| {
                rms_norm_bf16_cpu(
                    vector,
                    &final_norm_bytes,
                    1,
                    vector.len(),
                    config.rms_norm_eps,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let flat_logits = self.matmul_many("model.embed_tokens.weight", &normalized)?;
        let mut grouped_logits = (0..branches.len())
            .map(|_| Vec::with_capacity(depth))
            .collect::<Vec<_>>();
        for (index, logits) in flat_logits.into_iter().enumerate() {
            grouped_logits[index / depth].push(logits);
        }
        Ok(TreeForward {
            logits: grouped_logits,
            caches: branch_caches,
        })
    }

    /// Run a bounded Jacobi/lookahead candidate loop, then commit the exact
    /// accepted prefix from the final target pass. Every pass remains the
    /// original model and original BF16 weights; the scheduler only changes
    /// how many candidate positions share one streamed traversal.
    pub fn lookahead_step(
        &mut self,
        previous_logits: &[f32],
        position: usize,
        width: usize,
        iterations: usize,
    ) -> Result<LookaheadStep, String> {
        if !(2..=8).contains(&width) {
            return Err("lookahead width must be between two and eight".into());
        }
        if iterations == 0 {
            return Err("lookahead iterations must be non-zero".into());
        }
        let first = argmax(previous_logits)? as u32;
        let mut candidates = vec![first; width];
        let mut evaluated_candidates = candidates.clone();
        let mut final_logits = None;
        let mut final_caches = None;
        for _ in 0..iterations {
            if let Some(caches) = final_caches.take() {
                self.recycle_candidate_caches(caches, position);
            }
            evaluated_candidates.clone_from(&candidates);
            let (target_logits, candidate_caches) =
                self.forward_tokens_many_internal(&evaluated_candidates, position)?;
            let next = update_lookahead_candidates(&evaluated_candidates, &target_logits)?;
            final_logits = Some(target_logits);
            final_caches = Some(candidate_caches);
            if next == evaluated_candidates {
                break;
            }
            candidates = next;
        }
        let target_logits = final_logits.ok_or("lookahead produced no target logits")?;
        let mut candidate_caches = final_caches.ok_or("lookahead produced no candidate cache")?;
        let verification = self.commit_verified_forward(
            previous_logits,
            &evaluated_candidates,
            position,
            target_logits,
            &mut candidate_caches,
        )?;
        self.recycle_candidate_caches(candidate_caches, position);
        Ok(LookaheadStep {
            candidates: evaluated_candidates,
            verification,
        })
    }

    fn commit_verified_forward(
        &mut self,
        previous_logits: &[f32],
        candidate_tokens: &[u32],
        position: usize,
        target_logits: Vec<Vec<f32>>,
        candidate_caches: &mut Vec<KvCache>,
    ) -> Result<Verification, String> {
        let verification = assess_verification(previous_logits, candidate_tokens, target_logits)?;
        if verification.accepted_tokens > 0 {
            let committed_tokens = position
                .checked_add(verification.accepted_tokens)
                .ok_or("verified cache position overflows")?;
            for cache in &mut *candidate_caches {
                cache
                    .truncate_to(committed_tokens)
                    .map_err(|error| error.0)?;
            }
            std::mem::swap(&mut self.caches, candidate_caches);
        } else {
            for cache in &mut *candidate_caches {
                cache.truncate_to(position).map_err(|error| error.0)?;
            }
        }
        Ok(verification)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("prefill requires at least one token".into());
        }
        if tokens.len() > self.max_context {
            return Err("prompt exceeds configured context capacity".into());
        }
        self.reset();
        let mut logits = Vec::new();
        for (position, token) in tokens.iter().copied().enumerate() {
            logits = self.forward_token(token as usize, position)?;
        }
        Ok(logits)
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<Generation, String> {
        if max_new_tokens == 0 {
            return Err("max_new_tokens must be non-zero".into());
        }
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err("prompt tokenization produced no tokens".into());
        }
        if prompt_tokens.len().saturating_add(max_new_tokens) > self.max_context {
            return Err("prompt plus generation exceeds context capacity".into());
        }
        let mut logits = self.prefill(&prompt_tokens)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        for index in 0..max_new_tokens {
            let token = argmax(&logits)?;
            generated.push(token as u32);
            if token as u32 == self.store.config.eos_token_id {
                break;
            }
            logits = self.forward_token(token, prompt_tokens.len() + index)?;
        }
        let text = self.tokenizer.decode(&generated)?;
        Ok(Generation {
            prompt_tokens,
            tokens: generated,
            text,
        })
    }

    pub fn evaluate_quality(&mut self, suite: &QualitySuite) -> Result<QualitySummary, String> {
        suite.validate().map_err(|error| error.0)?;
        if suite.model_revision != self.store.manifest.revision {
            return Err(format!(
                "quality fixture model revision {} does not match {}",
                suite.model_revision, self.store.manifest.revision
            ));
        }
        if suite.tokenizer_revision != self.store.manifest.revision {
            return Err(format!(
                "quality fixture tokenizer revision {} does not match {}",
                suite.tokenizer_revision, self.store.manifest.revision
            ));
        }
        let mut likelihoods = std::collections::BTreeMap::new();
        for case in &suite.likelihood {
            let tokens = self.tokenizer.encode(&case.text)?;
            if tokens.len() < 2 {
                return Err(format!(
                    "likelihood case {} has fewer than two tokens",
                    case.id
                ));
            }
            self.reset();
            let mut logits = self.forward_token(tokens[0] as usize, 0)?;
            let mut total_nll = 0.0_f64;
            for index in 1..tokens.len() {
                total_nll += negative_log_likelihood(&logits, tokens[index] as usize)?;
                if index + 1 < tokens.len() {
                    logits = self.forward_token(tokens[index] as usize, index)?;
                }
            }
            likelihoods.insert(case.id.clone(), total_nll / (tokens.len() - 1) as f64);
        }
        let likelihood = suite
            .score_likelihood(&likelihoods)
            .map_err(|error| error.0)?;
        let mut outputs = std::collections::BTreeMap::new();
        for case in &suite.structured_completion {
            let generation = self.generate(&case.prompt, suite.generation.max_new_tokens)?;
            outputs.insert(case.id.clone(), generation.text);
        }
        let structured = suite.score_structured(&outputs).map_err(|error| error.0)?;
        Ok(QualitySummary {
            likelihood,
            structured,
            regression_cases: suite.regression_prompts.len(),
        })
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>, String> {
        self.store.tensor(name).map_err(|error| error.0)
    }

    fn embedding(&self, token_id: usize) -> Result<Vec<f32>, String> {
        let tensor = self.tensor("model.embed_tokens.weight")?;
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("embedding requires a rank-2 BF16 tensor".into());
        }
        let columns = tensor.info.shape[1];
        if token_id >= tensor.info.shape[0] {
            return Err("embedding token id exceeds vocabulary".into());
        }
        let row_start = token_id
            .checked_mul(columns)
            .and_then(|index| index.checked_mul(2))
            .ok_or("embedding row offset overflow")?;
        let row_end = row_start
            .checked_add(columns * 2)
            .ok_or("embedding row end overflow")?;
        let row = tensor
            .bytes
            .get(row_start..row_end)
            .ok_or("embedding tensor bytes are shorter than its shape")?;
        Ok(row.chunks_exact(2).map(bf16_to_f32).collect())
    }

    fn chained_attention_layer(
        &self,
        layer: usize,
        prefix: &str,
        input: &[f32],
    ) -> Result<crate::metal::ChainedAttentionOutput, String> {
        let q = self.tensor(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k = self.tensor(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v = self.tensor(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let o = self.tensor(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let q_norm = self.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm = self.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?;
        let config = &self.store.config;
        let cache = &self.caches[layer];
        let chain_config = ChainedAttentionConfig {
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            cached_tokens: cache.cached_tokens(),
            cache_capacity_tokens: cache.capacity_tokens(),
            position: cache.cached_tokens(),
            rope_theta: config.rope_theta,
            epsilon: config.rms_norm_eps,
        };
        if let Some(resident) = &self.resident_weights {
            let q_weight = resident
                .matrices
                .get(q.info.name.as_str())
                .ok_or_else(|| format!("resident weight is missing: {}", q.info.name))?;
            let k_weight = resident
                .matrices
                .get(k.info.name.as_str())
                .ok_or_else(|| format!("resident weight is missing: {}", k.info.name))?;
            let v_weight = resident
                .matrices
                .get(v.info.name.as_str())
                .ok_or_else(|| format!("resident weight is missing: {}", v.info.name))?;
            let o_weight = resident
                .matrices
                .get(o.info.name.as_str())
                .ok_or_else(|| format!("resident weight is missing: {}", o.info.name))?;
            return self
                .context
                .chained_qkv_attention_o_buffers(ChainedAttentionBufferRequest {
                    q: (&q_weight.weight, q_weight.rows, q_weight.columns),
                    k: (&k_weight.weight, k_weight.rows, k_weight.columns),
                    v: (&v_weight.weight, v_weight.rows, v_weight.columns),
                    o: (&o_weight.weight, o_weight.rows, o_weight.columns),
                    q_norm_bytes: q_norm.bytes,
                    k_norm_bytes: k_norm.bytes,
                    input,
                    key_cache: cache.key_storage(),
                    value_cache: cache.value_storage(),
                    config: chain_config,
                });
        }
        self.context
            .chained_qkv_attention_o_tensors(ChainedAttentionTensorRequest {
                q_tensor: &q,
                k_tensor: &k,
                v_tensor: &v,
                o_tensor: &o,
                q_norm_bytes: q_norm.bytes,
                k_norm_bytes: k_norm.bytes,
                input,
                key_cache: cache.key_storage(),
                value_cache: cache.value_storage(),
                config: chain_config,
            })
    }

    fn matvec(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(name)?;
        if name == "model.embed_tokens.weight" {
            if let Some(index) = &self.exact_head_index {
                return index.search(&self.context, &tensor, input);
            }
        }
        let profile = std::env::var_os("SI_PROFILE_OPS").is_some();
        let started = std::time::Instant::now();
        let result = if let Some(resident) = &self.resident_weights {
            let matrix = resident
                .matrices
                .get(name)
                .ok_or_else(|| format!("resident weight is missing: {name}"))?;
            self.context
                .bf16_matvec_buffer(&matrix.weight, matrix.rows, matrix.columns, input)
        } else if let Some(hot) = &self.hot_weights {
            if let Some(matrix) = hot.matrices.get(name) {
                self.context
                    .bf16_matvec_buffer(&matrix.weight, matrix.rows, matrix.columns, input)
            } else {
                match self.stream_chunk_rows {
                    Some(max_rows) => self
                        .context
                        .bf16_matvec_tensor_chunked(&tensor, input, max_rows),
                    None => self.context.bf16_matvec_tensor(&tensor, input),
                }
            }
        } else if let Some(retained) = &self.retained_layers {
            if let Some(matrix) = retained.matrices.get(name) {
                self.context
                    .bf16_matvec_buffer(&matrix.weight, matrix.rows, matrix.columns, input)
            } else {
                match self.stream_chunk_rows {
                    Some(max_rows) => self
                        .context
                        .bf16_matvec_tensor_chunked(&tensor, input, max_rows),
                    None => self.context.bf16_matvec_tensor(&tensor, input),
                }
            }
        } else {
            match self.stream_chunk_rows {
                Some(max_rows) => self
                    .context
                    .bf16_matvec_tensor_chunked(&tensor, input, max_rows),
                None => self.context.bf16_matvec_tensor(&tensor, input),
            }
        }?;
        if profile {
            eprintln!(
                "si_profile op={} rows={} cols={} ms={:.3}",
                name,
                tensor.info.shape[0],
                tensor.info.shape[1],
                started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
        Ok(result)
    }

    fn prefetch_stage(&mut self, names: &[String]) -> Result<(), String> {
        if self.stage_byte_budget == 0 {
            return Ok(());
        }
        if let Some(stager) = &mut self.stager {
            stager.prefetch(&self.store, names, self.stage_byte_budget)?;
        }
        Ok(())
    }

    fn wait_for_stage(&mut self, names: &[String]) -> Result<(), String> {
        if self.staged.is_some() {
            return Ok(());
        }
        if let Some(stager) = &mut self.stager {
            self.staged = stager.take(names)?;
        }
        Ok(())
    }

    fn take_staged(&mut self, names: &[String]) -> Result<Option<StagedProjectionGroup>, String> {
        if let Some(staged) = self.staged.take() {
            if staged.names != names {
                self.staged = Some(staged);
                return Err("staged projection order does not match execution".into());
            }
            return Ok(Some(staged));
        }
        if let Some(stager) = &mut self.stager {
            return stager.take(names);
        }
        Ok(None)
    }

    /// Execute one projection against several candidate hidden states. This
    /// intentionally rejects unsupported residency modes instead of silently
    /// falling back to K sequential matvecs, which would invalidate SI-004's
    /// weight-traffic measurement.
    fn matmul_many(&mut self, name: &str, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let (batch, columns, flattened) = flatten_batch_inputs(inputs)?;
        if name == "model.embed_tokens.weight" && batch == 1 && self.exact_head_index.is_some() {
            return Ok(vec![self.matvec(name, &inputs[0])?]);
        }
        if self.stream_chunk_rows.is_some() {
            return Err("batched verification does not support row-chunked projections".into());
        }
        if self.stage_byte_budget > 0 {
            return Err("batched verification does not support staged projections yet".into());
        }
        let tensor = self.tensor(name)?;
        if tensor.info.shape[1] != columns {
            return Err(format!(
                "projection {name} expects {} columns, got {columns}",
                tensor.info.shape[1]
            ));
        }
        if let Some(resident) = &self.resident_weights {
            let matrix = resident
                .matrices
                .get(name)
                .ok_or_else(|| format!("resident weight is missing: {name}"))?;
            return self.context.bf16_matmul_many_buffer(
                &matrix.weight,
                matrix.rows,
                matrix.columns,
                batch,
                &flattened,
            );
        }
        if let Some(hot) = &self.hot_weights {
            if let Some(matrix) = hot.matrices.get(name) {
                return self.context.bf16_matmul_many_buffer(
                    &matrix.weight,
                    matrix.rows,
                    matrix.columns,
                    batch,
                    &flattened,
                );
            }
        }
        if let Some(retained) = &self.retained_layers {
            if let Some(matrix) = retained.matrices.get(name) {
                return self.context.bf16_matmul_many_buffer(
                    &matrix.weight,
                    matrix.rows,
                    matrix.columns,
                    batch,
                    &flattened,
                );
            }
        }
        self.context
            .bf16_matmul_many_tensor(&tensor, batch, &flattened)
    }

    fn matmul_many_fused(
        &mut self,
        names: &[String],
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let group = fused_projection_group(names)
            .ok_or("batched fused projection names do not form QKV or gate/up")?;
        let (batch, columns, flattened) = flatten_batch_inputs(inputs)?;
        if self.stream_chunk_rows.is_some() || self.stage_byte_budget > 0 {
            return Err("batched fused projections require unchunked, unstaged weights".into());
        }
        let tensors = names
            .iter()
            .map(|name| self.tensor(name))
            .collect::<Result<Vec<_>, _>>()?;
        let shapes = tensors
            .iter()
            .map(|tensor| (tensor.info.shape[0], tensor.info.shape[1]))
            .collect::<Vec<_>>();
        if shapes
            .iter()
            .any(|(_, tensor_columns)| *tensor_columns != columns)
        {
            return Err("batched fused projections have mismatched input columns".into());
        }
        if let Some(resident) = &self.resident_weights {
            let matrices = names
                .iter()
                .zip(&shapes)
                .map(|(name, (rows, columns))| {
                    let matrix = resident
                        .matrices
                        .get(name)
                        .ok_or_else(|| format!("resident weight is missing: {name}"))?;
                    Ok((&matrix.weight, *rows, *columns))
                })
                .collect::<Result<Vec<_>, String>>()?;
            return match group {
                FusedProjectionGroup::Qkv => self
                    .context
                    .bf16_fused_qkv_many_buffer(&matrices, batch, &flattened),
                FusedProjectionGroup::GateUp => self
                    .context
                    .bf16_fused_gate_up_many_buffer(&matrices, batch, &flattened),
            };
        }
        if let Some(hot) = &self.hot_weights {
            if names.iter().all(|name| hot.matrices.contains_key(name)) {
                let matrices = names
                    .iter()
                    .zip(&shapes)
                    .map(|(name, (rows, columns))| {
                        let matrix = hot
                            .matrices
                            .get(name)
                            .ok_or_else(|| format!("hot weight is missing: {name}"))?;
                        Ok((&matrix.weight, *rows, *columns))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                return match group {
                    FusedProjectionGroup::Qkv => self
                        .context
                        .bf16_fused_qkv_many_buffer(&matrices, batch, &flattened),
                    FusedProjectionGroup::GateUp => self
                        .context
                        .bf16_fused_gate_up_many_buffer(&matrices, batch, &flattened),
                };
            }
        }
        if let Some(retained) = &self.retained_layers {
            if names
                .iter()
                .all(|name| retained.matrices.contains_key(name))
            {
                let matrices = names
                    .iter()
                    .zip(&shapes)
                    .map(|(name, (rows, columns))| {
                        let matrix = retained
                            .matrices
                            .get(name)
                            .ok_or_else(|| format!("retained weight is missing: {name}"))?;
                        Ok((&matrix.weight, *rows, *columns))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                return match group {
                    FusedProjectionGroup::Qkv => self
                        .context
                        .bf16_fused_qkv_many_buffer(&matrices, batch, &flattened),
                    FusedProjectionGroup::GateUp => self
                        .context
                        .bf16_fused_gate_up_many_buffer(&matrices, batch, &flattened),
                };
            }
        }
        let tensor_refs = tensors.iter().collect::<Vec<_>>();
        match group {
            FusedProjectionGroup::Qkv => {
                self.context
                    .bf16_fused_qkv_many_tensors(&tensor_refs, batch, &flattened)
            }
            FusedProjectionGroup::GateUp => {
                self.context
                    .bf16_fused_gate_up_many_tensors(&tensor_refs, batch, &flattened)
            }
        }
    }

    fn matvec_many_async(
        &mut self,
        names: &[String],
        input: &[f32],
    ) -> Result<Option<PendingMatvec>, String> {
        if !self.async_metal || self.stream_chunk_rows.is_some() {
            return Ok(None);
        }
        if self.stage_byte_budget > 0 {
            let Some(staged) = self.take_staged(names)? else {
                return Ok(None);
            };
            let group = fused_projection_group(names)
                .ok_or("staged weights must be a fused projection group")?;
            let shapes = names
                .iter()
                .map(|name| {
                    let tensor = self.tensor(name)?;
                    Ok((tensor.info.shape[0], tensor.info.shape[1]))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let matrices = staged
                .tensors
                .into_iter()
                .zip(shapes)
                .map(|(bytes, (rows, columns))| (bytes, rows, columns))
                .collect::<Vec<_>>();
            return match group {
                FusedProjectionGroup::Qkv => self
                    .context
                    .bf16_fused_qkv_owned_bytes_async(matrices, input)
                    .map(Some),
                FusedProjectionGroup::GateUp => self
                    .context
                    .bf16_fused_gate_up_owned_bytes_async(matrices, input)
                    .map(Some),
            };
        }
        if self.fused_streaming_projections {
            if let Some(resident) = &self.resident_weights {
                let matrices = names
                    .iter()
                    .map(|name| {
                        let tensor = self.tensor(name)?;
                        let matrix = resident
                            .matrices
                            .get(name)
                            .ok_or_else(|| format!("resident weight is missing: {name}"))?;
                        Ok((&matrix.weight, tensor.info.shape[0], tensor.info.shape[1]))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                return match fused_projection_group(names) {
                    Some(FusedProjectionGroup::Qkv) => self
                        .context
                        .bf16_fused_qkv_buffer_async(&matrices, input)
                        .map(Some),
                    Some(FusedProjectionGroup::GateUp) => self
                        .context
                        .bf16_fused_gate_up_buffer_async(&matrices, input)
                        .map(Some),
                    None => Ok(None),
                };
            }
            if self.hot_weights.is_some() || self.retained_layers.is_some() {
                return Ok(None);
            }
            let tensors = names
                .iter()
                .map(|name| self.tensor(name))
                .collect::<Result<Vec<_>, _>>()?;
            let tensor_refs = tensors.iter().collect::<Vec<_>>();
            return match fused_projection_group(names) {
                Some(FusedProjectionGroup::Qkv) => self
                    .context
                    .bf16_fused_qkv_tensors_async(&tensor_refs, input)
                    .map(Some),
                Some(FusedProjectionGroup::GateUp) => self
                    .context
                    .bf16_fused_gate_up_tensors_async(&tensor_refs, input)
                    .map(Some),
                None => Ok(None),
            };
        }
        if self.staged.is_some() {
            return Ok(None);
        }
        if let Some(resident) = &self.resident_weights {
            let matrices = names
                .iter()
                .map(|name| {
                    let tensor = self.tensor(name)?;
                    let matrix = resident
                        .matrices
                        .get(name)
                        .ok_or_else(|| format!("resident weight is missing: {name}"))?;
                    Ok((&matrix.weight, tensor.info.shape[0], tensor.info.shape[1]))
                })
                .collect::<Result<Vec<_>, String>>()?;
            return self
                .context
                .bf16_matvec_many_buffer_async(&matrices, input)
                .map(Some);
        }
        if self.hot_weights.is_some() || self.retained_layers.is_some() {
            return Ok(None);
        }
        let tensors = names
            .iter()
            .map(|name| self.tensor(name))
            .collect::<Result<Vec<_>, _>>()?;
        let tensor_refs = tensors.iter().collect::<Vec<_>>();
        self.context
            .bf16_matvec_many_tensors_async(&tensor_refs, input)
            .map(Some)
    }

    fn matvec_many(&mut self, names: &[String], input: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        if names.is_empty() {
            return Err("batched matvec requires at least one tensor".into());
        }
        if let Some(staged) = self.take_staged(names)? {
            let group = fused_projection_group(names)
                .ok_or("staged weights must be a fused projection group")?;
            let matrices = staged
                .tensors
                .iter()
                .zip(names)
                .map(|(bytes, name)| {
                    let tensor = self.tensor(name)?;
                    Ok((bytes.as_slice(), tensor.info.shape[0], tensor.info.shape[1]))
                })
                .collect::<Result<Vec<_>, String>>()?;
            return match group {
                FusedProjectionGroup::Qkv => self.context.bf16_fused_qkv_bytes(&matrices, input),
                FusedProjectionGroup::GateUp => {
                    self.context.bf16_fused_gate_up_bytes(&matrices, input)
                }
            };
        }
        if self.fused_streaming_projections && self.stream_chunk_rows.is_none() {
            if let Some(group) = fused_projection_group(names) {
                if let Some(resident) = &self.resident_weights {
                    let matrices = names
                        .iter()
                        .map(|name| {
                            let tensor = self.tensor(name)?;
                            let matrix = resident
                                .matrices
                                .get(name)
                                .ok_or_else(|| format!("resident weight is missing: {name}"))?;
                            Ok((&matrix.weight, tensor.info.shape[0], tensor.info.shape[1]))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    return match group {
                        FusedProjectionGroup::Qkv => {
                            self.context.bf16_fused_qkv_buffer(&matrices, input)
                        }
                        FusedProjectionGroup::GateUp => {
                            self.context.bf16_fused_gate_up_buffer(&matrices, input)
                        }
                    };
                }
                let tensors = names
                    .iter()
                    .map(|name| self.tensor(name))
                    .collect::<Result<Vec<_>, _>>()?;
                let tensor_refs = tensors.iter().collect::<Vec<_>>();
                return match group {
                    FusedProjectionGroup::Qkv => {
                        self.context.bf16_fused_qkv_tensors(&tensor_refs, input)
                    }
                    FusedProjectionGroup::GateUp => {
                        self.context.bf16_fused_gate_up_tensors(&tensor_refs, input)
                    }
                };
            }
        }
        if let Some(resident) = &self.resident_weights {
            let mut matrices = Vec::with_capacity(names.len());
            for name in names {
                let tensor = self.tensor(name)?;
                let matrix = resident
                    .matrices
                    .get(name)
                    .ok_or_else(|| format!("resident weight is missing: {name}"))?;
                matrices.push((&matrix.weight, matrix.rows, matrix.columns));
                if std::env::var_os("SI_PROFILE_OPS").is_some() {
                    eprintln!(
                        "si_profile batched_op={} rows={} cols={}",
                        name, tensor.info.shape[0], tensor.info.shape[1]
                    );
                }
            }
            return self.context.bf16_matvec_many_buffer(&matrices, input);
        }
        if let Some(max_rows) = self.stream_chunk_rows {
            return names
                .iter()
                .map(|name| {
                    let tensor = self.tensor(name)?;
                    self.context
                        .bf16_matvec_tensor_chunked(&tensor, input, max_rows)
                })
                .collect();
        }
        if !self.batch_streaming_projections {
            return names.iter().map(|name| self.matvec(name, input)).collect();
        }
        let tensors = names
            .iter()
            .map(|name| self.tensor(name))
            .collect::<Result<Vec<_>, _>>()?;
        let tensor_refs = tensors.iter().collect::<Vec<_>>();
        let outputs = self.context.bf16_matvec_many_tensors(&tensor_refs, input)?;
        if std::env::var_os("SI_PROFILE_OPS").is_some() {
            for (name, tensor) in names.iter().zip(&tensors) {
                eprintln!(
                    "si_profile batched_op={} rows={} cols={}",
                    name, tensor.info.shape[0], tensor.info.shape[1]
                );
            }
        }
        Ok(outputs)
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub prompt_tokens: Vec<u32>,
    pub tokens: Vec<u32>,
    pub text: String,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct Verification {
    pub accepted_tokens: usize,
    pub next_token: u32,
    pub target_logits: Vec<Vec<f32>>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct TreeVerification {
    pub candidates: Vec<Vec<u32>>,
    pub selected_branch: usize,
    pub verification: Verification,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct LookaheadStep {
    pub candidates: Vec<u32>,
    pub verification: Verification,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct ResidentReport {
    pub model_path: String,
    pub model_revision: String,
    pub backend: String,
    pub device: String,
    pub recommended_working_set_mib: u64,
    pub mapped_model_bytes: u64,
    pub active_weight_bytes: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub warmup_runs: u32,
    pub repetitions: u32,
    pub prefill: std::time::Duration,
    pub decode: std::time::Duration,
    pub peak_vram_mib: Option<u64>,
    pub peak_ram_mib: Option<u64>,
    pub median_peak_vram_mib: Option<u64>,
    pub median_peak_ram_mib: Option<u64>,
    pub output_match: Option<bool>,
    pub repetition_outputs_match: bool,
    pub quality: Option<QualitySummary>,
    pub generated_output: String,
    pub generated_token_ids: Vec<u32>,
}

/// Explain what the reported memory counters include. GGUF tensors are bound
/// through zero-copy shared buffers, so their file-backed pages do not appear
/// in Metal's private allocator even while they are resident in unified
/// memory.
fn memory_measurement_label(model_path: &str) -> &'static str {
    if std::path::Path::new(model_path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        "rss_plus_private_metal; mapped_weights_excluded_from_private_metal"
    } else {
        "rss_plus_private_metal"
    }
}

#[cfg(target_os = "macos")]
impl ResidentReport {
    pub fn total(&self) -> std::time::Duration {
        self.prefill + self.decode
    }

    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.decode)
    }

    pub fn total_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens + self.generated_tokens, self.total())
    }

    pub fn as_text(&self) -> String {
        let output = format!(
            "model={}\nmodel_revision={}\nbackend={}\ndevice={}\nprompt_tokens={}\ngenerated_tokens={}\nwarmup_runs={}\nrepetitions={}\nprefill_ms={:.3}\ndecode_ms={:.3}\nprefill_tok_s={:.3}\ndecode_tok_s={:.3}\ntotal_tok_s={:.3}\nrecommended_working_set_mib={}\nmapped_model_bytes={}\nactive_weight_bytes={}\nkv_bytes={}\nscratch_bytes={}\npeak_vram_mib={}\npeak_ram_mib={}\nmedian_peak_vram_mib={}\nmedian_peak_ram_mib={}\npeak_memory_aggregation=worst\nmemory_measurement={}\noutput_match={}\nrepetition_outputs_match={}\nquality_present={}\ngenerated_output={}",
            self.model_path,
            self.model_revision,
            self.backend,
            self.device,
            self.prompt_tokens,
            self.generated_tokens,
            self.warmup_runs,
            self.repetitions,
            self.prefill.as_secs_f64() * 1_000.0,
            self.decode.as_secs_f64() * 1_000.0,
            self.prefill_tokens_per_second(),
            self.decode_tokens_per_second(),
            self.total_tokens_per_second(),
            self.recommended_working_set_mib,
            self.mapped_model_bytes,
            self.active_weight_bytes,
            self.kv_bytes,
            self.scratch_bytes,
            option_u64(self.peak_vram_mib),
            option_u64(self.peak_ram_mib),
            option_u64(self.median_peak_vram_mib),
            option_u64(self.median_peak_ram_mib),
            memory_measurement_label(&self.model_path),
            option_bool(self.output_match),
            self.repetition_outputs_match,
            self.quality.is_some(),
            self.generated_output,
        );
        if let Some(quality) = &self.quality {
            format!(
                "{output}\nquality_mean_nll={:.6}\nquality_perplexity={:.6}\nquality_structured_accuracy={:.6}\nquality_regression_cases={}",
                quality.likelihood.mean_nll,
                quality.likelihood.perplexity,
                quality.structured.accuracy(),
                quality.regression_cases,
            )
        } else {
            output
        }
    }

    pub fn as_json(&self) -> String {
        let quality = self.quality.as_ref().map_or_else(
            || "null".into(),
            |value| serde_json::to_string(value).unwrap_or_else(|_| "null".into()),
        );
        format!(
            concat!(
                "{{\"model_path\":\"{}\",\"model_revision\":\"{}\",\"backend\":\"{}\",",
                "\"device\":\"{}\",\"prompt_tokens\":{},\"generated_tokens\":{},",
                "\"warmup_runs\":{},\"repetitions\":{},\"prefill_ms\":{:.3},\"decode_ms\":{:.3},",
                "\"prefill_tok_s\":{:.3},\"decode_tok_s\":{:.3},\"total_tok_s\":{:.3},",
                "\"recommended_working_set_mib\":{},\"mapped_model_bytes\":{},",
                "\"active_weight_bytes\":{},\"kv_bytes\":{},\"scratch_bytes\":{},",
                "\"peak_vram_mib\":{},\"peak_ram_mib\":{},",
                "\"median_peak_vram_mib\":{},\"median_peak_ram_mib\":{},",
                "\"peak_memory_aggregation\":\"worst\",",
                "\"memory_measurement\":\"{}\",\"output_match\":{},",
                "\"repetition_outputs_match\":{},\"quality\":{},",
                "\"generated_output\":\"{}\",\"generated_token_ids\":[{}]}}"
            ),
            crate::json_escape(&self.model_path),
            crate::json_escape(&self.model_revision),
            crate::json_escape(&self.backend),
            crate::json_escape(&self.device),
            self.prompt_tokens,
            self.generated_tokens,
            self.warmup_runs,
            self.repetitions,
            self.prefill.as_secs_f64() * 1_000.0,
            self.decode.as_secs_f64() * 1_000.0,
            self.prefill_tokens_per_second(),
            self.decode_tokens_per_second(),
            self.total_tokens_per_second(),
            self.recommended_working_set_mib,
            self.mapped_model_bytes,
            self.active_weight_bytes,
            self.kv_bytes,
            self.scratch_bytes,
            option_u64(self.peak_vram_mib),
            option_u64(self.peak_ram_mib),
            option_u64(self.median_peak_vram_mib),
            option_u64(self.median_peak_ram_mib),
            memory_measurement_label(&self.model_path),
            option_bool(self.output_match),
            self.repetition_outputs_match,
            quality,
            crate::json_escape(&self.generated_output),
            self.generated_token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MeasuredRun {
    prefill: std::time::Duration,
    decode: std::time::Duration,
    peak_vram_mib: u64,
    peak_ram_mib: u64,
    active_weight_bytes: u64,
    kv_bytes: u64,
    scratch_bytes: u64,
    generated_output: String,
    generated_token_ids: Vec<u32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct LookaheadConfig {
    width: usize,
    iterations: usize,
    dynamic_width: bool,
    draft_layers: Option<usize>,
    sidecar: Option<String>,
    sidecar_tree: bool,
    sidecar_auto: bool,
    sidecar_min_margin: Option<f32>,
    draft_model: Option<String>,
    ngram_order: Option<usize>,
    tree: Option<TreeConfig>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct TreeConfig {
    branches: usize,
    depth: usize,
    ngram_order: usize,
    memory_ceiling_mib: u64,
}

#[cfg(target_os = "macos")]
impl LookaheadConfig {
    fn from_env() -> Result<Option<Self>, String> {
        if std::env::var_os("SI_LOOKAHEAD").is_none() {
            return Ok(None);
        }
        let width = std::env::var("SI_LOOKAHEAD_WIDTH")
            .ok()
            .map_or(Ok(4), |value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "SI_LOOKAHEAD_WIDTH must be an integer".to_owned())
            })?;
        if !(2..=8).contains(&width) {
            return Err("SI_LOOKAHEAD_WIDTH must be between 2 and 8".into());
        }
        let iterations = std::env::var("SI_LOOKAHEAD_ITERS")
            .ok()
            .map_or(Ok(2), |value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "SI_LOOKAHEAD_ITERS must be an integer".to_owned())
            })?;
        if iterations == 0 {
            return Err("SI_LOOKAHEAD_ITERS must be non-zero".into());
        }
        let dynamic_width = std::env::var_os("SI_LOOKAHEAD_DYNAMIC").is_some();
        let draft_layers = std::env::var("SI_LOOKAHEAD_DRAFT_LAYERS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "SI_LOOKAHEAD_DRAFT_LAYERS must be an integer".to_owned())
            })
            .transpose()?;
        if draft_layers == Some(0) {
            return Err("SI_LOOKAHEAD_DRAFT_LAYERS must be non-zero".into());
        }
        let draft_model = std::env::var("SI_DRAFT_MODEL").ok();
        let sidecar = std::env::var("SI_DRAFT_SIDECAR").ok();
        let sidecar_tree = std::env::var_os("SI_DRAFT_SIDECAR_TREE").is_some();
        let sidecar_auto = std::env::var_os("SI_DRAFT_SIDECAR_AUTO").is_some();
        let sidecar_min_margin = std::env::var("SI_DRAFT_SIDECAR_MIN_MARGIN")
            .ok()
            .map(|value| {
                value
                    .parse::<f32>()
                    .map_err(|_| "SI_DRAFT_SIDECAR_MIN_MARGIN must be a number".to_owned())
            })
            .transpose()?;
        let ngram_order = if std::env::var_os("SI_LOOKAHEAD_NGRAM").is_some() {
            let order = std::env::var("SI_LOOKAHEAD_NGRAM_ORDER")
                .ok()
                .map_or(Ok(3), |value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| "SI_LOOKAHEAD_NGRAM_ORDER must be an integer".to_owned())
                })?;
            if order == 0 {
                return Err("SI_LOOKAHEAD_NGRAM_ORDER must be non-zero".into());
            }
            Some(order)
        } else {
            None
        };
        let tree = if std::env::var_os("SI_TREE").is_some() {
            let branches = std::env::var("SI_TREE_BRANCHES")
                .ok()
                .map_or(Ok(2), |value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| "SI_TREE_BRANCHES must be an integer".to_owned())
                })?;
            let depth = std::env::var("SI_TREE_DEPTH").ok().map_or(Ok(4), |value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "SI_TREE_DEPTH must be an integer".to_owned())
            })?;
            if !(2..=4).contains(&branches) {
                return Err("SI_TREE_BRANCHES must be between 2 and 4".into());
            }
            if !(2..=4).contains(&depth) {
                return Err("SI_TREE_DEPTH must be between 2 and 4".into());
            }
            if branches.saturating_mul(depth) > 8 {
                return Err("SI_TREE_BRANCHES * SI_TREE_DEPTH must be at most 8".into());
            }
            let ngram_order = std::env::var("SI_TREE_NGRAM_ORDER")
                .ok()
                .map_or(Ok(8), |value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| "SI_TREE_NGRAM_ORDER must be an integer".to_owned())
                })?;
            if ngram_order == 0 {
                return Err("SI_TREE_NGRAM_ORDER must be non-zero".into());
            }
            let memory_ceiling_mib =
                std::env::var("SI_TREE_MEMORY_MIB")
                    .ok()
                    .map_or(Ok(2_000_u64), |value| {
                        value
                            .parse::<u64>()
                            .map_err(|_| "SI_TREE_MEMORY_MIB must be an integer".to_owned())
                    })?;
            if memory_ceiling_mib == 0 || memory_ceiling_mib > 2_048 {
                return Err("SI_TREE_MEMORY_MIB must be between 1 and 2048".into());
            }
            Some(TreeConfig {
                branches,
                depth,
                ngram_order,
                memory_ceiling_mib,
            })
        } else {
            None
        };
        if sidecar.is_some() && draft_layers.is_none() {
            return Err("SI_DRAFT_SIDECAR requires SI_LOOKAHEAD_DRAFT_LAYERS".into());
        }
        if sidecar_tree && sidecar.is_none() {
            return Err("SI_DRAFT_SIDECAR_TREE requires SI_DRAFT_SIDECAR".into());
        }
        if sidecar_auto && sidecar.is_none() {
            return Err("SI_DRAFT_SIDECAR_AUTO requires SI_DRAFT_SIDECAR".into());
        }
        if sidecar_min_margin.is_some_and(|margin| !margin.is_finite() || margin < 0.0) {
            return Err("SI_DRAFT_SIDECAR_MIN_MARGIN must be finite and non-negative".into());
        }
        if sidecar_min_margin.is_some() && sidecar.is_none() {
            return Err("SI_DRAFT_SIDECAR_MIN_MARGIN requires SI_DRAFT_SIDECAR".into());
        }
        if tree.is_some()
            && (draft_model.is_some()
                || draft_layers.is_some()
                || ngram_order.is_some()
                || sidecar.is_some())
        {
            return Err(
                "SI_TREE cannot be combined with SI_DRAFT_MODEL, SI_LOOKAHEAD_DRAFT_LAYERS, or SI_LOOKAHEAD_NGRAM"
                    .into(),
            );
        }
        if draft_model.is_some() as u8 + draft_layers.is_some() as u8 + ngram_order.is_some() as u8
            > 1
        {
            return Err(
                "SI_DRAFT_MODEL, SI_LOOKAHEAD_DRAFT_LAYERS, and SI_LOOKAHEAD_NGRAM are mutually exclusive"
                    .into(),
            );
        }
        Ok(Some(Self {
            width,
            iterations,
            dynamic_width,
            draft_layers,
            sidecar,
            sidecar_tree,
            sidecar_auto,
            sidecar_min_margin,
            draft_model,
            ngram_order,
            tree,
        }))
    }
}

#[cfg(target_os = "macos")]
pub fn run_resident(config: &crate::Config) -> Result<ResidentReport, String> {
    if config.repetitions == 0 {
        return Err("measured repetitions must be non-zero".into());
    }
    let lookahead = LookaheadConfig::from_env()?;
    let residency = if config.backend == "metal-resident" {
        WeightResidency::Resident
    } else {
        WeightResidency::Streaming
    };
    let mut model = MetalQwen3::from_model_dir_with_residency(
        &config.model_path,
        config.verify_manifest,
        config.max_context,
        residency,
    )?;
    let mut draft_model = match lookahead
        .as_ref()
        .and_then(|value| value.draft_model.as_deref())
    {
        Some(path) => {
            if path == config.model_path {
                return Err("SI_DRAFT_MODEL must point to a separate model".into());
            }
            let draft_residency = if std::env::var_os("SI_DRAFT_SHARED").is_some() {
                WeightResidency::SharedResident
            } else if std::env::var_os("SI_DRAFT_RESIDENT").is_some() {
                WeightResidency::Resident
            } else {
                WeightResidency::Streaming
            };
            Some(MetalQwen3::from_model_dir_with_residency(
                path,
                false,
                config.max_context,
                draft_residency,
            )?)
        }
        None => None,
    };
    if let Some(draft) = draft_model.as_mut() {
        let retain_layers = std::env::var("SI_DRAFT_RETAIN_LAYERS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| "SI_DRAFT_RETAIN_LAYERS must be an integer".to_owned())
            })
            .transpose()?;
        if let Some(retain_layers) = retain_layers {
            draft.set_retain_layers(retain_layers)?;
        }
    }
    model.set_stream_chunk_rows(config.chunk_rows)?;
    model.set_retain_output_head(config.retain_output_head)?;
    model.set_retain_layers(config.retain_layers)?;
    if let Some(path) = lookahead
        .as_ref()
        .and_then(|value| value.sidecar.as_deref())
    {
        model.set_resident_sidecar(path)?;
    }
    if std::env::var_os("SI_EXACT_HEAD").is_some() {
        if config.quality_fixture.is_some() {
            return Err("SI_EXACT_HEAD is greedy-only and cannot run the quality suite".into());
        }
        if config.retain_output_head || residency == WeightResidency::Resident {
            return Err(
                "SI_EXACT_HEAD requires streaming weights without private output-head retention"
                    .into(),
            );
        }
        model.set_exact_head_search(true)?;
    }
    model.context.sample_allocated();
    let mut memory = ProcessMemorySampler::new();
    memory.sample()?;
    let profile_resources = std::env::var_os("SI_PROFILE_RESOURCES").is_some();
    let prompt_tokens = model.tokenizer.encode(&config.prompt)?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenization produced no tokens".into());
    }
    if prompt_tokens
        .len()
        .saturating_add(config.max_tokens as usize)
        > config.max_context
    {
        return Err("prompt plus generation exceeds context capacity".into());
    }
    let warmup_resources_before = if profile_resources {
        Some(sample_resources()?)
    } else {
        None
    };
    let warmup_stats_before = if profile_resources {
        Some(model.context.command_stats())
    } else {
        None
    };
    for _ in 0..config.warmup {
        let logits = model.prefill(&prompt_tokens)?;
        if let Some(draft) = draft_model.as_mut() {
            draft.prefill(&prompt_tokens)?;
        }
        if let Some(layers) = lookahead.as_ref().and_then(|value| value.draft_layers) {
            model.prepare_partial_draft(&prompt_tokens, layers)?;
        }
        model.context.sample_allocated();
        memory.sample()?;
        let _ = decode_with_schedule(
            &mut model,
            logits,
            &prompt_tokens,
            prompt_tokens.len(),
            config.max_tokens as usize,
            &mut memory,
            lookahead.clone(),
            draft_model.as_mut(),
        )?;
    }
    if profile_resources {
        let resources_after = sample_resources()?;
        let resources_before =
            warmup_resources_before.ok_or("resource profiling warmup snapshot was not captured")?;
        let resource_delta = resources_after.delta_since(resources_before);
        let stats_after = model.context.command_stats();
        let stats_before = warmup_stats_before
            .ok_or("resource profiling warmup Metal snapshot was not captured")?;
        let stats_delta = stats_after.delta_since(stats_before);
        eprintln!(
            "si_warmup_resources runs={} rss_before_mib={} rss_after_mib={} minor_faults={} major_faults={} voluntary_cs={} involuntary_cs={} metal_submitted={} metal_waited={} metal_wait_ms={:.3}",
            config.warmup,
            resources_before.rss_bytes / (1024 * 1024),
            resource_delta.rss_bytes / (1024 * 1024),
            resource_delta.minor_page_faults,
            resource_delta.major_page_faults,
            resource_delta.voluntary_context_switches,
            resource_delta.involuntary_context_switches,
            stats_delta.submitted,
            stats_delta.waited,
            stats_delta.wait_nanos as f64 / 1_000_000.0,
        );
    }

    let mut measured = Vec::with_capacity(config.repetitions as usize);
    for repetition in 0..config.repetitions {
        let resources_before = if profile_resources {
            Some(sample_resources()?)
        } else {
            None
        };
        let stats_before = if profile_resources {
            Some(model.context.command_stats())
        } else {
            None
        };
        model.context.reset_peaks();
        memory.reset()?;
        let prefill_start = std::time::Instant::now();
        let logits = model.prefill(&prompt_tokens)?;
        if let Some(draft) = draft_model.as_mut() {
            draft.prefill(&prompt_tokens)?;
        }
        if let Some(layers) = lookahead.as_ref().and_then(|value| value.draft_layers) {
            model.prepare_partial_draft(&prompt_tokens, layers)?;
        }
        model.context.sample_allocated();
        memory.sample()?;
        let prefill = prefill_start.elapsed();
        let decode_start = std::time::Instant::now();
        let generated_token_ids = decode_with_schedule(
            &mut model,
            logits,
            &prompt_tokens,
            prompt_tokens.len(),
            config.max_tokens as usize,
            &mut memory,
            lookahead.clone(),
            draft_model.as_mut(),
        )?;
        let decode = decode_start.elapsed();
        let generated_output = model.tokenizer.decode(&generated_token_ids)?;
        if profile_resources {
            let resources_after = sample_resources()?;
            let resources_before = resources_before
                .ok_or("resource profiling repetition snapshot was not captured")?;
            let resource_delta = resources_after.delta_since(resources_before);
            let stats_after = model.context.command_stats();
            let stats_before = stats_before
                .ok_or("resource profiling repetition Metal snapshot was not captured")?;
            let stats_delta = stats_after.delta_since(stats_before);
            eprintln!(
                "si_resource repetition={} rss_before_mib={} rss_after_mib={} peak_rss_mib={} minor_faults={} major_faults={} voluntary_cs={} involuntary_cs={} metal_submitted={} metal_waited={} metal_wait_ms={:.3}",
                repetition + 1,
                resources_before.rss_bytes / (1024 * 1024),
                resources_after.rss_bytes / (1024 * 1024),
                memory.peak_rss_bytes() / (1024 * 1024),
                resource_delta.minor_page_faults,
                resource_delta.major_page_faults,
                resource_delta.voluntary_context_switches,
                resource_delta.involuntary_context_switches,
                stats_delta.submitted,
                stats_delta.waited,
                stats_delta.wait_nanos as f64 / 1_000_000.0,
            );
        }
        measured.push(MeasuredRun {
            prefill,
            decode,
            peak_vram_mib: model.context.peak_allocated_bytes() / (1024 * 1024),
            peak_ram_mib: memory.peak_rss_bytes() / (1024 * 1024),
            active_weight_bytes: model.context.peak_active_weight_bytes(),
            kv_bytes: model.context.peak_kv_bytes().max(model.kv_cache_bytes()),
            scratch_bytes: model.context.peak_scratch_bytes(),
            generated_output,
            generated_token_ids,
        });
    }
    let prefill = median_duration(measured.iter().map(|run| run.prefill));
    let decode = median_duration(measured.iter().map(|run| run.decode));
    let peak_vram_mib = measured
        .iter()
        .map(|run| run.peak_vram_mib)
        .max()
        .unwrap_or(0);
    let peak_ram_mib = measured
        .iter()
        .map(|run| run.peak_ram_mib)
        .max()
        .unwrap_or(0);
    let median_peak_vram_mib = median_u64(measured.iter().map(|run| run.peak_vram_mib));
    let median_peak_ram_mib = median_u64(measured.iter().map(|run| run.peak_ram_mib));
    if let Some(tree) = lookahead.as_ref().and_then(|value| value.tree.as_ref()) {
        if peak_vram_mib > tree.memory_ceiling_mib || peak_ram_mib > tree.memory_ceiling_mib {
            return Err(format!(
                "SI tree memory ceiling exceeded: peak Metal={} MiB, peak RSS={} MiB, ceiling={} MiB",
                peak_vram_mib, peak_ram_mib, tree.memory_ceiling_mib
            ));
        }
    }
    if std::env::var_os("SI_EXACT_HEAD").is_some()
        && (peak_vram_mib > 2_000 || peak_ram_mib > 2_000)
    {
        return Err(format!(
            "SI exact-head memory ceiling exceeded: peak Metal={} MiB, peak RSS={} MiB, ceiling=2000 MiB",
            peak_vram_mib, peak_ram_mib
        ));
    }
    let active_weight_bytes = measured
        .iter()
        .map(|run| run.active_weight_bytes)
        .max()
        .unwrap_or(0);
    let kv_bytes = measured.iter().map(|run| run.kv_bytes).max().unwrap_or(0);
    let scratch_bytes = measured
        .iter()
        .map(|run| run.scratch_bytes)
        .max()
        .unwrap_or(0);
    let first = measured
        .first()
        .ok_or("no measured repetitions were completed")?;
    let repetition_outputs_match = measured.iter().all(|run| {
        run.generated_token_ids == first.generated_token_ids
            && run.generated_output == first.generated_output
    });
    let generated_output = first.generated_output.clone();
    let generated_token_ids = first.generated_token_ids.clone();
    let quality = if let Some(path) = &config.quality_fixture {
        let suite = QualitySuite::load(path).map_err(|error| error.0)?;
        Some(model.evaluate_quality(&suite)?)
    } else {
        None
    };
    if std::env::var_os("SI_PROFILE_METAL").is_some() {
        let stats = model.context.command_stats();
        eprintln!(
            "si_metal_stats submitted={} async_submitted={} waited={} wait_ms={:.3}",
            stats.submitted,
            stats.async_submitted,
            stats.waited,
            stats.wait_nanos as f64 / 1_000_000.0,
        );
    }
    let device = model.device_info()?;
    Ok(ResidentReport {
        model_path: config.model_path.clone(),
        model_revision: model.store.manifest.revision.clone(),
        backend: config.backend.clone(),
        device: device.name,
        recommended_working_set_mib: device.recommended_max_working_set_bytes / (1024 * 1024),
        mapped_model_bytes: model.store.mapped_bytes() as u64,
        active_weight_bytes,
        kv_bytes,
        scratch_bytes,
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_token_ids.len(),
        warmup_runs: config.warmup,
        repetitions: config.repetitions,
        prefill,
        decode,
        peak_vram_mib: Some(peak_vram_mib),
        peak_ram_mib: Some(peak_ram_mib),
        median_peak_vram_mib: Some(median_peak_vram_mib),
        median_peak_ram_mib: Some(median_peak_ram_mib),
        output_match: config
            .expected_output
            .as_ref()
            .map(|expected| expected == &generated_output),
        repetition_outputs_match,
        quality,
        generated_output,
        generated_token_ids,
    })
}

/// Run the canonical `si-bench` measurement loop against a Qwen3.6 GGUF.
///
/// The Safetensors path above is intentionally left unchanged: its model
/// layout, manifest checks, and canonical 4B benchmark are the reference SI
/// implementation.  GGUF uses the same warmup/repetition/timing/reporting
/// contract, but supplies the Qwen3.6 hybrid decoder and its mmap-backed
/// quantized tensors through `Qwen35Runtime`.
#[cfg(target_os = "macos")]
pub fn run_gguf_resident(config: &crate::Config) -> Result<ResidentReport, String> {
    if config.repetitions == 0 {
        return Err("measured repetitions must be non-zero".into());
    }
    if config.quality_fixture.is_some() {
        return Err(
            "the pinned quality fixture belongs to the Qwen3-4B Safetensors model; omit --quality-fixture for the GGUF canonical run"
                .into(),
        );
    }
    if config.chunk_rows.is_some() {
        return Err("GGUF canonical execution does not accept --chunk-rows".into());
    }

    let mut store = GgufModelStore::open(&config.model_path).map_err(|error| error.to_string())?;
    let model_config = store.qwen35_config().map_err(|error| error.to_string())?;
    if config.retain_layers > model_config.num_hidden_layers {
        return Err(format!(
            "requested {} retained layers exceeds GGUF model depth {}",
            config.retain_layers, model_config.num_hidden_layers
        ));
    }
    if config.max_context > model_config.context_length {
        return Err(format!(
            "requested context {} exceeds GGUF model capacity {}",
            config.max_context, model_config.context_length
        ));
    }
    if let Some(count) = std::env::var("SI_HOST_RETAIN_LAYERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
    {
        let cached_bytes = store
            .retain_prefix_layers(count)
            .map_err(|error| error.to_string())?;
        if std::env::var_os("SI_PROFILE_RESOURCES").is_some() {
            eprintln!(
                "si_gguf_host_cache retained_layers={} bytes={}",
                count, cached_bytes
            );
        }
    }
    let tokenizer = QwenTokenizer::from_gguf_model(&config.model_path)?;
    let prompt_tokens = tokenizer.encode(&config.prompt)?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenization produced no tokens".into());
    }
    if prompt_tokens
        .len()
        .saturating_add(config.max_tokens as usize)
        > config.max_context
    {
        return Err("prompt plus generation exceeds context capacity".into());
    }

    let context = MetalContext::new()?;
    let mut model = Qwen35Runtime::new_with_retained_layers_and_head(
        &context,
        &store,
        config.max_context,
        config.retain_layers,
        config.retain_output_head,
    )?;
    if config.retain_layers > 0 && std::env::var_os("SI_PROFILE_RESOURCES").is_some() {
        eprintln!(
            "si_gguf_resident retained_layers={} private_quant_bytes={}",
            config.retain_layers,
            model.retained_weight_bytes()
        );
    }
    let mut memory = ProcessMemorySampler::new();
    context.sample_allocated();
    memory.sample()?;

    for _ in 0..config.warmup {
        model.reset();
        let _ = run_gguf_generation(
            &mut model,
            &tokenizer,
            &prompt_tokens,
            config.max_tokens as usize,
            &context,
            &mut memory,
        )?;
    }

    let mut measured = Vec::with_capacity(config.repetitions as usize);
    for _ in 0..config.repetitions {
        model.reset();
        context.reset_peaks();
        memory.reset()?;
        let (prefill, decode, generated_token_ids, generated_output) = run_gguf_generation(
            &mut model,
            &tokenizer,
            &prompt_tokens,
            config.max_tokens as usize,
            &context,
            &mut memory,
        )?;
        measured.push(MeasuredRun {
            prefill,
            decode,
            peak_vram_mib: context.peak_allocated_bytes() / (1024 * 1024),
            peak_ram_mib: memory.peak_rss_bytes() / (1024 * 1024),
            active_weight_bytes: context.peak_active_weight_bytes(),
            kv_bytes: (model.state_bytes() as u64).saturating_add(context.peak_kv_bytes()),
            scratch_bytes: context.peak_scratch_bytes(),
            generated_output,
            generated_token_ids,
        });
    }

    let first = measured
        .first()
        .ok_or("no measured repetitions were completed")?;
    let repetition_outputs_match = measured.iter().all(|run| {
        run.generated_token_ids == first.generated_token_ids
            && run.generated_output == first.generated_output
    });
    let generated_output = first.generated_output.clone();
    let generated_token_ids = first.generated_token_ids.clone();
    let device = crate::metal::probe()?;
    let model_revision = store
        .metadata_string("general.name")
        .unwrap_or("qwen3.6-27b-q4-gguf")
        .to_owned();

    Ok(ResidentReport {
        model_path: config.model_path.clone(),
        model_revision,
        backend: config.backend.clone(),
        device: device.name,
        recommended_working_set_mib: device.recommended_max_working_set_bytes / (1024 * 1024),
        mapped_model_bytes: store.mapped_bytes() as u64,
        active_weight_bytes: measured
            .iter()
            .map(|run| run.active_weight_bytes)
            .max()
            .unwrap_or(0),
        kv_bytes: measured.iter().map(|run| run.kv_bytes).max().unwrap_or(0),
        scratch_bytes: measured
            .iter()
            .map(|run| run.scratch_bytes)
            .max()
            .unwrap_or(0),
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_token_ids.len(),
        warmup_runs: config.warmup,
        repetitions: config.repetitions,
        prefill: median_duration(measured.iter().map(|run| run.prefill)),
        decode: median_duration(measured.iter().map(|run| run.decode)),
        peak_vram_mib: Some(
            measured
                .iter()
                .map(|run| run.peak_vram_mib)
                .max()
                .unwrap_or(0),
        ),
        peak_ram_mib: Some(
            measured
                .iter()
                .map(|run| run.peak_ram_mib)
                .max()
                .unwrap_or(0),
        ),
        median_peak_vram_mib: Some(median_u64(measured.iter().map(|run| run.peak_vram_mib))),
        median_peak_ram_mib: Some(median_u64(measured.iter().map(|run| run.peak_ram_mib))),
        output_match: config
            .expected_output
            .as_ref()
            .map(|expected| expected == &generated_output),
        repetition_outputs_match,
        quality: None,
        generated_output,
        generated_token_ids,
    })
}

#[cfg(target_os = "macos")]
fn run_gguf_generation(
    model: &mut Qwen35Runtime<'_>,
    tokenizer: &QwenTokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    context: &MetalContext,
    memory: &mut ProcessMemorySampler,
) -> Result<(std::time::Duration, std::time::Duration, Vec<u32>, String), String> {
    let stats_before = context.command_stats();
    let prefill_start = std::time::Instant::now();
    let mut hidden = Vec::new();
    for (position, token_id) in prompt_tokens.iter().copied().enumerate() {
        let embedding = model.embed_token(token_id)?;
        hidden = model.decode_hidden(position, &embedding)?;
        context.sample_allocated();
        memory.sample()?;
    }
    let mut logits = model.logits(&hidden)?;
    context.sample_allocated();
    memory.sample()?;
    let prefill = prefill_start.elapsed();

    let decode_start = std::time::Instant::now();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let eos_token_id = model
        .config()
        .map_err(|error| error.to_string())?
        .eos_token_id;
    for index in 0..max_new_tokens {
        let token = argmax(&logits)? as u32;
        generated.push(token);
        if token == eos_token_id {
            break;
        }
        // Match the Safetensors canonical loop exactly: every non-EOS output
        // token, including the last requested token, advances the state once.
        let (_, next_logits) = model.decode_token(token, prompt_tokens.len() + index)?;
        logits = next_logits;
        context.sample_allocated();
        memory.sample()?;
    }
    let decode = decode_start.elapsed();
    let text = tokenizer.decode(&generated)?;
    if std::env::var_os("SI_PROFILE_QWEN35").is_some() {
        let stats = context.command_stats().delta_since(stats_before);
        eprintln!(
            "si_qwen35_commands submitted={} waited={} async={} wait_ms={:.3}",
            stats.submitted,
            stats.waited,
            stats.async_submitted,
            stats.wait_nanos as f64 / 1_000_000.0
        );
    }
    Ok((prefill, decode, generated, text))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn decode_with_schedule(
    model: &mut MetalQwen3,
    logits: Vec<f32>,
    initial_tokens: &[u32],
    prompt_len: usize,
    max_new_tokens: usize,
    memory: &mut ProcessMemorySampler,
    lookahead: Option<LookaheadConfig>,
    draft_model: Option<&mut MetalQwen3>,
) -> Result<Vec<u32>, String> {
    match lookahead {
        Some(config) => decode_lookahead(
            model,
            logits,
            initial_tokens,
            prompt_len,
            max_new_tokens,
            memory,
            config,
            draft_model,
        ),
        None => decode_greedy(model, logits, prompt_len, max_new_tokens, memory),
    }
}

#[cfg(target_os = "macos")]
fn decode_greedy(
    model: &mut MetalQwen3,
    mut logits: Vec<f32>,
    prompt_len: usize,
    max_new_tokens: usize,
    memory: &mut ProcessMemorySampler,
) -> Result<Vec<u32>, String> {
    let mut generated = Vec::with_capacity(max_new_tokens);
    for index in 0..max_new_tokens {
        let token = argmax(&logits)?;
        generated.push(token as u32);
        if token as u32 == model.config().eos_token_id {
            break;
        }
        logits = model.forward_token(token, prompt_len + index)?;
        model.context.sample_allocated();
        memory.sample()?;
    }
    Ok(generated)
}

#[cfg(target_os = "macos")]
fn next_dynamic_width(configured_width: usize, current_width: usize, accepted: usize) -> usize {
    if accepted <= 1 {
        2.min(configured_width)
    } else if accepted < current_width {
        (current_width / 2).max(2).min(configured_width)
    } else {
        current_width.saturating_mul(2).max(2).min(configured_width)
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn decode_lookahead(
    model: &mut MetalQwen3,
    mut previous_logits: Vec<f32>,
    initial_tokens: &[u32],
    prompt_len: usize,
    max_new_tokens: usize,
    memory: &mut ProcessMemorySampler,
    config: LookaheadConfig,
    mut draft_model: Option<&mut MetalQwen3>,
) -> Result<Vec<u32>, String> {
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut history = initial_tokens.to_vec();
    let mut position = prompt_len;
    let mut sidecar_enabled = config.sidecar.is_some();
    let mut dynamic_width = config.width;
    while generated.len() < max_new_tokens {
        if config.sidecar_auto && !sidecar_enabled {
            let token = argmax(&previous_logits)?;
            generated.push(token as u32);
            history.push(token as u32);
            if generated.len() == max_new_tokens || token as u32 == model.config().eos_token_id {
                break;
            }
            previous_logits = model.forward_token(token, position)?;
            position += 1;
            model.context.sample_allocated();
            memory.sample()?;
            continue;
        }
        let width = dynamic_width.min(model.max_context().saturating_sub(position));
        if width < 2 {
            let token = argmax(&previous_logits)?;
            generated.push(token as u32);
            history.push(token as u32);
            if token as u32 == model.config().eos_token_id {
                break;
            }
            previous_logits = model.forward_token(token, position)?;
            position += 1;
            model.context.sample_allocated();
            memory.sample()?;
            continue;
        }

        let (candidates, verification) = if let Some(tree) = &config.tree {
            let first = argmax(&previous_logits)? as u32;
            let branches = tree_candidate_branches(
                &history,
                first,
                tree.branches,
                tree.depth.min(width),
                tree.ngram_order,
            );
            if branches.iter().all(|branch| branch == &branches[0]) {
                let candidates = ngram_candidates(&history, first, width, tree.ngram_order);
                if candidates.len() < 2 {
                    generated.push(first);
                    history.push(first);
                    if generated.len() == max_new_tokens || first == model.config().eos_token_id {
                        return Ok(generated);
                    }
                    previous_logits = model.forward_token(first as usize, position)?;
                    position += 1;
                    model.context.sample_allocated();
                    memory.sample()?;
                    continue;
                }
                let verification = model.verify_many(&previous_logits, &candidates, position)?;
                (candidates, verification)
            } else {
                let tree_verification = model.lookahead_tree_step(
                    &previous_logits,
                    &branches,
                    position,
                    config.iterations,
                )?;
                let candidates =
                    tree_verification.candidates[tree_verification.selected_branch].clone();
                (candidates, tree_verification.verification)
            }
        } else if let Some(order) = config.ngram_order {
            let first = argmax(&previous_logits)? as u32;
            let candidates = ngram_candidates(&history, first, width, order);
            if candidates.len() < 2 {
                generated.push(first);
                history.push(first);
                if generated.len() == max_new_tokens || first == model.config().eos_token_id {
                    return Ok(generated);
                }
                previous_logits = model.forward_token(first as usize, position)?;
                position += 1;
                model.context.sample_allocated();
                memory.sample()?;
                continue;
            }
            let verification = model.verify_many(&previous_logits, &candidates, position)?;
            (candidates, verification)
        } else if let Some(draft) = draft_model.as_deref_mut() {
            let candidates = draft.draft_candidates(&previous_logits, position, width)?;
            let verification = model.verify_many(&previous_logits, &candidates, position)?;
            (candidates, verification)
        } else if sidecar_enabled && config.sidecar_tree && width >= 8 {
            let tree = model.resident_sidecar_tree_step(
                &previous_logits,
                position,
                config
                    .draft_layers
                    .ok_or("resident sidecar requires retained draft layers")?,
            )?;
            let candidates = tree.candidates[tree.selected_branch].clone();
            (candidates, tree.verification)
        } else if sidecar_enabled && config.sidecar.is_some() {
            let layers = config
                .draft_layers
                .ok_or("resident sidecar requires retained draft layers")?;
            let candidates = model.resident_sidecar_candidates(
                &previous_logits,
                position,
                width,
                layers,
                config.sidecar_min_margin,
            )?;
            let Some(candidates) = candidates else {
                let token = argmax(&previous_logits)?;
                generated.push(token as u32);
                history.push(token as u32);
                if generated.len() == max_new_tokens || token as u32 == model.config().eos_token_id
                {
                    break;
                }
                previous_logits = model.forward_token(token, position)?;
                position += 1;
                model.context.sample_allocated();
                memory.sample()?;
                continue;
            };
            let verification = model.verify_many(&previous_logits, &candidates, position)?;
            if verification.accepted_tokens < candidates.len() {
                model.truncate_partial_draft(position + verification.accepted_tokens)?;
            }
            (candidates, verification)
        } else if config.sidecar.is_none() {
            let layers = config
                .draft_layers
                .ok_or("partial draft requires retained draft layers")?;
            let candidates =
                model.partial_draft_candidates(&previous_logits, position, width, layers)?;
            let verification = model.verify_many(&previous_logits, &candidates, position)?;
            if verification.accepted_tokens < candidates.len() {
                model.truncate_partial_draft(position + verification.accepted_tokens)?;
            }
            (candidates, verification)
        } else {
            let step =
                model.lookahead_step(&previous_logits, position, width, config.iterations)?;
            (step.candidates, step.verification)
        };
        let accepted = verification.accepted_tokens.min(width);
        if std::env::var_os("SI_PROFILE_LOOKAHEAD").is_some() {
            eprintln!(
                "si_lookahead position={} width={} accepted={} next_token={}",
                position, width, accepted, verification.next_token
            );
        }
        if config.sidecar_auto && sidecar_enabled && accepted <= 1 {
            sidecar_enabled = false;
            if std::env::var_os("SI_PROFILE_LOOKAHEAD").is_some() {
                eprintln!("si_sidecar_auto fallback=canonical accepted={accepted}");
            }
        }
        if config.dynamic_width {
            dynamic_width = next_dynamic_width(config.width, width, accepted);
            if std::env::var_os("SI_PROFILE_LOOKAHEAD").is_some() {
                eprintln!("si_lookahead next_width={dynamic_width}");
            }
        }
        if accepted == 0 {
            let correction = verification.next_token;
            generated.push(correction);
            history.push(correction);
            if generated.len() == max_new_tokens || correction == model.config().eos_token_id {
                return Ok(generated);
            }
            previous_logits = model.forward_token(correction as usize, position)?;
            if let Some(draft) = draft_model.as_deref_mut() {
                draft.truncate_cache_to(position)?;
                let _ = draft.forward_token(correction as usize, position)?;
            }
            if config.sidecar.is_some() || config.draft_layers.is_some() {
                model.truncate_partial_draft(position)?;
                let _ = model.partial_draft_hidden(
                    correction as usize,
                    position,
                    config
                        .draft_layers
                        .ok_or("resident sidecar requires retained draft layers")?,
                )?;
            }
            position += 1;
            model.context.sample_allocated();
            memory.sample()?;
            continue;
        }
        for token in candidates.iter().take(accepted) {
            generated.push(*token);
            history.push(*token);
            if generated.len() == max_new_tokens || *token == model.config().eos_token_id {
                return Ok(generated);
            }
        }
        position += accepted;
        if accepted < width {
            // The verifier already computed the exact greedy correction after
            // the accepted prefix. Commit it immediately, rather than paying
            // for another wide candidate pass just to rediscover this token.
            let correction = verification.next_token;
            generated.push(correction);
            history.push(correction);
            if generated.len() == max_new_tokens || correction == model.config().eos_token_id {
                return Ok(generated);
            }
            previous_logits = model.forward_token(correction as usize, position)?;
            if let Some(draft) = draft_model.as_deref_mut() {
                draft.truncate_cache_to(position)?;
                let _ = draft.forward_token(correction as usize, position)?;
            }
            if config.sidecar.is_some() || config.draft_layers.is_some() {
                model.truncate_partial_draft(position)?;
                let _ = model.partial_draft_hidden(
                    correction as usize,
                    position,
                    config
                        .draft_layers
                        .ok_or("resident sidecar requires retained draft layers")?,
                )?;
            }
            position += 1;
        } else {
            // Candidate generation intentionally stops one prefix step short
            // of the window. If the verifier accepts the whole window, the
            // final accepted token is now needed as the starting state for
            // the next draft window, so advance the disposable prefix cache
            // exactly once on this path.
            if accepted == width && (config.sidecar.is_some() || config.draft_layers.is_some()) {
                let layers = config
                    .draft_layers
                    .ok_or("resident sidecar requires retained draft layers")?;
                let final_token = *candidates
                    .last()
                    .ok_or("lookahead verifier returned no candidates")?;
                model.partial_draft_hidden(final_token as usize, position - 1, layers)?;
            }
            previous_logits = verification
                .target_logits
                .last()
                .ok_or("lookahead verifier returned no target logits")?
                .clone();
        }
        model.context.sample_allocated();
        memory.sample()?;
    }
    Ok(generated)
}

#[cfg(target_os = "macos")]
fn bf16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
    f32::from_bits(bits << 16)
}

#[cfg(target_os = "macos")]
fn rms_norm_bf16_cpu(
    input: &[f32],
    weight_bytes: &[u8],
    heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if heads == 0
        || head_dim == 0
        || input.len() != heads * head_dim
        || weight_bytes.len() != head_dim * 2
    {
        return Err("CPU BF16 RMSNorm dimensions are invalid".into());
    }
    let mut output = vec![0.0_f32; input.len()];
    for head in 0..heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let sum = input[start..end]
            .iter()
            .fold(0.0_f32, |sum, value| sum + value * value);
        let inverse_rms = (sum / head_dim as f32 + epsilon).sqrt().recip();
        for index in 0..head_dim {
            output[start + index] = input[start + index]
                * inverse_rms
                * bf16_to_f32(&weight_bytes[index * 2..index * 2 + 2]);
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn rope_cpu(
    input: &[f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<Vec<f32>, String> {
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) || input.len() != heads * head_dim
    {
        return Err("CPU RoPE dimensions are invalid".into());
    }
    let mut output = input.to_vec();
    for head in 0..heads {
        for pair in 0..head_dim / 2 {
            let exponent = 2.0_f32 * pair as f32 / head_dim as f32;
            let angle = position as f32 * theta.powf(-exponent);
            let (sine, cosine) = angle.sin_cos();
            let offset = head * head_dim + pair * 2;
            let first = input[offset];
            let second = input[offset + 1];
            output[offset] = first * cosine - second * sine;
            output[offset + 1] = first * sine + second * cosine;
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn attention_decode_cpu(
    query: &[f32],
    cache: &KvCache,
    new_keys: &[f32],
    new_values: &[f32],
    query_heads: usize,
) -> Result<Vec<f32>, String> {
    let key_value_heads = cache.key_value_heads();
    let head_dim = cache.head_dim();
    if key_value_heads == 0
        || query_heads == 0
        || !query_heads.is_multiple_of(key_value_heads)
        || query.len() != query_heads * head_dim
        || new_keys.len() != key_value_heads * head_dim
        || new_values.len() != key_value_heads * head_dim
    {
        return Err("CPU attention dimensions are invalid".into());
    }
    let cached_tokens = cache.cached_tokens();
    let capacity = cache.capacity_tokens();
    let group_size = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0_f32; query.len()];
    for query_head in 0..query_heads {
        let key_value_head = query_head / group_size;
        let query_start = query_head * head_dim;
        let query_slice = &query[query_start..query_start + head_dim];
        let mut scores = Vec::with_capacity(cached_tokens + 1);
        for token in 0..=cached_tokens {
            let key_slice = if token < cached_tokens {
                let start = (key_value_head * capacity + token) * head_dim;
                &cache.key_storage()[start..start + head_dim]
            } else {
                let start = key_value_head * head_dim;
                &new_keys[start..start + head_dim]
            };
            let score = query_slice
                .iter()
                .zip(key_slice)
                .fold(0.0_f32, |sum, (query_value, key_value)| {
                    sum + query_value * key_value
                })
                * scale;
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut normalizer = 0.0_f32;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            normalizer += *score;
        }
        let output_slice = &mut output[query_start..query_start + head_dim];
        for (token, score) in scores.iter().enumerate() {
            let value_slice = if token < cached_tokens {
                let start = (key_value_head * capacity + token) * head_dim;
                &cache.value_storage()[start..start + head_dim]
            } else {
                let start = key_value_head * head_dim;
                &new_values[start..start + head_dim]
            };
            let weight = *score / normalizer;
            for (output_value, value) in output_slice.iter_mut().zip(value_slice) {
                *output_value += weight * value;
            }
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn median_duration<I>(values: I) -> std::time::Duration
where
    I: Iterator<Item = std::time::Duration>,
{
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return std::time::Duration::ZERO;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn median_u64<I>(values: I) -> u64
where
    I: Iterator<Item = u64>,
{
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn rate(tokens: usize, duration: std::time::Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

#[cfg(target_os = "macos")]
fn option_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

#[cfg(target_os = "macos")]
fn option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

#[cfg(target_os = "macos")]
fn load_resident_weights(
    context: &MetalContext,
    store: &ModelStore,
    config: &ModelConfig,
    private: bool,
) -> Result<ResidentWeights, String> {
    let mut names = vec!["model.embed_tokens.weight".to_owned()];
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{prefix}.{suffix}"));
        }
    }
    let mut matrices = HashMap::with_capacity(names.len());
    let mut resident_bytes = 0_u64;
    for name in names {
        let tensor = store.tensor(&name).map_err(|error| error.0)?;
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err(format!(
                "resident weight {name} is not a rank-2 BF16 tensor"
            ));
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        let weight = if private {
            context.upload_bf16_weight_private(tensor.bytes)?
        } else {
            context.upload_bf16_weight(tensor.bytes)?
        };
        resident_bytes = resident_bytes.saturating_add(weight.byte_len());
        matrices.insert(
            name,
            ResidentMatrix {
                rows,
                columns,
                weight,
            },
        );
    }
    context.note_resident_weight_bytes(resident_bytes);
    Ok(ResidentWeights { matrices })
}

#[cfg(target_os = "macos")]
fn load_output_head(context: &MetalContext, store: &ModelStore) -> Result<ResidentWeights, String> {
    let name = "model.embed_tokens.weight".to_owned();
    let tensor = store.tensor(&name).map_err(|error| error.0)?;
    if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
        return Err("output head must be a rank-2 BF16 tensor".into());
    }
    let rows = tensor.info.shape[0];
    let columns = tensor.info.shape[1];
    let weight = context.upload_bf16_weight_private(tensor.bytes)?;
    context.note_persistent_weight_bytes(weight.byte_len());
    let mut matrices = HashMap::new();
    matrices.insert(
        name,
        ResidentMatrix {
            rows,
            columns,
            weight,
        },
    );
    Ok(ResidentWeights { matrices })
}

#[cfg(target_os = "macos")]
fn load_retained_layers(
    context: &MetalContext,
    store: &ModelStore,
    count: usize,
) -> Result<ResidentWeights, String> {
    let mut matrices = HashMap::with_capacity(count.saturating_mul(7));
    let mut retained_bytes = 0_u64;
    for layer in 0..count {
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let name = format!("{prefix}.{suffix}");
            let tensor = store.tensor(&name).map_err(|error| error.0)?;
            if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
                return Err(format!(
                    "retained weight {name} is not a rank-2 BF16 tensor"
                ));
            }
            let weight = context.upload_bf16_weight_private(tensor.bytes)?;
            retained_bytes = retained_bytes.saturating_add(weight.byte_len());
            matrices.insert(
                name,
                ResidentMatrix {
                    rows: tensor.info.shape[0],
                    columns: tensor.info.shape[1],
                    weight,
                },
            );
        }
    }
    context.note_persistent_weight_bytes(retained_bytes);
    Ok(ResidentWeights { matrices })
}

#[cfg(target_os = "macos")]
fn validate_required_tensors(store: &ModelStore, config: &ModelConfig) -> Result<(), String> {
    let required = [
        "model.embed_tokens.weight".to_owned(),
        "model.norm.weight".to_owned(),
    ];
    for name in required {
        store.tensor(&name).map_err(|error| error.0)?;
    }
    for layer in 0..config.num_hidden_layers {
        for suffix in QWEN3_LAYER_TENSOR_SUFFIXES {
            let prefix = format!("model.layers.{layer}");
            store
                .tensor(&format!("{prefix}.{suffix}"))
                .map_err(|error| error.0)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return Err("residual vectors have different lengths".into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

#[cfg(target_os = "macos")]
fn flatten_batch_inputs(inputs: &[Vec<f32>]) -> Result<(usize, usize, Vec<f32>), String> {
    let first = inputs
        .first()
        .ok_or("batched projection requires at least one input")?;
    let columns = first.len();
    if columns == 0 {
        return Err("batched projection inputs must be non-empty".into());
    }
    if inputs.iter().any(|input| input.len() != columns) {
        return Err("batched projection inputs must have equal lengths".into());
    }
    let batch = inputs.len();
    if batch > 8 {
        return Err("batched projection supports at most eight inputs".into());
    }
    Ok((
        batch,
        columns,
        inputs.iter().flatten().copied().collect::<Vec<_>>(),
    ))
}

#[cfg(target_os = "macos")]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(target_os = "macos")]
fn argmax(values: &[f32]) -> Result<usize, String> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or_else(|| "logits contain no finite values".into())
}

#[cfg(target_os = "macos")]
fn assess_verification(
    previous_logits: &[f32],
    candidate_tokens: &[u32],
    target_logits: Vec<Vec<f32>>,
) -> Result<Verification, String> {
    if candidate_tokens.is_empty() {
        return Err("verification requires at least one candidate token".into());
    }
    let expected_first = argmax(previous_logits)? as u32;
    if candidate_tokens[0] != expected_first {
        return Ok(Verification {
            accepted_tokens: 0,
            next_token: expected_first,
            target_logits: Vec::new(),
        });
    }
    if target_logits.len() != candidate_tokens.len() {
        return Err("verification logits do not cover every candidate token".into());
    }
    let mut accepted_tokens = 1;
    for index in 0..candidate_tokens.len().saturating_sub(1) {
        let predicted = argmax(&target_logits[index])? as u32;
        if predicted != candidate_tokens[index + 1] {
            return Ok(Verification {
                accepted_tokens,
                next_token: predicted,
                target_logits,
            });
        }
        accepted_tokens += 1;
    }
    let next_token = argmax(
        target_logits
            .last()
            .ok_or("verification target logits are empty")?,
    )? as u32;
    Ok(Verification {
        accepted_tokens,
        next_token,
        target_logits,
    })
}

#[cfg(target_os = "macos")]
fn update_lookahead_candidates(
    candidates: &[u32],
    target_logits: &[Vec<f32>],
) -> Result<Vec<u32>, String> {
    if candidates.len() < 2 || candidates.len() > 8 {
        return Err("lookahead candidate width must be between two and eight".into());
    }
    if target_logits.len() != candidates.len() {
        return Err("lookahead logits must cover every candidate position".into());
    }
    let mut next = candidates.to_vec();
    for index in 0..candidates.len() - 1 {
        next[index + 1] = argmax(&target_logits[index])? as u32;
    }
    Ok(next)
}

#[cfg(target_os = "macos")]
fn ngram_candidates(history: &[u32], first_token: u32, width: usize, max_order: usize) -> Vec<u32> {
    if width == 0 {
        return Vec::new();
    }
    let mut sequence = history.to_vec();
    sequence.push(first_token);
    let mut candidates = vec![first_token];
    while candidates.len() < width {
        let mut next = None;
        let max_order = max_order.min(sequence.len().saturating_sub(1));
        for order in (1..=max_order).rev() {
            let suffix_start = sequence.len() - order;
            let suffix = &sequence[suffix_start..];
            for start in (0..suffix_start).rev() {
                if start + order < sequence.len() && sequence[start..start + order] == *suffix {
                    next = sequence.get(start + order).copied();
                    if next.is_some() {
                        break;
                    }
                }
            }
            if next.is_some() {
                break;
            }
        }
        let Some(next) = next else {
            break;
        };
        candidates.push(next);
        sequence.push(next);
    }
    candidates
}

#[cfg(target_os = "macos")]
fn tree_candidate_branches(
    history: &[u32],
    first_token: u32,
    branch_count: usize,
    depth: usize,
    max_order: usize,
) -> Vec<Vec<u32>> {
    (0..branch_count)
        .map(|branch| {
            let order = if branch == 0 {
                max_order
            } else if branch == 1 {
                1
            } else {
                max_order.saturating_sub(branch - 1).max(1)
            };
            let mut candidates = ngram_candidates(history, first_token, depth, order);
            let fill = candidates.last().copied().unwrap_or(first_token);
            candidates.resize(depth, fill);
            candidates
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn negative_log_likelihood(logits: &[f32], token_id: usize) -> Result<f64, String> {
    let target = *logits
        .get(token_id)
        .ok_or("likelihood target token exceeds vocabulary")?;
    if !target.is_finite() {
        return Err("target logit is not finite".into());
    }
    let maximum = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .ok_or("logits contain no finite values")?;
    let normalizer = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| f64::from(value - maximum).exp())
        .sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err("logit normalizer is invalid".into());
    }
    Ok(-(f64::from(target - maximum) - normalizer.ln()))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn flatten_batch_inputs_preserves_batch_major_layout() {
        let inputs = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        assert_eq!(
            super::flatten_batch_inputs(&inputs).expect("valid batch"),
            (2, 2, vec![1.0, 2.0, 3.0, 4.0])
        );
    }

    #[test]
    fn flatten_batch_inputs_rejects_mismatched_widths() {
        let error = super::flatten_batch_inputs(&[vec![1.0_f32], vec![2.0, 3.0]])
            .expect_err("mismatched widths must be rejected");
        assert!(error.contains("equal lengths"));
    }

    #[test]
    fn assess_verification_returns_rejection_token_and_prefix_length() {
        let verification = super::assess_verification(
            &[0.0, 4.0],
            &[1, 0, 2],
            vec![vec![3.0, 1.0], vec![0.0, 5.0], vec![8.0, 0.0]],
        )
        .expect("verification should succeed");
        assert_eq!(verification.accepted_tokens, 2);
        assert_eq!(verification.next_token, 1);
        assert_eq!(verification.target_logits.len(), 3);
    }

    #[test]
    fn lookahead_update_keeps_first_token_and_shifts_greedy_predictions() {
        let candidates = vec![7_u32, 7, 7, 7];
        let logits = vec![
            vec![0.0_f32, 3.0],
            vec![4.0, 0.0],
            vec![0.0, 5.0],
            vec![9.0, 0.0],
        ];
        assert_eq!(
            super::update_lookahead_candidates(&candidates, &logits).expect("valid update"),
            vec![7, 1, 0, 1]
        );
    }

    #[test]
    fn ngram_draft_extends_a_repeated_suffix_without_touching_target_logits() {
        let history = vec![1_u32, 2, 3, 1, 2];
        assert_eq!(super::ngram_candidates(&history, 3, 4, 2), vec![3, 1, 2, 3]);
    }

    #[test]
    fn ngram_draft_returns_only_exact_first_token_when_no_suffix_matches() {
        assert_eq!(super::ngram_candidates(&[1_u32, 2, 3], 9, 4, 3), vec![9]);
    }

    #[test]
    fn dynamic_width_shrinks_on_low_acceptance_and_grows_only_when_full() {
        assert_eq!(super::next_dynamic_width(8, 8, 1), 2);
        assert_eq!(super::next_dynamic_width(8, 8, 4), 4);
        assert_eq!(super::next_dynamic_width(8, 4, 4), 8);
        assert_eq!(super::next_dynamic_width(2, 2, 2), 2);
    }

    #[test]
    fn tree_candidates_share_first_token_and_have_bounded_depth() {
        let branches = super::tree_candidate_branches(&[1_u32, 2, 1, 2], 3, 2, 4, 3);
        assert_eq!(branches.len(), 2);
        assert!(branches.iter().all(|branch| branch.len() == 4));
        assert!(branches.iter().all(|branch| branch[0] == 3));
    }

    #[test]
    fn exact_head_index_builds_centroid_and_radius_metadata_from_bf16_rows() {
        let info = crate::model::TensorInfo {
            name: "head".into(),
            shard: "shard".into(),
            dtype: "BF16".into(),
            shape: vec![3, 2],
            data_start: 0,
            data_end: 12,
        };
        let bytes = [
            0x00, 0x3f, 0x00, 0x40, // 0.5, 2.0
            0x00, 0x3f, 0x00, 0x40, // 0.5, 2.0
            0x00, 0x40, 0x00, 0x3f, // 2.0, 0.5
        ];
        let tensor = crate::model::TensorView {
            info: &info,
            bytes: &bytes,
            backing: &bytes,
        };
        let index = super::OutputHeadIndex::build(&tensor).expect("valid BF16 head");
        assert_eq!(index.rows, 3);
        assert_eq!(index.columns, 2);
        assert_eq!(index.radii.len(), 1);
        assert!(index.radii[0] > 0.0);
        assert!((index.centroids[0] - 1.0).abs() < 0.01);
        assert!((index.centroids[1] - 1.5).abs() < 0.01);
    }

    #[test]
    fn identifies_fused_projection_groups_by_execution_order() {
        assert_eq!(
            super::fused_projection_group(&[
                "model.layers.0.self_attn.q_proj.weight".into(),
                "model.layers.0.self_attn.k_proj.weight".into(),
                "model.layers.0.self_attn.v_proj.weight".into(),
            ]),
            Some(super::FusedProjectionGroup::Qkv)
        );
        assert_eq!(
            super::fused_projection_group(&[
                "model.layers.0.mlp.gate_proj.weight".into(),
                "model.layers.0.mlp.up_proj.weight".into(),
            ]),
            Some(super::FusedProjectionGroup::GateUp)
        );
    }

    #[test]
    fn resident_sidecar_returns_descending_proposals() {
        let sidecar = super::ResidentSidecar {
            hidden_size: 2,
            rank: 2,
            vocab_size: 3,
            input_mean: vec![0.0, 0.0],
            input_to_latent: vec![1.0, 0.0, 0.0, 1.0],
            vocab_projection: vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
            vocab_bias: vec![0.0, 0.0, 0.0],
        };
        assert_eq!(
            sidecar
                .propose(&[2.0, 1.0], 2)
                .expect("proposal should score"),
            vec![0, 1]
        );
    }

    #[test]
    fn resident_report_serializes_optional_quality_summary() {
        let report = super::ResidentReport {
            model_path: "model".into(),
            model_revision: "revision".into(),
            backend: "metal-streaming".into(),
            device: "device".into(),
            recommended_working_set_mib: 1,
            mapped_model_bytes: 2,
            active_weight_bytes: 3,
            kv_bytes: 4,
            scratch_bytes: 5,
            prompt_tokens: 1,
            generated_tokens: 1,
            warmup_runs: 0,
            repetitions: 1,
            prefill: std::time::Duration::from_millis(1),
            decode: std::time::Duration::from_millis(1),
            peak_vram_mib: Some(6),
            peak_ram_mib: Some(7),
            median_peak_vram_mib: Some(6),
            median_peak_ram_mib: Some(7),
            output_match: None,
            repetition_outputs_match: true,
            quality: Some(crate::quality::QualitySummary {
                likelihood: crate::quality::LikelihoodScore {
                    mean_nll: 1.0,
                    perplexity: std::f64::consts::E,
                    cases: 1,
                },
                structured: crate::quality::StructuredScore {
                    passed: 1,
                    total: 1,
                    by_category: std::collections::BTreeMap::new(),
                },
                regression_cases: 1,
            }),
            generated_output: "ok".into(),
            generated_token_ids: vec![1],
        };
        let json = report.as_json();
        assert!(json.contains("\"quality\":{\"likelihood\""));
        assert!(report
            .as_text()
            .contains("quality_structured_accuracy=1.000000"));
    }

    #[test]
    fn memory_measurement_labels_mapped_gguf_separately() {
        assert_eq!(
            super::memory_measurement_label("model.Q4_K_M.gguf"),
            "rss_plus_private_metal; mapped_weights_excluded_from_private_metal"
        );
        assert_eq!(
            super::memory_measurement_label("model.safetensors"),
            "rss_plus_private_metal"
        );
    }

    #[test]
    fn runs_one_real_qwen_token_when_requested() {
        if std::env::var_os("SI_RUN_FULL_MODEL").is_none() {
            return;
        }
        let model_dir = std::env::var("SI_MODEL_DIR").expect("SI_MODEL_DIR is required");
        let mut model = super::MetalQwen3::from_model_dir(model_dir, false, 1)
            .expect("Qwen3 Metal runtime should initialize");
        let logits = model
            .forward_token(0, 0)
            .expect("one Qwen3 token should execute");
        assert_eq!(logits.len(), model.config().vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(model.cached_tokens(), 1);
    }
}
