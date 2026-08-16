//! Explainable memory-planner actions and a first sequential layer plan.
//!
//! The planner is intentionally backend-neutral. It emits logical actions and
//! validates residency ordering; a later scheduler will attach real queue
//! timestamps and asynchronous Metal events.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlannerMode {
    Resident,
    Mmap,
    LayerStream,
    SublayerStream,
    Overlap,
    Heterogeneous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryTier {
    MappedSource,
    PinnedHost,
    Metal,
    Cpu,
    Nvme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Prefetch,
    Execute,
    Retain,
    Evict,
    Wait,
    Release,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerAction {
    pub kind: ActionKind,
    pub operation_id: String,
    pub tensor_ids: Vec<String>,
    pub bytes: u64,
    pub tier: MemoryTier,
    pub started_ns: u64,
    pub ended_ns: u64,
    pub queue: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTensor {
    pub id: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    pub operation_id: String,
    pub tensors: Vec<LayerTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerTrace {
    pub mode: PlannerMode,
    pub actions: Vec<PlannerAction>,
    pub planned_peak_metal_bytes: u64,
    pub observed_peak_metal_bytes: Option<u64>,
    pub observed_peak_rss_bytes: Option<u64>,
    resident: BTreeMap<(MemoryTier, String), u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerError(pub String);

impl Display for PlannerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PlannerError {}

impl PlannerTrace {
    pub fn new(mode: PlannerMode) -> Self {
        Self {
            mode,
            actions: Vec::new(),
            planned_peak_metal_bytes: 0,
            observed_peak_metal_bytes: None,
            observed_peak_rss_bytes: None,
            resident: BTreeMap::new(),
        }
    }

    pub fn record(&mut self, action: PlannerAction) -> Result<(), PlannerError> {
        if action.tensor_ids.is_empty() {
            return Err(PlannerError("planner action must name a tensor".into()));
        }
        if action.ended_ns < action.started_ns {
            return Err(PlannerError(format!(
                "action {} ends before it starts",
                action.operation_id
            )));
        }
        match action.kind {
            ActionKind::Prefetch | ActionKind::Retain => {
                let per_tensor = action.bytes / action.tensor_ids.len() as u64;
                for tensor_id in &action.tensor_ids {
                    self.resident
                        .insert((action.tier, tensor_id.clone()), per_tensor);
                }
            }
            ActionKind::Execute => {
                for tensor_id in &action.tensor_ids {
                    if !self
                        .resident
                        .contains_key(&(action.tier, tensor_id.clone()))
                    {
                        return Err(PlannerError(format!(
                            "operation {} executes non-resident tensor {tensor_id}",
                            action.operation_id
                        )));
                    }
                }
            }
            ActionKind::Evict | ActionKind::Release => {
                for tensor_id in &action.tensor_ids {
                    if self
                        .resident
                        .remove(&(action.tier, tensor_id.clone()))
                        .is_none()
                    {
                        return Err(PlannerError(format!(
                            "operation {} evicts non-resident tensor {tensor_id}",
                            action.operation_id
                        )));
                    }
                }
            }
            ActionKind::Wait => {}
        }
        if action.tier == MemoryTier::Metal
            && matches!(action.kind, ActionKind::Prefetch | ActionKind::Retain)
        {
            let current = self
                .resident
                .iter()
                .filter(|((tier, _), _)| *tier == MemoryTier::Metal)
                .map(|(_, bytes)| *bytes)
                .sum();
            self.planned_peak_metal_bytes = self.planned_peak_metal_bytes.max(current);
        }
        self.actions.push(action);
        Ok(())
    }

    pub fn set_observed_peaks(&mut self, metal_bytes: u64, rss_bytes: u64) {
        self.observed_peak_metal_bytes = Some(metal_bytes);
        self.observed_peak_rss_bytes = Some(rss_bytes);
    }

    pub fn resident_tensors(&self, tier: MemoryTier) -> BTreeSet<&str> {
        self.resident
            .keys()
            .filter_map(|(resident_tier, tensor_id)| {
                (*resident_tier == tier).then_some(tensor_id.as_str())
            })
            .collect()
    }
}

pub fn plan_sequential_layers(
    layers: &[LayerPlan],
    metal_budget_bytes: u64,
) -> Result<PlannerTrace, PlannerError> {
    if layers.is_empty() {
        return Err(PlannerError(
            "layer plan must contain at least one layer".into(),
        ));
    }
    if metal_budget_bytes == 0 {
        return Err(PlannerError("Metal budget must be non-zero".into()));
    }
    let mut trace = PlannerTrace::new(PlannerMode::LayerStream);
    let mut timestamp = 0_u64;
    for layer in layers {
        if layer.tensors.is_empty() {
            return Err(PlannerError(format!(
                "layer {} has no tensors",
                layer.operation_id
            )));
        }
        let layer_bytes = layer
            .tensors
            .iter()
            .try_fold(0_u64, |sum, tensor| sum.checked_add(tensor.bytes))
            .ok_or_else(|| PlannerError("layer byte size overflow".into()))?;
        if layer_bytes > metal_budget_bytes {
            return Err(PlannerError(format!(
                "layer {} requires {} bytes but budget is {}",
                layer.operation_id, layer_bytes, metal_budget_bytes
            )));
        }
        for tensor in &layer.tensors {
            trace.record(PlannerAction {
                kind: ActionKind::Prefetch,
                operation_id: layer.operation_id.clone(),
                tensor_ids: vec![tensor.id.clone()],
                bytes: tensor.bytes,
                tier: MemoryTier::Metal,
                started_ns: timestamp,
                ended_ns: timestamp,
                queue: Some("transfer".into()),
                reason: "make current layer available before execution".into(),
            })?;
        }
        trace.record(PlannerAction {
            kind: ActionKind::Execute,
            operation_id: layer.operation_id.clone(),
            tensor_ids: layer
                .tensors
                .iter()
                .map(|tensor| tensor.id.clone())
                .collect(),
            bytes: layer_bytes,
            tier: MemoryTier::Metal,
            started_ns: timestamp,
            ended_ns: timestamp,
            queue: Some("compute".into()),
            reason: "execute layer with all required tensors resident".into(),
        })?;
        timestamp = timestamp.saturating_add(1);
        for tensor in &layer.tensors {
            trace.record(PlannerAction {
                kind: ActionKind::Evict,
                operation_id: layer.operation_id.clone(),
                tensor_ids: vec![tensor.id.clone()],
                bytes: tensor.bytes,
                tier: MemoryTier::Metal,
                started_ns: timestamp,
                ended_ns: timestamp,
                queue: Some("transfer".into()),
                reason: "release layer before advancing the bounded window".into(),
            })?;
        }
        timestamp = timestamp.saturating_add(1);
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_plan_evicts_each_layer_within_budget() {
        let layers = vec![
            LayerPlan {
                operation_id: "layer-0".into(),
                tensors: vec![
                    LayerTensor {
                        id: "q0".into(),
                        bytes: 30,
                    },
                    LayerTensor {
                        id: "mlp0".into(),
                        bytes: 40,
                    },
                ],
            },
            LayerPlan {
                operation_id: "layer-1".into(),
                tensors: vec![LayerTensor {
                    id: "q1".into(),
                    bytes: 50,
                }],
            },
        ];
        let trace = plan_sequential_layers(&layers, 70).expect("plan should fit");
        assert_eq!(trace.planned_peak_metal_bytes, 70);
        assert!(trace.resident_tensors(MemoryTier::Metal).is_empty());
        assert_eq!(
            trace
                .actions
                .iter()
                .filter(|action| action.kind == ActionKind::Execute)
                .count(),
            2
        );
    }

    #[test]
    fn rejects_layer_larger_than_budget() {
        let layers = vec![LayerPlan {
            operation_id: "layer-0".into(),
            tensors: vec![LayerTensor {
                id: "large".into(),
                bytes: 101,
            }],
        }];
        let error = plan_sequential_layers(&layers, 100).expect_err("budget must fail");
        assert!(error.0.contains("budget"));
    }

    #[test]
    fn trace_rejects_execute_before_prefetch() {
        let mut trace = PlannerTrace::new(PlannerMode::LayerStream);
        let error = trace
            .record(PlannerAction {
                kind: ActionKind::Execute,
                operation_id: "layer-0".into(),
                tensor_ids: vec!["q0".into()],
                bytes: 4,
                tier: MemoryTier::Metal,
                started_ns: 0,
                ended_ns: 0,
                queue: Some("compute".into()),
                reason: "invalid test".into(),
            })
            .expect_err("non-resident execution must fail");
        assert!(error.0.contains("non-resident"));
    }
}
