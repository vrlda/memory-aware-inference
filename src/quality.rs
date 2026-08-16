//! Versioned, small capability-quality suite.
//!
//! The scorer is backend-agnostic. Backends supply generated text or measured
//! negative log-likelihoods; the suite owns fixture validation and deterministic
//! aggregation so every memory profile is compared on the same contract.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityError(pub String);

impl Display for QualityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for QualityError {}

impl From<std::io::Error> for QualityError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for QualityError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct QualitySuite {
    pub suite: String,
    pub version: u32,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub generation: GenerationSettings,
    pub likelihood: Vec<LikelihoodCase>,
    pub structured_completion: Vec<StructuredCompletionCase>,
    pub regression_prompts: Vec<RegressionCase>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GenerationSettings {
    pub temperature: f32,
    pub sampling: bool,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LikelihoodCase {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct StructuredCompletionCase {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub criterion: Criterion,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum Criterion {
    #[serde(rename = "contains")]
    Contains { value: String },
    #[serde(rename = "contains_any")]
    ContainsAny { values: Vec<String> },
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct RegressionCase {
    pub id: String,
    pub prompt: String,
}

impl QualitySuite {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, QualityError> {
        let bytes = std::fs::read(path)?;
        let suite: Self = serde_json::from_slice(&bytes)?;
        suite.validate()?;
        Ok(suite)
    }

    pub fn validate(&self) -> Result<(), QualityError> {
        if self.suite != "si-quality-v0" || self.version != 1 {
            return Err(QualityError(
                "unsupported quality suite; expected si-quality-v0 v1".into(),
            ));
        }
        if self.model_revision.is_empty() || self.tokenizer_revision.is_empty() {
            return Err(QualityError(
                "quality suite revisions must be non-empty".into(),
            ));
        }
        if !self.generation.temperature.is_finite() || self.generation.temperature != 0.0 {
            return Err(QualityError(
                "quality suite requires deterministic temperature 0".into(),
            ));
        }
        if self.generation.sampling || self.generation.max_new_tokens == 0 {
            return Err(QualityError(
                "quality suite requires greedy generation with a positive token cap".into(),
            ));
        }
        if self.likelihood.is_empty()
            || self.structured_completion.is_empty()
            || self.regression_prompts.is_empty()
        {
            return Err(QualityError(
                "quality suite must contain all three non-empty categories".into(),
            ));
        }
        validate_unique_ids(
            self.likelihood.iter().map(|case| case.id.as_str()),
            "likelihood",
        )?;
        validate_unique_ids(
            self.structured_completion
                .iter()
                .map(|case| case.id.as_str()),
            "structured_completion",
        )?;
        validate_unique_ids(
            self.regression_prompts.iter().map(|case| case.id.as_str()),
            "regression_prompts",
        )?;
        for case in &self.structured_completion {
            if case.category.is_empty() || case.prompt.is_empty() {
                return Err(QualityError(format!(
                    "structured case {} has an empty category or prompt",
                    case.id
                )));
            }
            match &case.criterion {
                Criterion::Contains { value } if value.trim().is_empty() => {
                    return Err(QualityError(format!(
                        "structured case {} has an empty criterion",
                        case.id
                    )))
                }
                Criterion::ContainsAny { values } if values.is_empty() => {
                    return Err(QualityError(format!(
                        "structured case {} has no acceptable values",
                        case.id
                    )))
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn score_structured(
        &self,
        outputs: &BTreeMap<String, String>,
    ) -> Result<StructuredScore, QualityError> {
        let mut passed = 0;
        let mut by_category: BTreeMap<String, CategoryScore> = BTreeMap::new();
        for case in &self.structured_completion {
            let output = outputs.get(&case.id).ok_or_else(|| {
                QualityError(format!("missing structured output for {}", case.id))
            })?;
            let category =
                by_category
                    .entry(case.category.clone())
                    .or_insert_with(|| CategoryScore {
                        category: case.category.clone(),
                        passed: 0,
                        total: 0,
                    });
            category.total += 1;
            let case_passed = case.criterion.matches(output);
            if case_passed {
                passed += 1;
                category.passed += 1;
            }
        }
        Ok(StructuredScore {
            passed,
            total: self.structured_completion.len(),
            by_category,
        })
    }

    pub fn score_likelihood(
        &self,
        mean_negative_log_likelihood: &BTreeMap<String, f64>,
    ) -> Result<LikelihoodScore, QualityError> {
        let mut total = 0.0;
        for case in &self.likelihood {
            let value = mean_negative_log_likelihood
                .get(&case.id)
                .ok_or_else(|| QualityError(format!("missing likelihood score for {}", case.id)))?;
            if !value.is_finite() || *value < 0.0 {
                return Err(QualityError(format!(
                    "likelihood score for {} is invalid",
                    case.id
                )));
            }
            total += value;
        }
        let mean_nll = total / self.likelihood.len() as f64;
        Ok(LikelihoodScore {
            mean_nll,
            perplexity: mean_nll.exp(),
            cases: self.likelihood.len(),
        })
    }
}

impl Criterion {
    pub fn matches(&self, output: &str) -> bool {
        let output = output.to_lowercase();
        match self {
            Self::Contains { value } => output.contains(&value.to_lowercase()),
            Self::ContainsAny { values } => values
                .iter()
                .any(|value| output.contains(&value.to_lowercase())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructuredScore {
    pub passed: usize,
    pub total: usize,
    pub by_category: BTreeMap<String, CategoryScore>,
}

impl StructuredScore {
    pub fn accuracy(&self) -> f64 {
        ratio(self.passed, self.total)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CategoryScore {
    pub category: String,
    pub passed: usize,
    pub total: usize,
}

impl CategoryScore {
    pub fn accuracy(&self) -> f64 {
        ratio(self.passed, self.total)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LikelihoodScore {
    pub mean_nll: f64,
    pub perplexity: f64,
    pub cases: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualitySummary {
    pub likelihood: LikelihoodScore,
    pub structured: StructuredScore,
    pub regression_cases: usize,
}

fn validate_unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    category: &str,
) -> Result<(), QualityError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        if id.is_empty() || !seen.insert(id) {
            return Err(QualityError(format!(
                "{category} case IDs must be non-empty and unique"
            )));
        }
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_checked_in_suite() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("quality-v0.json");
        let suite = QualitySuite::load(path).expect("fixture should validate");
        assert_eq!(suite.likelihood.len(), 12);
        assert_eq!(suite.structured_completion.len(), 12);
        assert_eq!(suite.regression_prompts.len(), 4);
    }

    #[test]
    fn scores_structured_outputs_by_category() {
        let suite = test_suite();
        let outputs = BTreeMap::from([
            ("completion-1".into(), "The answer is H2O".into()),
            ("completion-2".into(), "not correct".into()),
        ]);
        let score = suite.score_structured(&outputs).expect("all outputs exist");
        assert_eq!(score.passed, 1);
        assert_eq!(score.total, 2);
        assert_eq!(score.by_category["factual"].passed, 1);
        assert_eq!(score.by_category["arithmetic"].passed, 0);
    }

    #[test]
    fn scores_mean_nll_and_perplexity() {
        let suite = test_suite();
        let values = BTreeMap::from([("likelihood-1".into(), 0.0), ("likelihood-2".into(), 2.0)]);
        let score = suite.score_likelihood(&values).expect("scores are valid");
        assert_eq!(score.mean_nll, 1.0);
        assert!((score.perplexity - std::f64::consts::E).abs() < 1.0e-12);
    }

    fn test_suite() -> QualitySuite {
        QualitySuite {
            suite: "si-quality-v0".into(),
            version: 1,
            model_revision: "model".into(),
            tokenizer_revision: "tokenizer".into(),
            generation: GenerationSettings {
                temperature: 0.0,
                sampling: false,
                max_new_tokens: 4,
            },
            likelihood: vec![
                LikelihoodCase {
                    id: "likelihood-1".into(),
                    text: "one".into(),
                },
                LikelihoodCase {
                    id: "likelihood-2".into(),
                    text: "two".into(),
                },
            ],
            structured_completion: vec![
                StructuredCompletionCase {
                    id: "completion-1".into(),
                    category: "factual".into(),
                    prompt: "water".into(),
                    criterion: Criterion::Contains {
                        value: "H2O".into(),
                    },
                },
                StructuredCompletionCase {
                    id: "completion-2".into(),
                    category: "arithmetic".into(),
                    prompt: "one plus one".into(),
                    criterion: Criterion::Contains { value: "2".into() },
                },
            ],
            regression_prompts: vec![RegressionCase {
                id: "regression-1".into(),
                prompt: "continue".into(),
            }],
        }
    }
}
