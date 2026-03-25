use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use serde::Serialize;
#[cfg(feature = "inference")]
use std::time::Instant;

#[cfg(feature = "inference")]
use {
    fastly::log::Endpoint, ndarray::Array2, std::io::Write, tokenizers::Tokenizer,
    tract_onnx::prelude::*,
};

#[cfg(feature = "inference")]
const MAX_SEQ_LEN: usize = 128;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
struct ModelThresholds {
    #[serde(rename = "T_block_at_1pct_FPR")]
    t_block: f32,
}

#[derive(Debug, serde::Deserialize)]
struct ThresholdConfig {
    injection: ModelThresholds,
}

#[derive(Debug, Serialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub injection_score: f32,
    pub elapsed_ms: f64,
    pub tokenization_ms: f64,
    pub injection_inference_ms: f64,
    pub postprocess_ms: f64,
}

static THRESHOLDS: Lazy<Result<ThresholdConfig>> = Lazy::new(|| {
    serde_json::from_slice(include_bytes!("../assets/calibrated_thresholds.json"))
        .map_err(|e| anyhow!("Failed to load thresholds: {}", e))
});

#[cfg(feature = "inference")]
type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[cfg(feature = "inference")]
static TOKENIZER: Lazy<Result<Tokenizer>> = Lazy::new(|| build_raw_tokenizer());

#[cfg(feature = "inference")]
static INJECTION_MODEL: Lazy<Result<RunnableModel>> =
    Lazy::new(|| load_model(include_bytes!("../assets/injection_1x128_int8.onnx")));

#[cfg(feature = "inference")]
static CLASSIFY_LOG: Lazy<Option<Endpoint>> =
    Lazy::new(|| Endpoint::try_from_name("classify_metrics").ok());

/// Wizer pre-initialization entry point.
///
/// Forces the three lazy statics that depend only on embedded bytes (pure computation)
/// to be evaluated before Wizer snapshots the heap. This eliminates the ~160ms
/// lazy-static init cost from every production request.
///
/// IMPORTANT: CLASSIFY_LOG must NOT be touched here — it calls
/// `Endpoint::try_from_name()` which is a Fastly host function and will trap
/// under Wizer (which has no Fastly runtime).
#[cfg(all(feature = "inference", feature = "wizer-init"))]
#[export_name = "wizer.initialize"]
pub extern "C" fn wizer_initialize() {
    let _ = THRESHOLDS.as_ref();
    let _ = TOKENIZER.as_ref();
    let _ = INJECTION_MODEL.as_ref();
}

#[cfg(feature = "inference")]
fn load_tokenizer_from_bytes() -> Result<Tokenizer> {
    Tokenizer::from_bytes(include_bytes!("../assets/tokenizer.json"))
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))
}

#[cfg(feature = "inference")]
fn build_raw_tokenizer() -> Result<Tokenizer> {
    let mut tokenizer = load_tokenizer_from_bytes()?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!("Failed to clear tokenizer truncation: {}", e))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

fn decide_label(injection_score: f32, thresholds: &ThresholdConfig) -> (&'static str, f32) {
    if injection_score >= thresholds.injection.t_block {
        ("INJECTION", injection_score)
    } else {
        ("SAFE", 1.0 - injection_score)
    }
}

#[cfg(feature = "inference")]
fn load_model(model_bytes: &[u8]) -> Result<RunnableModel> {
    let mut model = tract_onnx::onnx().model_for_read(&mut &model_bytes[..])?;
    let input_count = model.inputs.len();
    for i in 0..input_count {
        model = model.with_input_fact(i, i64::fact([1usize, MAX_SEQ_LEN]).into())?;
    }
    model.into_optimized()?.into_runnable()
}

#[cfg(feature = "inference")]
fn prepare_inputs(tokenizer: &Tokenizer, text: &str) -> Result<(Array2<i64>, Array2<i64>)> {
    let (padded_ids, padded_attention_mask) = prepare_head_tail_vectors(tokenizer, text)?;

    Ok((
        Array2::from_shape_vec((1, MAX_SEQ_LEN), padded_ids)?,
        Array2::from_shape_vec((1, MAX_SEQ_LEN), padded_attention_mask)?,
    ))
}

#[cfg(feature = "inference")]
fn prepare_head_tail_vectors(tokenizer: &Tokenizer, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    // The serialized tokenizer carries fixed right-truncation and padding to length 128.
    // We clear those settings at load time so this path works on raw content ids, matching
    // training before we manually rebuild [CLS] content [SEP] and the attention mask.
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
    let raw_ids = encoding.get_ids();

    const SPECIAL_TOKENS: usize = 2;
    const CONTENT_BUDGET: usize = MAX_SEQ_LEN - SPECIAL_TOKENS;
    let head_n = CONTENT_BUDGET / 2;
    let tail_n = CONTENT_BUDGET - head_n;

    let cls_id = tokenizer
        .token_to_id("[CLS]")
        .ok_or_else(|| anyhow!("Tokenizer is missing [CLS] token"))? as i64;
    let sep_id = tokenizer
        .token_to_id("[SEP]")
        .ok_or_else(|| anyhow!("Tokenizer is missing [SEP] token"))? as i64;
    let pad_id = tokenizer
        .token_to_id("[PAD]")
        .ok_or_else(|| anyhow!("Tokenizer is missing [PAD] token"))? as i64;

    let mut padded_ids = vec![pad_id; MAX_SEQ_LEN];
    let mut padded_attention_mask = vec![0i64; MAX_SEQ_LEN];
    padded_ids[0] = cls_id;
    padded_attention_mask[0] = 1;

    let content_len = if raw_ids.len() <= CONTENT_BUDGET {
        for (idx, &id) in raw_ids.iter().enumerate() {
            padded_ids[idx + 1] = id as i64;
            padded_attention_mask[idx + 1] = 1;
        }
        raw_ids.len()
    } else {
        for (idx, &id) in raw_ids[..head_n].iter().enumerate() {
            padded_ids[idx + 1] = id as i64;
            padded_attention_mask[idx + 1] = 1;
        }

        for (offset, &id) in raw_ids[raw_ids.len() - tail_n..].iter().enumerate() {
            let idx = head_n + offset;
            padded_ids[idx + 1] = id as i64;
            padded_attention_mask[idx + 1] = 1;
        }

        CONTENT_BUDGET
    };
    let sep_pos = content_len + 1;
    padded_ids[sep_pos] = sep_id;
    padded_attention_mask[sep_pos] = 1;

    Ok((padded_ids, padded_attention_mask))
}

#[cfg(all(test, feature = "inference"))]
fn prepare_right_truncation_vectors(
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
    let ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let seq_len = ids.len().min(MAX_SEQ_LEN);

    let mut padded_ids = vec![0i64; MAX_SEQ_LEN];
    let mut padded_attention_mask = vec![0i64; MAX_SEQ_LEN];
    for idx in 0..seq_len {
        padded_ids[idx] = ids[idx] as i64;
        padded_attention_mask[idx] = attention_mask[idx] as i64;
    }

    Ok((padded_ids, padded_attention_mask))
}

#[cfg(all(test, feature = "inference"))]
fn prepare_failed_v4_vectors(tokenizer: &Tokenizer, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
    let all_ids = encoding.get_ids();

    const CONTENT_BUDGET: usize = MAX_SEQ_LEN - 2;
    let head_n = CONTENT_BUDGET / 2;
    let tail_n = CONTENT_BUDGET - head_n;

    let cls_id = all_ids[0] as i64;
    let sep_id = all_ids[all_ids.len() - 1] as i64;
    let content_ids = &all_ids[1..all_ids.len() - 1];

    let selected: Vec<u32> = if content_ids.len() <= CONTENT_BUDGET {
        content_ids.to_vec()
    } else {
        let mut v = content_ids[..head_n].to_vec();
        v.extend_from_slice(&content_ids[content_ids.len() - tail_n..]);
        v
    };

    let mut padded_ids = vec![0i64; MAX_SEQ_LEN];
    let mut padded_attention_mask = vec![0i64; MAX_SEQ_LEN];
    padded_ids[0] = cls_id;
    padded_attention_mask[0] = 1;
    for (idx, &id) in selected.iter().enumerate() {
        padded_ids[idx + 1] = id as i64;
        padded_attention_mask[idx + 1] = 1;
    }
    let sep_pos = selected.len() + 1;
    padded_ids[sep_pos] = sep_id;
    padded_attention_mask[sep_pos] = 1;

    Ok((padded_ids, padded_attention_mask))
}

#[cfg(feature = "inference")]
fn probability_for_class_one(logits: &[f32]) -> Result<f32> {
    if logits.len() != 2 {
        return Err(anyhow!(
            "Unexpected output size: {} (expected 2)",
            logits.len()
        ));
    }

    let safe_logit = logits[0];
    let positive_logit = logits[1];
    let max_logit = safe_logit.max(positive_logit);
    let exp_safe = (safe_logit - max_logit).exp();
    let exp_positive = (positive_logit - max_logit).exp();
    let sum_exp = exp_safe + exp_positive;

    Ok(exp_positive / sum_exp)
}

#[cfg(feature = "inference")]
fn run_model(
    model: &RunnableModel,
    input_ids: &Array2<i64>,
    attention_mask: &Array2<i64>,
) -> Result<f32> {
    let t_ids: Tensor = input_ids.clone().into();
    let t_mask: Tensor = attention_mask.clone().into();
    // If model has 3 inputs (input_ids, attention_mask, token_type_ids), add zeros
    let outputs = if model.model().inputs.len() == 3 {
        let token_type_ids = Array2::<i64>::zeros((1, MAX_SEQ_LEN));
        let t_types: Tensor = token_type_ids.into();
        model.run(tvec!(t_ids.into(), t_mask.into(), t_types.into()))?
    } else {
        model.run(tvec!(t_ids.into(), t_mask.into()))?
    };
    let logits_view = outputs[0].to_array_view::<f32>()?;
    let logits = logits_view
        .as_slice()
        .ok_or_else(|| anyhow!("Failed to extract logits"))?;

    probability_for_class_one(logits)
}

#[cfg(feature = "inference")]
fn emit_log(result: &ClassificationResult) {
    if let Some(endpoint) = CLASSIFY_LOG.as_ref() {
        let line = format!(
            r#"{{"event":"classify","label":"{}","injection_score":{:.4},"elapsed_ms":{:.2},"tokenization_ms":{:.2},"injection_inference_ms":{:.2}}}"#,
            result.label,
            result.injection_score,
            result.elapsed_ms,
            result.tokenization_ms,
            result.injection_inference_ms
        );
        let mut endpoint = endpoint.clone();
        let _ = endpoint.write_all(line.as_bytes());
    }
}

#[cfg(not(feature = "inference"))]
fn emit_log(_result: &ClassificationResult) {}

#[cfg(feature = "inference")]
pub fn classify(text: &str) -> Result<ClassificationResult> {
    let start = Instant::now();
    let tokenizer = TOKENIZER
        .as_ref()
        .map_err(|e| anyhow!("Tokenizer not initialized: {}", e))?;
    let injection_model = INJECTION_MODEL
        .as_ref()
        .map_err(|e| anyhow!("Injection model not initialized: {}", e))?;
    let thresholds = THRESHOLDS
        .as_ref()
        .map_err(|e| anyhow!("Thresholds not initialized: {}", e))?;

    let tokenization_start = Instant::now();
    let (input_ids, attention_mask) = prepare_inputs(tokenizer, text)?;
    let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;

    let injection_inference_start = Instant::now();
    let injection_score = run_model(injection_model, &input_ids, &attention_mask)?;
    let injection_inference_ms = injection_inference_start.elapsed().as_secs_f64() * 1000.0;

    let postprocess_start = Instant::now();
    let (label, score) = decide_label(injection_score, thresholds);
    let postprocess_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0;

    let result = ClassificationResult {
        label: label.to_string(),
        score,
        injection_score,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        tokenization_ms,
        injection_inference_ms,
        postprocess_ms,
    };
    emit_log(&result);

    Ok(result)
}

#[cfg(all(test, feature = "inference"))]
mod inference_tests {
    use super::*;

    #[test]
    fn head_tail_matches_right_truncation_for_short_input() -> Result<()> {
        let configured = load_tokenizer_from_bytes()?;
        let raw = build_raw_tokenizer()?;
        let text = "What is the weather today?";

        let (right_ids, right_mask) = prepare_right_truncation_vectors(&configured, text)?;
        let (head_tail_ids, head_tail_mask) = prepare_head_tail_vectors(&raw, text)?;

        assert_eq!(head_tail_ids, right_ids);
        assert_eq!(head_tail_mask, right_mask);
        Ok(())
    }

    #[test]
    fn naive_rebuild_from_configured_encoding_attends_padding() -> Result<()> {
        let configured = load_tokenizer_from_bytes()?;
        let text = "What is the weather today?";

        let (right_ids, right_mask) = prepare_right_truncation_vectors(&configured, text)?;
        let (failed_ids, failed_mask) = prepare_failed_v4_vectors(&configured, text)?;

        assert_ne!(failed_mask, right_mask);
        assert_eq!(right_ids, failed_ids);

        let first_pad = right_ids
            .iter()
            .position(|&id| id == 0)
            .ok_or_else(|| anyhow!("Expected padded test encoding"))?;
        assert_eq!(right_mask[first_pad], 0);
        assert_eq!(failed_mask[first_pad], 1);
        Ok(())
    }
}

#[cfg(not(feature = "inference"))]
pub fn classify(text: &str) -> Result<ClassificationResult> {
    let text_lower = text.to_lowercase();
    let injection_score = if ["ignore", "instruction", "prompt", "system", "override"]
        .iter()
        .any(|kw| text_lower.contains(kw))
    {
        0.75
    } else {
        0.05
    };

    let thresholds = THRESHOLDS
        .as_ref()
        .map_err(|e| anyhow!("Thresholds not initialized: {}", e))?;
    let (label, score) = decide_label(injection_score, thresholds);

    let result = ClassificationResult {
        label: label.to_string(),
        score,
        injection_score,
        elapsed_ms: 1.0,
        tokenization_ms: 0.2,
        injection_inference_ms: 0.3,
        postprocess_ms: 0.2,
    };
    emit_log(&result);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{decide_label, ModelThresholds, ThresholdConfig};

    fn assert_close(actual: f32, expected: f32) {
        let delta = (actual - expected).abs();
        assert!(delta < 1e-6, "expected {expected}, got {actual}");
    }

    #[test]
    fn deserializes_nested_thresholds() {
        let thresholds: ThresholdConfig = serde_json::from_str(
            r#"{
                "injection": {
                    "T_block_at_1pct_FPR": 0.9912,
                    "T_review_lower_at_2pct_FPR": 0.9611
                }
            }"#,
        )
        .expect("thresholds should deserialize");

        assert_close(thresholds.injection.t_block, 0.9912);
    }

    #[test]
    fn injection_above_threshold_is_labelled_injection() {
        let thresholds = ThresholdConfig {
            injection: ModelThresholds { t_block: 0.80 },
        };

        let (label, score) = decide_label(0.85, &thresholds);

        assert_eq!(label, "INJECTION");
        assert_close(score, 0.85);
    }

    #[test]
    fn injection_below_threshold_is_labelled_safe() {
        let thresholds = ThresholdConfig {
            injection: ModelThresholds { t_block: 0.90 },
        };

        let (label, score) = decide_label(0.45, &thresholds);

        assert_eq!(label, "SAFE");
        assert_close(score, 1.0 - 0.45);
    }

    #[test]
    fn safe_score_is_one_minus_injection_score() {
        let thresholds = ThresholdConfig {
            injection: ModelThresholds { t_block: 0.90 },
        };

        let (label, score) = decide_label(0.20, &thresholds);

        assert_eq!(label, "SAFE");
        assert_close(score, 0.80);
    }
}
