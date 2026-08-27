//! DeepSeek-V4 configuration.
//!
//! `DeepseekV4Config` fully deserializes the DeepSeek-V4 `config.json` (as
//! shipped with the tiny sample model) and exposes the V4-specific fields that
//! `deepseek2` does not carry: `head_dim`, `o_lora_rank`, `o_groups`,
//! `compress_rates`/`compress_ratios`, `num_hash_layers`, the `index_*` group,
//! the `hc_*` group, `layer_types`, `num_nextn_predict_layers`,
//! `partial_rotary_factor` and `topk_method`. It also maps `layer_types` to a
//! per-layer compression rate and RoPE type (main layers use `rope_theta`,
//! compressor layers use `compress_rope_theta` with Yarn scaling).

use candle_nn::Activation;
use serde::Deserialize;

/// Per-layer attention type used by DeepSeek-V4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
}

/// Expert selection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopkMethod {
    Greedy,
    GroupLimitedGreedy,
    NoauxTc,
}

/// Router scoring function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunc {
    Softmax,
    #[serde(rename = "sqrtsoftplus")]
    SqrtSoftplus,
}

/// RoPE scaling type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

/// Compression rates keyed by compressor layer type.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CompressRates {
    #[serde(rename = "compressed_sparse_attention")]
    pub compressed_sparse_attention: usize,
    #[serde(rename = "heavily_compressed_attention")]
    pub heavily_compressed_attention: usize,
}

/// RoPE scaling parameters (Yarn for compressor layers).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    pub scaling_type: ScaledRopeType,
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_norm_topk_prob() -> bool {
    false
}

fn default_routed_scaling_factor() -> f64 {
    1.0
}

fn default_topk_method() -> TopkMethod {
    TopkMethod::Greedy
}

fn default_scoring_func() -> ScoringFunc {
    ScoringFunc::Softmax
}

/// DeepSeek-V4 configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekV4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_nextn_predict_layers: usize,
    pub o_groups: usize,
    pub num_hash_layers: usize,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub partial_rotary_factor: f64,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub mlp_bias: bool,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    pub output_router_logits: bool,
    pub router_aux_loss_coef: f64,
    pub router_jitter_noise: f64,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    pub initializer_range: f64,
    #[serde(default = "default_topk_method")]
    pub topk_method: TopkMethod,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: ScoringFunc,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: Option<usize>,
    pub compress_rates: CompressRates,
    pub compress_ratios: Vec<usize>,
    pub layer_types: Vec<LayerType>,
    pub rope_scaling: RopeScaling,
}

/// Per-layer derived configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    /// Compression rate for this layer's attention; `0` for sliding-attention layers.
    pub compress_rate: usize,
    /// RoPE base theta for this layer.
    pub rope_theta: f32,
    /// Whether Yarn RoPE scaling applies (compressor layers).
    pub use_yarn: bool,
}

impl DeepseekV4Config {
    /// Compression rate associated with a given layer type.
    pub fn compress_rate_for(&self, layer_type: LayerType) -> usize {
        match layer_type {
            LayerType::SlidingAttention => 0,
            LayerType::CompressedSparseAttention => self.compress_rates.compressed_sparse_attention,
            LayerType::HeavilyCompressedAttention => {
                self.compress_rates.heavily_compressed_attention
            }
        }
    }

    /// Derive per-layer compression rate and RoPE type.
    ///
    /// Compressor layers (CSA/HCA) use `compress_rope_theta` (e.g. 160000) with
    /// Yarn scaling; plain sliding-attention layers use `rope_theta` (e.g. 10000).
    pub fn layer_configs(&self) -> Vec<LayerConfig> {
        self.layer_types
            .iter()
            .map(|&layer_type| {
                let compress = layer_type != LayerType::SlidingAttention;
                LayerConfig {
                    layer_type,
                    compress_rate: self.compress_rate_for(layer_type),
                    rope_theta: if compress {
                        self.compress_rope_theta
                    } else {
                        self.rope_theta
                    },
                    use_yarn: compress,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_config_path() -> PathBuf {
        let candidates = [
            "sample_models/deepseek-v4-tiny/config.json",
            "../sample_models/deepseek-v4-tiny/config.json",
        ];
        for c in candidates {
            let p = PathBuf::from(c);
            if p.exists() {
                return p;
            }
        }
        // Fall back to CARGO_MANIFEST_DIR-relative (workspace root is one level up).
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sample_models/deepseek-v4-tiny/config.json")
    }

    #[test]
    fn deepseek_v4_config() {
        let path = sample_config_path();
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let cfg: DeepseekV4Config = serde_json::from_str(&raw).unwrap();

        // Key field assertions from the story acceptance criteria.
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.layer_types.len(), 7);
        assert_eq!(cfg.index_topk, 1024);
        assert_eq!(cfg.o_groups, 8);

        // Core deserialized fields.
        assert_eq!(cfg.hidden_size, 8);
        assert_eq!(cfg.num_hidden_layers, 7);
        assert_eq!(cfg.o_lora_rank, 128);
        assert_eq!(cfg.num_hash_layers, 3);
        assert_eq!(cfg.num_nextn_predict_layers, 1);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.compress_rope_theta, 160_000.0);
        assert_eq!(cfg.compress_rates.compressed_sparse_attention, 4);
        assert_eq!(cfg.compress_rates.heavily_compressed_attention, 128);
        assert_eq!(cfg.compress_ratios, vec![0, 0, 4, 128, 4, 128, 4, 0]);
        assert_eq!(cfg.topk_method, TopkMethod::NoauxTc);
        assert_eq!(cfg.scoring_func, ScoringFunc::SqrtSoftplus);
        assert_eq!(cfg.rope_scaling.scaling_type, ScaledRopeType::Yarn);

        // Per-layer derivation: main layers use theta=10000, compressors use
        // theta=160000 (Yarn) with their per-type compression rate.
        let layers = cfg.layer_configs();
        assert_eq!(layers.len(), 7);
        assert_eq!(layers[0].layer_type, LayerType::SlidingAttention);
        assert_eq!(layers[0].compress_rate, 0);
        assert_eq!(layers[0].rope_theta, 10_000.0);
        assert!(!layers[0].use_yarn);
        assert_eq!(layers[2].layer_type, LayerType::CompressedSparseAttention);
        assert_eq!(layers[2].compress_rate, 4);
        assert_eq!(layers[2].rope_theta, 160_000.0);
        assert!(layers[2].use_yarn);
        assert_eq!(layers[3].layer_type, LayerType::HeavilyCompressedAttention);
        assert_eq!(layers[3].compress_rate, 128);
        assert_eq!(layers[3].rope_theta, 160_000.0);
        assert!(layers[3].use_yarn);
    }
}
