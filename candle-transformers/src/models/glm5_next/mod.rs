//! GLM-5.3-Flash text configuration (story #4353).
//!
//! `Glm5NextTextConfig` deserializes the official `zai-org/GLM-5.3-Flash`
//! `text_config` (a `glm5_next_text` transformers config) into a minimal set
//! of fields needed for inference, resolving the per-layer attention / MLP /
//! indexer schedules with the same defaults the reference `transformers`
//! `Glm5NextTextConfig` applies.
//!
//! This slice establishes the weight/config contract only: it does not build
//! the GLM-5.3 attention/MLP blocks. Where GLM-5.3 reuses an architecture
//! primitive that already exists in the `deepseek_v4` module (Manifold
//! Constrained Hyper-Connection, and the routed/shared MoE expert weights),
//! this module reuses that code by implementing the small config traits those
//! primitives are generic over, rather than copying them.

use crate::models::deepseek_v4::{HyperConnectionConfig, MoeExpertsConfig};
use crate::serde_default_fn;
use candle_nn::Activation;
use serde::Deserialize;

serde_default_fn!(usize, default_num_key_value_heads, 64);
serde_default_fn!(usize, default_n_shared_experts, 1);
serde_default_fn!(f64, default_routed_scaling_factor, 1.0);
serde_default_fn!(usize, default_qk_rope_head_dim, 0);
serde_default_fn!(usize, default_n_group, 1);
serde_default_fn!(usize, default_topk_group, 1);
serde_default_fn!(bool, default_norm_topk_prob, true);
serde_default_fn!(Activation, default_hidden_act, Activation::Silu);
serde_default_fn!(f32, default_initializer_range, 0.02);
serde_default_fn!(f64, default_rms_norm_eps, 1e-5);
serde_default_fn!(bool, default_use_cache, true);
serde_default_fn!(usize, default_pad_token_id, 154820);
serde_default_fn!(bool, default_tie_word_embeddings, false);
serde_default_fn!(bool, default_attention_bias, false);
serde_default_fn!(f64, default_attention_dropout, 0.0);
serde_default_fn!(usize, default_index_topk, 2048);
serde_default_fn!(usize, default_index_head_dim, 128);
serde_default_fn!(usize, default_index_n_heads, 32);
serde_default_fn!(usize, default_head_dim, 0);
serde_default_fn!(f64, default_swiglu_limit, 10.0);
serde_default_fn!(Option<f64>, default_linear_lower_bound, Some(-5.0));
serde_default_fn!(usize, default_hc_mult, 4);
serde_default_fn!(f64, default_hc_eps, 1e-6);
serde_default_fn!(usize, default_hc_sinkhorn_iters, 20);
serde_default_fn!(bool, default_output_router_logits, false);
serde_default_fn!(f64, default_router_aux_loss_coef, 0.001);
serde_default_fn!(usize, default_index_kpool, 16);
serde_default_fn!(bool, default_index_kpool_always_select_tail, true);
serde_default_fn!(usize, default_first_k_dense_replace, 3);
serde_default_fn!(bool, default_mhc, true);
serde_default_fn!(bool, default_mla_use_nope, true);
serde_default_fn!(bool, default_index_kpool_compress, true);
serde_default_fn!(bool, default_index_share_for_mtp_iteration, false);
serde_default_fn!(bool, default_indexer_rope_interleave, true);

/// Per-layer attention kind. GLM-5.3 alternates linear (KDA) layers with
/// deepseek-sparse-attention (DSA / MLA) layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    DeepseekSparseAttention,
}

/// Per-layer feed-forward kind: dense MLP or sparse (routed) MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpLayerType {
    Dense,
    Sparse,
}

/// Per-layer DSA indexer mode: `full` runs the indexer, `shared` reuses the
/// previous full layer's top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexerType {
    Full,
    Shared,
}

/// KDA (linear-attention) parameters shipped as the nested `linear_attn_config`
/// dict in the official checkpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LinearAttnConfig {
    pub num_heads: usize,
    pub gate_lower_bound: Option<f64>,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
    #[serde(rename = "kda_layers")]
    pub kda_layers: Vec<usize>,
    #[serde(rename = "full_attn_layers")]
    pub full_attn_layers: Vec<usize>,
}

impl Default for LinearAttnConfig {
    fn default() -> Self {
        Self {
            num_heads: 64,
            gate_lower_bound: Some(-5.0),
            head_dim: 128,
            short_conv_kernel_size: 4,
            kda_layers: Vec::new(),
            full_attn_layers: Vec::new(),
        }
    }
}

/// GLM-5.3-Flash text configuration (`glm5_next_text`).
///
/// Fields mirror the official `zai-org/GLM-5.3-Flash` `text_config`. Fields the
/// reference config always ships are required; fields it may omit (or that a
/// reduced test config omits) carry serde defaults matching `transformers`.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm5NextTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: usize,
    pub n_routed_experts: usize,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
    #[serde(default)]
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<Vec<usize>>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub mlp_layer_types: Vec<MlpLayerType>,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,
    #[serde(default = "default_index_topk")]
    pub index_topk: usize,
    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: usize,
    #[serde(default = "default_index_n_heads")]
    pub index_n_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub indexer_types: Vec<IndexerType>,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f64,
    #[serde(default)]
    pub linear_attn_config: LinearAttnConfig,
    #[serde(default = "default_linear_lower_bound")]
    pub linear_lower_bound: Option<f64>,
    #[serde(default = "default_hc_mult")]
    pub hc_mult: usize,
    #[serde(default = "default_hc_eps")]
    pub hc_eps: f64,
    #[serde(default = "default_hc_sinkhorn_iters")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "default_output_router_logits")]
    pub output_router_logits: bool,
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f64,
    #[serde(default = "default_index_kpool")]
    pub index_kpool: usize,
    #[serde(default = "default_index_kpool_always_select_tail")]
    pub index_kpool_always_select_tail: bool,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_mhc")]
    pub mhc: bool,
    #[serde(default = "default_mla_use_nope")]
    pub mla_use_nope: bool,
    #[serde(default = "default_index_kpool_compress")]
    pub index_kpool_compress: bool,
    #[serde(default = "default_index_share_for_mtp_iteration")]
    pub index_share_for_mtp_iteration: bool,
    #[serde(default = "default_indexer_rope_interleave")]
    pub indexer_rope_interleave: bool,
}

impl Glm5NextTextConfig {
    /// Resolve the per-layer attention schedule. The official checkpoint ships
    /// an explicit `layer_types`; when absent, GLM-5.3 uses a DSA layer on every
    /// 4th index (`idx % 4 == 3`) and linear (KDA) elsewhere, matching
    /// `transformers`.
    pub fn effective_layer_types(&self) -> Vec<LayerType> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if i % 4 == 3 {
                    LayerType::DeepseekSparseAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    /// Resolve the per-layer MLP schedule. The official checkpoint ships an
    /// explicit `mlp_layer_types`; when absent, the first
    /// `first_k_dense_replace` layers are dense and the rest sparse.
    pub fn effective_mlp_layer_types(&self) -> Vec<MlpLayerType> {
        if !self.mlp_layer_types.is_empty() {
            return self.mlp_layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if i < self.first_k_dense_replace {
                    MlpLayerType::Dense
                } else {
                    MlpLayerType::Sparse
                }
            })
            .collect()
    }

    /// Resolve the per-layer DSA indexer schedule. The official checkpoint
    /// ships an explicit `indexer_types` (all `full`); when absent, every DSA
    /// layer runs the indexer in `full` mode.
    pub fn effective_indexer_types(&self) -> Vec<IndexerType> {
        if !self.indexer_types.is_empty() {
            return self.indexer_types.clone();
        }
        vec![IndexerType::Full; self.num_hidden_layers]
    }

    /// Indices of the DSA (sparse-attention) layers.
    pub fn dsa_layer_indices(&self) -> Vec<usize> {
        self.effective_layer_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::DeepseekSparseAttention)
            .map(|(i, _)| i)
            .collect()
    }

    /// Indices of the linear (KDA) attention layers.
    pub fn kda_layer_indices(&self) -> Vec<usize> {
        self.effective_layer_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::LinearAttention)
            .map(|(i, _)| i)
            .collect()
    }

    /// Indices of the sparse (routed MoE) MLP layers.
    pub fn sparse_mlp_layer_indices(&self) -> Vec<usize> {
        self.effective_mlp_layer_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == MlpLayerType::Sparse)
            .map(|(i, _)| i)
            .collect()
    }

    /// Effective number of KDA linear-attention heads.
    pub fn linear_num_heads(&self) -> usize {
        if self.linear_attn_config.num_heads != 0 {
            self.linear_attn_config.num_heads
        } else {
            64
        }
    }

    /// Effective KDA linear-attention head dim.
    pub fn linear_head_dim(&self) -> usize {
        if self.linear_attn_config.head_dim != 0 {
            self.linear_attn_config.head_dim
        } else {
            128
        }
    }

    /// Effective KDA short-conv kernel size.
    pub fn linear_conv_kernel_dim(&self) -> usize {
        if self.linear_attn_config.short_conv_kernel_size != 0 {
            self.linear_attn_config.short_conv_kernel_size
        } else {
            4
        }
    }

    /// Effective KDA gate lower bound (forget-gate decay floor).
    pub fn linear_lower_bound(&self) -> Option<f64> {
        self.linear_attn_config
            .gate_lower_bound
            .or(self.linear_lower_bound)
    }
}

/// GLM-5.3 reuses the DeepSeek-V4 Manifold-Constrained Hyper-Connection
/// components directly (they are generic over this trait).
impl HyperConnectionConfig for Glm5NextTextConfig {
    fn hc_mult(&self) -> usize {
        self.hc_mult
    }
    fn hc_sinkhorn_iters(&self) -> usize {
        self.hc_sinkhorn_iters
    }
    fn hc_eps(&self) -> f64 {
        self.hc_eps
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
}

/// GLM-5.3 reuses the DeepSeek-V4 routed/shared MoE expert weights directly
/// (they are generic over this trait). FP8 compute is a non-goal for this slice.
impl MoeExpertsConfig for Glm5NextTextConfig {
    fn n_routed_experts(&self) -> usize {
        self.n_routed_experts
    }
    fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn swiglu_limit(&self) -> f64 {
        self.swiglu_limit
    }
    #[cfg(feature = "cuda")]
    fn fp8_compute(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deepseek_v4::DeepseekV4HyperConnection;
    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    /// A reduced, official-shaped `glm5_next_text` config: real field names and
    /// the nested `linear_attn_config`, but tiny sizes so it fits a unit test.
    /// Deliberately omits the optional fields (`layer_types`, `mlp_layer_types`,
    /// `indexer_types`) to exercise the schedule defaults.
    fn reduced_config() -> Glm5NextTextConfig {
        serde_json::from_str(
            r#"{
                "vocab_size": 1024, "hidden_size": 32, "intermediate_size": 64,
                "moe_intermediate_size": 16, "num_hidden_layers": 8,
                "num_attention_heads": 4, "num_key_value_heads": 4,
                "n_shared_experts": 1, "n_routed_experts": 8,
                "routed_scaling_factor": 2.5, "kv_lora_rank": 8, "q_lora_rank": 16,
                "qk_rope_head_dim": 0, "v_head_dim": 8, "qk_nope_head_dim": 8,
                "n_group": 1, "topk_group": 1, "num_experts_per_tok": 2,
                "norm_topk_prob": true, "hidden_act": "silu",
                "max_position_embeddings": 1024, "rms_norm_eps": 1e-5,
                "use_cache": true, "pad_token_id": 1020,
                "tie_word_embeddings": false, "attention_bias": false,
                "attention_dropout": 0.0, "index_topk": 32, "index_head_dim": 8,
                "index_n_heads": 2, "head_dim": 0, "swiglu_limit": 10.0,
                "linear_attn_config": {"num_heads": 2, "gate_lower_bound": -5.0,
                    "head_dim": 8, "short_conv_kernel_size": 2,
                    "kda_layers": [0,1,2,4,5,6], "full_attn_layers": [3,7]},
                "hc_mult": 2, "hc_eps": 1e-6, "hc_sinkhorn_iters": 5,
                "output_router_logits": false, "router_aux_loss_coef": 0.001,
                "index_kpool": 2, "index_kpool_always_select_tail": true,
                "first_k_dense_replace": 2, "mhc": true, "mla_use_nope": true,
                "index_kpool_compress": true, "index_share_for_mtp_iteration": false,
                "indexer_rope_interleave": true
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn official_config_deserializes_with_defaults() {
        let cfg = reduced_config();
        assert_eq!(cfg.vocab_size, 1024);
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.moe_intermediate_size, 16);
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.qk_rope_head_dim, 0);
        assert_eq!(cfg.hc_mult, 2);
        assert_eq!(cfg.linear_attn_config.num_heads, 2);
        assert_eq!(cfg.linear_head_dim(), 8);
        assert_eq!(cfg.linear_conv_kernel_dim(), 2);
        assert_eq!(cfg.linear_lower_bound(), Some(-5.0));
        assert_eq!(cfg.head_dim, 0);
    }

    #[test]
    fn layer_schedule_validation() {
        let cfg = reduced_config();

        // Default attention schedule (no explicit layer_types): DSA on every
        // 4th layer (idx % 4 == 3), linear elsewhere.
        let attn = cfg.effective_layer_types();
        assert_eq!(attn.len(), 8);
        assert_eq!(attn[0], LayerType::LinearAttention);
        assert_eq!(attn[3], LayerType::DeepseekSparseAttention);
        assert_eq!(attn[7], LayerType::DeepseekSparseAttention);
        assert_eq!(cfg.dsa_layer_indices(), vec![3, 7]);
        assert_eq!(cfg.kda_layer_indices(), vec![0, 1, 2, 4, 5, 6]);

        // Default MLP schedule (no explicit mlp_layer_types): first
        // first_k_dense_replace (2) layers dense, rest sparse.
        let mlp = cfg.effective_mlp_layer_types();
        assert_eq!(mlp.len(), 8);
        assert_eq!(mlp[0], MlpLayerType::Dense);
        assert_eq!(mlp[1], MlpLayerType::Dense);
        assert_eq!(mlp[2], MlpLayerType::Sparse);
        assert_eq!(cfg.sparse_mlp_layer_indices(), vec![2, 3, 4, 5, 6, 7]);

        // Default indexer schedule: all full.
        let idx = cfg.effective_indexer_types();
        assert_eq!(idx.len(), 8);
        assert!(idx.iter().all(|t| *t == IndexerType::Full));
    }

    /// GLM-5.3 reuses the DeepSeek-V4 mHC component via the
    /// `HyperConnectionConfig` trait rather than copying it.
    #[test]
    fn reuses_deepseek_v4_mhc() -> candle::Result<()> {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        let hc = cfg.hc_mult; // 2
        let mix = (2 + hc) * hc; // 8
        let mut tensors = HashMap::new();
        tensors.insert(
            "fn".to_string(),
            Tensor::zeros((mix, hc * cfg.hidden_size), DType::F32, &dev)?,
        );
        tensors.insert("base".to_string(), Tensor::zeros((mix,), DType::F32, &dev)?);
        tensors.insert("scale".to_string(), Tensor::zeros((3,), DType::F32, &dev)?);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let _comp = DeepseekV4HyperConnection::new(&cfg, vb)?;
        Ok(())
    }
}
