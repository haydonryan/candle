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

use super::deepseek2::{BincountOp, TopKLastDimOp};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{rms_norm, Activation, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

/// DeepSeek-V4 fp8/fp4 quantized weight loading + CPU/disk offload.
pub mod quantized;

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

/// Unweighted RMS normalization, matching DeepSeek-V4 `UnweightedRMSNorm`:
/// `x * rsqrt(mean(x^2) + eps)`. The norm factor is computed in f32 and the
/// multiplication is performed in the input's dtype (as in the reference).
#[derive(Debug, Clone, Copy)]
pub struct UnweightedRMSNorm {
    eps: f64,
}

impl UnweightedRMSNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f = x.to_dtype(DType::F32)?;
        let norm = (x_f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        let norm = norm.to_dtype(x.dtype())?;
        x.broadcast_mul(&norm)
    }
}

/// Block-diagonal grouped linear (DeepSeek-V4 `o_a_proj`).
///
/// The `n_groups` heads are split into independent blocks; each block projects
/// its `in_per_group`-dim slice to `out_per_group` via a block-diagonal bmm.
/// Input is `(..., n_groups, in_per_group)`, output `(..., n_groups, out_per_group)`.
#[derive(Debug, Clone)]
pub struct GroupedLinear {
    weight: Tensor,
    n_groups: usize,
}

impl GroupedLinear {
    /// `weight` has the flat shape `(n_groups * out_per_group, in_per_group)`.
    pub fn new(weight: Tensor, n_groups: usize) -> Self {
        Self { weight, n_groups }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ndim = x.rank();
        if ndim < 2 {
            candle::bail!(
                "GroupedLinear expects input of rank >= 2 (..., n_groups, in_per_group), got {x:?}"
            );
        }
        let in_per_group = self.weight.dim(1)?;
        let out_per_group = self.weight.dim(0)? / self.n_groups;
        let batch: usize = x.dims()[..ndim - 2].iter().product();
        // (n_groups, out_per_group, in_per_group) -> (n_groups, in_per_group, out_per_group)
        let w = self
            .weight
            .reshape((self.n_groups, out_per_group, in_per_group))?
            .transpose(1, 2)?
            .contiguous()?;
        let xr = x
            .reshape((batch, self.n_groups, in_per_group))?
            .transpose(0, 1)?
            .contiguous()?;
        let y = xr.matmul(&w)?.transpose(0, 1)?; // (batch, n_groups, out_per_group)
        let mut out_dims = x.dims()[..ndim - 2].to_vec();
        out_dims.push(self.n_groups);
        out_dims.push(out_per_group);
        y.reshape(out_dims)
    }
}

/// RoPE variant used by a V4 layer: main (plain theta) or compress (Yarn).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeVariant {
    Main,
    Compress,
}

/// Precomputed interleaved RoPE cos/sin tables for one variant.
///
/// Each table stores one entry per interleaved pair, so `cos`/`sin` have
/// `rope_dim / 2` columns (the layout `rope_i` expects).
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// `(max_position_embeddings, rope_dim / 2)` in F32.
    pub sin: Tensor,
    pub cos: Tensor,
}

impl RopeTable {
    /// Plain (unscaled) RoPE: `inv_freq = 1 / theta^(2i/rope_dim)`.
    fn unscaled(theta: f32, rope_dim: usize, max_seq: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / rope_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Yarn-scaled RoPE for the compressor layers (`compress_rope_theta`,
    /// factor 16, beta_fast 32, beta_slow 1, orig 65536). Attention factor is
    /// forced to 1.0, so no mscale is applied to the cos/sin.
    fn yarn(cfg: &DeepseekV4Config, rope_dim: usize, dev: &Device) -> Result<Self> {
        let base = cfg.compress_rope_theta;
        let factor = cfg.rope_scaling.factor;
        let beta_fast = cfg.rope_scaling.beta_fast;
        let beta_slow = cfg.rope_scaling.beta_slow;
        let original_max = cfg.rope_scaling.original_max_position_embeddings;
        let max_seq = cfg.max_position_embeddings;

        let pos_freqs: Vec<_> = (0..rope_dim)
            .step_by(2)
            .map(|i| base.powf(i as f32 / rope_dim as f32))
            .collect();
        let inv_freq_extra: Vec<_> = pos_freqs.iter().map(|f| 1.0 / f).collect();
        let inv_freq_inter: Vec<_> = pos_freqs.iter().map(|f| 1.0 / (factor * f)).collect();
        let n_extra = inv_freq_extra.len();
        let n_inter = inv_freq_inter.len();
        let inv_freq_extra = Tensor::from_vec(inv_freq_extra, (1, n_extra), dev)?;
        let inv_freq_inter = Tensor::from_vec(inv_freq_inter, (1, n_inter), dev)?;

        let (low, high) =
            yarn_find_correction_range(beta_fast, beta_slow, rope_dim, base, original_max);
        let inv_freq_mask = (1. - yarn_linear_ramp_mask(low, high, rope_dim / 2, dev)?)?;
        let inv_freq = inv_freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&inv_freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }
}

fn yarn_find_correction_dim(
    num_rot: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * std::f32::consts::PI)).ln())
        / (2. * base.ln())
}

fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> (f32, f32) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
    (low.max(0.), high.min(dim as f32 - 1.))
}

fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
    if min == max {
        max += 0.001;
    }
    let linear_func =
        ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
    linear_func.clamp(0., 1.)
}

/// Multi-variant interleaved RoPE for DeepSeek-V4.
///
/// Only the trailing `rope_dim = head_dim * partial_rotary_factor` slice of each
/// head is rotated (leading nope channels pass through unchanged); the rotation
/// runs in f32 via the interleaved `rope_i` kernel.
#[derive(Debug, Clone)]
pub struct DeepseekV4RotaryEmbedding {
    main: RopeTable,
    compress: RopeTable,
    rope_dim: usize,
}

impl DeepseekV4RotaryEmbedding {
    pub fn new(cfg: &DeepseekV4Config, dev: &Device) -> Result<Self> {
        Self::new_with_head_dim(cfg, cfg.head_dim, dev)
    }

    /// Build a rotary embedding sized for a custom `head_dim` (used by the
    /// indexer, which runs its own scaled-down compressor at `index_head_dim`).
    /// `rope_dim` is rounded down to even because the interleaved `rope_i`
    /// kernel requires an even rotation width.
    pub fn new_with_head_dim(
        cfg: &DeepseekV4Config,
        head_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_dim = (head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let rope_dim = if rope_dim.is_multiple_of(2) {
            rope_dim
        } else {
            rope_dim.saturating_sub(1)
        };
        Self::new_with_rope_dim(cfg, rope_dim, dev)
    }

    /// Build a rotary embedding with an explicit `rope_dim` rotation width
    /// (used by the indexer, which rotates the full `index_head_dim` slice).
    pub fn new_with_rope_dim(
        cfg: &DeepseekV4Config,
        rope_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let main = RopeTable::unscaled(cfg.rope_theta, rope_dim, cfg.max_position_embeddings, dev)?;
        let compress = RopeTable::yarn(cfg, rope_dim, dev)?;
        Ok(Self {
            main,
            compress,
            rope_dim,
        })
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    pub fn table(&self, variant: RopeVariant) -> &RopeTable {
        match variant {
            RopeVariant::Main => &self.main,
            RopeVariant::Compress => &self.compress,
        }
    }

    /// Apply RoPE to the trailing `rope_dim` slice of a `(b, h, t, head_dim)`
    /// tensor, leaving the leading nope channels unchanged.
    pub fn forward(
        &self,
        x: &Tensor,
        variant: RopeVariant,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.apply(x, variant, seqlen_offset, 1.0)
    }

    /// Apply the conjugate rotation (`-sin`) at the query positions. Used to undo
    /// the RoPE that V (==K) carries on its rope slice from the attention output.
    pub fn forward_conjugate(
        &self,
        x: &Tensor,
        variant: RopeVariant,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.apply(x, variant, seqlen_offset, -1.0)
    }

    fn apply(
        &self,
        x: &Tensor,
        variant: RopeVariant,
        seqlen_offset: usize,
        sin_scale: f64,
    ) -> Result<Tensor> {
        if self.rope_dim == 0 {
            return Ok(x.clone());
        }
        let (_, _, seq_len, head_dim) = x.dims4()?;
        let nope_dim = head_dim - self.rope_dim;
        let table = self.table(variant);
        let nope = x.narrow(D::Minus1, 0, nope_dim)?;
        let rope = x.narrow(D::Minus1, nope_dim, self.rope_dim)?;
        let cos = table.cos.narrow(0, seqlen_offset, seq_len)?.contiguous()?;
        let sin = (table.sin.narrow(0, seqlen_offset, seq_len)? * sin_scale)?.contiguous()?;
        let rope_f = rope.to_dtype(DType::F32)?.contiguous()?;
        let rotated = candle_nn::rotary_emb::rope_i(&rope_f, &cos, &sin)?.to_dtype(x.dtype())?;
        Tensor::cat(&[nope, rotated], D::Minus1)
    }

    /// Apply RoPE to the trailing `rope_dim` slice of a `(b, h, t, head_dim)`
    /// tensor using explicit per-position indices (for compressed KV entries
    /// whose absolute positions are strided by `compress_rate`).
    pub fn forward_at_positions(
        &self,
        x: &Tensor,
        variant: RopeVariant,
        positions: &Tensor,
    ) -> Result<Tensor> {
        if self.rope_dim == 0 {
            return Ok(x.clone());
        }
        let (_, _, _, head_dim) = x.dims4()?;
        let nope_dim = head_dim - self.rope_dim;
        let table = self.table(variant);
        let nope = x.narrow(D::Minus1, 0, nope_dim)?;
        let rope = x.narrow(D::Minus1, nope_dim, self.rope_dim)?;
        let cos = table.cos.index_select(positions, 0)?.contiguous()?;
        let sin = table.sin.index_select(positions, 0)?.contiguous()?;
        let rope_f = rope.to_dtype(DType::F32)?.contiguous()?;
        let rotated = candle_nn::rotary_emb::rope_i(&rope_f, &cos, &sin)?.to_dtype(x.dtype())?;
        Tensor::cat(&[nope, rotated], D::Minus1)
    }
}

/// Sinkhorn double-normalization (DeepSeek-V4 HC `comb`).
///
/// Starting from a logits matrix `x`, applies row softmax + `eps`, then column
/// normalization, then `iters - 1` alternating row/column normalizations.
pub fn sinkhorn(x: &Tensor, eps: f64, iters: usize) -> Result<Tensor> {
    let soft = candle_nn::ops::softmax_last_dim(x)?;
    let mut comb = (soft + eps)?;
    let col_sum = (comb.sum_keepdim(D::Minus2)? + eps)?;
    comb = comb.broadcast_div(&col_sum)?;
    for _ in 0..iters.saturating_sub(1) {
        let row_sum = (comb.sum_keepdim(D::Minus1)? + eps)?;
        comb = comb.broadcast_div(&row_sum)?;
        let col_sum = (comb.sum_keepdim(D::Minus2)? + eps)?;
        comb = comb.broadcast_div(&col_sum)?;
    }
    Ok(comb)
}

/// `sqrt(softplus(x))` = `sqrt(ln(1 + e^x))` — DeepSeek-V4 router scoring.
pub fn sqrt_softplus(xs: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(xs)?;
    let sp = xs.exp()?.add(&one)?.log()?;
    sp.sqrt()
}

/// DeepSeek-V4 sliding-window attention (`DeepseekV4Attention`, sliding path).
///
/// Exact MLA math from `modeling_deepseek_v4.py`:
/// `q_a_proj(hidden -> q_lora_rank) -> RMSNorm -> q_b_proj(-> num_heads*head_dim)
/// -> UnweightedRMSNorm`; a single shared KV head (`num_key_value_heads=1`,
/// K == V) via `kv_proj(hidden -> head_dim) -> RMSNorm`; partial/interleaved RoPE
/// on the trailing `qk_rope_head_dim` slice of q and kv; a per-head learnable
/// sink appended pre-softmax and dropped after; a grouped output `o_a_proj`
/// (block-diagonal over `o_groups`) then `o_b_proj`; and a sliding-window KV
/// cache that keeps the last `sliding_window - 1` tokens.
pub struct DeepseekV4Attention {
    q_a_proj: Linear,
    q_a_norm: RmsNorm,
    q_b_proj: Linear,
    q_b_norm: UnweightedRMSNorm,
    kv_proj: Linear,
    kv_norm: RmsNorm,
    o_a_proj: GroupedLinear,
    o_b_proj: Linear,
    sinks: Tensor,
    rotary_emb: Arc<DeepseekV4RotaryEmbedding>,
    rope_variant: RopeVariant,
    softmax_scale: f64,
    sliding_window: usize,
    kv_cache: Option<Tensor>,
    compressor: Option<DeepseekV4Compressor>,
    use_flash_attn: bool,
    cfg: DeepseekV4Config,
}

/// Flash DSA kernel wrapper for DeepSeek-V4, gated by the `flash-attn` feature.
///
/// Fuses the exact eager op: `QK^T * scale + additive_mask` (sliding-window
/// local prefix + per-query compressed `block_bias`), a per-head sink logit
/// appended pre-softmax, max-subtract (sink included), softmax, drop-sink, and
/// `@V` — all in one CUDA kernel. Layouts: `q` `[B, Sq, H, D]`, `k`/`v`
/// `[B, Skv, Hk, D]` (MQA `Hk` divides `H`), `block_bias_mask` `[B, 1, Sq, Skv]`
/// F32 `0`/`-inf`, `sink_logits` `[H]` F32. Returns `[B, Sq, H, D]`.
#[cfg(feature = "flash-attn")]
fn flash_attn_blockmask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block_bias_mask: &Tensor,
    sink_logits: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn_windowed_blockmask(
        q,
        k,
        v,
        block_bias_mask,
        sink_logits,
        softmax_scale,
    )
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn_blockmask(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _block_bias_mask: &Tensor,
    _sink_logits: &Tensor,
    _softmax_scale: f32,
) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl DeepseekV4Attention {
    pub fn new(
        cfg: &DeepseekV4Config,
        layer_idx: usize,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let rope_variant = if cfg.layer_types[layer_idx] == LayerType::SlidingAttention {
            RopeVariant::Main
        } else {
            RopeVariant::Compress
        };
        let q_a_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.q_lora_rank, vb.pp("q_a_proj"))?;
        let q_a_norm = rms_norm(cfg.q_lora_rank, cfg.rms_norm_eps, vb.pp("q_a_norm"))?;
        let q_b_proj =
            candle_nn::linear_no_bias(cfg.q_lora_rank, num_heads * head_dim, vb.pp("q_b_proj"))?;
        let q_b_norm = UnweightedRMSNorm::new(cfg.rms_norm_eps);
        let kv_proj = candle_nn::linear_no_bias(cfg.hidden_size, head_dim, vb.pp("kv_proj"))?;
        let kv_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("kv_norm"))?;
        // `o_a_proj` weight is `(o_groups * o_lora_rank, num_heads*head_dim / o_groups)`.
        let o_a_weight = vb.get(
            (
                cfg.o_groups * cfg.o_lora_rank,
                num_heads * head_dim / cfg.o_groups,
            ),
            "o_a_proj.weight",
        )?;
        let o_a_proj = GroupedLinear::new(o_a_weight, cfg.o_groups);
        let o_b_proj = candle_nn::linear_no_bias(
            cfg.o_groups * cfg.o_lora_rank,
            cfg.hidden_size,
            vb.pp("o_b_proj"),
        )?;
        let sinks = vb.get((num_heads,), "sinks")?;
        let compressor =
            DeepseekV4Compressor::new(cfg, cfg.layer_types[layer_idx], vb.pp("compressor"))?;
        Ok(Self {
            q_a_proj,
            q_a_norm,
            q_b_proj,
            q_b_norm,
            kv_proj,
            kv_norm,
            o_a_proj,
            o_b_proj,
            sinks,
            rotary_emb: Arc::new(DeepseekV4RotaryEmbedding::new(cfg, vb.device())?),
            rope_variant,
            softmax_scale: (head_dim as f64).powf(-0.5),
            sliding_window: cfg.sliding_window,
            kv_cache: None,
            compressor,
            use_flash_attn,
            cfg: cfg.clone(),
        })
    }
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        if let Some(c) = &mut self.compressor {
            c.clear_cache();
        }
    }

    /// Sliding-window causal additive mask over `(q_len, kv_len)`.
    ///
    /// Query `i` (absolute position `seqlen_offset + i`) may attend to KV `j`
    /// (absolute position `seqlen_offset - prev_len + j`) iff the distance
    /// `i + prev_len - j` is in `[0, sliding_window)`; everything else is `-inf`.
    fn sliding_window_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        prev_len: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let inf = f32::NEG_INFINITY;
        let mut mask = vec![inf; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let d = i as isize + prev_len as isize - j as isize;
                if d >= 0 && (d as usize) < self.sliding_window {
                    mask[i * kv_len + j] = 0.0;
                }
            }
        }
        Tensor::from_vec(mask, (q_len, kv_len), dev)?
            .unsqueeze(0)?
            .unsqueeze(0)
    }

    /// Keep only the last `sliding_window - 1` tokens of the K==V cache.
    fn trim_cache(&self, kv: &Tensor) -> Result<Tensor> {
        let t = kv.dim(2)?;
        let keep = self.sliding_window.saturating_sub(1).min(t);
        if keep == t {
            Ok(kv.clone())
        } else {
            kv.narrow(D::Minus2, t - keep, keep)
        }
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let num_heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim;
        let variant = self.rope_variant;

        // Query: q_a -> RMSNorm -> q_b -> UnweightedRMSNorm.
        let q_residual = self.q_a_norm.forward(&self.q_a_proj.forward(xs)?)?;
        let q = self
            .q_b_proj
            .forward(&q_residual)?
            .reshape((bs, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?;
        let q = self.q_b_norm.forward(&q)?;
        let q = self.rotary_emb.forward(&q, variant, seqlen_offset)?;

        // Single shared KV head (K == V).
        let kv = self
            .kv_norm
            .forward(&self.kv_proj.forward(xs)?)?
            .reshape((bs, seq_len, 1, head_dim))?
            .transpose(1, 2)?;
        let kv = self.rotary_emb.forward(&kv, variant, seqlen_offset)?;

        // Sliding-window cache update: return `full` (prev + current) for
        // attention, store only the last `sliding_window - 1` tokens.
        let (k, v) = match &self.kv_cache {
            None => {
                self.kv_cache = Some(self.trim_cache(&kv)?);
                (kv.clone(), kv.clone())
            }
            Some(prev) => {
                let full = Tensor::cat(&[prev, &kv], 2)?;
                self.kv_cache = Some(self.trim_cache(&full)?);
                (full.clone(), full.clone())
            }
        };

        let kv_len = k.dim(2)?;
        let prev_len = kv_len - seq_len;

        // CSA layer: run the compressor, append the long-range compressed KV
        // entries to the sliding window and extend the mask with its block_bias.
        // An external/padding attention mask (additive `[B, 1, Sq, kv_len]`,
        // `0` attend / `-inf` pad) is folded into the local sliding columns so
        // batched padded prompts stay correct for both eager and flash paths.
        let (k, v, mask) = match &mut self.compressor {
            None => {
                let mut mask = self.sliding_window_mask(seq_len, kv_len, prev_len, xs.device())?;
                if let Some(ext) = attention_mask {
                    mask = mask.broadcast_add(ext)?;
                }
                (k, v, mask)
            }
            Some(comp) => {
                let (compressed_kv, block_bias) = comp.forward(xs, &q_residual, seqlen_offset)?;
                let k = Tensor::cat(&[k.clone(), compressed_kv.clone()], 2)?;
                let v = Tensor::cat(&[v, compressed_kv], 2)?;
                let sliding = self
                    .sliding_window_mask(seq_len, kv_len, prev_len, xs.device())?
                    .broadcast_as((bs, 1, seq_len, kv_len))?;
                let sliding = match attention_mask {
                    Some(ext) => sliding.broadcast_add(ext)?,
                    None => sliding,
                };
                let mask = Tensor::cat(&[sliding, block_bias], 3)?;
                (k, v, mask)
            }
        };

        let kv_len = k.dim(2)?;

        // Attention over the combined [sliding | compressed] KV. The flash DSA
        // kernel fuses mask + per-head sink + max-subtract + softmax + drop-sink
        // internally; the eager path materializes them explicitly.
        let attn_out = if self.use_flash_attn {
            // The kernel needs a batch-materialized additive mask [B, 1, Sq, Skv].
            let mask = mask.broadcast_as((bs, 1, seq_len, kv_len))?;
            // Transpose to the kernel layout [B, S, H, D] (MQA: Hkv == 1).
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let sink = self.sinks.to_dtype(DType::F32)?;
            let out = flash_attn_blockmask(&q, &k, &v, &mask, &sink, self.softmax_scale as f32)?;
            // Kernel returns [B, Sq, H, D]; undo RoPE in [B, H, Sq, D] layout.
            self.rotary_emb
                .forward_conjugate(&out.transpose(1, 2)?, variant, seqlen_offset)?
        } else {
            // Single shared KV head is broadcast to every query head for the bmm.
            let k = k.broadcast_as((bs, num_heads, kv_len, head_dim))?;
            let v = v.broadcast_as((bs, num_heads, kv_len, head_dim))?;
            let orig_dtype = q.dtype();
            let qf = q.to_dtype(DType::F32)?;
            let kf = k.to_dtype(DType::F32)?;
            let vf = v.to_dtype(DType::F32)?;
            let att = (qf.contiguous()?.matmul(&kf.t()?.contiguous()?)? * self.softmax_scale)?;
            let att = att.broadcast_add(&mask.to_dtype(DType::F32)?)?;

            // Per-head learnable sink appended pre-softmax, dropped after.
            let sinks = self
                .sinks
                .to_dtype(DType::F32)?
                .reshape((1, num_heads, 1, 1))?
                .broadcast_as((bs, num_heads, seq_len, 1))?;
            let att = Tensor::cat(&[att, sinks], D::Minus1)?.contiguous()?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let att = att.narrow(D::Minus1, 0, kv_len)?;

            let attn_out = att.contiguous()?.matmul(&vf.contiguous()?)?;
            let attn_out = attn_out.to_dtype(orig_dtype)?;

            // K == V, so V carries RoPE on its rope slice; undo it at the query
            // positions so each KV contribution stays a function of relative distance.
            self.rotary_emb
                .forward_conjugate(&attn_out, variant, seqlen_offset)?
        };

        // Grouped low-rank output: o_a (block-diagonal over o_groups) then o_b.
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?; // (bs, seq_len, num_heads, head_dim)
        let grouped = attn_out.reshape((
            bs,
            seq_len,
            self.cfg.o_groups,
            num_heads * head_dim / self.cfg.o_groups,
        ))?;
        let grouped = self.o_a_proj.forward(&grouped)?;
        let grouped = grouped.reshape((bs, seq_len, self.cfg.o_groups * self.cfg.o_lora_rank))?;
        self.o_b_proj.forward(&grouped)
    }
}

/// Per-row top-k along the last dim, sorted descending (largest first).
/// Returns the indices of the `k` largest values of each row.
fn topk_last_dim(x: &Tensor, k: usize) -> Result<Tensor> {
    let sorted = x.contiguous()?.arg_sort_last_dim(false)?;
    sorted
        .narrow(D::Minus1, 0, k)?
        .contiguous()?
        .to_dtype(DType::I64)
}

/// Persistent CSA compression state shared by the outer compressor and the
/// indexer (mirrors `DeepseekV4CSACache`'s `"compressor"`/`"indexer"` entries).
///
/// Tracks the per-name buffer of un-consumed source-token projections, the
/// running list of compressed KV entries emitted so far, the window count, and
/// the previous window's Ca overlap slice (carried across forward calls so the
/// first window of the next call can fold in the prior window's Ca series).
#[derive(Debug, Clone)]
pub struct CsaCompressionState {
    buffer_kv: Option<Tensor>,
    buffer_gate: Option<Tensor>,
    compressed_kv: Option<Tensor>,
    entry_count: usize,
    overlap_kv: Option<Tensor>,
    overlap_gate: Option<Tensor>,
}

impl Default for CsaCompressionState {
    fn default() -> Self {
        Self::new()
    }
}

impl CsaCompressionState {
    pub fn new() -> Self {
        Self {
            buffer_kv: None,
            buffer_gate: None,
            compressed_kv: None,
            entry_count: 0,
            overlap_kv: None,
            overlap_gate: None,
        }
    }

    /// Running compressed KV `[B, T, head_dim]`, or empty `[B, 0, head_dim]`.
    fn running_compressed(&self, device: &Device, head_dim: usize, dtype: DType) -> Result<Tensor> {
        match &self.compressed_kv {
            Some(c) => Ok(c.clone()),
            None => Tensor::zeros((1, 0, head_dim), dtype, device),
        }
    }

    /// `store_compression_weights`: fold new projections into the buffer, peel
    /// off the longest window-aligned prefix, keep the remainder buffered, and
    /// return `(chunk_kv, chunk_gate, first_window_position)`.
    fn store(
        &mut self,
        kv: &Tensor,
        gate: &Tensor,
        compress_rate: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let first_window_position = self.entry_count * compress_rate;
        let (kv, gate) = match (&self.buffer_kv, &self.buffer_gate) {
            (Some(bk), Some(bg)) => (Tensor::cat(&[bk, kv], 1)?, Tensor::cat(&[bg, gate], 1)?),
            _ => (kv.clone(), gate.clone()),
        };
        let len = kv.dim(1)?;
        let usable = (len / compress_rate) * compress_rate;
        let rem = len - usable;
        if rem > 0 {
            self.buffer_kv = Some(kv.narrow(1, usable, rem)?);
            self.buffer_gate = Some(gate.narrow(1, usable, rem)?);
        } else {
            self.buffer_kv = None;
            self.buffer_gate = None;
        }
        let chunk_kv = kv.narrow(1, 0, usable)?;
        let chunk_gate = gate.narrow(1, 0, usable)?;
        Ok((chunk_kv, chunk_gate, first_window_position))
    }

    /// `update_overlap_state`: return the previous window's Ca slice (zero-kv /
    /// `-inf`-gate on the first call) and persist the current chunk's last-window
    /// Ca slice for the next forward call.
    fn update_overlap(&mut self, ca_kv: &Tensor, ca_gate: &Tensor) -> Result<(Tensor, Tensor)> {
        let (prior_kv, prior_gate) = (self.overlap_kv.take(), self.overlap_gate.take());
        let batch = ca_kv.dim(0)?;
        let rate = ca_kv.dim(2)?;
        let head_dim = ca_kv.dim(3)?;
        let n_windows = ca_kv.dim(1)?;
        let device = ca_kv.device();
        let prior_kv = match prior_kv {
            Some(p) => p,
            None => Tensor::zeros((batch, rate, head_dim), ca_kv.dtype(), device)?,
        };
        let prior_gate = match prior_gate {
            Some(p) => p,
            None => Tensor::full(f32::NEG_INFINITY, (batch, rate, head_dim), device)?
                .to_dtype(ca_gate.dtype())?,
        };
        let last_kv = ca_kv
            .narrow(1, n_windows - 1, 1)?
            .squeeze(1)?
            .contiguous()?;
        let last_gate = ca_gate
            .narrow(1, n_windows - 1, 1)?
            .squeeze(1)?
            .contiguous()?;
        self.overlap_kv = Some(last_kv);
        self.overlap_gate = Some(last_gate);
        Ok((prior_kv, prior_gate))
    }

    /// Feed new `kv`/`gate` source projections (shape `[B, S, 2*head_dim]`),
    /// compress every complete window into one entry via the Ca/Cb overlap
    /// layout, apply compress-RoPE at the strided absolute positions, append to
    /// the running compressed KV, and return `(running_compressed, n_new)`.
    #[allow(clippy::too_many_arguments)]
    fn compress(
        &mut self,
        kv: &Tensor,
        gate: &Tensor,
        position_bias: &Tensor,
        compress_rate: usize,
        head_dim: usize,
        kv_norm: &RmsNorm,
        rotary_emb: &DeepseekV4RotaryEmbedding,
    ) -> Result<(Tensor, usize)> {
        let (chunk_kv, chunk_gate, first_window_position) = self.store(kv, gate, compress_rate)?;
        let batch = chunk_kv.dim(0)?;
        let len = chunk_kv.dim(1)?;
        let n_windows = len / compress_rate;
        if n_windows == 0 {
            return Ok((
                self.running_compressed(chunk_kv.device(), head_dim, chunk_kv.dtype())?,
                0,
            ));
        }
        let feat = 2 * head_dim;
        let chunk_kv = chunk_kv.reshape((batch, n_windows, compress_rate, feat))?;
        let chunk_gate = chunk_gate
            .reshape((batch, n_windows, compress_rate, feat))?
            .broadcast_add(position_bias)?;
        // Ca = [..., :head_dim] (contributes to the *next* window's entry),
        // Cb = [..., head_dim:] (contributes to the *current* window's entry).
        let ca_kv = chunk_kv.narrow(3, 0, head_dim)?;
        let ca_gate = chunk_gate.narrow(3, 0, head_dim)?;
        let cb_kv = chunk_kv.narrow(3, head_dim, head_dim)?;
        let cb_gate = chunk_gate.narrow(3, head_dim, head_dim)?;

        // First window's Ca half comes from the previous forward call's overlap
        // (zero-kv / -inf-gate on the very first call -> softmax weight 0).
        let (prior_kv, prior_gate) = self.update_overlap(&ca_kv, &ca_gate)?;
        let ca_full_kv = if n_windows > 1 {
            Tensor::cat(
                &[prior_kv.unsqueeze(1)?, ca_kv.narrow(1, 0, n_windows - 1)?],
                1,
            )?
        } else {
            prior_kv.unsqueeze(1)?
        };
        let ca_full_gate = if n_windows > 1 {
            Tensor::cat(
                &[
                    prior_gate.unsqueeze(1)?,
                    ca_gate.narrow(1, 0, n_windows - 1)?,
                ],
                1,
            )?
        } else {
            prior_gate.unsqueeze(1)?
        };

        // Lay out `[B, n_win, 2*rate, head_dim]`: Cb (current window) in the
        // second half, Ca of the previous window in the first half.
        let new_kv = Tensor::cat(&[ca_full_kv, cb_kv], 2)?;
        let new_gate = Tensor::cat(&[ca_full_gate, cb_gate], 2)?;
        // Softmax in fp32 for stability, then gated weighted sum over width 2m.
        let gate_softmax = candle_nn::ops::softmax(&new_gate.to_dtype(DType::F32)?, 2)?
            .to_dtype(chunk_kv.dtype())?;
        let summed = (new_kv * gate_softmax)?.sum(2)?; // [B, n_win, head_dim]
        let compressed = kv_norm.forward(&summed)?; // [B, n_win, head_dim]

        // Compress-RoPE at `i * rate + first_window_position` for each new entry.
        let positions: Vec<u32> = (0..n_windows)
            .map(|w| (w * compress_rate + first_window_position) as u32)
            .collect();
        let positions = Tensor::from_vec(positions, (n_windows,), chunk_kv.device())?;
        let compressed = compressed.unsqueeze(1)?; // [B, 1, n_win, head_dim]
        let compressed = rotary_emb
            .forward_at_positions(&compressed, RopeVariant::Compress, &positions)?
            .squeeze(1)?; // [B, n_win, head_dim]

        // Append to the running compressed KV and bump the window count.
        let running = match &self.compressed_kv {
            None => compressed.clone(),
            Some(old) => Tensor::cat(&[old, &compressed], 1)?,
        };
        self.compressed_kv = Some(running.clone());
        self.entry_count += n_windows;
        Ok((running, n_windows))
    }
}

/// Lightning-indexer scoring head: `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`.
pub struct DeepseekV4IndexerScorer {
    softmax_scale: f64,
    weights_scaling: f64,
    weights_proj: Linear,
}

impl DeepseekV4IndexerScorer {
    fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            softmax_scale: (cfg.index_head_dim as f64).powf(-0.5),
            weights_scaling: (cfg.index_n_heads as f64).powf(-0.5),
            weights_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.index_n_heads,
                vb.pp("weights_proj"),
            )?,
        })
    }

    /// `q` is `[B, S, H, D]` (post-RoPE), `compressed_kv` is `[B, T, D]`.
    /// Returns `[B, S, T]`.
    fn forward(
        &self,
        q: &Tensor,
        compressed_kv: &Tensor,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _h, _d) = q.dims4()?;
        let d = compressed_kv.dim(2)?;
        let t = compressed_kv.dim(1)?;
        // Broadcast the single KV head to the query batch/seq dims before matmul.
        let ck = compressed_kv
            .transpose(1, 2)?
            .unsqueeze(1)?
            .broadcast_as((batch, seq_len, d, t))?; // [B, S, D, T]
        let scores = q.to_dtype(DType::F32)?.matmul(&ck.to_dtype(DType::F32)?)?; // [B,S,H,T]
        let scores = (scores.relu()? * self.softmax_scale)?;
        let weights = (self
            .weights_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?
            * self.weights_scaling)?; // [B, S, H]
        let weights = weights.unsqueeze(3)?; // [B, S, H, 1]
        scores.broadcast_mul(&weights)?.sum(2)
    }
}

/// Lightning Indexer (paper §2.3.1, eqs. 13-17): per-query top-`index_topk`
/// compressed entries. Runs its own scaled-down compressor at `index_head_dim`
/// over the same windows as the outer CSA compressor, then scores queries with
/// `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and returns the top-`index_topk`
/// compressed indices per query, with `-1` sentinels for invalid (future /
/// not-yet-ready) picks.
pub struct DeepseekV4Indexer {
    compress_rate: usize,
    num_heads: usize,
    head_dim: usize,
    index_topk: usize,
    kv_proj: Linear,
    gate_proj: Linear,
    position_bias: Tensor,
    kv_norm: RmsNorm,
    q_b_proj: Linear,
    rotary_emb: Arc<DeepseekV4RotaryEmbedding>,
    scorer: DeepseekV4IndexerScorer,
    state: CsaCompressionState,
}

impl DeepseekV4Indexer {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.index_head_dim;
        Ok(Self {
            compress_rate: cfg.compress_rates.compressed_sparse_attention,
            num_heads: cfg.index_n_heads,
            head_dim,
            index_topk: cfg.index_topk,
            kv_proj: candle_nn::linear_no_bias(cfg.hidden_size, 2 * head_dim, vb.pp("kv_proj"))?,
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                2 * head_dim,
                vb.pp("gate_proj"),
            )?,
            position_bias: vb.get(
                (cfg.compress_rates.compressed_sparse_attention, 2 * head_dim),
                "position_bias",
            )?,
            kv_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("kv_norm"))?,
            q_b_proj: candle_nn::linear_no_bias(
                cfg.q_lora_rank,
                cfg.index_n_heads * head_dim,
                vb.pp("q_b_proj"),
            )?,
            rotary_emb: Arc::new(DeepseekV4RotaryEmbedding::new_with_rope_dim(
                cfg,
                cfg.index_head_dim,
                vb.device(),
            )?),
            scorer: DeepseekV4IndexerScorer::new(cfg, vb.pp("scorer"))?,
            state: CsaCompressionState::new(),
        })
    }

    /// Returns per-query top-`index_topk` compressed indices `[B, S, k]` (i64),
    /// with `-1` marking picks that point past the causal threshold.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        q_residual: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let kv = self.kv_proj.forward(hidden_states)?; // [B, S, 2*ihd]
        let gate = self.gate_proj.forward(hidden_states)?;
        let (compressed_kv, _n) = self.state.compress(
            &kv,
            &gate,
            &self.position_bias,
            self.compress_rate,
            self.head_dim,
            &self.kv_norm,
            &self.rotary_emb,
        )?;
        // compressed_kv: [B, T, index_head_dim]

        // Query: q_b(q_residual) -> [B, S, H, D], compress-RoPE at position_ids.
        let q = self
            .q_b_proj
            .forward(q_residual)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, H, S, D]
        let q = self
            .rotary_emb
            .forward(&q, RopeVariant::Compress, seqlen_offset)?;
        let q = q.transpose(1, 2)?; // [B, S, H, D]

        // No compressed entries yet (early decode): nothing to score/index.
        let compressed_len = compressed_kv.dim(1)?;
        let top_k = self.index_topk.min(compressed_len);
        if compressed_len == 0 || top_k == 0 {
            return Tensor::zeros((batch, seq_len, 0), DType::I64, device);
        }

        let index_scores = self.scorer.forward(&q, &compressed_kv, hidden_states)?; // [B, S, T]

        // Query `t` may only attend to compressed entries `w` with `t > w*rate`.
        let threshold: Vec<f32> = (0..seq_len)
            .map(|t| ((seqlen_offset + t + 1) / self.compress_rate) as f32)
            .collect();
        let threshold =
            Tensor::from_vec(threshold, (1, seq_len), device)?.broadcast_as((batch, seq_len))?;
        let entry = Tensor::arange(0u32, compressed_len as u32, device)?.to_dtype(DType::F32)?;
        let entry_b = entry.reshape((1, 1, compressed_len))?.broadcast_as((
            batch,
            seq_len,
            compressed_len,
        ))?;
        let thresh_b = threshold.reshape((batch, seq_len, 1))?.broadcast_as((
            batch,
            seq_len,
            compressed_len,
        ))?;
        let future_mask = entry_b.ge(&thresh_b)?; // [B, S, T]
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch, seq_len, compressed_len), device)?;
        let masked = future_mask.where_cond(&neg_inf, &index_scores)?;
        let topk_indices = topk_last_dim(&masked, top_k)?; // [B, S, k]

        // Picks that still point past the threshold are invalid -> `-1` sentinel.
        let thresh_k = threshold
            .reshape((batch, seq_len, 1))?
            .broadcast_as((batch, seq_len, top_k))?;
        let invalid = topk_indices.to_dtype(DType::F32)?.ge(&thresh_k)?;
        let minus_one = Tensor::full(-1i64, (batch, seq_len, top_k), device)?;
        invalid.where_cond(&minus_one, &topk_indices)
    }
}

/// Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9-17). Compresses
/// every `compress_rate` source tokens into a single KV entry via the Ca/Cb
/// two-series overlap layout and runs a Lightning Indexer on top that scores
/// queries against the compressed KV to gather the per-query top-`index_topk`
/// entries. Produces `(compressed_kv [B, 1, T, head_dim], block_bias [B, 1, S, T])`.
pub struct DeepseekV4CSACompressor {
    compress_rate: usize,
    head_dim: usize,
    kv_proj: Linear,
    gate_proj: Linear,
    position_bias: Tensor,
    kv_norm: RmsNorm,
    rotary_emb: Arc<DeepseekV4RotaryEmbedding>,
    indexer: DeepseekV4Indexer,
    state: CsaCompressionState,
}

impl DeepseekV4CSACompressor {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let compress_rate = cfg.compress_rates.compressed_sparse_attention;
        Ok(Self {
            compress_rate,
            head_dim,
            kv_proj: candle_nn::linear_no_bias(cfg.hidden_size, 2 * head_dim, vb.pp("kv_proj"))?,
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                2 * head_dim,
                vb.pp("gate_proj"),
            )?,
            position_bias: vb.get((compress_rate, 2 * head_dim), "position_bias")?,
            kv_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("kv_norm"))?,
            rotary_emb: Arc::new(DeepseekV4RotaryEmbedding::new(cfg, vb.device())?),
            indexer: DeepseekV4Indexer::new(cfg, vb.pp("indexer"))?,
            state: CsaCompressionState::new(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.state = CsaCompressionState::new();
        self.indexer.state = CsaCompressionState::new();
    }

    /// `hidden_states` `[B, S, hidden]`, `q_residual` `[B, S, q_lora_rank]`.
    /// Returns `(compressed_kv [B, 1, T, head_dim], block_bias [B, 1, S, T])`.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        q_residual: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let kv = self.kv_proj.forward(hidden_states)?; // [B, S, 2*hd]
        let gate = self.gate_proj.forward(hidden_states)?;
        let (compressed, _n) = self.state.compress(
            &kv,
            &gate,
            &self.position_bias,
            self.compress_rate,
            self.head_dim,
            &self.kv_norm,
            &self.rotary_emb,
        )?;
        let compressed_kv = compressed.unsqueeze(1)?; // [B, 1, T, head_dim]

        // Per-query top-k picks from the Lightning Indexer.
        let top_k_indices = self
            .indexer
            .forward(hidden_states, q_residual, seqlen_offset)?; // [B, S, k]
        let compressed_len = compressed_kv.dim(2)?;
        let top_k = top_k_indices.dim(2)?;
        // No compressed entries yet (early decode): empty block_bias over the
        // empty compressed suffix (avoids elementwise ops on 0-element CUDA
        // tensors, which candle-core's CUDA kernels reject).
        if compressed_len == 0 || top_k == 0 {
            let empty_bb = Tensor::zeros((batch, seq_len, 0), DType::F32, device)?.unsqueeze(1)?;
            return Ok((compressed_kv, empty_bb));
        }
        let c_len = compressed_len as i64;
        // Safe indices: valid picks keep their index, invalid (-1) point at a
        // padding column that is sliced off below.
        let valid = top_k_indices.ge(0i64)?; // [B, S, k]
        let padding = Tensor::full(c_len, (batch, seq_len, top_k), device)?;
        let safe = valid.where_cond(&top_k_indices, &padding)?; // [B, S, k]

        // block_bias: 0 where a query selected the compressed entry, -inf else.
        let cand = Tensor::arange(0u32, compressed_len as u32, device)?
            .to_dtype(DType::I64)?
            .reshape((1, 1, 1, compressed_len))?;
        let safe_b = safe
            .unsqueeze(3)?
            .broadcast_as((batch, seq_len, top_k, compressed_len))?;
        let cand_b = cand.broadcast_as((batch, seq_len, top_k, compressed_len))?;
        let eq = safe_b.eq(&cand_b)?; // [B, S, k, T]
        let selected = eq.to_dtype(DType::F32)?.sum(2)?.gt(0f32)?; // [B, S, T]
        let zeros = Tensor::zeros((batch, seq_len, compressed_len), DType::F32, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch, seq_len, compressed_len), device)?;
        let block_bias = selected.where_cond(&zeros, &neg_inf)?.unsqueeze(1)?; // [B, 1, S, T]
        Ok((compressed_kv, block_bias))
    }
}

/// A compressor layer — Compressed Sparse Attention (with its Lightning
/// Indexer) or Heavily Compressed Attention (causality-only block bias).
pub enum DeepseekV4Compressor {
    Csa(Box<DeepseekV4CSACompressor>),
    Hca(Box<DeepseekV4HCACompressor>),
}

impl DeepseekV4Compressor {
    pub fn new(
        cfg: &DeepseekV4Config,
        layer_type: LayerType,
        vb: VarBuilder,
    ) -> Result<Option<Self>> {
        match layer_type {
            LayerType::CompressedSparseAttention => Ok(Some(Self::Csa(Box::new(
                DeepseekV4CSACompressor::new(cfg, vb)?,
            )))),
            LayerType::HeavilyCompressedAttention => Ok(Some(Self::Hca(Box::new(
                DeepseekV4HCACompressor::new(cfg, vb)?,
            )))),
            LayerType::SlidingAttention => Ok(None),
        }
    }

    pub fn clear_cache(&mut self) {
        match self {
            Self::Csa(c) => c.clear_cache(),
            Self::Hca(c) => c.clear_cache(),
        }
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        q_residual: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Csa(c) => c.forward(hidden_states, q_residual, seqlen_offset),
            Self::Hca(c) => c.forward(hidden_states, q_residual, seqlen_offset),
        }
    }
}

/// HCA compression state: buffered source projections, the running list of
/// emitted compressed entries, and the total window count.
#[derive(Default)]
struct HcaCompressionState {
    buffer_kv: Option<Tensor>,
    buffer_gate: Option<Tensor>,
    compressed_kv: Option<Tensor>,
    entry_count: usize,
}

impl HcaCompressionState {
    /// Running compressed KV `[B, T, head_dim]`, or empty `[B, 0, head_dim]`.
    fn running_compressed(&self, device: &Device, head_dim: usize, dtype: DType) -> Result<Tensor> {
        match &self.compressed_kv {
            Some(c) => Ok(c.clone()),
            None => Tensor::zeros((1, 0, head_dim), dtype, device),
        }
    }

    /// Concatenate new `(kv, gate)` projections with the buffered remainder,
    /// peel off the longest window-aligned prefix, keep the leftover, and
    /// return `(chunk_kv, chunk_gate, first_window_position)`.
    fn store(
        &mut self,
        kv: &Tensor,
        gate: &Tensor,
        compress_rate: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let first_window_position = self.entry_count * compress_rate;
        let (kv, gate) = match (&self.buffer_kv, &self.buffer_gate) {
            (Some(bk), Some(bg)) => (Tensor::cat(&[bk, kv], 1)?, Tensor::cat(&[bg, gate], 1)?),
            _ => (kv.clone(), gate.clone()),
        };
        let len = kv.dim(1)?;
        let usable = (len / compress_rate) * compress_rate;
        let rem = len - usable;
        if rem > 0 {
            self.buffer_kv = Some(kv.narrow(1, usable, rem)?);
            self.buffer_gate = Some(gate.narrow(1, usable, rem)?);
        } else {
            self.buffer_kv = None;
            self.buffer_gate = None;
        }
        let chunk_kv = kv.narrow(1, 0, usable)?;
        let chunk_gate = gate.narrow(1, 0, usable)?;
        Ok((chunk_kv, chunk_gate, first_window_position))
    }

    /// Compress every complete non-overlapping window into one KV entry via a
    /// softmax-gated weighted sum, apply compress-RoPE at the strided absolute
    /// positions, append to the running compressed KV, and return
    /// `(running_compressed, n_new)`.
    #[allow(clippy::too_many_arguments)]
    fn compress(
        &mut self,
        kv: &Tensor,
        gate: &Tensor,
        position_bias: &Tensor,
        compress_rate: usize,
        head_dim: usize,
        kv_norm: &RmsNorm,
        rotary_emb: &DeepseekV4RotaryEmbedding,
    ) -> Result<(Tensor, usize)> {
        let (chunk_kv, chunk_gate, first_window_position) = self.store(kv, gate, compress_rate)?;
        let batch = chunk_kv.dim(0)?;
        let len = chunk_kv.dim(1)?;
        let n_windows = len / compress_rate;
        if n_windows == 0 {
            return Ok((
                self.running_compressed(chunk_kv.device(), head_dim, chunk_kv.dtype())?,
                0,
            ));
        }
        let chunk_kv = chunk_kv.reshape((batch, n_windows, compress_rate, head_dim))?;
        let chunk_gate = chunk_gate
            .reshape((batch, n_windows, compress_rate, head_dim))?
            .broadcast_add(position_bias)?;
        // Softmax over the window in fp32, then the gated weighted sum.
        let gate_softmax = candle_nn::ops::softmax(&chunk_gate.to_dtype(DType::F32)?, 2)?
            .to_dtype(chunk_kv.dtype())?;
        let summed = (chunk_kv * gate_softmax)?.sum(2)?; // [B, n_win, head_dim]
        let compressed = kv_norm.forward(&summed)?; // [B, n_win, head_dim]

        // Compress-RoPE at `i * rate + first_window_position`.
        let positions: Vec<u32> = (0..n_windows)
            .map(|w| (w * compress_rate + first_window_position) as u32)
            .collect();
        let positions = Tensor::from_vec(positions, (n_windows,), chunk_gate.device())?;
        let compressed = compressed.unsqueeze(1)?; // [B, 1, n_win, head_dim]
        let compressed = rotary_emb
            .forward_at_positions(&compressed, RopeVariant::Compress, &positions)?
            .squeeze(1)?; // [B, n_win, head_dim]

        let running = match &self.compressed_kv {
            None => compressed.clone(),
            Some(old) => Tensor::cat(&[old, &compressed], 1)?,
        };
        self.compressed_kv = Some(running.clone());
        self.entry_count += n_windows;
        Ok((running, n_windows))
    }
}

/// Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20-23).
/// Compresses every `compress_rate` source tokens into a single compressed KV
/// entry via a softmax-gated weighted sum over each non-overlapping window
/// (kv_proj + gate_proj + position_bias at head_dim). Produces
/// `(compressed_kv [B, 1, T, head_dim], block_bias [B, 1, S, T])` where
/// block_bias is causality-only: entry `w` is visible to a query at absolute
/// position `p` iff `w < (p + 1) // compress_rate` (no indexer).
pub struct DeepseekV4HCACompressor {
    compress_rate: usize,
    head_dim: usize,
    kv_proj: Linear,
    gate_proj: Linear,
    position_bias: Tensor,
    kv_norm: RmsNorm,
    rotary_emb: Arc<DeepseekV4RotaryEmbedding>,
    state: HcaCompressionState,
}

impl DeepseekV4HCACompressor {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let compress_rate = cfg.compress_rates.heavily_compressed_attention;
        Ok(Self {
            compress_rate,
            head_dim,
            kv_proj: candle_nn::linear_no_bias(cfg.hidden_size, head_dim, vb.pp("kv_proj"))?,
            gate_proj: candle_nn::linear_no_bias(cfg.hidden_size, head_dim, vb.pp("gate_proj"))?,
            position_bias: vb.get((compress_rate, head_dim), "position_bias")?,
            kv_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("kv_norm"))?,
            rotary_emb: Arc::new(DeepseekV4RotaryEmbedding::new(cfg, vb.device())?),
            state: HcaCompressionState::default(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.state = HcaCompressionState::default();
    }

    /// `hidden_states` `[B, S, hidden]`, `q_residual` `[B, S, q_lora_rank]`.
    /// Returns `(compressed_kv [B, 1, T, head_dim], block_bias [B, 1, S, T])`.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        _q_residual: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let kv = self.kv_proj.forward(hidden_states)?; // [B, S, head_dim]
        let gate = self.gate_proj.forward(hidden_states)?;
        let (compressed, _n) = self.state.compress(
            &kv,
            &gate,
            &self.position_bias,
            self.compress_rate,
            self.head_dim,
            &self.kv_norm,
            &self.rotary_emb,
        )?;
        let compressed_kv = compressed.unsqueeze(1)?; // [B, 1, T, head_dim]
        let compressed_len = compressed_kv.dim(2)?;
        // No compressed entries yet (early decode): empty block_bias (avoids
        // elementwise/to_dtype ops on 0-element CUDA tensors).
        if compressed_len == 0 {
            let empty_bb = Tensor::zeros((batch, seq_len, 0), DType::F32, device)?.unsqueeze(1)?;
            return Ok((compressed_kv, empty_bb));
        }

        // Causality-only block_bias: entry `w` is masked for query `t` (at
        // absolute position `seqlen_offset + t`) when `w >= (pos + 1) // rate`.
        let threshold: Vec<f32> = (0..seq_len)
            .map(|t| ((seqlen_offset + t + 1) / self.compress_rate) as f32)
            .collect();
        let entry = Tensor::arange(0u32, compressed_len as u32, device)?.to_dtype(DType::F32)?;
        let entry_b = entry.reshape((1, 1, 1, compressed_len))?.broadcast_as((
            1,
            1,
            seq_len,
            compressed_len,
        ))?;
        let thresh_b = Tensor::from_vec(threshold, (1, seq_len), device)?
            .reshape((1, 1, seq_len, 1))?
            .broadcast_as((1, 1, seq_len, compressed_len))?;
        let future = entry_b.ge(&thresh_b)?; // [1, 1, S, T]
        let zeros = Tensor::zeros((1, 1, seq_len, compressed_len), DType::F32, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (1, 1, seq_len, compressed_len), device)?;
        let block_bias = future.where_cond(&neg_inf, &zeros)?.broadcast_as((
            batch,
            1,
            seq_len,
            compressed_len,
        ))?;
        Ok((compressed_kv, block_bias))
    }
}

/// Manifold-Constrained Hyper-Connections (mHC) mixer (paper §2.2, eq. 8;
/// `DeepseekV4HyperConnection` in `modeling_deepseek_v4.py`).
///
/// Owns the learned `fn` / `base` / `scale` parameters that turn the incoming
/// `hc_mult` parallel residual streams (shape `[B, S, hc_mult, D]`) into three
/// sets of weights: `pre` (stream-collapse weights for the sublayer input),
/// `post` (block-output placement, range `[0, 2]`) and `comb` (an `H×H` stream
/// mixer, Sinkhorn-projected onto the doubly-stochastic manifold).
pub struct DeepseekV4HyperConnection {
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    input_norm: UnweightedRMSNorm,
    fn_: Tensor,
    base: Tensor,
    scale: Tensor,
}

impl DeepseekV4HyperConnection {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let hc = cfg.hc_mult;
        let mix = (2 + hc) * hc;
        let fn_ = vb.get((mix, hc * cfg.hidden_size), "fn")?;
        let base = vb.get((mix,), "base")?;
        let scale = vb.get((3,), "scale")?;
        Ok(Self {
            hc_mult: hc,
            hc_sinkhorn_iters: cfg.hc_sinkhorn_iters,
            hc_eps: cfg.hc_eps,
            input_norm: UnweightedRMSNorm::new(cfg.rms_norm_eps),
            fn_,
            base,
            scale,
        })
    }

    /// Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8)
    /// and the collapsed single sequence fed into the sublayer.
    ///
    /// Returns `(post [B,S,H], comb [B,S,H,H], collapsed [B,S,D])`.
    pub fn forward(&self, hidden_streams: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let hc = self.hc_mult;
        // Flatten streams to [B, S, H*D] and run the unweighted RMS norm in f32.
        let flat = self
            .input_norm
            .forward(&hidden_streams.flatten_from(2)?.to_dtype(DType::F32)?)?;
        // F.linear(flat, fn) = flat @ fn^T (flattened to 2D for matmul).
        let (b, sl, k) = flat.dims3()?;
        let lin = flat
            .reshape((b * sl, k))?
            .matmul(&self.fn_.t()?)?
            .reshape((b, sl, (2 + hc) * hc))?; // [B, S, (2+H)*H]
        let pre_w = lin.narrow(D::Minus1, 0, hc)?;
        let post_w = lin.narrow(D::Minus1, hc, hc)?;
        let comb_w = lin.narrow(D::Minus1, 2 * hc, hc * hc)?;
        let pre_b = self.base.narrow(0, 0, hc)?;
        let post_b = self.base.narrow(0, hc, hc)?;
        let comb_b = self.base.narrow(0, 2 * hc, hc * hc)?;
        let s = self.scale.to_vec1::<f32>()?;
        let (pre_scale, post_scale, comb_scale) = (s[0] as f64, s[1] as f64, s[2] as f64);

        let pre =
            (candle_nn::ops::sigmoid(&(pre_w * pre_scale)?.broadcast_add(&pre_b)?)? + self.hc_eps)?;
        let post =
            (candle_nn::ops::sigmoid(&(post_w * post_scale)?.broadcast_add(&post_b)?)? * 2.0)?;

        let comb_logits = {
            let (b, s_len, _) = lin.dims3()?;
            let cw = comb_w.reshape((b, s_len, hc, hc))?;
            let cb = comb_b.reshape((hc, hc))?;
            (cw * comb_scale)?.broadcast_add(&cb)?
        };
        let comb = sinkhorn(&comb_logits, self.hc_eps, self.hc_sinkhorn_iters)?; // [B,S,H,H]

        // Collapse the hc_mult streams into a single sequence with the `pre` weights.
        let collapsed = pre
            .unsqueeze(D::Minus1)?
            .broadcast_mul(hidden_streams)?
            .sum(D::Minus2)?
            .to_dtype(hidden_streams.dtype())?;
        Ok((post, comb, collapsed))
    }
}

/// Final HC-stream collapse (paper §2.2; `DeepseekV4HyperHead`). Reduces the
/// `hc_mult` residual streams down to a single sequence with per-stream
/// sigmoid-gated weights before the shared final RMSNorm.
pub struct DeepseekV4HyperHead {
    hc_eps: f64,
    input_norm: UnweightedRMSNorm,
    hc_fn: Tensor,
    hc_base: Tensor,
    hc_scale: Tensor,
}

impl DeepseekV4HyperHead {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let hc = cfg.hc_mult;
        let hc_fn = vb.get((hc, hc * cfg.hidden_size), "hc_fn")?;
        let hc_base = vb.get((hc,), "hc_base")?;
        let hc_scale = vb.get((1,), "hc_scale")?;
        Ok(Self {
            hc_eps: cfg.hc_eps,
            input_norm: UnweightedRMSNorm::new(cfg.rms_norm_eps),
            hc_fn,
            hc_base,
            hc_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let flat = self
            .input_norm
            .forward(&x.flatten_from(2)?.to_dtype(DType::F32)?)?;
        let (b, sl, k) = flat.dims3()?;
        let mixes = flat
            .reshape((b * sl, k))?
            .matmul(&self.hc_fn.t()?)?
            .reshape((b, sl, self.hc_fn.dim(0)?))?; // [B,S,H]
        let scale = self.hc_scale.to_vec1::<f32>()?[0] as f64;
        let pre = (candle_nn::ops::sigmoid(&(mixes * scale)?.broadcast_add(&self.hc_base)?)?
            + self.hc_eps)?;
        pre.unsqueeze(D::Minus1)?
            .broadcast_mul(x)?
            .sum(D::Minus2)?
            .to_dtype(x.dtype())
    }
}

/// Shared-expert MLP (`DeepseekV4MLP`): gate/up/down with SiLU and the
/// `swiglu_limit` clamp. Uses `moe_intermediate_size` as the hidden width (the
/// tiny config carries no separate dense `intermediate_size`).
pub struct DeepseekV4MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    limit: f64,
}

impl DeepseekV4MLP {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let inter = cfg.moe_intermediate_size;
        let hidden = cfg.hidden_size;
        let gate_proj = candle_nn::linear_no_bias(hidden, inter, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden, inter, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(inter, hidden, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            limit: cfg.swiglu_limit,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(x)?
            .clamp(f64::NEG_INFINITY, self.limit)?;
        let up = self.up_proj.forward(x)?.clamp(-self.limit, self.limit)?;
        let act = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&act)
    }
}

/// Routed expert weights stored as 3D tensors (`DeepseekV4Experts`):
/// `gate_up_proj` `[E, 2*inter, hidden]` and `down_proj` `[E, hidden, inter]`.
pub struct DeepseekV4Experts {
    num_experts: usize,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    limit: f64,
}

impl DeepseekV4Experts {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let e = cfg.n_routed_experts;
        let inter = cfg.moe_intermediate_size;
        let hidden = cfg.hidden_size;
        let gate_up_proj = vb.get((e, 2 * inter, hidden), "gate_up_proj")?;
        let down_proj = vb.get((e, hidden, inter), "down_proj")?;
        Ok(Self {
            num_experts: e,
            gate_up_proj,
            down_proj,
            limit: cfg.swiglu_limit,
        })
    }

    /// `hidden_states` is `[N, hidden]`; `top_k_index` and `top_k_weights` are
    /// `[N, K]`. Non-routed tokens contribute weight 0, matching the reference's
    /// `index_add_` over only the routed tokens.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        top_k_index: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        let n = hidden_states.dim(0)?;
        let k = top_k_index.dim(1)?;
        let dev = hidden_states.device();
        let dtype = hidden_states.dtype();
        let mut out = hidden_states.zeros_like()?;
        let zero = Tensor::zeros((n, k), dtype, dev)?;
        for i in 0..self.num_experts {
            let sel = top_k_index.eq(i as u32)?; // [N,K] bool
            let w = sel.where_cond(top_k_weights, &zero)?;
            let wsum = w.sum(D::Minus1)?; // [N]
            let gu = self.gate_up_proj.narrow(0, i, 1)?.squeeze(0)?; // [2*inter, hidden]
            let all = hidden_states.matmul(&gu.t()?)?; // [N, 2*inter]
            let chunks = all.chunk(2, D::Minus1)?;
            let gate = chunks[0].clamp(f64::NEG_INFINITY, self.limit)?;
            let up = chunks[1].clamp(-self.limit, self.limit)?;
            let act = (candle_nn::ops::silu(&gate)? * up)?; // [N, inter]
            let down_w = self.down_proj.narrow(0, i, 1)?.squeeze(0)?; // [hidden, inter]
            let down = act.matmul(&down_w.t()?)?; // [N, hidden]
            let contrib = down.broadcast_mul(&wsum.unsqueeze(D::Minus1)?)?;
            out = out.add(&contrib)?;
        }
        Ok(out)
    }
}

/// Learned top-k router (`DeepseekV4TopKRouter`): `sqrt_softplus` scores,
/// top-k indices, weights normalized over the selected experts and scaled by
/// `routed_scaling_factor`.
pub struct DeepseekV4TopKRouter {
    top_k: usize,
    hidden_size: usize,
    weight: Tensor,
    e_score_correction_bias: Tensor,
    routed_scaling_factor: f64,
}

impl DeepseekV4TopKRouter {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        let bias = match vb.get((cfg.n_routed_experts,), "e_score_correction_bias") {
            Ok(b) => b,
            Err(_) => Tensor::zeros(cfg.n_routed_experts, DType::F32, vb.device())?,
        };
        Ok(Self {
            top_k: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            weight: vb.get((cfg.n_routed_experts, cfg.hidden_size), "weight")?,
            e_score_correction_bias: bias,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// `x` is `[N, hidden]`; returns `(weights [N,K], indices [N,K])`.
    ///
    /// Selection uses `scores + e_score_correction_bias` (matching
    /// `modeling_deepseek_v4.py`), while the weights are gathered from the raw
    /// (unbiased) scores.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let n: usize = x.dims()[..x.rank() - 1].iter().product();
        let flat = x.reshape((n, self.hidden_size))?;
        let logits = flat.matmul(&self.weight.t()?)?; // [N, E]
        let scores = sqrt_softplus(&logits)?;
        let biased = scores.broadcast_add(&self.e_score_correction_bias)?;
        let indices = topk_last_dim(&biased, self.top_k)?; // [N, K]
        let weights = scores.gather(&indices, D::Minus1)?; // [N, K]
        let denom = (weights.sum(D::Minus1)?.unsqueeze(D::Minus1)? + 1e-20)?;
        let weights = weights.broadcast_div(&denom)?;
        let weights = (weights * self.routed_scaling_factor)?;
        Ok((weights, indices))
    }
}

/// Hash router (`DeepseekV4HashRouter`): expert selection via a frozen
/// token-id→expert-id `tid2eid` lookup; the learned gate still scores the
/// selected experts' activations.
pub struct DeepseekV4HashRouter {
    hidden_size: usize,
    weight: Tensor,
    tid2eid: Tensor,
    routed_scaling_factor: f64,
}

impl DeepseekV4HashRouter {
    pub fn new(cfg: &DeepseekV4Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            hidden_size: cfg.hidden_size,
            weight: vb.get((cfg.n_routed_experts, cfg.hidden_size), "weight")?,
            tid2eid: vb.get((cfg.vocab_size, cfg.num_experts_per_tok), "tid2eid")?,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// `x` is `[N, hidden]`, `input_ids` is `[N]`; returns
    /// `(weights [N,K], indices [N,K])`.
    pub fn forward(&self, x: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let n: usize = x.dims()[..x.rank() - 1].iter().product();
        let flat = x.reshape((n, self.hidden_size))?;
        let logits = flat.matmul(&self.weight.t()?)?;
        let scores = sqrt_softplus(&logits)?;
        let ids = input_ids.flatten_all()?.to_dtype(DType::I64)?; // [N]
        let indices = self
            .tid2eid
            .index_select(&ids, 0)?
            .to_dtype(DType::I64)?
            .contiguous()?; // [N, K]
        let weights = scores.gather(&indices, D::Minus1)?;
        let denom = (weights.sum(D::Minus1)?.unsqueeze(D::Minus1)? + 1e-20)?;
        let weights = weights.broadcast_div(&denom)?;
        let weights = (weights * self.routed_scaling_factor)?;
        Ok((weights, indices))
    }
}

enum DeepseekV4MoeGate {
    TopK(DeepseekV4TopKRouter),
    Hash(DeepseekV4HashRouter),
}

/// Sparse MoE block (`DeepseekV4SparseMoeBlock`): top-k or hash routing plus
/// routed experts plus a shared expert.
pub struct DeepseekV4SparseMoeBlock {
    gate: DeepseekV4MoeGate,
    experts: DeepseekV4Experts,
    shared_experts: DeepseekV4MLP,
}

impl DeepseekV4SparseMoeBlock {
    pub fn new(cfg: &DeepseekV4Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let gate = if layer_idx < cfg.num_hash_layers {
            DeepseekV4MoeGate::Hash(DeepseekV4HashRouter::new(cfg, vb.pp("gate"))?)
        } else {
            DeepseekV4MoeGate::TopK(DeepseekV4TopKRouter::new(cfg, vb.pp("gate"))?)
        };
        let experts = DeepseekV4Experts::new(cfg, vb.pp("experts"))?;
        let shared_experts = DeepseekV4MLP::new(cfg, vb.pp("shared_experts"))?;
        Ok(Self {
            gate,
            experts,
            shared_experts,
        })
    }

    /// `x` is `[B, S, D]`; `input_ids` (when hash-routed) matches `x`'s token
    /// count. Returns `[B, S, D]`.
    pub fn forward(&self, x: &Tensor, input_ids: Option<&Tensor>) -> Result<Tensor> {
        let (bs, seq, d) = x.dims3()?;
        let flat = x.reshape((bs * seq, d))?;
        let (weights, indices) = match &self.gate {
            DeepseekV4MoeGate::TopK(g) => g.forward(&flat)?,
            DeepseekV4MoeGate::Hash(g) => g.forward(&flat, input_ids.unwrap())?,
        };
        let routed = self
            .experts
            .forward(&flat, &indices, &weights)?
            .reshape((bs, seq, d))?;
        let shared = self.shared_experts.forward(x)?;
        routed.add(&shared)
    }
}

/// Auxiliary load-balancing loss (Switch Transformer), matching
/// `load_balancing_loss_func` in `modeling_deepseek_v4.py`. `gate_logits` are
/// the per-layer router logits (`[N, num_experts]`); `attention_mask` is the
/// flat per-token mask (weights padding tokens to zero).
pub fn load_balancing_loss(
    gate_logits: &[Tensor],
    num_experts: usize,
    top_k: usize,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if gate_logits.is_empty() {
        return Tensor::zeros(1, DType::F32, &Device::Cpu);
    }
    let mut tokens_per_expert_sum = vec![0.0f32; num_experts];
    let mut router_prob_sum = vec![0.0f32; num_experts];
    let mut total_rows = 0.0f32;
    let flat_mask: Option<Vec<f32>> = match attention_mask {
        Some(m) => Some(m.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    for layer_gate in gate_logits {
        let rw = candle_nn::ops::softmax_last_dim(layer_gate)?.to_dtype(DType::F32)?;
        let n = rw.dim(0)?;
        match &flat_mask {
            None => {
                let counts = rw
                    .topk_unsorted(top_k)?
                    .indices
                    .flatten_all()?
                    .bincount(num_experts as u32)?;
                for (i, c) in counts.iter().enumerate() {
                    tokens_per_expert_sum[i] += *c as f32;
                }
                let col = rw.sum_keepdim(0)?.flatten_all()?;
                let col = col.to_vec1::<f32>()?;
                for (i, s) in col.iter().enumerate() {
                    router_prob_sum[i] += s;
                }
                total_rows += n as f32;
            }
            Some(mask) => {
                let sel = rw.topk_unsorted(top_k)?.indices.flatten_all()?;
                let sel = sel.to_vec1::<u32>()?;
                for (pos, &e) in sel.iter().enumerate() {
                    tokens_per_expert_sum[e as usize] += mask[pos / top_k];
                }
                let mask_t = Tensor::from_vec(mask.clone(), n, &Device::Cpu)?;
                let col = rw
                    .broadcast_mul(&mask_t.unsqueeze(1)?)?
                    .sum_keepdim(0)?
                    .flatten_all()?;
                let col = col.to_vec1::<f32>()?;
                for (i, s) in col.iter().enumerate() {
                    router_prob_sum[i] += s;
                }
                total_rows += mask.iter().sum::<f32>();
            }
        }
    }
    let total = total_rows.max(1e-9);
    let mut overall = 0.0f32;
    for i in 0..num_experts {
        overall += (tokens_per_expert_sum[i] / total) * (router_prob_sum[i] / total);
    }
    Tensor::new(overall * num_experts as f32, &Device::Cpu)
}

/// One V4 decoder block (paper §2): an mHC hyper-connection around each of the
/// attention and MoE sublayers, with the `hc_mult` residual streams kept in
/// shape `[B, S, hc_mult, D]` throughout.
pub struct DeepseekV4DecoderLayer {
    self_attn: DeepseekV4Attention,
    mlp: DeepseekV4SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn_hc: DeepseekV4HyperConnection,
    ffn_hc: DeepseekV4HyperConnection,
}

impl DeepseekV4DecoderLayer {
    pub fn new(
        cfg: &DeepseekV4Config,
        layer_idx: usize,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn =
            DeepseekV4Attention::new(cfg, layer_idx, use_flash_attn, vb.pp("self_attn"))?;
        let mlp = DeepseekV4SparseMoeBlock::new(cfg, layer_idx, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let attn_hc = DeepseekV4HyperConnection::new(cfg, vb.pp("attn_hc"))?;
        let ffn_hc = DeepseekV4HyperConnection::new(cfg, vb.pp("ffn_hc"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attn_hc,
            ffn_hc,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        input_ids: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let (post, comb, collapsed) = self.attn_hc.forward(hidden_states)?;
        let normed = self.input_layernorm.forward(&collapsed)?;
        let attn_output = self.self_attn.forward(&normed, seqlen_offset, None)?;
        let hidden_states = post
            .to_dtype(dtype)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&attn_output.unsqueeze(D::Minus2)?)?
            .broadcast_add(
                &comb
                    .to_dtype(dtype)?
                    .transpose(D::Minus1, D::Minus2)?
                    .matmul(hidden_states)?,
            )?;

        let (post, comb, collapsed) = self.ffn_hc.forward(&hidden_states)?;
        let normed = self.post_attention_layernorm.forward(&collapsed)?;
        let mlp_output = self.mlp.forward(&normed, input_ids)?;
        post.to_dtype(dtype)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&mlp_output.unsqueeze(D::Minus2)?)?
            .broadcast_add(
                &comb
                    .to_dtype(dtype)?
                    .transpose(D::Minus1, D::Minus2)?
                    .matmul(&hidden_states)?,
            )
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Full V4 stack: token embedding, per-`layer_types` decoder layers, final mHC
/// head collapse and RMSNorm.
pub struct DeepseekV4Model {
    hidden_size: usize,
    hc_mult: usize,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DeepseekV4DecoderLayer>,
    norm: RmsNorm,
    hc_head: DeepseekV4HyperHead,
}

impl DeepseekV4Model {
    pub fn new(cfg: &DeepseekV4Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                DeepseekV4DecoderLayer::new(cfg, i, use_flash_attn, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let hc_head = DeepseekV4HyperHead::new(cfg, vb.pp("hc_head"))?;
        Ok(Self {
            hidden_size: cfg.hidden_size,
            hc_mult: cfg.hc_mult,
            embed_tokens,
            layers,
            norm,
            hc_head,
        })
    }

    /// Runs the full stack. `seqlen_offset` is the absolute position of the
    /// first token in `input_ids` (0 for a fresh prefill; the running token
    /// count during KV-cache decode).
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (bs, seq) = input_ids.dims2()?;
        let emb = self.embed_tokens.forward(input_ids)?; // [B,S,D]
        let hidden = emb
            .unsqueeze(2)?
            .broadcast_as((bs, seq, self.hc_mult, self.hidden_size))?
            .contiguous()?; // [B,S,H,D]
        let mut hidden = hidden;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, Some(input_ids), seqlen_offset)?;
        }
        let collapsed = self.hc_head.forward(&hidden)?; // [B,S,D]
        self.norm.forward(&collapsed)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

/// `DeepseekV4ForCausalLM`: the model plus a separate (untied) `lm_head`.
pub struct DeepseekV4ForCausalLM {
    model: DeepseekV4Model,
    lm_head: Linear,
}

impl DeepseekV4ForCausalLM {
    pub fn new(cfg: &DeepseekV4Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let model = DeepseekV4Model::new(cfg, use_flash_attn, vb.pp("model"))?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self { model, lm_head })
    }

    /// Returns `[B, S, vocab]` logits.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, seqlen_offset)?;
        self.lm_head.forward(&hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
    /// Autoregressive generation over the incremental compressed-KV cache.
    ///
    /// `prompt` is `[B, S]` (B == 1) of integer token ids. The first
    /// `forward` prefills the whole prompt at `seqlen_offset = 0`; each later
    /// step decodes a single token at the running absolute position, so the
    /// attention/compressor caches grow incrementally (one token per step).
    /// `sample` receives the last-position `[vocab]` logits and returns the
    /// next token id (the caller chooses the sampler: ArgMax/TopK/TopP/etc.).
    /// Returns the generated token ids (excluding the prompt, including the
    /// EOS token if it is emitted).
    pub fn generate(
        &mut self,
        prompt: &Tensor,
        max_new_tokens: usize,
        eos_token_id: usize,
        mut sample: impl FnMut(&Tensor) -> Result<u32>,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();
        let mut pos = 0usize; // absolute position of the first token of `next`
        let mut next = prompt.clone();
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let logits = self.forward(&next, pos)?; // [1, S, vocab]
            let last = logits
                .narrow(D::Minus2, logits.dim(D::Minus2)? - 1, 1)?
                .squeeze(0)?
                .squeeze(0)?; // [vocab]
            let tok = sample(&last)?;
            out.push(tok);
            if tok as usize == eos_token_id {
                break;
            }
            pos += next.dim(D::Minus1)?;
            next = Tensor::new(&[tok], prompt.device())?.unsqueeze(0)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
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

    /// Small self-contained config for op unit tests: head_dim=8,
    /// partial_rotary_factor=0.5 -> rope_dim=4, max_position_embeddings=8.
    fn test_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 100, "hidden_size": 16, "moe_intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 1,
                "head_dim": 8, "q_lora_rank": 8, "o_lora_rank": 8, "qk_rope_head_dim": 4,
                "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 2, "num_hash_layers": 0,
                "index_head_dim": 4, "index_n_heads": 2, "index_topk": 8, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "partial_rotary_factor": 0.5,
                "sliding_window": 4, "max_position_embeddings": 8, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
                "attention_bias": false, "attention_dropout": 0.0, "mlp_bias": false,
                "output_router_logits": false, "router_aux_loss_coef": 0.001, "router_jitter_noise": 0.0,
                "swiglu_limit": 10.0, "initializer_range": 0.02, "use_cache": true,
                "bos_token_id": 0, "eos_token_id": 1,
                "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
                "compress_ratios": [0, 0],
                "layer_types": ["sliding_attention", "sliding_attention"],
                "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                                 "original_max_position_embeddings": 65536, "type": "yarn"}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn unweighted_rms_norm() -> candle::Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3.], (3,), &Device::Cpu)?;
        let out = UnweightedRMSNorm::new(1e-6).forward(&x)?.to_vec1::<f32>()?;
        let expected = [0.46291f32, 0.92582f32, 1.38873f32];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
        Ok(())
    }

    #[test]
    fn grouped_linear_block_diagonal() -> candle::Result<()> {
        // weight (4, 3): n_groups=2, out_per_group=2, in_per_group=3
        let w = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            (4, 3),
            &Device::Cpu,
        )?;
        let gl = GroupedLinear::new(w, 2);
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu)?;
        let out = gl.forward(&x)?.to_vec3::<f32>()?;
        assert_eq!(out, vec![vec![vec![14., 32.], vec![122., 167.]]]);
        Ok(())
    }

    #[test]
    fn sqrt_softplus_values() -> candle::Result<()> {
        let x = Tensor::from_vec(vec![0f32, 1.], (2,), &Device::Cpu)?;
        let out = sqrt_softplus(&x)?.to_vec1::<f32>()?;
        // sqrt(ln(2)) and sqrt(ln(1+e))
        let expected = [0.8325546, 1.1459745];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
        Ok(())
    }

    #[test]
    fn sinkhorn_double_normalization() -> candle::Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 0.5], (2, 2), &Device::Cpu)?;
        let out = sinkhorn(&x, 1e-6, 2)?.to_vec2::<f32>()?;
        let expected = [[0.1826139, 0.8809369], [0.8173861, 0.1190631]];
        for (i, row) in out.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(
                    (v - expected[i][j]).abs() < 1e-3,
                    "sinkhorn[{i}][{j}] got {v}, expected {}",
                    expected[i][j]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rope_partial_rotary_dim_and_tables() -> candle::Result<()> {
        let cfg = test_config();
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &Device::Cpu)?;
        // head_dim=8, partial_rotary_factor=0.5 -> rope_dim=4.
        assert_eq!(emb.rope_dim(), 4);
        assert_eq!(emb.table(RopeVariant::Main).cos.dim(1)? * 2, 4);
        assert_eq!(emb.table(RopeVariant::Compress).cos.dim(1)? * 2, 4);

        // Main table: plain theta=10000. inv_freq = [10000^0, 10000^-0.5] = [1, 0.01].
        let main = emb.table(RopeVariant::Main).cos.to_vec2::<f32>()?;
        assert!(
            (main[1][0] - 1f32.cos()).abs() < 1e-5,
            "main[1][0]={}",
            main[1][0]
        );
        assert!(
            (main[1][1] - 0.01f32.cos()).abs() < 1e-5,
            "main[1][1]={}",
            main[1][1]
        );

        // Compress table: Yarn with base=160000, factor=16, beta_fast=32,
        // beta_slow=1, orig=65536. dim=4 -> ramp dim=2, mask=[1,0.5],
        // inv_freq[1] = 0.00015625*0.5 + 0.0025*0.5 = 0.001328125.
        let comp = emb.table(RopeVariant::Compress).cos.to_vec2::<f32>()?;
        assert!(
            (comp[1][0] - 1f32.cos()).abs() < 1e-5,
            "comp[1][0]={}",
            comp[1][0]
        );
        assert!(
            (comp[1][1] - 0.001328125f32.cos()).abs() < 1e-5,
            "comp[1][1]={}",
            comp[1][1]
        );

        // Real config: partial_rotary_factor=0.5 yields 64/128 rope dim.
        let path = sample_config_path();
        let raw = std::fs::read_to_string(&path).unwrap();
        let real: DeepseekV4Config = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            (real.head_dim as f64 * real.partial_rotary_factor) as usize,
            64
        );
        assert_eq!(real.qk_rope_head_dim, 64);
        Ok(())
    }

    #[test]
    fn rope_applies_to_trailing_slice() -> candle::Result<()> {
        let cfg = test_config();
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &Device::Cpu)?;
        // q: (1, 1, 1, 8); nope = [1,2,3,4], rope = [5,6,7,8].
        let q = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 1, 8),
            &Device::Cpu,
        )?;
        // seqlen_offset=1 picks the position-1 table row (cos(1),sin(1) and cos(0.01),sin(0.01)).
        let out = emb
            .forward(&q, RopeVariant::Main, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let c0 = 1f32.cos();
        let s0 = 1f32.sin();
        let c1 = 0.01f32.cos();
        let s1 = 0.01f32.sin();
        let expected = [
            1.,
            2.,
            3.,
            4.,
            5. * c0 - 6. * s0,
            5. * s0 + 6. * c0,
            7. * c1 - 8. * s1,
            7. * s1 + 8. * c1,
        ];
        for (i, (a, b)) in out.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 1e-4, "out[{i}] got {a}, expected {b}");
        }
        Ok(())
    }

    /// Tiny config for the eager-parity test: hidden=8, heads=2, head_dim=4,
    /// o_groups=2, sliding_window=3, partial_rotary_factor=0.5 (rope_dim=2).
    fn parity_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 100, "hidden_size": 8, "moe_intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "q_lora_rank": 4, "o_lora_rank": 4, "qk_rope_head_dim": 2,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 2, "num_hash_layers": 0,
                "index_head_dim": 2, "index_n_heads": 1, "index_topk": 4, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "partial_rotary_factor": 0.5,
                "sliding_window": 3, "max_position_embeddings": 8, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
                "attention_bias": false, "attention_dropout": 0.0, "mlp_bias": false,
                "output_router_logits": false, "router_aux_loss_coef": 0.001, "router_jitter_noise": 0.0,
                "swiglu_limit": 10.0, "initializer_range": 0.02, "use_cache": true,
                "bos_token_id": 0, "eos_token_id": 1,
                "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
                "compress_ratios": [0, 0],
                "layer_types": ["sliding_attention", "sliding_attention"],
                "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                                 "original_max_position_embeddings": 65536, "type": "yarn"}
            }"#,
        )
        .unwrap()
    }
    /// Decode-parity config: small dims for CPU speed but the real compressor
    /// rates (CSA 4, HCA 128) and a large `max_position_embeddings` so the
    /// RoPE tables cover a 300+ token decode (compressed entries are RoPE'd at
    /// strided absolute positions up to `n_windows * rate`).
    fn decode_parity_config() -> DeepseekV4Config {
        let mut cfg = parity_config();
        cfg.compress_rates.compressed_sparse_attention = 4;
        cfg.compress_rates.heavily_compressed_attention = 128;
        cfg.max_position_embeddings = 4096;
        cfg
    }

    /// Deterministic weight tensor: `w[i] = sin((i + base) * 0.13) * 0.5`.
    fn det_tensor(shape: &[usize], base: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + base) * 0.13).sin() * 0.5)
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn parity_weights(cfg: &DeepseekV4Config) -> HashMap<String, Tensor> {
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let mut m = HashMap::new();
        m.insert(
            "q_a_proj.weight".into(),
            det_tensor(&[cfg.q_lora_rank, cfg.hidden_size], 1.0),
        );
        m.insert(
            "q_a_norm.weight".into(),
            det_tensor(&[cfg.q_lora_rank, 1], 2.0)
                .flatten_all()
                .unwrap(),
        );
        m.insert(
            "q_b_proj.weight".into(),
            det_tensor(&[h * d, cfg.q_lora_rank], 3.0),
        );
        m.insert(
            "kv_proj.weight".into(),
            det_tensor(&[d, cfg.hidden_size], 4.0),
        );
        m.insert(
            "kv_norm.weight".into(),
            det_tensor(&[d, 1], 5.0).flatten_all().unwrap(),
        );
        m.insert(
            "o_a_proj.weight".into(),
            det_tensor(&[cfg.o_groups * cfg.o_lora_rank, h * d / cfg.o_groups], 6.0),
        );
        m.insert(
            "o_b_proj.weight".into(),
            det_tensor(&[cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank], 7.0),
        );
        m.insert(
            "sinks".into(),
            det_tensor(&[h, 1], 8.0).flatten_all().unwrap(),
        );
        m
    }

    /// Bias-free linear: `x @ w^T` with `w` shaped `(out, in)`.
    fn lin(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let wt = w.t()?;
        if x.rank() == 3 {
            let (bsize, m, k) = x.dims3()?;
            let out = wt.dim(1)?;
            x.reshape((bsize * m, k))?
                .matmul(&wt)?
                .reshape((bsize, m, out))
        } else {
            x.matmul(&wt)
        }
    }

    /// Weighted RMSNorm (DeepseekV4RMSNorm): `x * rsqrt(mean(x^2) + eps) * w`.
    fn rms_w(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let n = (var + eps)?.sqrt()?.recip()?;
        xf.broadcast_mul(&n)?.broadcast_mul(w)?.to_dtype(x.dtype())
    }

    /// Reference `eager_attention_forward` transcribed from
    /// `modeling_deepseek_v4.py` L694-724: repeat_kv, `q @ k^T * scaling`,
    /// add mask, cat per-head sink, subtract row max, softmax, drop sink, then
    /// `attn @ v`.
    fn eager_op(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sinks: &Tensor,
        mask: &Tensor,
        scaling: f64,
    ) -> Result<Tensor> {
        let (b, h, sq, d) = q.dims4()?;
        let skv = k.dim(2)?;
        let k = k.broadcast_as((b, h, skv, d))?.contiguous()?;
        let v = v.broadcast_as((b, h, skv, d))?.contiguous()?;
        let attn = (q.matmul(&k.t()?)? * scaling)?.broadcast_add(mask)?;
        let sinks = sinks.reshape((1, h, 1, 1))?.broadcast_as((b, h, sq, 1))?;
        let combined = Tensor::cat(&[attn, sinks], D::Minus1)?;
        let max = combined.max_keepdim(D::Minus1)?;
        let combined = combined.broadcast_sub(&max)?;
        let probs = candle_nn::ops::softmax_last_dim(&combined)?;
        let scores = probs.narrow(D::Minus1, 0, skv)?;
        scores.matmul(&v)
    }

    /// Sliding-window causal additive mask, same formula as `DeepseekV4Attention`.
    fn ref_mask(
        q_len: usize,
        kv_len: usize,
        prev_len: usize,
        sliding_window: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let inf = f32::NEG_INFINITY;
        let mut mask = vec![inf; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let d = i as isize + prev_len as isize - j as isize;
                if d >= 0 && (d as usize) < sliding_window {
                    mask[i * kv_len + j] = 0.0;
                }
            }
        }
        Tensor::from_vec(mask, (q_len, kv_len), dev)?
            .unsqueeze(0)?
            .unsqueeze(0)
    }

    fn ref_trim(kv: &Tensor, sliding_window: usize) -> Result<Tensor> {
        let t = kv.dim(2)?;
        let keep = sliding_window.saturating_sub(1).min(t);
        if keep == t {
            Ok(kv.clone())
        } else {
            kv.narrow(D::Minus2, t - keep, keep)
        }
    }

    /// Independent transcription of the transformers MLA path for a sliding
    /// layer (K == V), driven by the same weight tensors as the candle module.
    fn reference_forward(
        cfg: &DeepseekV4Config,
        emb: &DeepseekV4RotaryEmbedding,
        w: &HashMap<String, Tensor>,
        x: &Tensor,
        seqlen_offset: usize,
        cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let variant = RopeVariant::Main;

        let q_res = rms_w(
            &lin(x, &w["q_a_proj.weight"])?,
            &w["q_a_norm.weight"],
            cfg.rms_norm_eps,
        )?;
        let q = lin(&q_res, &w["q_b_proj.weight"])?
            .reshape((bs, seq_len, h, d))?
            .transpose(1, 2)?;
        let q = UnweightedRMSNorm::new(cfg.rms_norm_eps).forward(&q)?;
        let q = emb.forward(&q, variant, seqlen_offset)?;

        let kv = rms_w(
            &lin(x, &w["kv_proj.weight"])?,
            &w["kv_norm.weight"],
            cfg.rms_norm_eps,
        )?
        .reshape((bs, seq_len, 1, d))?
        .transpose(1, 2)?;
        let kv = emb.forward(&kv, variant, seqlen_offset)?;

        let (k, v) = match cache {
            None => {
                *cache = Some(ref_trim(&kv, cfg.sliding_window)?);
                (kv.clone(), kv.clone())
            }
            Some(prev) => {
                let full = Tensor::cat(&[prev, &kv], 2)?;
                *cache = Some(ref_trim(&full, cfg.sliding_window)?);
                (full.clone(), full.clone())
            }
        };
        let kv_len = k.dim(2)?;
        let prev_len = kv_len - seq_len;

        let mask = ref_mask(seq_len, kv_len, prev_len, cfg.sliding_window, x.device())?;
        let attn = eager_op(&q, &k, &v, &w["sinks"], &mask, (d as f64).powf(-0.5))?;
        let attn = emb.forward_conjugate(&attn, variant, seqlen_offset)?;
        let attn = attn.transpose(1, 2)?;
        let grouped = attn.reshape((bs, seq_len, cfg.o_groups, h * d / cfg.o_groups))?;
        let oa =
            GroupedLinear::new(w["o_a_proj.weight"].clone(), cfg.o_groups).forward(&grouped)?;
        let oa = oa.reshape((bs, seq_len, cfg.o_groups * cfg.o_lora_rank))?;
        lin(&oa, &w["o_b_proj.weight"])
    }

    /// Weights for the CSA compressor + indexer (keys match the sub-namespace
    /// `DeepseekV4CSACompressor::new` expects).
    fn csa_parity_weights(cfg: &DeepseekV4Config) -> HashMap<String, Tensor> {
        let hd = cfg.head_dim;
        let ihd = cfg.index_head_dim;
        let rate = cfg.compress_rates.compressed_sparse_attention;
        let mut m = HashMap::new();
        m.insert(
            "kv_proj.weight".into(),
            det_tensor(&[2 * hd, cfg.hidden_size], 21.0),
        );
        m.insert(
            "gate_proj.weight".into(),
            det_tensor(&[2 * hd, cfg.hidden_size], 22.0),
        );
        m.insert("position_bias".into(), det_tensor(&[rate, 2 * hd], 23.0));
        m.insert(
            "kv_norm.weight".into(),
            det_tensor(&[hd, 1], 24.0).flatten_all().unwrap(),
        );
        m.insert(
            "indexer.kv_proj.weight".into(),
            det_tensor(&[2 * ihd, cfg.hidden_size], 31.0),
        );
        m.insert(
            "indexer.gate_proj.weight".into(),
            det_tensor(&[2 * ihd, cfg.hidden_size], 32.0),
        );
        m.insert(
            "indexer.position_bias".into(),
            det_tensor(&[rate, 2 * ihd], 33.0),
        );
        m.insert(
            "indexer.kv_norm.weight".into(),
            det_tensor(&[ihd, 1], 34.0).flatten_all().unwrap(),
        );
        m.insert(
            "indexer.q_b_proj.weight".into(),
            det_tensor(&[cfg.index_n_heads * ihd, cfg.q_lora_rank], 35.0),
        );
        m.insert(
            "indexer.scorer.weights_proj.weight".into(),
            det_tensor(&[cfg.index_n_heads, cfg.hidden_size], 36.0),
        );
        m
    }

    /// Reference CSA compressor + indexer (transcribed from
    /// `modeling_deepseek_v4.py` `DeepseekV4CSACompressor.forward` +
    /// `DeepseekV4Indexer.forward`, stateless single call). Returns
    /// `(compressed_kv [B,1,T,hd], block_bias [B,1,S,T])`.
    fn ref_csa(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Tensor>,
        rotary: &DeepseekV4RotaryEmbedding,
        x: &Tensor,
        q_residual: &Tensor,
        seqlen_offset: usize,
        overlap: &mut (Option<Tensor>, Option<Tensor>),
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let rate = cfg.compress_rates.compressed_sparse_attention;
        let hd = cfg.head_dim;
        let ihd = cfg.index_head_dim;
        let dev = x.device();

        // Outer compressor.
        let kv = lin(x, &w["kv_proj.weight"])?;
        let gate = lin(x, &w["gate_proj.weight"])?;
        let usable = (seq_len / rate) * rate;
        let chunk_kv = kv.narrow(1, 0, usable)?;
        let chunk_gate = gate.narrow(1, 0, usable)?;
        let n_windows = usable / rate;
        let mut compressed = if n_windows > 0 {
            let ck = chunk_kv.reshape((batch, n_windows, rate, 2 * hd))?;
            let cg = chunk_gate
                .reshape((batch, n_windows, rate, 2 * hd))?
                .broadcast_add(&w["position_bias"])?;
            let ca_kv = ck.narrow(3, 0, hd)?;
            let ca_gate = cg.narrow(3, 0, hd)?;
            let cb_kv = ck.narrow(3, hd, hd)?;
            let cb_gate = cg.narrow(3, hd, hd)?;
            let (prior_kv, prior_gate) = match overlap {
                (Some(pk), Some(pg)) => (pk.clone(), pg.clone()),
                _ => (
                    Tensor::zeros((batch, rate, hd), x.dtype(), dev)?,
                    Tensor::full(f32::NEG_INFINITY, (batch, rate, hd), dev)?.to_dtype(x.dtype())?,
                ),
            };
            let ca_full_kv = if n_windows > 1 {
                Tensor::cat(
                    &[prior_kv.unsqueeze(1)?, ca_kv.narrow(1, 0, n_windows - 1)?],
                    1,
                )?
            } else {
                prior_kv.unsqueeze(1)?
            };
            let ca_full_gate = if n_windows > 1 {
                Tensor::cat(
                    &[
                        prior_gate.unsqueeze(1)?,
                        ca_gate.narrow(1, 0, n_windows - 1)?,
                    ],
                    1,
                )?
            } else {
                prior_gate.unsqueeze(1)?
            };
            *overlap = (
                Some(
                    ca_kv
                        .narrow(1, n_windows - 1, 1)?
                        .squeeze(1)?
                        .contiguous()?,
                ),
                Some(
                    ca_gate
                        .narrow(1, n_windows - 1, 1)?
                        .squeeze(1)?
                        .contiguous()?,
                ),
            );
            let new_kv = Tensor::cat(&[ca_full_kv, cb_kv], 2)?;
            let new_gate = Tensor::cat(&[ca_full_gate, cb_gate], 2)?;
            let soft =
                candle_nn::ops::softmax(&new_gate.to_dtype(DType::F32)?, 2)?.to_dtype(x.dtype())?;
            let summed = (new_kv * soft)?.sum(2)?;
            let comp = rms_w(&summed, &w["kv_norm.weight"], cfg.rms_norm_eps)?;
            let positions: Vec<u32> = (0..n_windows).map(|ww| (ww * rate) as u32).collect();
            let positions = Tensor::from_vec(positions, (n_windows,), dev)?;
            let comp = rotary
                .forward_at_positions(&comp.unsqueeze(1)?, RopeVariant::Compress, &positions)?
                .squeeze(1)?;
            vec![comp]
        } else {
            vec![]
        };
        let compressed_kv = if compressed.is_empty() {
            Tensor::zeros((batch, 0, hd), x.dtype(), dev)?
        } else {
            compressed.remove(0)
        };
        let compressed_len = compressed_kv.dim(1)?;

        // Indexer compressor (at index_head_dim).
        let ikv = lin(x, &w["indexer.kv_proj.weight"])?;
        let igate = lin(x, &w["indexer.gate_proj.weight"])?;
        let icompressed = if n_windows > 0 {
            let ck = ikv
                .narrow(1, 0, usable)?
                .reshape((batch, n_windows, rate, 2 * ihd))?;
            let cg = igate
                .narrow(1, 0, usable)?
                .reshape((batch, n_windows, rate, 2 * ihd))?
                .broadcast_add(&w["indexer.position_bias"])?;
            let ica_kv = ck.narrow(3, 0, ihd)?;
            let ica_gate = cg.narrow(3, 0, ihd)?;
            let icb_kv = ck.narrow(3, ihd, ihd)?;
            let icb_gate = cg.narrow(3, ihd, ihd)?;
            let zero_kv = Tensor::zeros((batch, rate, ihd), x.dtype(), dev)?;
            let neg_gate = Tensor::full(f32::NEG_INFINITY, (batch, rate, ihd), dev)?;
            let ica_full_kv = if n_windows > 1 {
                Tensor::cat(
                    &[zero_kv.unsqueeze(1)?, ica_kv.narrow(1, 0, n_windows - 1)?],
                    1,
                )?
            } else {
                zero_kv.unsqueeze(1)?
            };
            let ica_full_gate = if n_windows > 1 {
                Tensor::cat(
                    &[
                        neg_gate.unsqueeze(1)?,
                        ica_gate.narrow(1, 0, n_windows - 1)?,
                    ],
                    1,
                )?
            } else {
                neg_gate.unsqueeze(1)?
            };
            let inew_kv = Tensor::cat(&[ica_full_kv, icb_kv], 2)?;
            let inew_gate = Tensor::cat(&[ica_full_gate, icb_gate], 2)?;
            let isoft = candle_nn::ops::softmax(&inew_gate.to_dtype(DType::F32)?, 2)?
                .to_dtype(x.dtype())?;
            let isummed = (inew_kv * isoft)?.sum(2)?;
            let icomp = rms_w(&isummed, &w["indexer.kv_norm.weight"], cfg.rms_norm_eps)?;
            let ipositions: Vec<u32> = (0..n_windows).map(|ww| (ww * rate) as u32).collect();
            let ipositions = Tensor::from_vec(ipositions, (n_windows,), dev)?;
            rotary
                .forward_at_positions(&icomp.unsqueeze(1)?, RopeVariant::Compress, &ipositions)?
                .squeeze(1)?
        } else {
            Tensor::zeros((batch, 0, ihd), x.dtype(), dev)?
        };

        // Indexer query + scorer.
        let q = lin(q_residual, &w["indexer.q_b_proj.weight"])?
            .reshape((batch, seq_len, cfg.index_n_heads, ihd))?
            .transpose(1, 2)?;
        let q = rotary
            .forward(&q, RopeVariant::Compress, seqlen_offset)?
            .transpose(1, 2)?;
        let (_, _, _, _d) = q.dims4()?;
        let ck = icompressed.transpose(1, 2)?.unsqueeze(1)?.broadcast_as((
            batch,
            seq_len,
            ihd,
            icompressed.dim(1)?,
        ))?;
        let scores = q.to_dtype(DType::F32)?.matmul(&ck.to_dtype(DType::F32)?)?;
        let scores = (scores.relu()? * (ihd as f64).powf(-0.5))?;
        let wts = (lin(x, &w["indexer.scorer.weights_proj.weight"])?.to_dtype(DType::F32)?
            * (cfg.index_n_heads as f64).powf(-0.5))?;
        let scores = scores.broadcast_mul(&wts.unsqueeze(3)?)?.sum(2)?;

        // Per-query top-k with causal + invalid clamping.
        let icomp_len = icompressed.dim(1)?;
        let top_k = cfg.index_topk.min(icomp_len);
        let topk = if icomp_len == 0 || top_k == 0 {
            // No compressed entries yet (early decode steps): empty pick set.
            Tensor::zeros((batch, seq_len, 0), DType::I64, dev)?
        } else {
            let threshold: Vec<f32> = (0..seq_len)
                .map(|t| ((seqlen_offset + t + 1) / rate) as f32)
                .collect();
            let threshold =
                Tensor::from_vec(threshold, (1, seq_len), dev)?.broadcast_as((batch, seq_len))?;
            let entry = Tensor::arange(0u32, icomp_len as u32, dev)?.to_dtype(DType::F32)?;
            let entry_b = entry
                .reshape((1, 1, icomp_len))?
                .broadcast_as((batch, seq_len, icomp_len))?;
            let thresh_b = threshold
                .reshape((batch, seq_len, 1))?
                .broadcast_as((batch, seq_len, icomp_len))?;
            let future = entry_b.ge(&thresh_b)?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch, seq_len, icomp_len), dev)?;
            let masked = future.where_cond(&neg_inf, &scores)?;
            let topk = topk_last_dim(&masked, top_k)?;
            let thresh_k = threshold
                .reshape((batch, seq_len, 1))?
                .broadcast_as((batch, seq_len, top_k))?;
            let invalid = topk.to_dtype(DType::F32)?.ge(&thresh_k)?;
            let minus_one = Tensor::full(-1i64, (batch, seq_len, top_k), dev)?;
            invalid.where_cond(&minus_one, &topk)?
        };

        // block_bias: 0 at valid selected compressed positions, -inf else.
        let c_len = compressed_len as i64;
        let valid = topk.ge(0i64)?;
        let padding = Tensor::full(c_len, (batch, seq_len, top_k), dev)?;
        let safe = valid.where_cond(&topk, &padding)?;
        let cand = Tensor::arange(0u32, compressed_len as u32, dev)?
            .to_dtype(DType::I64)?
            .reshape((1, 1, 1, compressed_len))?;
        let safe_b = safe
            .unsqueeze(3)?
            .broadcast_as((batch, seq_len, top_k, compressed_len))?;
        let cand_b = cand.broadcast_as((batch, seq_len, top_k, compressed_len))?;
        let eq = safe_b.eq(&cand_b)?;
        let selected = eq.to_dtype(DType::F32)?.sum(2)?.gt(0f32)?;
        let zeros = Tensor::zeros((batch, seq_len, compressed_len), DType::F32, dev)?;
        let neg_inf2 = Tensor::full(f32::NEG_INFINITY, (batch, seq_len, compressed_len), dev)?;
        let bias = selected.where_cond(&zeros, &neg_inf2)?.unsqueeze(1)?;
        Ok((compressed_kv.unsqueeze(1)?, bias))
    }

    /// Single-layer CSA attention config (mirrors `parity_config` but the layer
    /// is `compressed_sparse_attention`, so `DeepseekV4Attention` wires the
    /// compressor and uses compress-RoPE).
    fn csa_attention_config() -> DeepseekV4Config {
        let mut cfg = parity_config();
        cfg.layer_types = vec![LayerType::CompressedSparseAttention];
        cfg
    }

    /// Reference CSA attention: sliding MLA path + compressor concat + block_bias
    /// mask concat + eager attention over `[sliding | compressed]`. `cw` holds
    /// the uncompressed sub-namespace (`kv_proj.*`, `indexer.*`) for `ref_csa`.
    fn reference_csa_attention(
        cfg: &DeepseekV4Config,
        emb: &DeepseekV4RotaryEmbedding,
        w: &HashMap<String, Tensor>,
        cw: &HashMap<String, Tensor>,
        x: &Tensor,
        seqlen_offset: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let variant = RopeVariant::Compress;

        let q_res = rms_w(
            &lin(x, &w["q_a_proj.weight"])?,
            &w["q_a_norm.weight"],
            cfg.rms_norm_eps,
        )?;
        let q = lin(&q_res, &w["q_b_proj.weight"])?
            .reshape((bs, seq_len, h, d))?
            .transpose(1, 2)?;
        let q = UnweightedRMSNorm::new(cfg.rms_norm_eps).forward(&q)?;
        let q = emb.forward(&q, variant, seqlen_offset)?;

        let kv = rms_w(
            &lin(x, &w["kv_proj.weight"])?,
            &w["kv_norm.weight"],
            cfg.rms_norm_eps,
        )?
        .reshape((bs, seq_len, 1, d))?
        .transpose(1, 2)?;
        let kv = emb.forward(&kv, variant, seqlen_offset)?;

        // Single call, no prior sliding cache: k_sliding == current kv.
        let kv_len = seq_len;
        let (compressed_kv, block_bias) =
            ref_csa(cfg, cw, emb, x, &q_res, seqlen_offset, &mut (None, None))?;
        let k = Tensor::cat(&[kv.clone(), compressed_kv.clone()], 2)?;
        let v = Tensor::cat(&[kv, compressed_kv], 2)?;
        let sliding = ref_mask(seq_len, kv_len, 0, cfg.sliding_window, x.device())?
            .broadcast_as((bs, 1, seq_len, kv_len))?;
        let sliding = match attention_mask {
            Some(ext) => sliding.broadcast_add(ext)?,
            None => sliding,
        };
        let mask = Tensor::cat(&[sliding, block_bias], 3)?;

        let attn = eager_op(&q, &k, &v, &w["sinks"], &mask, (d as f64).powf(-0.5))?;
        let attn = emb.forward_conjugate(&attn, variant, seqlen_offset)?;
        let attn = attn.transpose(1, 2)?;
        let grouped = attn.reshape((bs, seq_len, cfg.o_groups, h * d / cfg.o_groups))?;
        let oa =
            GroupedLinear::new(w["o_a_proj.weight"].clone(), cfg.o_groups).forward(&grouped)?;
        let oa = oa.reshape((bs, seq_len, cfg.o_groups * cfg.o_lora_rank))?;
        lin(&oa, &w["o_b_proj.weight"])
    }

    #[test]
    fn csa_attention_parity_with_transformers() -> candle::Result<()> {
        let cfg = csa_attention_config();
        let dev = Device::Cpu;
        let attn_w = parity_weights(&cfg);
        let csa_w = csa_parity_weights(&cfg);
        let mut merged = attn_w.clone();
        for (k, v) in &csa_w {
            merged.insert(format!("compressor.{k}"), v.clone());
        }
        let vb = VarBuilder::from_tensors(merged, DType::F32, &dev);
        let mut attn = DeepseekV4Attention::new(&cfg, 0, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let x = det_tensor(&[1, 5, cfg.hidden_size], 51.0);
        let out = attn.forward(&x, 0, None)?;
        let ref_out = reference_csa_attention(&cfg, &emb, &attn_w, &csa_w, &x, 0, None)?;
        assert_close(&out, &ref_out, 1e-4, "csa-attn");
        Ok(())
    }

    /// External/padding attention mask folded into the combined
    /// [sliding | block_bias] mask for a batched CSA layer: a `-inf` at a local
    /// sliding position must suppress that column, while the compressed
    /// block_bias columns are unaffected. Verified against the eager reference
    /// that applies the same external mask to the sliding part.
    #[test]
    fn csa_external_mask_combined_with_block_bias() -> candle::Result<()> {
        let cfg = csa_attention_config();
        let dev = Device::Cpu;
        let attn_w = parity_weights(&cfg);
        let csa_w = csa_parity_weights(&cfg);
        let x = det_tensor(&[1, 5, cfg.hidden_size], 91.0);

        // All-attend external mask is a no-op: identical to no mask.
        let zeros = Tensor::zeros((1, 1, 5, 5), DType::F32, &dev)?;
        let out_none = {
            let vb = VarBuilder::from_tensors(merged_weights(&attn_w, &csa_w), DType::F32, &dev);
            DeepseekV4Attention::new(&cfg, 0, false, vb)?.forward(&x, 0, None)?
        };
        let out_zero = {
            let vb = VarBuilder::from_tensors(merged_weights(&attn_w, &csa_w), DType::F32, &dev);
            DeepseekV4Attention::new(&cfg, 0, false, vb)?.forward(&x, 0, Some(&zeros))?
        };
        assert_close(&out_none, &out_zero, 1e-6, "ext-zero-noop");

        // Masking the first in-window sliding position (query 3 -> KV 1) must
        // change the output and match the reference that folds the same mask in.
        let mut m = vec![0.0f32; 5 * 5];
        m[3 * 5 + 1] = f32::NEG_INFINITY;
        let ext = Tensor::from_vec(m, (1, 1, 5, 5), &dev)?;
        let vb = VarBuilder::from_tensors(merged_weights(&attn_w, &csa_w), DType::F32, &dev);
        let mut attn = DeepseekV4Attention::new(&cfg, 0, false, vb)?;
        let out_masked = attn.forward(&x, 0, Some(&ext))?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let ref_masked = reference_csa_attention(&cfg, &emb, &attn_w, &csa_w, &x, 0, Some(&ext))?;
        assert_close(&out_masked, &ref_masked, 1e-4, "ext-masked");
        // And masking must actually perturb the output vs the unmasked case.
        let flat_a = out_none.flatten_all()?.to_vec1::<f32>()?;
        let flat_b = out_masked.flatten_all()?.to_vec1::<f32>()?;
        let diff: f32 = flat_a
            .iter()
            .zip(flat_b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-3,
            "external mask had no effect on output (max diff {diff})"
        );
        Ok(())
    }

    /// Merge the attention and compressor sub-namespace weight maps into the
    /// single `compressor.*`-prefixed map `DeepseekV4Attention::new` expects.
    fn merged_weights(
        attn_w: &HashMap<String, Tensor>,
        extra_w: &HashMap<String, Tensor>,
    ) -> HashMap<String, Tensor> {
        let mut merged = attn_w.clone();
        for (k, v) in extra_w {
            merged.insert(format!("compressor.{k}"), v.clone());
        }
        merged
    }

    #[test]
    fn csa_compressor_indexer_parity_with_transformers() -> candle::Result<()> {
        let cfg = parity_config();
        let dev = Device::Cpu;
        let weights = csa_parity_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut comp = DeepseekV4CSACompressor::new(&cfg, vb)?;
        let rotary = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;

        // Prefill of 5 tokens -> 2 complete windows (rate=2), first window has
        // zero-kv / -inf-gate Ca (no prior overlap).
        let x = det_tensor(&[1, 5, cfg.hidden_size], 41.0);
        let qr = det_tensor(&[1, 5, cfg.q_lora_rank], 42.0);
        let (ckv, bb) = comp.forward(&x, &qr, 0)?;
        let (r_ckv, r_bb) = ref_csa(&cfg, &weights, &rotary, &x, &qr, 0, &mut (None, None))?;
        assert_eq!(ckv.dims(), r_ckv.dims(), "compressed_kv shape");
        assert_eq!(bb.dims(), r_bb.dims(), "block_bias shape");
        assert_close(&ckv, &r_ckv, 1e-4, "compressed_kv");
        assert_close(&bb, &r_bb, 1e-4, "block_bias");
        Ok(())
    }

    #[test]
    fn attention_eager_parity_with_transformers() -> candle::Result<()> {
        let cfg = parity_config();
        let dev = Device::Cpu;
        let weights = parity_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut attn = DeepseekV4Attention::new(&cfg, 0, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let mut cache: Option<Tensor> = None;

        let x1 = det_tensor(&[1, 5, cfg.hidden_size], 11.0);
        let o1 = attn.forward(&x1, 0, None)?;
        let r1 = reference_forward(&cfg, &emb, &weights, &x1, 0, &mut cache)?;
        assert_close(&o1, &r1, 1e-4, "step1");

        // Step 2: a two-token continuation after the cache has been trimmed.
        let x2 = det_tensor(&[1, 2, cfg.hidden_size], 13.0);
        let o2 = attn.forward(&x2, 5, None)?;
        let r2 = reference_forward(&cfg, &emb, &weights, &x2, 5, &mut cache)?;
        assert_close(&o2, &r2, 1e-4, "step2");

        // Step 3: single-token generation off the sliding cache.
        let x3 = det_tensor(&[1, 1, cfg.hidden_size], 17.0);
        let o3 = attn.forward(&x3, 7, None)?;
        let r3 = reference_forward(&cfg, &emb, &weights, &x3, 7, &mut cache)?;
        assert_close(&o3, &r3, 1e-4, "step3");
        Ok(())
    }

    /// Weights for the HCA compressor (keys match the sub-namespace
    /// `DeepseekV4HCACompressor::new` expects): single head_dim series, no
    /// indexer.
    fn hca_parity_weights(cfg: &DeepseekV4Config) -> HashMap<String, Tensor> {
        let hd = cfg.head_dim;
        let rate = cfg.compress_rates.heavily_compressed_attention;
        let mut m = HashMap::new();
        m.insert(
            "kv_proj.weight".into(),
            det_tensor(&[hd, cfg.hidden_size], 61.0),
        );
        m.insert(
            "gate_proj.weight".into(),
            det_tensor(&[hd, cfg.hidden_size], 62.0),
        );
        m.insert("position_bias".into(), det_tensor(&[rate, hd], 63.0));
        m.insert(
            "kv_norm.weight".into(),
            det_tensor(&[hd, 1], 64.0).flatten_all().unwrap(),
        );
        m
    }

    /// Reference HCA compressor (transcribed from `modeling_deepseek_v4.py`
    /// `DeepseekV4HCACompressor.forward`, stateless single call): non-overlapping
    /// windows, softmax-gated sum at head_dim, compress-RoPE at `i * rate`,
    /// causality-only block_bias. Returns `(compressed_kv [B,1,T,hd],
    /// block_bias [B,1,S,T])`.
    fn ref_hca(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Tensor>,
        rotary: &DeepseekV4RotaryEmbedding,
        x: &Tensor,
        _q_residual: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let rate = cfg.compress_rates.heavily_compressed_attention;
        let hd = cfg.head_dim;
        let dev = x.device();
        let kv = lin(x, &w["kv_proj.weight"])?;
        let gate = lin(x, &w["gate_proj.weight"])?;
        let usable = (seq_len / rate) * rate;
        let chunk_kv = kv.narrow(1, 0, usable)?;
        let chunk_gate = gate.narrow(1, 0, usable)?;
        let n_windows = usable / rate;
        let compressed = if n_windows > 0 {
            let ck = chunk_kv.reshape((batch, n_windows, rate, hd))?;
            let cg = chunk_gate
                .reshape((batch, n_windows, rate, hd))?
                .broadcast_add(&w["position_bias"])?;
            let soft =
                candle_nn::ops::softmax(&cg.to_dtype(DType::F32)?, 2)?.to_dtype(x.dtype())?;
            let summed = (ck * soft)?.sum(2)?;
            let comp = rms_w(&summed, &w["kv_norm.weight"], cfg.rms_norm_eps)?;
            let positions: Vec<u32> = (0..n_windows).map(|ww| (ww * rate) as u32).collect();
            let positions = Tensor::from_vec(positions, (n_windows,), dev)?;
            rotary
                .forward_at_positions(&comp.unsqueeze(1)?, RopeVariant::Compress, &positions)?
                .squeeze(1)?
        } else {
            Tensor::zeros((batch, 0, hd), x.dtype(), dev)?
        };
        let compressed_kv = compressed.unsqueeze(1)?; // [B, 1, T, hd]
        let compressed_len = compressed_kv.dim(2)?;

        // Causality-only block_bias: `w >= (pos + 1) // rate` -> -inf.
        let threshold: Vec<f32> = (0..seq_len)
            .map(|t| ((seqlen_offset + t + 1) / rate) as f32)
            .collect();
        let entry = Tensor::arange(0u32, compressed_len as u32, dev)?.to_dtype(DType::F32)?;
        let entry_b = entry.reshape((1, 1, 1, compressed_len))?.broadcast_as((
            1,
            1,
            seq_len,
            compressed_len,
        ))?;
        let thresh_b = Tensor::from_vec(threshold, (1, seq_len), dev)?
            .reshape((1, 1, seq_len, 1))?
            .broadcast_as((1, 1, seq_len, compressed_len))?;
        let future = entry_b.ge(&thresh_b)?;
        let zeros = Tensor::zeros((1, 1, seq_len, compressed_len), DType::F32, dev)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (1, 1, seq_len, compressed_len), dev)?;
        let block_bias = future.where_cond(&neg_inf, &zeros)?.broadcast_as((
            batch,
            1,
            seq_len,
            compressed_len,
        ))?;
        Ok((compressed_kv, block_bias))
    }

    /// Single-layer HCA attention config (mirrors `parity_config` but the layer
    /// is `heavily_compressed_attention`).
    fn hca_attention_config() -> DeepseekV4Config {
        let mut cfg = parity_config();
        cfg.layer_types = vec![LayerType::HeavilyCompressedAttention];
        cfg
    }

    /// Reference HCA attention: sliding MLA path + HCA compressor concat +
    /// block_bias mask concat + eager attention over `[sliding | compressed]`.
    /// `hw` holds the uncompressed sub-namespace (`kv_proj.*`) for `ref_hca`.
    fn reference_hca_attention(
        cfg: &DeepseekV4Config,
        emb: &DeepseekV4RotaryEmbedding,
        w: &HashMap<String, Tensor>,
        hw: &HashMap<String, Tensor>,
        x: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let variant = RopeVariant::Compress;

        let q_res = rms_w(
            &lin(x, &w["q_a_proj.weight"])?,
            &w["q_a_norm.weight"],
            cfg.rms_norm_eps,
        )?;
        let q = lin(&q_res, &w["q_b_proj.weight"])?
            .reshape((bs, seq_len, h, d))?
            .transpose(1, 2)?;
        let q = UnweightedRMSNorm::new(cfg.rms_norm_eps).forward(&q)?;
        let q = emb.forward(&q, variant, seqlen_offset)?;

        let kv = rms_w(
            &lin(x, &w["kv_proj.weight"])?,
            &w["kv_norm.weight"],
            cfg.rms_norm_eps,
        )?
        .reshape((bs, seq_len, 1, d))?
        .transpose(1, 2)?;
        let kv = emb.forward(&kv, variant, seqlen_offset)?;

        // Single call, no prior sliding cache: k_sliding == current kv.
        let kv_len = seq_len;
        let (compressed_kv, block_bias) = ref_hca(cfg, hw, emb, x, &q_res, seqlen_offset)?;
        let k = Tensor::cat(&[kv.clone(), compressed_kv.clone()], 2)?;
        let v = Tensor::cat(&[kv, compressed_kv], 2)?;
        let sliding = ref_mask(seq_len, kv_len, 0, cfg.sliding_window, x.device())?
            .broadcast_as((bs, 1, seq_len, kv_len))?;
        let mask = Tensor::cat(&[sliding, block_bias], 3)?;

        let attn = eager_op(&q, &k, &v, &w["sinks"], &mask, (d as f64).powf(-0.5))?;
        let attn = emb.forward_conjugate(&attn, variant, seqlen_offset)?;
        let attn = attn.transpose(1, 2)?;
        let grouped = attn.reshape((bs, seq_len, cfg.o_groups, h * d / cfg.o_groups))?;
        let oa =
            GroupedLinear::new(w["o_a_proj.weight"].clone(), cfg.o_groups).forward(&grouped)?;
        let oa = oa.reshape((bs, seq_len, cfg.o_groups * cfg.o_lora_rank))?;
        lin(&oa, &w["o_b_proj.weight"])
    }

    #[test]
    fn hca_compressor_parity_with_transformers() -> candle::Result<()> {
        let cfg = parity_config();
        let dev = Device::Cpu;
        let weights = hca_parity_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut comp = DeepseekV4HCACompressor::new(&cfg, vb)?;
        let rotary = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let x = det_tensor(&[1, 5, cfg.hidden_size], 71.0);
        let qr = det_tensor(&[1, 5, cfg.q_lora_rank], 72.0);
        let (ckv, bb) = comp.forward(&x, &qr, 0)?;
        let (r_ckv, r_bb) = ref_hca(&cfg, &weights, &rotary, &x, &qr, 0)?;
        assert_eq!(ckv.dims(), r_ckv.dims(), "compressed_kv shape");
        assert_eq!(bb.dims(), r_bb.dims(), "block_bias shape");
        assert_close(&ckv, &r_ckv, 1e-4, "compressed_kv");
        assert_close(&bb, &r_bb, 1e-4, "block_bias");
        Ok(())
    }

    #[test]
    fn hca_attention_parity_with_transformers() -> candle::Result<()> {
        let cfg = hca_attention_config();
        let dev = Device::Cpu;
        let attn_w = parity_weights(&cfg);
        let hca_w = hca_parity_weights(&cfg);
        let mut merged = attn_w.clone();
        for (k, v) in &hca_w {
            merged.insert(format!("compressor.{k}"), v.clone());
        }
        let vb = VarBuilder::from_tensors(merged, DType::F32, &dev);
        let mut attn = DeepseekV4Attention::new(&cfg, 0, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let x = det_tensor(&[1, 5, cfg.hidden_size], 81.0);
        let out = attn.forward(&x, 0, None)?;
        let ref_out = reference_hca_attention(&cfg, &emb, &attn_w, &hca_w, &x, 0)?;
        assert_close(&out, &ref_out, 1e-4, "hca-attn");
        Ok(())
    }

    /// Flash-vs-eager parity for the DSA kernel, covering all three layer types
    /// (sliding, CSA, HCA). Both paths run the same BF16 weights on CUDA; the
    /// only difference is `use_flash_attn`, so a match isolates the kernel's
    /// fused mask/sink/softmax/drop-sink math from the eager transcription.
    ///
    /// Requires `--features flash-attn` (pulls in `cuda`) and a CUDA device;
    /// skipped with a warning when no GPU is present.
    #[cfg(feature = "flash-attn")]
    fn check_flash_eager_parity(
        cfg: &DeepseekV4Config,
        label: &str,
        dev: &Device,
    ) -> candle::Result<()> {
        for layer in [
            LayerType::SlidingAttention,
            LayerType::CompressedSparseAttention,
            LayerType::HeavilyCompressedAttention,
        ] {
            let mut cfg = cfg.clone();
            cfg.layer_types = vec![layer];
            let attn_w = parity_weights(&cfg);
            let extra_w = match layer {
                LayerType::CompressedSparseAttention => csa_parity_weights(&cfg),
                LayerType::HeavilyCompressedAttention => hca_parity_weights(&cfg),
                _ => HashMap::new(),
            };
            let merged: HashMap<String, Tensor> = merged_weights(&attn_w, &extra_w)
                .into_iter()
                .map(|(k, v)| (k, v.to_device(dev).unwrap().to_dtype(DType::BF16).unwrap()))
                .collect();
            let vb_eager = VarBuilder::from_tensors(merged.clone(), DType::BF16, dev);
            let vb_flash = VarBuilder::from_tensors(merged, DType::BF16, dev);
            let mut eager = DeepseekV4Attention::new(&cfg, 0, false, vb_eager)?;
            let mut flash = DeepseekV4Attention::new(&cfg, 0, true, vb_flash)?;
            let x = det_tensor(&[1, 5, cfg.hidden_size], 51.0)
                .to_device(dev)?
                .to_dtype(DType::BF16)?;
            let e = eager.forward(&x, 0, None)?.to_dtype(DType::F32)?;
            let f = flash.forward(&x, 0, None)?.to_dtype(DType::F32)?;
            assert_close(&e, &f, 2e-2, &format!("flash-vs-eager {layer:?} ({label})"));
        }
        Ok(())
    }

    /// Flash-vs-eager per-step decode parity for the DSA kernel, running the
    /// full `DeepseekV4Attention` with the **incremental** compressed-KV cache:
    /// at each step a single new token (`Sq = 1`) is fed at absolute position
    /// `step`, exactly as the `DeepseekV4ForCausalLM::generate` loop decodes,
    /// and the flash output must match eager at every step for all three layer
    /// types (sliding, CSA, HCA). This is the acceptance test for decode-mode
    /// DSA flash driven by the incremental cache + blockmask/block-sparse
    /// kernels.
    ///
    /// Requires `--features flash-attn` (pulls in `cuda`) and a CUDA device;
    /// skipped with a warning when no GPU is present.
    #[cfg(feature = "flash-attn")]
    fn check_flash_eager_decode_parity(
        cfg: &DeepseekV4Config,
        label: &str,
        dev: &Device,
        n_steps: usize,
    ) -> candle::Result<()> {
        for layer in [
            LayerType::SlidingAttention,
            LayerType::CompressedSparseAttention,
            LayerType::HeavilyCompressedAttention,
        ] {
            let mut cfg = cfg.clone();
            cfg.layer_types = vec![layer];
            let attn_w = parity_weights(&cfg);
            let extra_w = match layer {
                LayerType::CompressedSparseAttention => csa_parity_weights(&cfg),
                LayerType::HeavilyCompressedAttention => hca_parity_weights(&cfg),
                _ => HashMap::new(),
            };
            let merged: HashMap<String, Tensor> = merged_weights(&attn_w, &extra_w)
                .into_iter()
                .map(|(k, v)| (k, v.to_device(dev).unwrap().to_dtype(DType::BF16).unwrap()))
                .collect();
            let vb_eager = VarBuilder::from_tensors(merged.clone(), DType::BF16, dev);
            let vb_flash = VarBuilder::from_tensors(merged, DType::BF16, dev);
            let mut eager = DeepseekV4Attention::new(&cfg, 0, false, vb_eager)?;
            let mut flash = DeepseekV4Attention::new(&cfg, 0, true, vb_flash)?;
            // Per-step decode: Sq=1 at running absolute position, incremental cache.
            for step in 0..n_steps {
                let x = det_tensor(&[1, 1, cfg.hidden_size], (51 + step) as f32)
                    .to_device(dev)?
                    .to_dtype(DType::BF16)?;
                let e = eager.forward(&x, step, None)?.to_dtype(DType::F32)?;
                let f = flash.forward(&x, step, None)?.to_dtype(DType::F32)?;
                assert_close(
                    &e,
                    &f,
                    1e-1,
                    &format!("flash-vs-eager decode step {step} {layer:?} ({label})"),
                );
            }
        }
        Ok(())
    }

    /// Per-step decode parity at the tiny-model shape (head_dim 128, 8 heads,
    /// MQA Hkv=1), 300 decode steps through the incremental cache.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_eager_decode_parity_head_dim_128() -> candle::Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device; skipping flash-eager decode parity head_dim 128 ({e})");
                return Ok(());
            }
        };
        let mut cfg = decode_parity_config();
        cfg.num_attention_heads = 8;
        cfg.head_dim = 128;
        cfg.q_lora_rank = 128;
        cfg.o_lora_rank = 128;
        check_flash_eager_decode_parity(&cfg, "tiny-head_dim-128", &dev, 300)
    }

    /// Per-step decode parity at the real V4-Flash shape (head_dim 512, 64
    /// heads, MQA Hkv=1), 300 decode steps through the incremental cache on
    /// the 96GB GPU.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_eager_decode_parity_v4flash_512_64h() -> candle::Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device; skipping flash-eager decode parity 512/64-head ({e})");
                return Ok(());
            }
        };
        let mut cfg = decode_parity_config();
        cfg.num_attention_heads = 64;
        cfg.head_dim = 512;
        cfg.q_lora_rank = 512;
        cfg.o_lora_rank = 512;
        check_flash_eager_decode_parity(&cfg, "v4flash-head_dim-512-64h", &dev, 300)
    }

    /// Base head_dim 4 smoke parity.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_eager_parity() -> candle::Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device; skipping flash-eager parity ({e})");
                return Ok(());
            }
        };
        check_flash_eager_parity(&parity_config(), "head_dim-4", &dev)
    }

    /// Real tiny-model shape (head_dim 128, 8 heads, MQA Hkv=1).
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_eager_parity_head_dim_128() -> candle::Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device; skipping flash-eager parity head_dim 128 ({e})");
                return Ok(());
            }
        };
        let mut cfg = parity_config();
        cfg.num_attention_heads = 8;
        cfg.head_dim = 128;
        cfg.q_lora_rank = 128;
        cfg.o_lora_rank = 128;
        check_flash_eager_parity(&cfg, "tiny-head_dim-128", &dev)
    }

    /// Real V4-Flash shape (head_dim 512, 64 heads, MQA Hkv=1), run on the
    /// 96GB GPU for the flash wiring verification (story #4270).
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_eager_parity_v4flash_512_64h() -> candle::Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device; skipping flash-eager parity 512/64-head ({e})");
                return Ok(());
            }
        };
        let mut cfg = parity_config();
        cfg.num_attention_heads = 64;
        cfg.head_dim = 512;
        cfg.q_lora_rank = 512;
        cfg.o_lora_rank = 512;
        check_flash_eager_parity(&cfg, "v4flash-head_dim-512-64h", &dev)
    }
    /// Reference mHC mixer (transcribed from `modeling_deepseek_v4.py`
    /// `DeepseekV4HyperConnection.forward`): returns `(post, comb, collapsed)`.
    fn ref_hc(
        cfg: &DeepseekV4Config,
        fn_: &Tensor,
        base: &Tensor,
        scale: &Tensor,
        streams: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let hc = cfg.hc_mult;
        let flat = UnweightedRMSNorm::new(cfg.rms_norm_eps)
            .forward(&streams.flatten_from(2)?.to_dtype(DType::F32)?)?;
        let (b, sl, k) = flat.dims3()?;
        let lin = flat
            .reshape((b * sl, k))?
            .matmul(&fn_.t()?)?
            .reshape((b, sl, (2 + hc) * hc))?;
        let pre_w = lin.narrow(D::Minus1, 0, hc)?;
        let post_w = lin.narrow(D::Minus1, hc, hc)?;
        let comb_w = lin.narrow(D::Minus1, 2 * hc, hc * hc)?;
        let pre_b = base.narrow(0, 0, hc)?;
        let post_b = base.narrow(0, hc, hc)?;
        let comb_b = base.narrow(0, 2 * hc, hc * hc)?;
        let s = scale.to_vec1::<f32>()?;
        let (pre_scale, post_scale, comb_scale) = (s[0] as f64, s[1] as f64, s[2] as f64);
        let pre =
            (candle_nn::ops::sigmoid(&(pre_w * pre_scale)?.broadcast_add(&pre_b)?)? + cfg.hc_eps)?;
        let post =
            (candle_nn::ops::sigmoid(&(post_w * post_scale)?.broadcast_add(&post_b)?)? * 2.0)?;
        let (b, sl, _) = lin.dims3()?;
        let comb_logits = ((comb_w.reshape((b, sl, hc, hc))? * comb_scale)?)
            .broadcast_add(&comb_b.reshape((hc, hc))?)?;
        let comb = sinkhorn(&comb_logits, cfg.hc_eps, cfg.hc_sinkhorn_iters)?;
        let collapsed = pre
            .unsqueeze(D::Minus1)?
            .broadcast_mul(streams)?
            .sum(D::Minus2)?
            .to_dtype(streams.dtype())?;
        Ok((post, comb, collapsed))
    }

    /// Reference final HC-head collapse (`DeepseekV4HyperHead.forward`).
    fn ref_hc_head(
        cfg: &DeepseekV4Config,
        hc_fn: &Tensor,
        hc_base: &Tensor,
        hc_scale: &Tensor,
        x: &Tensor,
    ) -> Result<Tensor> {
        let flat = UnweightedRMSNorm::new(cfg.rms_norm_eps)
            .forward(&x.flatten_from(2)?.to_dtype(DType::F32)?)?;
        let (b, sl, k) = flat.dims3()?;
        let mixes =
            flat.reshape((b * sl, k))?
                .matmul(&hc_fn.t()?)?
                .reshape((b, sl, hc_fn.dim(0)?))?;
        let scale = hc_scale.to_vec1::<f32>()?[0] as f64;
        let pre =
            (candle_nn::ops::sigmoid(&(mixes * scale)?.broadcast_add(hc_base)?)? + cfg.hc_eps)?;
        pre.unsqueeze(D::Minus1)?
            .broadcast_mul(x)?
            .sum(D::Minus2)?
            .to_dtype(x.dtype())
    }

    /// Reference shared-expert MLP (`DeepseekV4MLP.forward`).
    fn ref_mlp(
        cfg: &DeepseekV4Config,
        gw: &Tensor,
        uw: &Tensor,
        dw: &Tensor,
        x: &Tensor,
    ) -> Result<Tensor> {
        let gate = lin(x, gw)?.clamp(f64::NEG_INFINITY, cfg.swiglu_limit)?;
        let up = lin(x, uw)?.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)?;
        lin(&(candle_nn::ops::silu(&gate)? * up)?, dw)
    }

    /// Reference sparse MoE block (top-k router + routed experts + shared
    /// expert), transcribed from `DeepseekV4SparseMoeBlock.forward` +
    /// `DeepseekV4Experts.forward`. `p` is the `model.layers.{i}` prefix.
    fn ref_moe(
        cfg: &DeepseekV4Config,
        p: &str,
        w: &HashMap<String, Tensor>,
        x: &Tensor,
    ) -> Result<Tensor> {
        let (bs, seq, d) = x.dims3()?;
        let flat = x.reshape((bs * seq, d))?;
        let scores = sqrt_softplus(&flat.matmul(&w[&format!("{p}.mlp.gate.weight")].t()?)?)?;
        let indices = topk_last_dim(&scores, cfg.num_experts_per_tok)?;
        let weights = scores.gather(&indices, D::Minus1)?;
        let denom = (weights.sum(D::Minus1)?.unsqueeze(D::Minus1)? + 1e-20)?;
        let weights = (weights.broadcast_div(&denom)? * cfg.routed_scaling_factor)?;
        let n = flat.dim(0)?;
        let k = indices.dim(1)?;
        let zero = Tensor::zeros((n, k), DType::F32, x.device())?;
        let mut out = flat.zeros_like()?;
        for i in 0..cfg.n_routed_experts {
            let sel = indices.eq(i as u32)?;
            let wsel = sel.where_cond(&weights, &zero)?;
            let wsum = wsel.sum(D::Minus1)?;
            let gu = w[&format!("{p}.mlp.experts.gate_up_proj")]
                .narrow(0, i, 1)?
                .squeeze(0)?;
            let all = flat.matmul(&gu.t()?)?;
            let c = all.chunk(2, D::Minus1)?;
            let gate = c[0].clamp(f64::NEG_INFINITY, cfg.swiglu_limit)?;
            let up = c[1].clamp(-cfg.swiglu_limit, cfg.swiglu_limit)?;
            let act = (candle_nn::ops::silu(&gate)? * up)?;
            let dw = w[&format!("{p}.mlp.experts.down_proj")]
                .narrow(0, i, 1)?
                .squeeze(0)?;
            let down = act.matmul(&dw.t()?)?;
            out = out.add(&down.broadcast_mul(&wsum.unsqueeze(D::Minus1)?)?)?;
        }
        let shared = ref_mlp(
            cfg,
            &w[&format!("{p}.mlp.shared_experts.gate_proj.weight")],
            &w[&format!("{p}.mlp.shared_experts.up_proj.weight")],
            &w[&format!("{p}.mlp.shared_experts.down_proj.weight")],
            x,
        )?;
        let routed = out.reshape((bs, seq, d))?;
        routed.add(&shared)
    }

    /// Reference full-model forward for an all-sliding-attention stack,
    /// transcribing `DeepseekV4Model.forward` and `DeepseekV4ForCausalLM.forward`
    /// and returning `[B, S, vocab]` logits.
    #[allow(clippy::type_complexity)]
    fn ref_model_forward(
        cfg: &DeepseekV4Config,
        emb: &DeepseekV4RotaryEmbedding,
        w: &HashMap<String, Tensor>,
        input_ids: &Tensor,
        seqlen_offset: usize,
        caches: &mut [Option<Tensor>],
    ) -> Result<Tensor> {
        let (bs, seq) = input_ids.dims2()?;
        let hc = cfg.hc_mult;
        let hidden = w["model.embed_tokens.weight"]
            .index_select(&input_ids.flatten_all()?, 0)?
            .reshape((bs, seq, cfg.hidden_size))?;
        let mut streams = hidden
            .unsqueeze(2)?
            .broadcast_as((bs, seq, hc, cfg.hidden_size))?
            .contiguous()?;
        for (i, cache) in caches.iter_mut().enumerate() {
            let p = format!("model.layers.{i}");
            // Attention half.
            let (post, comb, collapsed) = ref_hc(
                cfg,
                &w[&format!("{p}.attn_hc.fn")],
                &w[&format!("{p}.attn_hc.base")],
                &w[&format!("{p}.attn_hc.scale")],
                &streams,
            )?;
            let normed = rms_w(
                &collapsed,
                &w[&format!("{p}.input_layernorm.weight")],
                cfg.rms_norm_eps,
            )?;
            let mut aw = HashMap::new();
            for k in [
                "q_a_proj.weight",
                "q_a_norm.weight",
                "q_b_proj.weight",
                "kv_proj.weight",
                "kv_norm.weight",
                "o_a_proj.weight",
                "o_b_proj.weight",
                "sinks",
            ] {
                aw.insert(k.to_string(), w[&format!("{p}.self_attn.{k}")].clone());
            }
            let attn_out = reference_forward(cfg, emb, &aw, &normed, seqlen_offset, cache)?;
            streams = post
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&attn_out.unsqueeze(D::Minus2)?)?
                .broadcast_add(&comb.transpose(D::Minus1, D::Minus2)?.matmul(&streams)?)?;
            // FFN half.
            let (post, comb, collapsed) = ref_hc(
                cfg,
                &w[&format!("{p}.ffn_hc.fn")],
                &w[&format!("{p}.ffn_hc.base")],
                &w[&format!("{p}.ffn_hc.scale")],
                &streams,
            )?;
            let normed = rms_w(
                &collapsed,
                &w[&format!("{p}.post_attention_layernorm.weight")],
                cfg.rms_norm_eps,
            )?;
            let mlp_out = ref_moe(cfg, &p, w, &normed)?;
            streams = post
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&mlp_out.unsqueeze(D::Minus2)?)?
                .broadcast_add(&comb.transpose(D::Minus1, D::Minus2)?.matmul(&streams)?)?;
        }
        let collapsed = ref_hc_head(
            cfg,
            &w["model.hc_head.hc_fn"],
            &w["model.hc_head.hc_base"],
            &w["model.hc_head.hc_scale"],
            &streams,
        )?;
        let normed = rms_w(&collapsed, &w["model.norm.weight"], cfg.rms_norm_eps)?;
        lin(&normed, &w["lm_head.weight"])
    }

    /// Small all-sliding 2-layer config for the full-model parity tests.
    fn model_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 64, "hidden_size": 8, "moe_intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "q_lora_rank": 4, "o_lora_rank": 4, "qk_rope_head_dim": 2,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 2, "num_hash_layers": 0,
                "index_head_dim": 2, "index_n_heads": 1, "index_topk": 4, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "partial_rotary_factor": 0.5,
                "sliding_window": 6, "max_position_embeddings": 32, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
                "attention_bias": false, "attention_dropout": 0.0, "mlp_bias": false,
                "output_router_logits": false, "router_aux_loss_coef": 0.001, "router_jitter_noise": 0.0,
                "swiglu_limit": 10.0, "initializer_range": 0.02, "use_cache": true,
                "bos_token_id": 0, "eos_token_id": 1,
                "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
                "compress_ratios": [0, 0, 0, 0],
                "layer_types": ["sliding_attention", "sliding_attention"],
                "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                                 "original_max_position_embeddings": 65536, "type": "yarn"}
            }"#,
        )
        .unwrap()
    }

    /// Deterministic weights for the full model under `DeepseekV4ForCausalLM`'s
    /// `VarBuilder` namespace.
    fn model_weights(cfg: &DeepseekV4Config) -> HashMap<String, Tensor> {
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let mut m = HashMap::new();
        m.insert(
            "model.embed_tokens.weight".into(),
            det_tensor(&[cfg.vocab_size, cfg.hidden_size], 201.0),
        );
        m.insert(
            "lm_head.weight".into(),
            det_tensor(&[cfg.vocab_size, cfg.hidden_size], 202.0),
        );
        m.insert(
            "model.norm.weight".into(),
            det_tensor(&[cfg.hidden_size, 1], 203.0)
                .flatten_all()
                .unwrap(),
        );
        m.insert(
            "model.hc_head.hc_fn".into(),
            det_tensor(&[cfg.hc_mult, cfg.hc_mult * cfg.hidden_size], 204.0),
        );
        m.insert(
            "model.hc_head.hc_base".into(),
            det_tensor(&[cfg.hc_mult], 205.0).flatten_all().unwrap(),
        );
        m.insert(
            "model.hc_head.hc_scale".into(),
            det_tensor(&[1], 206.0).flatten_all().unwrap(),
        );
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let b = 300.0 + i as f32 * 100.0;
            m.insert(
                format!("{p}.self_attn.q_a_proj.weight"),
                det_tensor(&[cfg.q_lora_rank, cfg.hidden_size], b + 1.0),
            );
            m.insert(
                format!("{p}.self_attn.q_a_norm.weight"),
                det_tensor(&[cfg.q_lora_rank, 1], b + 2.0)
                    .flatten_all()
                    .unwrap(),
            );
            m.insert(
                format!("{p}.self_attn.q_b_proj.weight"),
                det_tensor(&[h * d, cfg.q_lora_rank], b + 3.0),
            );
            m.insert(
                format!("{p}.self_attn.kv_proj.weight"),
                det_tensor(&[d, cfg.hidden_size], b + 4.0),
            );
            m.insert(
                format!("{p}.self_attn.kv_norm.weight"),
                det_tensor(&[d, 1], b + 5.0).flatten_all().unwrap(),
            );
            m.insert(
                format!("{p}.self_attn.o_a_proj.weight"),
                det_tensor(
                    &[cfg.o_groups * cfg.o_lora_rank, h * d / cfg.o_groups],
                    b + 6.0,
                ),
            );
            m.insert(
                format!("{p}.self_attn.o_b_proj.weight"),
                det_tensor(&[cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank], b + 7.0),
            );
            m.insert(
                format!("{p}.self_attn.sinks"),
                det_tensor(&[h], b + 8.0).flatten_all().unwrap(),
            );
            let mix = (2 + cfg.hc_mult) * cfg.hc_mult;
            m.insert(
                format!("{p}.attn_hc.fn"),
                det_tensor(&[mix, cfg.hc_mult * cfg.hidden_size], b + 9.0),
            );
            m.insert(
                format!("{p}.attn_hc.base"),
                det_tensor(&[mix], b + 10.0).flatten_all().unwrap(),
            );
            m.insert(
                format!("{p}.attn_hc.scale"),
                det_tensor(&[3], b + 11.0).flatten_all().unwrap(),
            );
            m.insert(
                format!("{p}.ffn_hc.fn"),
                det_tensor(&[mix, cfg.hc_mult * cfg.hidden_size], b + 12.0),
            );
            m.insert(
                format!("{p}.ffn_hc.base"),
                det_tensor(&[mix], b + 13.0).flatten_all().unwrap(),
            );
            m.insert(
                format!("{p}.ffn_hc.scale"),
                det_tensor(&[3], b + 14.0).flatten_all().unwrap(),
            );
            m.insert(
                format!("{p}.input_layernorm.weight"),
                det_tensor(&[cfg.hidden_size, 1], b + 15.0)
                    .flatten_all()
                    .unwrap(),
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                det_tensor(&[cfg.hidden_size, 1], b + 16.0)
                    .flatten_all()
                    .unwrap(),
            );
            m.insert(
                format!("{p}.mlp.gate.weight"),
                det_tensor(&[cfg.n_routed_experts, cfg.hidden_size], b + 17.0),
            );
            m.insert(
                format!("{p}.mlp.experts.gate_up_proj"),
                det_tensor(
                    &[
                        cfg.n_routed_experts,
                        2 * cfg.moe_intermediate_size,
                        cfg.hidden_size,
                    ],
                    b + 18.0,
                ),
            );
            m.insert(
                format!("{p}.mlp.experts.down_proj"),
                det_tensor(
                    &[
                        cfg.n_routed_experts,
                        cfg.hidden_size,
                        cfg.moe_intermediate_size,
                    ],
                    b + 19.0,
                ),
            );
            m.insert(
                format!("{p}.mlp.shared_experts.gate_proj.weight"),
                det_tensor(&[cfg.moe_intermediate_size, cfg.hidden_size], b + 20.0),
            );
            m.insert(
                format!("{p}.mlp.shared_experts.up_proj.weight"),
                det_tensor(&[cfg.moe_intermediate_size, cfg.hidden_size], b + 21.0),
            );
            m.insert(
                format!("{p}.mlp.shared_experts.down_proj.weight"),
                det_tensor(&[cfg.hidden_size, cfg.moe_intermediate_size], b + 22.0),
            );
        }
        m
    }

    #[test]
    fn hyper_connection_matches_reference() -> candle::Result<()> {
        let cfg = parity_config();
        let dev = Device::Cpu;
        let hc = cfg.hc_mult;
        let mut m = HashMap::new();
        m.insert(
            "fn".into(),
            det_tensor(&[(2 + hc) * hc, hc * cfg.hidden_size], 91.0),
        );
        m.insert(
            "base".into(),
            det_tensor(&[(2 + hc) * hc], 92.0).flatten_all().unwrap(),
        );
        m.insert(
            "scale".into(),
            det_tensor(&[3], 93.0).flatten_all().unwrap(),
        );
        let vb = VarBuilder::from_tensors(m.clone(), DType::F32, &dev);
        let hcmod = DeepseekV4HyperConnection::new(&cfg, vb)?;
        let streams = det_tensor(&[1, 4, hc, cfg.hidden_size], 94.0);
        let (post, comb, collapsed) = hcmod.forward(&streams)?;
        let (rpost, rcomb, rcollapsed) = ref_hc(&cfg, &m["fn"], &m["base"], &m["scale"], &streams)?;
        assert_eq!(post.dims(), rpost.dims(), "post shape");
        assert_eq!(comb.dims(), rcomb.dims(), "comb shape");
        assert_eq!(collapsed.dims(), rcollapsed.dims(), "collapsed shape");
        assert_close(&post, &rpost, 1e-4, "hc-post");
        assert_close(&comb, &rcomb, 1e-4, "hc-comb");
        assert_close(&collapsed, &rcollapsed, 1e-4, "hc-collapsed");
        Ok(())
    }

    #[test]
    fn hyper_head_matches_reference() -> candle::Result<()> {
        let cfg = parity_config();
        let dev = Device::Cpu;
        let hc = cfg.hc_mult;
        let mut m = HashMap::new();
        m.insert(
            "hc_fn".into(),
            det_tensor(&[hc, hc * cfg.hidden_size], 95.0),
        );
        m.insert(
            "hc_base".into(),
            det_tensor(&[hc], 96.0).flatten_all().unwrap(),
        );
        m.insert(
            "hc_scale".into(),
            det_tensor(&[1], 97.0).flatten_all().unwrap(),
        );
        let vb = VarBuilder::from_tensors(m.clone(), DType::F32, &dev);
        let hh = DeepseekV4HyperHead::new(&cfg, vb)?;
        let x = det_tensor(&[1, 3, hc, cfg.hidden_size], 98.0);
        let out = hh.forward(&x)?;
        let ref_out = ref_hc_head(&cfg, &m["hc_fn"], &m["hc_base"], &m["hc_scale"], &x)?;
        assert_close(&out, &ref_out, 1e-4, "hc-head");
        Ok(())
    }

    #[test]
    fn deepseek_v4_model_forward_parity_with_transformers() -> candle::Result<()> {
        let cfg = model_config();
        let dev = Device::Cpu;
        let weights = model_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut model = DeepseekV4ForCausalLM::new(&cfg, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let ids = Tensor::new(&[1u32, 3, 7, 2, 5][..], &dev)?.unsqueeze(0)?; // [1,5]
        let logits = model.forward(&ids, 0)?;
        let mut caches: Vec<Option<Tensor>> = vec![None; cfg.num_hidden_layers];
        let ref_logits = ref_model_forward(&cfg, &emb, &weights, &ids, 0, &mut caches)?;
        assert_eq!(logits.dims(), ref_logits.dims(), "logits shape");
        assert_close(&logits, &ref_logits, 1e-4, "model-forward");
        Ok(())
    }

    #[test]
    fn deepseek_v4_generation_matches_reference() -> candle::Result<()> {
        let cfg = model_config();
        let dev = Device::Cpu;
        let weights = model_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut model = DeepseekV4ForCausalLM::new(&cfg, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;

        // Prefill of 4 tokens.
        let prompt = Tensor::new(&[2u32, 5, 1, 4][..], &dev)?.unsqueeze(0)?;
        let l0 = model.forward(&prompt, 0)?;
        let mut caches: Vec<Option<Tensor>> = vec![None; cfg.num_hidden_layers];
        let r0 = ref_model_forward(&cfg, &emb, &weights, &prompt, 0, &mut caches)?;
        assert_close(&l0, &r0, 1e-4, "prefill");

        // Decode 3 tokens greedily, comparing logits and chosen token per step.
        let mut cur = l0
            .narrow(D::Minus2, l0.dim(D::Minus2)? - 1, 1)?
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?;
        for step in 0..3 {
            let l = model.forward(&cur, 4 + step)?;
            let r = ref_model_forward(&cfg, &emb, &weights, &cur, 4 + step, &mut caches)?;
            assert_close(&l, &r, 1e-4, &format!("decode{step}"));
            cur = l
                .narrow(D::Minus2, 0, 1)?
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?;
            let ref_cur = r
                .narrow(D::Minus2, 0, 1)?
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?;
            let a = cur.flatten_all()?.to_vec1::<u32>()?;
            let b = ref_cur.flatten_all()?.to_vec1::<u32>()?;
            assert_eq!(a, b, "decode step {step} token mismatch");
        }
        Ok(())
    }
    #[test]
    fn deepseek_v4_generate_matches_reference() -> candle::Result<()> {
        let cfg = model_config();
        let dev = Device::Cpu;
        let weights = model_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let mut model = DeepseekV4ForCausalLM::new(&cfg, false, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;

        let prompt = Tensor::new(&[2u32, 5, 1, 4][..], &dev)?.unsqueeze(0)?;
        let eos = 1u32;
        let max_new = 5usize;

        // Public `generate()` with an ArgMax sampler over last-position logits.
        let tokens = model.generate(&prompt, max_new, eos as usize, |logits| {
            let t = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec0::<u32>()?;
            Ok(t)
        })?;

        // Independent reference greedy decode (stateless full-history recompute).
        let mut caches: Vec<Option<Tensor>> = vec![None; cfg.num_hidden_layers];
        let r0 = ref_model_forward(&cfg, &emb, &weights, &prompt, 0, &mut caches)?;
        let mut cur = r0
            .narrow(D::Minus2, r0.dim(D::Minus2)? - 1, 1)?
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?;
        let mut ref_tokens = Vec::new();
        for step in 0..max_new {
            let r = ref_model_forward(&cfg, &emb, &weights, &cur, 4 + step, &mut caches)?;
            let rt = r
                .narrow(D::Minus2, 0, 1)?
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?[0];
            ref_tokens.push(rt);
            if rt == eos {
                break;
            }
            cur = Tensor::new(&[rt], &dev)?.unsqueeze(0)?;
        }
        assert_eq!(tokens, ref_tokens, "generated token sequence mismatch");
        Ok(())
    }

    /// MoE parity config: hidden=8, moe_intermediate=16, n_routed_experts=8,
    /// num_experts_per_tok=2, num_hash_layers=1 (layer 0 hash, layer 1 top-k),
    /// norm_topk_prob, routed_scaling_factor=2.5, scoring sqrtsoftplus.
    fn moe_parity_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 32, "hidden_size": 8, "moe_intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "q_lora_rank": 4, "o_lora_rank": 4, "qk_rope_head_dim": 2,
                "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 2, "num_hash_layers": 1,
                "index_head_dim": 2, "index_n_heads": 1, "index_topk": 4, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "partial_rotary_factor": 0.5,
                "sliding_window": 3, "max_position_embeddings": 8, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
                "attention_bias": false, "attention_dropout": 0.0, "mlp_bias": false,
                "norm_topk_prob": true, "routed_scaling_factor": 2.5,
                "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
                "swiglu_limit": 10.0, "initializer_range": 0.02,
                "output_router_logits": false, "router_aux_loss_coef": 0.001, "router_jitter_noise": 0.0,
                "use_cache": true, "bos_token_id": 0, "eos_token_id": 1,
                "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
                "compress_ratios": [0, 0],
                "layer_types": ["sliding_attention", "sliding_attention"],
                "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                                 "original_max_position_embeddings": 65536, "type": "yarn"}
            }"#,
        )
        .unwrap()
    }

    /// MoE weights keyed by their state-dict path (`mlp.…`).
    fn moe_parity_weights(cfg: &DeepseekV4Config) -> HashMap<String, Tensor> {
        let e = cfg.n_routed_experts;
        let h = cfg.hidden_size;
        let int = cfg.moe_intermediate_size;
        let mut m = HashMap::new();
        m.insert("mlp.gate.weight".into(), det_tensor(&[e, h], 1.0));
        m.insert(
            "mlp.gate.e_score_correction_bias".into(),
            det_tensor(&[e], 2.0),
        );
        let vocab = cfg.vocab_size;
        let tk = cfg.num_experts_per_tok;
        let v: Vec<u32> = (0..vocab * tk).map(|i| (i as u32) % (e as u32)).collect();
        m.insert(
            "mlp.gate.tid2eid".into(),
            Tensor::from_vec(v, &[vocab, tk], &Device::Cpu).unwrap(),
        );
        m.insert(
            "mlp.experts.gate_up_proj".into(),
            det_tensor(&[e, 2 * int, h], 3.0),
        );
        m.insert(
            "mlp.experts.down_proj".into(),
            det_tensor(&[e, h, int], 4.0),
        );
        m.insert(
            "mlp.shared_experts.gate_proj.weight".into(),
            det_tensor(&[int, h], 5.0),
        );
        m.insert(
            "mlp.shared_experts.up_proj.weight".into(),
            det_tensor(&[int, h], 6.0),
        );
        m.insert(
            "mlp.shared_experts.down_proj.weight".into(),
            det_tensor(&[h, int], 7.0),
        );
        m
    }

    /// Independent reference for the sparse MoE block covering both router
    /// types, transcribed from `modeling_deepseek_v4.py` (TopK/Hash router +
    /// `DeepseekV4Experts` + shared MLP). Includes `e_score_correction_bias`.
    fn ref_moe_parity(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Tensor>,
        x: &Tensor,
        input_ids: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bs, seq, d) = x.dims3()?;
        let h = cfg.hidden_size;
        let e = cfg.n_routed_experts;
        let top_k = cfg.num_experts_per_tok;
        let flat = x.reshape((bs * seq, h))?;
        let scores = sqrt_softplus(&flat.matmul(&w["mlp.gate.weight"].t()?)?)?;
        let indices = if layer_idx < cfg.num_hash_layers {
            let ids = input_ids.unwrap().flatten_all()?.to_dtype(DType::I64)?;
            w["mlp.gate.tid2eid"]
                .index_select(&ids, 0)?
                .to_dtype(DType::I64)?
                .contiguous()?
        } else {
            let biased = scores.broadcast_add(&w["mlp.gate.e_score_correction_bias"])?;
            topk_last_dim(&biased, top_k)?
        };
        let mut weights = scores.gather(&indices, D::Minus1)?;
        let denom = (weights.sum(D::Minus1)?.unsqueeze(D::Minus1)? + 1e-20)?;
        weights = (weights.broadcast_div(&denom)? * cfg.routed_scaling_factor)?;
        // routed experts (all tokens, weighted by the per-expert top-k mask)
        let gu = &w["mlp.experts.gate_up_proj"];
        let dw = &w["mlp.experts.down_proj"];
        let n = flat.dim(0)?;
        let k = indices.dim(1)?;
        let zero = Tensor::zeros((n, k), DType::F32, x.device())?;
        let mut out = flat.zeros_like()?;
        for i in 0..e {
            let sel = indices.eq(i as u32)?;
            let wsel = sel.where_cond(&weights, &zero)?;
            let wsum = wsel.sum(D::Minus1)?;
            let gui = gu.narrow(0, i, 1)?.squeeze(0)?;
            let all = flat.matmul(&gui.t()?)?;
            let c = all.chunk(2, D::Minus1)?;
            let gate = c[0].clamp(f64::NEG_INFINITY, cfg.swiglu_limit)?;
            let up = c[1].clamp(-cfg.swiglu_limit, cfg.swiglu_limit)?;
            let act = (candle_nn::ops::silu(&gate)? * up)?;
            let dwi = dw.narrow(0, i, 1)?.squeeze(0)?;
            let down = act.matmul(&dwi.t()?)?;
            out = out.add(&down.broadcast_mul(&wsum.unsqueeze(D::Minus1)?)?)?;
        }
        let shared = ref_mlp(
            cfg,
            &w["mlp.shared_experts.gate_proj.weight"],
            &w["mlp.shared_experts.up_proj.weight"],
            &w["mlp.shared_experts.down_proj.weight"],
            x,
        )?;
        let routed = out.reshape((bs, seq, d))?;
        routed.add(&shared)
    }

    #[test]
    fn moe_topk_bias_parity() -> candle::Result<()> {
        let cfg = moe_parity_config();
        let dev = Device::Cpu;
        let weights = moe_parity_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let moe = DeepseekV4SparseMoeBlock::new(&cfg, 1, vb.pp("mlp"))?;
        let x = det_tensor(&[2, 3, cfg.hidden_size], 91.0);
        let out = moe.forward(&x, None)?;
        let ref_out = ref_moe_parity(&cfg, &weights, &x, None, 1)?;
        assert_eq!(out.dims(), ref_out.dims(), "moe-topk shape");
        assert_close(&out, &ref_out, 1e-4, "moe-topk-bias");
        Ok(())
    }

    #[test]
    fn moe_hash_parity() -> candle::Result<()> {
        let cfg = moe_parity_config();
        let dev = Device::Cpu;
        let weights = moe_parity_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let moe = DeepseekV4SparseMoeBlock::new(&cfg, 0, vb.pp("mlp"))?;
        let ids = Tensor::new(&[3u32, 7, 2, 1, 0, 5][..], &dev)?;
        let x = det_tensor(&[2, 3, cfg.hidden_size], 92.0);
        let out = moe.forward(&x, Some(&ids))?;
        let ref_out = ref_moe_parity(&cfg, &weights, &x, Some(&ids), 0)?;
        assert_eq!(out.dims(), ref_out.dims(), "moe-hash shape");
        assert_close(&out, &ref_out, 1e-4, "moe-hash");
        Ok(())
    }

    /// Independent reference for the auxiliary load-balancing loss (no mask).
    fn ref_load_balancing(logits: &[Tensor], e: usize, top_k: usize) -> f32 {
        let mut tps = vec![0.0f32; e];
        let mut rps = vec![0.0f32; e];
        let mut total = 0.0f32;
        for lg in logits {
            let rw = candle_nn::ops::softmax_last_dim(lg)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let (n, ne) = rw.dims2().unwrap();
            assert_eq!(ne, e);
            let counts = rw
                .topk_unsorted(top_k)
                .unwrap()
                .indices
                .flatten_all()
                .unwrap()
                .bincount(e as u32)
                .unwrap();
            for (i, c) in counts.iter().enumerate() {
                tps[i] += *c as f32;
            }
            let col = rw.sum_keepdim(0).unwrap().flatten_all().unwrap();
            let col = col.to_vec1::<f32>().unwrap();
            for (i, s) in col.iter().enumerate() {
                rps[i] += s;
            }
            total += n as f32;
        }
        let mut overall = 0.0;
        for i in 0..e {
            overall += (tps[i] / total) * (rps[i] / total);
        }
        overall * e as f32
    }

    #[test]
    fn moe_aux_loss_matches_reference() -> candle::Result<()> {
        let cfg = moe_parity_config();
        let weights = moe_parity_weights(&cfg);
        let x = det_tensor(&[2, 3, cfg.hidden_size], 95.0);
        let flat = x.reshape(((), cfg.hidden_size))?;
        let logits = flat.matmul(&weights["mlp.gate.weight"].t()?)?;
        let l = load_balancing_loss(
            std::slice::from_ref(&logits),
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            None,
        )?;
        let expected = ref_load_balancing(&[logits], cfg.n_routed_experts, cfg.num_experts_per_tok);
        let got: f32 = l.to_scalar()?;
        assert!(
            (got - expected).abs() < 1e-4,
            "aux loss got {got}, expected {expected}"
        );
        Ok(())
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            // Handle NaN and infinities (e.g. -inf in block_bias) explicitly.
            let ok = (x.is_nan() && y.is_nan()) || (x == y) || (x - y).abs() < tol;
            assert!(ok, "{label}[{i}]: got {x}, expected {y}");
        }
    }
    /// Multi-step (300+) decode parity: the incremental compressor cache (one
    /// token per `forward` call, carrying buffer / overlap / running state
    /// across calls) must produce the same `compressed_kv` + `block_bias` as a
    /// stateless full-history recompute of the whole prefix at every step, for
    /// both CSA (rate 4, overlap carry-across-forward) and HCA (rate 128).
    #[test]
    fn incremental_decode_parity_vs_full_history() -> candle::Result<()> {
        let cfg = decode_parity_config();
        let dev = Device::Cpu;
        let n_steps = 320usize;

        // ---- CSA (rate 4, overlap carry-across-forward) ----
        {
            let weights = csa_parity_weights(&cfg);
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
            let mut comp = DeepseekV4CSACompressor::new(&cfg, vb)?;
            let rotary = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;

            let mut hist: Option<Tensor> = None; // [B, H, hidden]
            let mut qhist: Option<Tensor> = None; // [B, H, q_lora_rank]

            for t in 0..n_steps {
                let x = det_tensor(&[1, 1, cfg.hidden_size], (100 + t) as f32);
                let qr = det_tensor(&[1, 1, cfg.q_lora_rank], (200 + t) as f32);
                hist = Some(match hist {
                    Some(h) => Tensor::cat(&[&h, &x], 1)?,
                    None => x.clone(),
                });
                qhist = Some(match qhist {
                    Some(h) => Tensor::cat(&[&h, &qr], 1)?,
                    None => qr.clone(),
                });
                let h = hist.as_ref().unwrap();
                let qh = qhist.as_ref().unwrap();

                // Incremental cache: single-token forward at absolute position t.
                let (ckv, bb) = comp.forward(&x, &qr, t)?;
                // Full-history eager recompute over the whole prefix.
                let (r_ckv, r_bb) = ref_csa(&cfg, &weights, &rotary, h, qh, 0, &mut (None, None))?;
                assert_eq!(
                    ckv.dims(),
                    r_ckv.dims(),
                    "CSA step {t}: compressed_kv shape"
                );
                assert_close(&ckv, &r_ckv, 1e-4, &format!("CSA step {t}: compressed_kv"));

                // Last full-history query row == the single-token block_bias.
                let h_len = h.dim(1)?;
                let r_bb_last = r_bb.narrow(2, h_len - 1, 1)?;
                assert_eq!(
                    bb.dims(),
                    r_bb_last.dims(),
                    "CSA step {t}: block_bias shape"
                );
                assert_close(&bb, &r_bb_last, 1e-4, &format!("CSA step {t}: block_bias"));
            }
        }

        // ---- HCA (rate 128) ----
        {
            let weights = hca_parity_weights(&cfg);
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
            let mut comp = DeepseekV4HCACompressor::new(&cfg, vb)?;
            let rotary = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;

            let mut hist: Option<Tensor> = None; // [B, H, hidden]
            let mut qhist: Option<Tensor> = None; // [B, H, q_lora_rank]

            for t in 0..n_steps {
                let x = det_tensor(&[1, 1, cfg.hidden_size], (300 + t) as f32);
                let qr = det_tensor(&[1, 1, cfg.q_lora_rank], (400 + t) as f32);
                hist = Some(match hist {
                    Some(h) => Tensor::cat(&[&h, &x], 1)?,
                    None => x.clone(),
                });
                qhist = Some(match qhist {
                    Some(h) => Tensor::cat(&[&h, &qr], 1)?,
                    None => qr.clone(),
                });
                let h = hist.as_ref().unwrap();
                let qh = qhist.as_ref().unwrap();

                let (ckv, bb) = comp.forward(&x, &qr, t)?;
                let (r_ckv, r_bb) = ref_hca(&cfg, &weights, &rotary, h, qh, 0)?;
                assert_eq!(
                    ckv.dims(),
                    r_ckv.dims(),
                    "HCA step {t}: compressed_kv shape"
                );
                assert_close(&ckv, &r_ckv, 1e-4, &format!("HCA step {t}: compressed_kv"));

                let h_len = h.dim(1)?;
                let r_bb_last = r_bb.narrow(2, h_len - 1, 1)?;
                assert_eq!(
                    bb.dims(),
                    r_bb_last.dims(),
                    "HCA step {t}: block_bias shape"
                );
                assert_close(&bb, &r_bb_last, 1e-4, &format!("HCA step {t}: block_bias"));
            }
        }
        Ok(())
    }
}
