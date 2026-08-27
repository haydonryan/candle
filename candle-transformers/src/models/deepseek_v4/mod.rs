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

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{rms_norm, Activation, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

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
            .transpose(1, 2)?;
        let xr = x
            .reshape((batch, self.n_groups, in_per_group))?
            .transpose(0, 1)?;
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
        let rope_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
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
    compressor: Option<DeepseekV4CSACompressor>,
    cfg: DeepseekV4Config,
}

impl DeepseekV4Attention {
    pub fn new(cfg: &DeepseekV4Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
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
        let compressor = if cfg.layer_types[layer_idx] == LayerType::CompressedSparseAttention {
            Some(DeepseekV4CSACompressor::new(cfg, vb.pp("compressor"))?)
        } else {
            None
        };
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

    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
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
        let (k, v, mask) = match &mut self.compressor {
            None => {
                let mask = self.sliding_window_mask(seq_len, kv_len, prev_len, xs.device())?;
                (k, v, mask)
            }
            Some(comp) => {
                let (compressed_kv, block_bias) = comp.forward(xs, &q_residual, seqlen_offset)?;
                let k = Tensor::cat(&[k.clone(), compressed_kv.clone()], 2)?;
                let v = Tensor::cat(&[v, compressed_kv], 2)?;
                let sliding = self
                    .sliding_window_mask(seq_len, kv_len, prev_len, xs.device())?
                    .broadcast_as((bs, 1, seq_len, kv_len))?;
                let mask = Tensor::cat(&[sliding, block_bias], 3)?;
                (k, v, mask)
            }
        };

        let kv_len = k.dim(2)?;

        // Single shared KV head is broadcast to every query head for the bmm.
        let k = k.broadcast_as((bs, num_heads, kv_len, head_dim))?;
        let v = v.broadcast_as((bs, num_heads, kv_len, head_dim))?;
        let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.softmax_scale)?;
        let att = att.broadcast_add(&mask)?;

        // Per-head learnable sink appended pre-softmax, dropped after.
        let sinks = self
            .sinks
            .reshape((1, num_heads, 1, 1))?
            .broadcast_as((bs, num_heads, seq_len, 1))?;
        let att = Tensor::cat(&[att, sinks], D::Minus1)?.contiguous()?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att.narrow(D::Minus1, 0, kv_len)?;

        let attn_out = att.matmul(&v.contiguous()?)?;

        // K == V, so V carries RoPE on its rope slice; undo it at the query
        // positions so each KV contribution stays a function of relative distance.
        let attn_out = self
            .rotary_emb
            .forward_conjugate(&attn_out, variant, seqlen_offset)?;

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
    fn running_compressed(&self, device: &Device, head_dim: usize) -> Result<Tensor> {
        match &self.compressed_kv {
            Some(c) => Ok(c.clone()),
            None => Tensor::zeros((1, 0, head_dim), DType::F32, device),
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
            None => Tensor::full(f32::NEG_INFINITY, (batch, rate, head_dim), device)?,
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
            return Ok((self.running_compressed(chunk_kv.device(), head_dim)?, 0));
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
            rotary_emb: Arc::new(DeepseekV4RotaryEmbedding::new(cfg, vb.device())?),
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

        let index_scores = self.scorer.forward(&q, &compressed_kv, hidden_states)?; // [B, S, T]
        let compressed_len = compressed_kv.dim(1)?;
        let top_k = self.index_topk.min(compressed_len);
        if compressed_len == 0 || top_k == 0 {
            return Tensor::zeros((batch, seq_len, 0), DType::I64, device);
        }

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
                    Tensor::full(f32::NEG_INFINITY, (batch, rate, hd), dev)?,
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
        let topk = invalid.where_cond(&minus_one, &topk)?;

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
        let mut attn = DeepseekV4Attention::new(&cfg, 0, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let x = det_tensor(&[1, 5, cfg.hidden_size], 51.0);
        let out = attn.forward(&x, 0)?;
        let ref_out = reference_csa_attention(&cfg, &emb, &attn_w, &csa_w, &x, 0)?;
        assert_close(&out, &ref_out, 1e-4, "csa-attn");
        Ok(())
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
        let mut attn = DeepseekV4Attention::new(&cfg, 0, vb)?;
        let emb = DeepseekV4RotaryEmbedding::new(&cfg, &dev)?;
        let mut cache: Option<Tensor> = None;

        let x1 = det_tensor(&[1, 5, cfg.hidden_size], 11.0);
        let o1 = attn.forward(&x1, 0)?;
        let r1 = reference_forward(&cfg, &emb, &weights, &x1, 0, &mut cache)?;
        assert_close(&o1, &r1, 1e-4, "step1");

        // Step 2: a two-token continuation after the cache has been trimmed.
        let x2 = det_tensor(&[1, 2, cfg.hidden_size], 13.0);
        let o2 = attn.forward(&x2, 5)?;
        let r2 = reference_forward(&cfg, &emb, &weights, &x2, 5, &mut cache)?;
        assert_close(&o2, &r2, 1e-4, "step2");

        // Step 3: single-token generation off the sliding cache.
        let x3 = det_tensor(&[1, 1, cfg.hidden_size], 17.0);
        let o3 = attn.forward(&x3, 7)?;
        let r3 = reference_forward(&cfg, &emb, &weights, &x3, 7, &mut cache)?;
        assert_close(&o3, &r3, 1e-4, "step3");
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
}
