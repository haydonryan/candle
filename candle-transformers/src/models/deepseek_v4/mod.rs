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
            cfg: cfg.clone(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
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

        // Single shared KV head is broadcast to every query head for the bmm.
        let k = k.broadcast_as((bs, num_heads, kv_len, head_dim))?;
        let v = v.broadcast_as((bs, num_heads, kv_len, head_dim))?;
        let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.softmax_scale)?;
        let mask = self.sliding_window_mask(seq_len, kv_len, prev_len, xs.device())?;
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
            assert!((x - y).abs() < tol, "{label}[{i}]: got {x}, expected {y}");
        }
    }
}
