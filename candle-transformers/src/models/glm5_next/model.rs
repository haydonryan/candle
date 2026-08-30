//! GLM-5.3-Flash text causal LM assembly (story #4355).
//!
//! Assembles the text-only `Glm5Next` model from the KDA linear-attention block
//! (`kda::KdaLinearAttention`), a new DeepSeek-Sparse-Attention (MLA + indexer)
//! block for the DSA layers, and reused Candle components from `deepseek_v4`:
//! the Manifold-Constrained Hyper-Connection (`DeepseekV4HyperConnection`) and
//! the routed MoE expert weights (`DeepseekV4Experts`). Dense and shared MLPs
//! mirror the reference `transformers` `Glm5NextTextMLP` (clamped SiLU); the
//! grouped top-k MoE router and the k-pool DSA indexer are GLM-5.3-specific and
//! are implemented here matching `transformers` `Glm5NextTextTopkRouter` and
//! `Glm5NextTextIndexer`.
//!
//! Weight names match `transformers` `Glm5NextTextModel` / `Glm5NextTextDecoderLayer`
//! (`embed_tokens`, `layers.{i}.{input_layernorm,self_attn,post_attention_layernorm,
//! mlp,attn_hc,ffn_hc}`, `norm`, `lm_head`), so official checkpoints load
//! directly.
//!
//! Non-goals (per the story): vision/video inputs, MTP/speculative decoding,
//! training, FP8/NVFP4/GGUF, tensor parallelism, or new optimized kernels.

use super::kda::KdaLinearAttention;
use super::{Glm5NextTextConfig, IndexerType, LayerType, MlpLayerType};
use crate::models::deepseek_v4::{DeepseekV4Experts, DeepseekV4HyperConnection};
use candle::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

/// Boolean AND of two (boolean or 0/1 numeric) tensors, returning a bool tensor.
fn bool_and(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let x = a
        .to_dtype(DType::F32)?
        .broadcast_mul(&b.to_dtype(DType::F32)?)?;
    x.gt(0.0)
}

/// Indices of the `k` largest elements along the last dim, descending.
fn topk_last_dim(x: &Tensor, k: usize) -> Result<Tensor> {
    let sorted = x.contiguous()?.arg_sort_last_dim(false)?;
    sorted
        .narrow(D::Minus1, 0, k)?
        .contiguous()?
        .to_dtype(DType::I64)
}

/// `(values, indices)` of the `k` largest elements along the last dim.
fn topk_last_dim_values(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let idx = topk_last_dim(x, k)?;
    let vals = x.gather(&idx, D::Minus1)?;
    Ok((vals, idx))
}

/// Any over a dim: convert to f32, sum, check `> 0`.
fn any_gt_zero<Dm: candle::shape::Dim>(x: &Tensor, dim: Dm) -> Result<Tensor> {
    let s = x.to_dtype(DType::F32)?.sum(dim)?;
    s.gt(0.0)
}

/// All over a dim for a `[.., N]` boolean tensor: sum equals `N`.
fn all_eq_len<Dm: candle::shape::Dim>(x: &Tensor, dim: Dm, n: usize) -> Result<Tensor> {
    let s = x.to_dtype(DType::F32)?.sum(dim)?;
    s.eq(n as f32)
}

/// Dense / shared-expert MLP, mirroring `transformers` `Glm5NextTextMLP`:
/// `silu(clamp(gate)) * clamp(up)` then down-projection, with the GLM-5.3
/// swiglu clamp limits.
pub struct Glm5NextMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    limit: f64,
}

impl Glm5NextMLP {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        swiglu_limit: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            limit: swiglu_limit,
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

/// Grouped top-k MoE router, mirroring `transformers` `Glm5NextTextTopkRouter`.
///
/// Scores are sigmoid logits; the `num_experts` are split into `n_group`
/// groups and the top `topk_group` groups are selected by the sum of their
/// two best expert scores. Weights are gathered from the raw scores, optionally
/// normalised over the selected experts, and scaled by `routed_scaling_factor`.
pub struct Glm5NextTopkRouter {
    weight: Tensor,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    num_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl Glm5NextTopkRouter {
    pub fn new(cfg: &Glm5NextTextConfig, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((cfg.n_routed_experts, cfg.hidden_size), "weight")?;
        Ok(Self {
            weight,
            top_k: cfg.num_experts_per_tok,
            num_experts: cfg.n_routed_experts,
            hidden: cfg.hidden_size,
            num_group: cfg.n_group,
            topk_group: cfg.topk_group,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// `hidden_states` is `[B, S, D]`; returns `(router_logits [N, E],
    /// topk_weights [N, K], topk_indices [N, K])` with `N = B*S`.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (bs, seq, _) = hidden_states.dims3()?;
        let n = bs * seq;
        let dev = hidden_states.device();
        let flat = hidden_states.reshape((n, self.hidden))?;
        let router_logits = flat.matmul(&self.weight.t()?)?; // [N, E]
        let scores = candle_nn::ops::sigmoid(&router_logits)?; // [N, E]

        let per_grp = self.num_experts / self.num_group;
        // Group scores = sum of the top-2 expert scores within each group.
        let grouped = scores.reshape((n, self.num_group, per_grp))?;
        let (gtop_vals, _) = topk_last_dim_values(&grouped, 2)?;
        let group_scores = gtop_vals.sum(D::Minus1)?; // [N, num_group]

        let group_idx = topk_last_dim(&group_scores, self.topk_group)?; // [N, topk_group]
        let ones = Tensor::ones((n, self.topk_group), DType::F32, dev)?;
        let group_mask = Tensor::zeros((n, self.num_group), DType::F32, dev)?.scatter_add(
            &group_idx,
            &ones,
            D::Minus1,
        )?;
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as((n, self.num_group, per_grp))?
            .reshape((n, self.num_experts))?;

        let neg_inf = Tensor::full(f32::NEG_INFINITY, (n, self.num_experts), dev)?;
        let masked = score_mask.gt(0.0)?.where_cond(&scores, &neg_inf)?;
        let (_, topk_indices) = topk_last_dim_values(&masked, self.top_k)?; // [N, K]
        let topk_weights = scores.contiguous()?.gather(&topk_indices, D::Minus1)?; // [N, K]
        let topk_weights = if self.norm_topk_prob {
            let denom = (topk_weights.sum(D::Minus1)?.unsqueeze(D::Minus1)? + 1e-20)?;
            topk_weights.broadcast_div(&denom)?
        } else {
            topk_weights
        };
        let topk_weights = (topk_weights * self.routed_scaling_factor)?;
        Ok((router_logits, topk_weights, topk_indices))
    }
}

/// Routed + shared MoE block, mirroring `transformers` `Glm5NextTextMoE`.
///
/// Reuses the DeepSeek-V4 routed expert weights (`DeepseekV4Experts`, generic
/// over `MoeExpertsConfig`) and a GLM-5.3 shared MLP sized
/// `moe_intermediate_size * n_shared_experts`.
pub struct Glm5NextTextMoE {
    experts: DeepseekV4Experts,
    gate: Glm5NextTopkRouter,
    shared_experts: Glm5NextMLP,
}

impl Glm5NextTextMoE {
    pub fn new(cfg: &Glm5NextTextConfig, vb: VarBuilder) -> Result<Self> {
        let experts = DeepseekV4Experts::new(cfg, vb.pp("experts"))?;
        let gate = Glm5NextTopkRouter::new(cfg, vb.pp("gate"))?;
        let shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let shared_experts = Glm5NextMLP::new(
            cfg.hidden_size,
            shared_inter,
            cfg.swiglu_limit,
            vb.pp("shared_experts"),
        )?;
        Ok(Self {
            experts,
            gate,
            shared_experts,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, seq, d) = x.dims3()?;
        let (_, topk_weights, topk_indices) = self.gate.forward(x)?;
        let flat = x.reshape((bs * seq, d))?;
        let routed = self
            .experts
            .forward(&flat, &topk_indices, &topk_weights)?
            .reshape((bs, seq, d))?;
        let shared = self.shared_experts.forward(x)?;
        routed.add(&shared)
    }
}

/// DSA k-pool indexer, mirroring `transformers` `Glm5NextTextIndexer`.
///
/// Scores compressed k-pool candidates with lightweight projections, expands
/// the selected pools back into raw cache indices, and (optionally) appends the
/// current incomplete tail pool. The accumulated per-layer packed state is
/// cached so cached decode can select over the full past context.
pub struct Glm5NextTextIndexer {
    wq_b: Linear,
    wk: Linear,
    k_norm: candle_nn::LayerNorm,
    weights_proj: Linear,
    ape: Tensor,
    gate: Tensor,
    num_heads: usize,
    head_dim: usize,
    index_topk: usize,
    index_kpool: usize,
    always_select_tail: bool,
    softmax_scale: f64,
    packed_cache: Option<Tensor>,
}

impl Glm5NextTextIndexer {
    pub fn new(cfg: &Glm5NextTextConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.index_n_heads;
        let head_dim = cfg.index_head_dim;
        let wq_b = linear_no_bias(cfg.q_lora_rank, num_heads * head_dim, vb.pp("wq_b"))?;
        let wk = linear_no_bias(cfg.hidden_size, head_dim, vb.pp("wk"))?;
        let k_norm = candle_nn::layer_norm(head_dim, 1e-6, vb.pp("k_norm"))?;
        let weights_proj = linear_no_bias(cfg.hidden_size, num_heads, vb.pp("weights_proj"))?;
        let ape = vb.get((cfg.index_kpool, head_dim), "index_kpool_compress_ape")?;
        let gate = vb.get((head_dim, cfg.hidden_size), "index_kpool_compress_gate")?;
        Ok(Self {
            wq_b,
            wk,
            k_norm,
            weights_proj,
            ape,
            gate,
            num_heads,
            head_dim,
            index_topk: cfg.index_topk,
            index_kpool: cfg.index_kpool,
            always_select_tail: cfg.index_kpool_always_select_tail,
            softmax_scale: (head_dim as f64).powf(-0.5),
            packed_cache: None,
        })
    }

    pub fn clear_cache(&mut self) {
        self.packed_cache = None;
    }

    /// Select top-k token indices per query for DSA.
    ///
    /// * `hidden_states`: `[B, S, hidden]`.
    /// * `q_resid`: query residual `[B, S, q_lora_rank]` from the MLA
    ///   `q_a_layernorm(q_a_proj(x))`.
    /// * `attention_mask`: optional `[B, S]` 0/1 float mask (1 = real token).
    ///
    /// Returns `int64` top-k indices `[B, S, output_width]` with `-1` sentinels
    /// for invalid selections (matching the reference's `-1` convention).
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        q_resid: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq, _) = hidden_states.dims3()?;
        let dev = hidden_states.device();
        let hd = self.head_dim;
        let k = self.index_kpool;

        let valid_channel = match attention_mask {
            Some(m) => m.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?,
            None => Tensor::ones((batch, seq, 1), DType::F32, dev)?,
        };

        let q = self
            .wq_b
            .forward(q_resid)?
            .reshape((batch, seq, self.num_heads, hd))?;
        let kk = self
            .k_norm
            .forward(&self.wk.forward(hidden_states)?)?
            .reshape((batch, seq, hd))?;
        let gate_scores = hidden_states.matmul(&self.gate.t()?.unsqueeze(0)?)?; // [B, S, hd]
        let packed = Tensor::cat(&[&kk, &gate_scores, &valid_channel], D::Minus1)?;

        // Accumulate across calls so cached decode selects over the full context.
        let (packed, kv_len) = match &self.packed_cache {
            Some(prev) => {
                let full = Tensor::cat(&[prev, &packed], D::Minus2)?;
                let len = full.dim(D::Minus2)?;
                self.packed_cache = Some(full.clone());
                (full, len)
            }
            None => {
                let len = packed.dim(D::Minus2)?;
                self.packed_cache = Some(packed.clone());
                (packed, len)
            }
        };
        let current_length = kv_len;
        let q_length = seq;

        let valid_keys = packed
            .narrow(D::Minus1, 2 * hd, 1)?
            .squeeze(D::Minus1)?
            .gt(0.0)?; // [B, kv_len]

        // Visibility per query position over the accumulated KV.
        let visible_tokens = self.get_visible_tokens(&valid_keys, q_length, current_length)?; // [B,S,kv_len]

        let (pool_keys, pool_indices, pool_valid) = self.get_pooled_states(&packed)?;
        let n_pools = pool_keys.dim(1)?;

        // Score across pools (not per token).
        let scores = q
            .to_dtype(DType::F32)?
            .broadcast_matmul(&pool_keys.t()?.unsqueeze(1)?)?;
        let scores = (scores * self.softmax_scale)?.clamp(0.0, f32::MAX)?; // [B, S, n_heads, n_pools]

        let weights = (self
            .weights_proj
            .forward(hidden_states)?
            .to_dtype(DType::F32)?
            * (self.num_heads as f64).powf(-0.5))?; // [B, S, n_heads]
        let index_scores = weights
            .unsqueeze(D::Minus2)?
            .broadcast_matmul(&scores)?
            .squeeze(D::Minus2)?; // [B, S, n_pools]

        // A pool is selectable only if its final token is visible to the query.
        let pool_end = pool_indices
            .narrow(D::Minus1, k - 1, 1)?
            .squeeze(D::Minus1)?
            .clamp(0, (kv_len - 1) as i64)?; // [B, n_pools]
        let pool_visible = visible_tokens.contiguous()?.gather(
            &pool_end
                .unsqueeze(1)?
                .broadcast_as((batch, seq, n_pools))?
                .contiguous()?,
            D::Minus1,
        )?; // [B, S, n_pools]
        let valid_candidates = bool_and(&pool_visible, &pool_valid.unsqueeze(1)?)?;

        let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch, seq, n_pools), dev)?;
        let index_scores = valid_candidates.where_cond(&index_scores, &neg_inf)?;

        let select_k = self.index_topk / k;
        let select_k = select_k.min(n_pools);
        let (_, selected) = topk_last_dim_values(&index_scores, select_k)?; // [B, S, select_k]

        let selected_valid = valid_candidates
            .contiguous()?
            .gather(&selected, D::Minus1)?; // [B, S, select_k]

        // Gather the raw token indices of each selected pool into
        // [B, S, select_k*K] via flat indexing over the [B, P*K] pool rows.
        let pk = pool_indices.reshape((batch, n_pools * k))?.contiguous()?;
        let k_off = Tensor::arange(0i64, k as i64, dev)?.reshape((1, 1, 1, k))?;
        let flat_idx = selected
            .unsqueeze(D::Minus1)?
            .to_dtype(DType::I64)?
            .broadcast_mul(&Tensor::full(k as i64, (batch, seq, select_k, 1), dev)?)?
            .broadcast_add(&k_off)?
            .reshape((batch, seq * select_k * k))?
            .contiguous()?;
        let gathered = pk.gather(&flat_idx, 1)?; // [B, seq*select_k*K]
        let mut topk_indices = gathered.reshape((batch, seq, select_k * k))?; // [B,S,select_k*K]

        // Mask invalid pool selections to -1.
        let neg1 = Tensor::full(-1i64, (batch, seq, select_k * k), dev)?;
        let valid_b = selected_valid
            .unsqueeze(D::Minus1)?
            .broadcast_as((batch, seq, select_k, k))?
            .reshape((batch, seq, select_k * k))?;
        topk_indices = valid_b.where_cond(&topk_indices, &neg1)?;

        let mut output_width = self.index_topk;
        if self.always_select_tail {
            topk_indices = self.append_visible_tail(&topk_indices, &visible_tokens, &valid_keys)?;
            output_width += k - 1;
        }

        // Pad/trim to the output width and mask padding tokens.
        let cur = topk_indices.dim(D::Minus1)?;
        if cur < output_width {
            let fill = Tensor::full(-1i64, (batch, seq, output_width - cur), dev)?;
            topk_indices = Tensor::cat(&[&topk_indices, &fill], D::Minus1)?;
        } else if cur > output_width {
            topk_indices = topk_indices.narrow(D::Minus1, 0, output_width)?;
        }

        if let Some(m) = attention_mask {
            let valid_am = m.gt(0.0)?.unsqueeze(D::Minus1)?;
            let neg1 = Tensor::full(-1i64, (batch, seq, output_width), dev)?;
            topk_indices = valid_am
                .broadcast_as((batch, seq, output_width))?
                .where_cond(&topk_indices, &neg1)?;
        }
        Ok(topk_indices)
    }

    fn get_visible_tokens(
        &self,
        valid_keys: &Tensor,
        q_length: usize,
        current_length: usize,
    ) -> Result<Tensor> {
        let (_batch, kv_len) = valid_keys.dims2()?;
        let dev = valid_keys.device();
        let kv_positions = Tensor::arange(0i64, kv_len as i64, dev)?
            .reshape((1, 1, kv_len))?
            .broadcast_as((1, q_length, kv_len))?; // [1,q_len,kv_len]
        let q_positions = Tensor::arange(
            (current_length - q_length) as i64,
            current_length as i64,
            dev,
        )?
        .reshape((1, q_length, 1))?
        .broadcast_as((1, q_length, kv_len))?; // [1,q_len,kv_len]
        let causal = kv_positions.le(&q_positions)?; // [1,q_len,kv_len]
        bool_and(&causal, &valid_keys.unsqueeze(1)?)
    }

    fn get_pooled_states(&self, packed_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = packed_states.dims3()?;
        let dev = packed_states.device();
        let hd = self.head_dim;
        let k = self.index_kpool;
        let keys = packed_states.narrow(D::Minus1, 0, hd)?;
        let gate_scores = packed_states.narrow(D::Minus1, hd, hd)?;
        let valid_keys = packed_states
            .narrow(D::Minus1, 2 * hd, 1)?
            .squeeze(D::Minus1)?
            .gt(0.0)?; // [B, seq_len]

        let n_pools = seq_len.div_ceil(k);
        let any_valid = any_gt_zero(&valid_keys, D::Minus1)?; // [B]
        let first_key = any_valid.where_cond(
            &valid_keys.argmax(D::Minus1)?.to_dtype(DType::I64)?,
            &Tensor::full(seq_len as i64, (batch,), dev)?,
        )?; // [B]

        let pool_offsets =
            Tensor::arange(0i64, (n_pools * k) as i64, dev)?.reshape((1, n_pools, k))?;
        let pool_indices = first_key
            .unsqueeze(1)?
            .unsqueeze(D::Minus1)?
            .broadcast_add(&pool_offsets)?; // [B, n_pools, k]

        let safe_indices = pool_indices
            .reshape((batch, n_pools * k))?
            .clamp(0, (seq_len - 1) as i64)?;
        // 3D gathers (keys, gate_scores) need a per-feature index; valid_keys is 2D.
        let safe_idx_3d = safe_indices
            .unsqueeze(D::Minus1)?
            .broadcast_as((batch, n_pools * k, hd))?
            .contiguous()?;
        let grouped_keys = keys
            .contiguous()?
            .gather(&safe_idx_3d, 1)?
            .reshape((batch, n_pools, k, hd))?;
        let grouped_gate = gate_scores
            .contiguous()?
            .gather(&safe_idx_3d, 1)?
            .reshape((batch, n_pools, k, hd))?;
        let grouped_valid = valid_keys
            .contiguous()?
            .gather(&safe_indices, 1)?
            .reshape((batch, n_pools, k))?;
        let in_range = pool_indices.lt(seq_len as i64)?;
        let grouped_valid = bool_and(&grouped_valid, &in_range)?; // [B,n_pools,k]
        let pool_valid = all_eq_len(&grouped_valid, D::Minus1, k)?; // [B,n_pools]

        // Learned weighted average over the tokens inside each complete pool.
        let logits = grouped_gate
            .to_dtype(DType::F32)?
            .broadcast_add(&self.ape.unsqueeze(0)?.unsqueeze(0)?)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (batch, n_pools, k, hd), dev)?;
        let logits = grouped_valid
            .unsqueeze(D::Minus1)?
            .broadcast_as((batch, n_pools, k, hd))?
            .where_cond(&logits, &neg_inf)?;
        let probabilities = candle_nn::ops::softmax(&logits, 2)?; // over pool members
        let pool_keys =
            (probabilities.to_dtype(DType::F32)? * grouped_keys.to_dtype(DType::F32)?)?.sum(2)?; // [B, n_pools, hd]

        // Keep pools valid for any batch row.
        let keep = any_gt_zero(&pool_valid, 0usize)?; // [n_pools]
        let keep_vec: Vec<u8> = keep.to_dtype(DType::U8)?.to_vec1()?;
        let keep_idx: Vec<u32> = keep_vec
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, _)| i as u32)
            .collect();
        let keep_t = Tensor::new(keep_idx.as_slice(), dev)?;
        let pool_keys = pool_keys.index_select(&keep_t, 1)?;
        let pool_indices = pool_indices.index_select(&keep_t, 1)?;
        let pool_valid = pool_valid.index_select(&keep_t, 1)?;
        Ok((pool_keys, pool_indices, pool_valid))
    }

    fn append_visible_tail(
        &self,
        topk_indices: &Tensor,
        token_visible: &Tensor,
        key_valid: &Tensor,
    ) -> Result<Tensor> {
        let max_tail_width = self.index_kpool - 1;
        if max_tail_width == 0 {
            return Ok(topk_indices.clone());
        }
        let (batch, q_len, _) = token_visible.dims3()?;
        let (_, kv_len) = key_valid.dims2()?;
        let dev = topk_indices.device();
        let k = self.index_kpool;

        let any_valid = any_gt_zero(key_valid, D::Minus1)?;
        let first_key = any_valid.where_cond(
            &key_valid.argmax(D::Minus1)?.to_dtype(DType::I64)?,
            &Tensor::full(kv_len as i64, (batch,), dev)?,
        )?; // [B]
        let visible_count = token_visible.to_dtype(DType::I64)?.sum(D::Minus1)?; // [B, q_len]
        let kt = Tensor::new(k as i64, dev)?;
        let tail_count =
            visible_count.broadcast_sub(&visible_count.broadcast_div(&kt)?.broadcast_mul(&kt)?)?; // [B,q_len]
        let tail_offsets = Tensor::arange(0i64, max_tail_width as i64, dev)?; // [max_tail_width]

        let tail_start = first_key
            .unsqueeze(1)?
            .broadcast_add(&visible_count)?
            .sub(&tail_count)?; // [B,q_len]
        let tail_indices = tail_start
            .unsqueeze(D::Minus1)?
            .broadcast_add(&tail_offsets.unsqueeze(0)?.unsqueeze(0)?)?; // [B,q_len,mtw]

        let off_lt = tail_offsets
            .reshape((1, 1, max_tail_width))?
            .broadcast_as((batch, q_len, max_tail_width))?
            .lt(&tail_count.unsqueeze(D::Minus1)?.broadcast_as((
                batch,
                q_len,
                max_tail_width,
            ))?)?; // [B,q_len,mtw]
        let len_ok = tail_indices.lt(kv_len as i64)?;
        let tail_valid = bool_and(&off_lt, &len_ok)?;

        let kv_idx = tail_indices.clamp(0, (kv_len - 1) as i64)?;
        let tail_visible = token_visible.contiguous()?.gather(&kv_idx, D::Minus1)?; // [B,q_len,mtw]
        let ok = bool_and(&tail_valid, &tail_visible)?;
        let neg1 = Tensor::full(-1i64, (batch, q_len, max_tail_width), dev)?;
        let tail_indices = ok.where_cond(&tail_indices, &neg1)?;
        Tensor::cat(&[topk_indices, &tail_indices], D::Minus1)
    }
}

/// GLM-5.3-Flash DeepSeek-Sparse-Attention (MLA + indexer) block, mirroring
/// `transformers` `Glm5NextTextAttention`.
///
/// This is a clean DeepSeek-V3-style MLA (no sliding window / compressor,
/// unlike DeepSeek-V4's attention) with a DSA indexer for sparse top-k
/// selection and optional cross-layer top-k sharing (`indexer_types`).
pub struct Glm5NextTextAttention {
    q_a_proj: Linear,
    q_a_layernorm: RmsNorm,
    q_b_proj: Linear,
    kv_a_proj: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    scaling: f64,
    indexer: Option<Glm5NextTextIndexer>,
    next_skip_topk: bool,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Glm5NextTextAttention {
    pub fn new(cfg: &Glm5NextTextConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let num_heads_v = num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        let attention_bias = cfg.attention_bias;
        Ok(Self {
            q_a_proj: attn_linear(
                cfg.hidden_size,
                cfg.q_lora_rank,
                attention_bias,
                vb.pp("q_a_proj"),
            )?,
            q_a_layernorm: rms_norm(cfg.q_lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?,
            q_b_proj: linear_no_bias(cfg.q_lora_rank, num_heads * qk_head_dim, vb.pp("q_b_proj"))?,
            kv_a_proj: attn_linear(
                cfg.hidden_size,
                cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                attention_bias,
                vb.pp("kv_a_proj_with_mqa"),
            )?,
            kv_a_layernorm: rms_norm(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?,
            kv_b_proj: linear_no_bias(cfg.kv_lora_rank, num_heads_v, vb.pp("kv_b_proj"))?,
            o_proj: attn_linear(
                num_heads * cfg.v_head_dim,
                cfg.hidden_size,
                attention_bias,
                vb.pp("o_proj"),
            )?,
            num_heads,
            qk_head_dim,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            scaling: (qk_head_dim as f64).powf(-0.5),
            indexer: build_indexer(cfg, layer_idx, vb)?,
            next_skip_topk: next_is_shared(cfg, layer_idx),
            kv_cache: None,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        if let Some(i) = &mut self.indexer {
            i.clear_cache();
        }
    }

    /// Runs the MLA attention with the sparse indexer.
    ///
    /// * `xs`: post-layernorm input `[B, S, hidden]`.
    /// * `attention_mask`: optional `[B, S]` 0/1 float mask.
    /// * `prev_topk`: top-k indices from the previous full indexer layer (for
    ///   shared layers).
    ///
    /// Returns `(output [B, S, hidden], topk_indices Option<[B,S,W]>)`.
    pub fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        prev_topk: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (bs, seq, _) = xs.dims3()?;
        let dtype = xs.dtype();

        // Query (LoRA path): q_a -> RMSNorm -> q_b -> view/transpose.
        let q_resid = self.q_a_layernorm.forward(&self.q_a_proj.forward(xs)?)?;
        let query_states = self
            .q_b_proj
            .forward(&q_resid)?
            .reshape((bs, seq, self.num_heads, self.qk_head_dim))?
            .transpose(1, 2)?; // [B, H, S, qk]

        // Compressed KV: kv_a -> split latent + rope slice -> kv_b expand.
        let compressed = self.kv_a_proj.forward(xs)?;
        let kv_pass = compressed.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_rot = compressed.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;
        let k_pass =
            self.kv_a_layernorm
                .forward(&kv_pass)?
                .reshape((bs, 1, seq, self.kv_lora_rank))?;
        let k_rot = k_rot.reshape((bs, 1, seq, self.qk_rope_head_dim))?;

        let kv_nope = self
            .kv_b_proj
            .forward(&k_pass)?
            .reshape((
                bs,
                seq,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ))?
            .transpose(1, 2)?; // [B, H, S, qk_nope+v_head]
        let k_nope = kv_nope.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let value_states = kv_nope.narrow(D::Minus1, self.qk_nope_head_dim, self.v_head_dim)?;
        let k_rot = k_rot.broadcast_as((bs, self.num_heads, seq, self.qk_rope_head_dim))?;
        let key_states = Tensor::cat(&[&k_nope, &k_rot], D::Minus1)?; // [B, H, S, qk]

        // KV cache: accumulate for cached decode.
        let (key_states, value_states) = if let Some((ck, cv)) = &self.kv_cache {
            let k = Tensor::cat(&[ck, &key_states], 2)?;
            let v = Tensor::cat(&[cv, &value_states], 2)?;
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        } else {
            self.kv_cache = Some((key_states.clone(), value_states.clone()));
            (key_states, value_states)
        };
        let kv_len = key_states.dim(2)?;

        let topk_indices = match &mut self.indexer {
            Some(idx) => idx.forward(xs, &q_resid, attention_mask)?,
            None => prev_topk
                .ok_or_else(|| candle::Error::msg("shared DSA layer requires prev_topk_indices"))?
                .clone(),
        };

        let mask = self.build_attention_mask_from_topk(&topk_indices, kv_len)?;

        // Eager attention over the sparse top-k mask.
        let qf = query_states.to_dtype(DType::F32)?;
        let kf = key_states.to_dtype(DType::F32)?;
        let vf = value_states.to_dtype(DType::F32)?;
        let att = (qf.matmul(&kf.t()?)? * self.scaling)?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let attn_out = att.matmul(&vf)?.to_dtype(dtype)?; // [B, H, S, v_head]

        let out = attn_out.transpose(1, 2)?.contiguous()?.reshape((
            bs,
            seq,
            self.num_heads * self.v_head_dim,
        ))?;
        let out = self.o_proj.forward(&out)?;

        let ret_topk = if self.next_skip_topk {
            Some(topk_indices)
        } else {
            None
        };
        Ok((out, ret_topk))
    }

    /// Convert the indexer's top-k token indices into an additive attention
    /// mask `[B, 1, S, kv_len]` (0 attend / `-inf` otherwise).
    fn build_attention_mask_from_topk(
        &self,
        topk_indices: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor> {
        let (bs, seq, _) = topk_indices.dims3()?;
        let dev = topk_indices.device();
        let topk_valid = bool_and(&topk_indices.ge(0i64)?, &topk_indices.lt(kv_len as i64)?)?;
        let safe_indices = topk_indices.clamp(0, (kv_len - 1) as i64)?;
        let src = topk_valid.to_dtype(DType::I64)?;
        let counts = Tensor::zeros((bs, seq, kv_len), DType::I64, dev)?.scatter_add(
            &safe_indices,
            &src,
            D::Minus1,
        )?;
        let mask_bool = counts.ne(0i64)?.unsqueeze(1)?; // [B,1,S,kv_len]
        let zeros = Tensor::zeros((bs, 1, seq, kv_len), DType::F32, dev)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (bs, 1, seq, kv_len), dev)?;
        mask_bool.where_cond(&zeros, &neg_inf)
    }
}

/// Helper: linear with optional bias (matches `attention_bias`).
fn attn_linear(in_: usize, out: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_, out, vb)
    } else {
        linear_no_bias(in_, out, vb)
    }
}

/// Whether this layer runs the indexer (returns `Some`) or shares the previous
/// full layer's top-k (returns `None`).
fn build_indexer(
    cfg: &Glm5NextTextConfig,
    layer_idx: usize,
    vb: VarBuilder,
) -> Result<Option<Glm5NextTextIndexer>> {
    let types = cfg.effective_indexer_types();
    let this = *types.get(layer_idx).unwrap_or(&IndexerType::Full);
    if this == IndexerType::Shared {
        Ok(None)
    } else {
        Ok(Some(Glm5NextTextIndexer::new(cfg, vb.pp("indexer"))?))
    }
}

/// Whether this layer's top-k should be returned for the next (shared) layer.
fn next_is_shared(cfg: &Glm5NextTextConfig, layer_idx: usize) -> bool {
    let types = cfg.effective_indexer_types();
    let next = *types
        .get(layer_idx + 1)
        .unwrap_or_else(|| types.last().unwrap_or(&IndexerType::Full));
    next == IndexerType::Shared
}

/// Attention submodule of a decoder layer (KDA linear or DSA sparse).
pub enum Glm5NextSelfAttn {
    Linear(KdaLinearAttention),
    Sparse(Glm5NextTextAttention),
}

/// MLP submodule of a decoder layer (dense or sparse MoE).
pub enum Glm5NextMlp {
    Dense(Glm5NextMLP),
    MoE(Glm5NextTextMoE),
}

/// One GLM-5.3-Flash text decoder layer, mirroring
/// `transformers` `Glm5NextTextDecoderLayer`.
pub struct Glm5NextDecoderLayer {
    self_attn: Glm5NextSelfAttn,
    mlp: Glm5NextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn_hc: DeepseekV4HyperConnection,
    ffn_hc: DeepseekV4HyperConnection,
}

impl Glm5NextDecoderLayer {
    pub fn new(cfg: &Glm5NextTextConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let attn_types = cfg.effective_layer_types();
        let block_type = *attn_types
            .get(layer_idx)
            .unwrap_or(&LayerType::LinearAttention);
        let self_attn = if block_type == LayerType::DeepseekSparseAttention {
            Glm5NextSelfAttn::Sparse(Glm5NextTextAttention::new(
                cfg,
                layer_idx,
                vb.pp("self_attn"),
            )?)
        } else {
            Glm5NextSelfAttn::Linear(KdaLinearAttention::new(cfg, layer_idx, vb.pp("self_attn"))?)
        };

        let mlp_types = cfg.effective_mlp_layer_types();
        let mlp_type = *mlp_types.get(layer_idx).unwrap_or(&MlpLayerType::Sparse);
        let mlp = if mlp_type == MlpLayerType::Dense {
            Glm5NextMlp::Dense(Glm5NextMLP::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.swiglu_limit,
                vb.pp("mlp"),
            )?)
        } else {
            Glm5NextMlp::MoE(Glm5NextTextMoE::new(cfg, vb.pp("mlp"))?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
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
        attention_mask: Option<&Tensor>,
        prev_topk: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let dtype = hidden_states.dtype();

        let (post, comb, collapsed) = self.attn_hc.forward(hidden_states)?;
        let normed = self.input_layernorm.forward(&collapsed)?;
        let (attn_out, topk) = match &mut self.self_attn {
            Glm5NextSelfAttn::Linear(a) => (a.forward(&normed, attention_mask)?, None),
            Glm5NextSelfAttn::Sparse(a) => {
                let (o, t) = a.forward(&normed, attention_mask, prev_topk)?;
                (o, t)
            }
        };
        let hidden = post
            .to_dtype(dtype)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&attn_out.unsqueeze(D::Minus2)?)?
            .broadcast_add(
                &comb
                    .to_dtype(dtype)?
                    .transpose(D::Minus1, D::Minus2)?
                    .matmul(hidden_states)?,
            )?;

        let (post, comb, collapsed) = self.ffn_hc.forward(&hidden)?;
        let normed = self.post_attention_layernorm.forward(&collapsed)?;
        let mlp_out = match &self.mlp {
            Glm5NextMlp::Dense(m) => m.forward(&normed)?,
            Glm5NextMlp::MoE(m) => m.forward(&normed)?,
        };
        let hidden = post
            .to_dtype(dtype)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&mlp_out.unsqueeze(D::Minus2)?)?
            .broadcast_add(
                &comb
                    .to_dtype(dtype)?
                    .transpose(D::Minus1, D::Minus2)?
                    .matmul(&hidden)?,
            )?;

        Ok((hidden, topk))
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.self_attn {
            Glm5NextSelfAttn::Linear(a) => a.clear_cache(),
            Glm5NextSelfAttn::Sparse(a) => a.clear_kv_cache(),
        }
    }

    /// Whether this layer uses the DSA (sparse MLA + indexer) attention.
    pub fn is_sparse_attention(&self) -> bool {
        matches!(self.self_attn, Glm5NextSelfAttn::Sparse(_))
    }

    /// Whether this layer uses the sparse (routed MoE) MLP.
    pub fn is_sparse_mlp(&self) -> bool {
        matches!(self.mlp, Glm5NextMlp::MoE(_))
    }
}

/// The GLM-5.3-Flash text stack: token embedding, scheduled decoder layers,
/// final unweighted mean stream collapse, and RMSNorm.
pub struct Glm5NextTextModel {
    hidden_size: usize,
    hc_mult: usize,
    embed_tokens: Embedding,
    layers: Vec<Glm5NextDecoderLayer>,
    norm: RmsNorm,
}

impl Glm5NextTextModel {
    pub fn new(cfg: &Glm5NextTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Glm5NextDecoderLayer::new(cfg, i, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            hidden_size: cfg.hidden_size,
            hc_mult: cfg.hc_mult,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Runs the full stack. `input_ids` is `[B, S]`; returns final hidden
    /// states `[B, S, hidden]`. `_seqlen_offset` is accepted for API
    /// consistency with nearby Candle causal models; the KDA / DSA caches are
    /// stateful and drive decode internally.
    pub fn forward(&mut self, input_ids: &Tensor, _seqlen_offset: usize) -> Result<Tensor> {
        let (bs, seq) = input_ids.dims2()?;
        let dev = input_ids.device();
        let emb = self.embed_tokens.forward(input_ids)?; // [B, S, D]
        let hidden = emb
            .unsqueeze(2)?
            .broadcast_as((bs, seq, self.hc_mult, self.hidden_size))?
            .contiguous()?; // [B, S, hc_mult, D]
        let mask = Tensor::ones((bs, seq), DType::F32, dev)?;
        let mut topk: Option<Tensor> = None;
        let mut hidden = hidden;
        for layer in &mut self.layers {
            (hidden, topk) = layer.forward(&hidden, Some(&mask), topk.as_ref())?;
        }
        // Glm5NextTextHyperHead: unweighted mean over the hc_mult streams.
        let collapsed = hidden.mean(D::Minus2)?; // [B, S, D]
        self.norm.forward(&collapsed)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

/// `Glm5NextForCausalLM`: the text model plus a separate (untied) `lm_head`.
pub struct Glm5NextForCausalLM {
    model: Glm5NextTextModel,
    lm_head: Linear,
}

impl Glm5NextForCausalLM {
    pub fn new(cfg: &Glm5NextTextConfig, vb: VarBuilder) -> Result<Self> {
        let model = Glm5NextTextModel::new(cfg, vb.pp("model"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};
    use std::collections::HashMap;

    /// A tiny, official-shaped `glm5_next_text` config (same reduced config as
    /// the config story's tests) so the full model fits a unit test.
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

    /// Deterministic pseudo-random weights in `[-1, 1)`.
    fn rng_seq() -> impl FnMut() -> f32 {
        let mut s = 12345u64;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) as f32 - 1.0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn put(
        m: &mut HashMap<String, Tensor>,
        name: &str,
        shape: &[usize],
        dev: &Device,
        r: &mut impl FnMut() -> f32,
    ) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| r()).collect();
        m.insert(
            name.to_string(),
            Tensor::from_vec(data, shape, dev).unwrap(),
        );
    }

    /// Build the full official-named weight map for the reduced config.
    fn build_weights(cfg: &Glm5NextTextConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut r = rng_seq();
        let mut m = HashMap::new();
        let h = cfg.hidden_size;
        let hc = cfg.hc_mult;
        let mix = (2 + hc) * hc;
        let nh = cfg.linear_num_heads();
        let hd = cfg.linear_head_dim();
        let qkv = nh * hd;
        let conv_dim = qkv * 3;
        let kernel = cfg.linear_conv_kernel_dim();
        let qk_head = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let nheads = cfg.num_attention_heads;
        let ihd = cfg.index_head_dim;
        let inheads = cfg.index_n_heads;
        let kpool = cfg.index_kpool;
        let inter = cfg.intermediate_size;
        let moe = cfg.moe_intermediate_size;
        let n_routed = cfg.n_routed_experts;
        let shared_inter = moe * cfg.n_shared_experts;

        put(
            &mut m,
            "model.embed_tokens.weight",
            &[cfg.vocab_size, h],
            dev,
            &mut r,
        );
        put(&mut m, "model.norm.weight", &[h], dev, &mut r);
        put(&mut m, "lm_head.weight", &[cfg.vocab_size, h], dev, &mut r);

        let attn_types = cfg.effective_layer_types();
        let mlp_types = cfg.effective_mlp_layer_types();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            put(
                &mut m,
                &format!("{p}.input_layernorm.weight"),
                &[h],
                dev,
                &mut r,
            );
            put(
                &mut m,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
                dev,
                &mut r,
            );
            put(
                &mut m,
                &format!("{p}.attn_hc.fn"),
                &[mix, hc * h],
                dev,
                &mut r,
            );
            put(&mut m, &format!("{p}.attn_hc.base"), &[mix], dev, &mut r);
            put(&mut m, &format!("{p}.attn_hc.scale"), &[3], dev, &mut r);
            put(
                &mut m,
                &format!("{p}.ffn_hc.fn"),
                &[mix, hc * h],
                dev,
                &mut r,
            );
            put(&mut m, &format!("{p}.ffn_hc.base"), &[mix], dev, &mut r);
            put(&mut m, &format!("{p}.ffn_hc.scale"), &[3], dev, &mut r);
            let sa = format!("{p}.self_attn");
            match attn_types[i] {
                LayerType::LinearAttention => {
                    put(
                        &mut m,
                        &format!("{sa}.q_proj.weight"),
                        &[qkv, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.k_proj.weight"),
                        &[qkv, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.v_proj.weight"),
                        &[qkv, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.conv1d.weight"),
                        &[conv_dim, 1, kernel],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.forget_gate.f_a_proj.weight"),
                        &[hd, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.forget_gate.f_b_proj.weight"),
                        &[qkv, hd],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.forget_gate.dt_bias"),
                        &[qkv],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.forget_gate.A_log"),
                        &[nh],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.b_proj.weight"),
                        &[nh, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.g_a_proj.weight"),
                        &[hd, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.g_b_proj.weight"),
                        &[qkv, hd],
                        dev,
                        &mut r,
                    );
                    put(&mut m, &format!("{sa}.o_norm.weight"), &[hd], dev, &mut r);
                    put(
                        &mut m,
                        &format!("{sa}.o_proj.weight"),
                        &[h, qkv],
                        dev,
                        &mut r,
                    );
                }
                LayerType::DeepseekSparseAttention => {
                    put(
                        &mut m,
                        &format!("{sa}.q_a_proj.weight"),
                        &[cfg.q_lora_rank, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.q_a_layernorm.weight"),
                        &[cfg.q_lora_rank],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.q_b_proj.weight"),
                        &[nheads * qk_head, cfg.q_lora_rank],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.kv_a_proj_with_mqa.weight"),
                        &[cfg.kv_lora_rank + cfg.qk_rope_head_dim, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.kv_a_layernorm.weight"),
                        &[cfg.kv_lora_rank],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.kv_b_proj.weight"),
                        &[
                            nheads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                            cfg.kv_lora_rank,
                        ],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{sa}.o_proj.weight"),
                        &[h, nheads * cfg.v_head_dim],
                        dev,
                        &mut r,
                    );
                    let ix = format!("{sa}.indexer");
                    put(
                        &mut m,
                        &format!("{ix}.wq_b.weight"),
                        &[inheads * ihd, cfg.q_lora_rank],
                        dev,
                        &mut r,
                    );
                    put(&mut m, &format!("{ix}.wk.weight"), &[ihd, h], dev, &mut r);
                    put(&mut m, &format!("{ix}.k_norm.weight"), &[ihd], dev, &mut r);
                    put(&mut m, &format!("{ix}.k_norm.bias"), &[ihd], dev, &mut r);
                    put(
                        &mut m,
                        &format!("{ix}.weights_proj.weight"),
                        &[inheads, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{ix}.index_kpool_compress_ape"),
                        &[kpool, ihd],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{ix}.index_kpool_compress_gate"),
                        &[ihd, h],
                        dev,
                        &mut r,
                    );
                }
            }
            let mlp = format!("{p}.mlp");
            match mlp_types[i] {
                MlpLayerType::Dense => {
                    put(
                        &mut m,
                        &format!("{mlp}.gate_proj.weight"),
                        &[inter, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.up_proj.weight"),
                        &[inter, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.down_proj.weight"),
                        &[h, inter],
                        dev,
                        &mut r,
                    );
                }
                MlpLayerType::Sparse => {
                    put(
                        &mut m,
                        &format!("{mlp}.gate.weight"),
                        &[n_routed, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.experts.gate_up_proj"),
                        &[n_routed, 2 * moe, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.experts.down_proj"),
                        &[n_routed, h, moe],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.shared_experts.gate_proj.weight"),
                        &[shared_inter, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.shared_experts.up_proj.weight"),
                        &[shared_inter, h],
                        dev,
                        &mut r,
                    );
                    put(
                        &mut m,
                        &format!("{mlp}.shared_experts.down_proj.weight"),
                        &[h, shared_inter],
                        dev,
                        &mut r,
                    );
                }
            }
        }
        m
    }

    fn build_model(cfg: &Glm5NextTextConfig, dev: &Device) -> Glm5NextForCausalLM {
        let weights = build_weights(cfg, dev);
        let vb = VarBuilder::from_tensors(weights, DType::F32, dev);
        Glm5NextForCausalLM::new(cfg, vb).unwrap()
    }

    #[test]
    fn constructs_from_official_weight_names() {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        let _model = build_model(&cfg, &dev);
    }

    #[test]
    fn layer_dispatch_follows_schedule() {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        let model = build_model(&cfg, &dev);
        // Default schedules: DSA on idx%4==3 (3,7); first 2 MLPs dense, rest MoE.
        let schedule: Vec<(bool, bool)> = (0..cfg.num_hidden_layers)
            .map(|i| {
                (
                    model.model.layers[i].is_sparse_attention(),
                    model.model.layers[i].is_sparse_mlp(),
                )
            })
            .collect();
        let expected = vec![
            (false, false),
            (false, false),
            (false, true),
            (true, true),
            (false, true),
            (false, true),
            (false, true),
            (true, true),
        ];
        assert_eq!(schedule, expected);
    }

    #[test]
    fn logits_shape() {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        let mut model = build_model(&cfg, &dev);
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let logits = model.forward(&ids, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn prefill_decode_matches_full_forward() {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        // One-shot forward over all 5 tokens.
        let mut full = build_model(&cfg, &dev);
        let ids5 = Tensor::new(&[[1u32, 2, 3, 4, 5]], &dev).unwrap();
        let full_logits = full.forward(&ids5, 0).unwrap(); // [1,5,vocab]
        let full_last = full_logits
            .narrow(1, 4, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Prefill 4 tokens, then cached single-token decode.
        let mut step = build_model(&cfg, &dev);
        let ids4 = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let _ = step.forward(&ids4, 0).unwrap();
        let id5 = Tensor::new(&[[5u32]], &dev).unwrap();
        let dec = step.forward(&id5, 4).unwrap().flatten_all().unwrap();
        let dec_v = dec.to_vec1::<f32>().unwrap();

        assert_eq!(full_last.len(), dec_v.len());
        for (a, b) in full_last.iter().zip(dec_v.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "decode logit mismatch: full={a} decode={b}"
            );
        }
    }

    #[test]
    fn cache_reset_restores_fresh_prefill() {
        let cfg = reduced_config();
        let dev = Device::Cpu;
        let ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &dev).unwrap();

        let mut model = build_model(&cfg, &dev);
        let first = model.forward(&ids, 0).unwrap();
        // Accumulate cache across a second call, then reset.
        let _ = model.forward(&ids, 0).unwrap();
        model.clear_kv_cache();
        let after_reset = model.forward(&ids, 0).unwrap();

        let a = first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = after_reset.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b);
    }
}
