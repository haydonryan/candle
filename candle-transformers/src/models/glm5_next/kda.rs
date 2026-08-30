//! GLM-5.3-Flash KDA linear attention (story #4354).
//!
//! The KDA (Kimi-style Delta Attention) block is GLM-5.3-Flash's recurrent
//! linear-attention primitive. It mirrors `transformers` `Glm5NextTextLinearAttention`
//! (which itself wraps the `fla` package): a single depthwise short-conv over the
//! concatenated Q/K/V, per-dimension forget-gate decay (`g`) plus an input gate
//! (`beta = sigmoid(b_proj(x))`), a gated delta-rule recurrent update, a final
//! gated RMSNorm, and an output projection.
//!
//! The recurrence is the `fla` `recurrent_kimi_delta_attention` (delta-rule):
//!
//! ```text
//! S_t = S_{t-1} * exp(g_t) + (beta_t * k_t) ⊗ (v_t - k_tᵀ S_{t-1})
//! o_t = q_tᵀ S_t
//! ```
//!
//! with `q`, `k` L2-normalised over the head dim and `q` scaled by
//! `1/sqrt(head_dim)`.
//!
//! Non-goals (per the story): custom CUDA/Metal kernels, chunked parallel
//! training, quantization, performance tuning. Inference uses ordinary Candle
//! tensor ops; prefill and cached single-token decode share the same recurrent
//! state and short-conv state.

use super::Glm5NextTextConfig;
use candle::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, Activation, Linear, Module, VarBuilder};

/// The subset of GLM-5.3-Flash config that drives the KDA block. Kept as a
/// small trait (matching the `HyperConnectionConfig` / `MoeExpertsConfig`
/// pattern) so a reduced, deterministic config can drive the unit tests.
pub trait KdaConfig {
    fn hidden_size(&self) -> usize;
    fn linear_num_heads(&self) -> usize;
    fn linear_head_dim(&self) -> usize;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn linear_lower_bound(&self) -> Option<f64>;
    fn rms_norm_eps(&self) -> f64;
    fn hidden_act(&self) -> Activation;
}

impl KdaConfig for Glm5NextTextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn linear_num_heads(&self) -> usize {
        self.linear_num_heads()
    }
    fn linear_head_dim(&self) -> usize {
        self.linear_head_dim()
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim()
    }
    fn linear_lower_bound(&self) -> Option<f64> {
        self.linear_lower_bound()
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn hidden_act(&self) -> Activation {
        self.hidden_act
    }
}

/// `l2norm` over the last dim with `+eps` inside the sqrt, matching the FLA
/// kernel (not `F.normalize`, which uses `max(eps, ...)`).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = (x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&inv_norm)
}

/// Strict FP32 RMSNorm followed by a sigmoid gate, matching `transformers`
/// `Glm5NextTextRMSNormGated` (no downcast of the weight).
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let variance = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let gate_f = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?;
        Ok(normed.broadcast_mul(&gate_f)?.to_dtype(dtype)?)
    }
}

/// KDA forget gate: a low-rank projection producing per-dimension log-space
/// decays `g` of shape `[B, T, num_heads, head_dim]`.
///
/// With a safe lower bound (GLM-5.3 ships `-5.0`):
/// `g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`.
struct KdaForgetGate {
    f_a_proj: Linear,
    f_b_proj: Linear,
    dt_bias: Tensor,
    a_log: Tensor,
    num_heads: usize,
    head_dim: usize,
    lower_bound: Option<f64>,
}

impl KdaForgetGate {
    fn new(cfg: &impl KdaConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.linear_head_dim();
        let num_heads = cfg.linear_num_heads();
        let qkv_dim = head_dim * num_heads;
        let f_a_proj = linear_no_bias(cfg.hidden_size(), head_dim, vb.pp("f_a_proj"))?;
        let f_b_proj = linear_no_bias(head_dim, qkv_dim, vb.pp("f_b_proj"))?;
        let dt_bias = vb.get((qkv_dim,), "dt_bias")?;
        let a_log = vb.get((num_heads,), "A_log")?;
        Ok(Self {
            f_a_proj,
            f_b_proj,
            dt_bias,
            a_log,
            num_heads,
            head_dim,
            lower_bound: cfg.linear_lower_bound(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv_dim = self.num_heads * self.head_dim;
        let forget_gate = self.f_b_proj.forward(&self.f_a_proj.forward(x)?)?;
        let g = forget_gate
            .to_dtype(DType::F32)?
            .broadcast_add(&self.dt_bias.reshape((1, 1, qkv_dim))?)?
            .reshape((b, t, self.num_heads, self.head_dim))?;
        let decay_rate =
            self.a_log
                .to_dtype(DType::F32)?
                .exp()?
                .reshape((1, 1, self.num_heads, 1))?;
        let g = decay_rate.broadcast_mul(&g)?;
        match self.lower_bound {
            Some(lb) => {
                Ok(candle_nn::ops::sigmoid(&g)?
                    .broadcast_mul(&Tensor::new(lb as f32, x.device())?)?)
            }
            None => {
                // softplus with a guard against overflow for large g.
                let sp = (g.exp()? + 1.0)?.log()?;
                let gt20 = g.broadcast_gt(&Tensor::new(20.0f32, x.device())?)?;
                Ok(gt20.where_cond(&g, &sp)?)
            }
        }
    }
}

/// Recurrent delta-rule KDA kernel over `[B, T, H, D]` inputs.
///
/// All math runs in FP32 (states are more susceptible to rounding error).
/// Returns `(output [B, T, H, D], final_state [B, H, D, D])`.
fn recurrent_kda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (b, t, h, d) = q.dims4()?;
    let dev = q.device();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;

    let mut state = match initial_state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((b, h, d, d), DType::F32, dev)?,
    };
    let mut outs: Vec<Tensor> = Vec::with_capacity(t);
    for i in 0..t {
        let q_i = q.narrow(1, i, 1)?.squeeze(1)?; // [B, H, D]
        let k_i = k.narrow(1, i, 1)?.squeeze(1)?;
        let v_i = v.narrow(1, i, 1)?.squeeze(1)?;
        let g_i = g.narrow(1, i, 1)?.squeeze(1)?.exp()?; // [B, H, D]
        let b_i = beta.narrow(1, i, 1)?.squeeze(1)?; // [B, H]

        state = state.broadcast_mul(&g_i.unsqueeze(D::Minus1)?)?;
        let kv_mem = state
            .broadcast_mul(&k_i.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?; // [B, H, D]
        let delta = v_i
            .sub(&kv_mem)?
            .broadcast_mul(&b_i.unsqueeze(D::Minus1)?)?; // [B, H, D]
        state = state.broadcast_add(
            &k_i.unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?,
        )?;
        let o_i = state
            .broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?; // [B, H, D]
        outs.push(o_i.unsqueeze(1)?);
    }
    let out = Tensor::cat(&outs, 1)?;
    Ok((out, state))
}

/// Depthwise causal short-conv. `x` is `[B, conv_dim, L]`; returns the causal
/// outputs for the last `t` positions (padding applied on the left, matching
/// `F.conv1d` with `padding=kernel-1` then taking the last `t`).
fn causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    conv_kernel: usize,
    conv_dim: usize,
    t: usize,
    take_last: bool,
) -> Result<Tensor> {
    let out = x.conv1d_with_algo(weight, conv_kernel - 1, 1, 1, conv_dim, None)?;
    let len = out.dim(2)?;
    if take_last {
        out.narrow(2, len - t, t)
    } else {
        out.narrow(2, 0, t)
    }
}

/// Single-token causal conv update, mirroring `causal_conv1d_update`: append
/// `x_t` to the cached `conv_state` (the previous `kernel-1` inputs), run a
/// zero-padded valid conv over the `kernel` window, and keep the new
/// `kernel-1` inputs as state.
fn causal_conv1d_step(
    x_t: &Tensor,
    conv_state: &Tensor,
    weight: &Tensor,
    conv_kernel: usize,
    conv_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let full = Tensor::cat(&[conv_state, x_t], 2)?; // [B, conv_dim, kernel]
    let out = full.conv1d_with_algo(weight, 0, 1, 1, conv_dim, None)?;
    let new_state = full.narrow(2, 1, conv_kernel - 1)?;
    Ok((out, new_state))
}

/// GLM-5.3-Flash KDA linear-attention block.
///
/// Weights are named to match `transformers` `Glm5NextTextLinearAttention`
/// (`q_proj`, `k_proj`, `v_proj`, `conv1d`, `forget_gate.{f_a_proj,f_b_proj,
/// dt_bias,A_log}`, `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm`, `o_proj`),
/// so a decoder layer instantiates it with `vb.pp("self_attn")`.
pub struct KdaLinearAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    conv_weight: Tensor,
    forget_gate: KdaForgetGate,
    b_proj: Linear,
    g_a_proj: Linear,
    g_b_proj: Linear,
    o_norm: RmsNormGated,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    qkv_dim: usize,
    conv_kernel: usize,
    conv_dim: usize,
    activation: Activation,
    // Cache: the short-conv state holds the last `kernel-1` inputs as
    // `[B, conv_dim, kernel-1]`; the recurrent state is `[B, H, D, D]` in FP32.
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    layer_idx: usize,
}

impl KdaLinearAttention {
    pub fn new<C: KdaConfig>(cfg: &C, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let num_heads = cfg.linear_num_heads();
        let head_dim = cfg.linear_head_dim();
        let qkv_dim = head_dim * num_heads;
        let conv_kernel = cfg.linear_conv_kernel_dim();
        let conv_dim = qkv_dim * 3;

        let q_proj = linear_no_bias(hidden_size, qkv_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, qkv_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, qkv_dim, vb.pp("v_proj"))?;
        let conv_weight = vb.get((conv_dim, 1, conv_kernel), "conv1d.weight")?;
        let forget_gate = KdaForgetGate::new(cfg, vb.pp("forget_gate"))?;
        let b_proj = linear_no_bias(hidden_size, num_heads, vb.pp("b_proj"))?;
        let g_a_proj = linear_no_bias(hidden_size, head_dim, vb.pp("g_a_proj"))?;
        let g_b_proj = linear_no_bias(head_dim, qkv_dim, vb.pp("g_b_proj"))?;
        let o_norm = RmsNormGated::new(vb.get((head_dim,), "o_norm.weight")?, cfg.rms_norm_eps());
        let o_proj = linear_no_bias(qkv_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            conv_weight,
            forget_gate,
            b_proj,
            g_a_proj,
            g_b_proj,
            o_norm,
            o_proj,
            num_heads,
            head_dim,
            qkv_dim,
            conv_kernel,
            conv_dim,
            activation: cfg.hidden_act(),
            conv_state: None,
            recurrent_state: None,
            layer_idx,
        })
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    /// Run the KDA block.
    ///
    /// * Fresh prefill (`seq_len > 1`): causal short-conv over the full
    ///   sequence, delta-rule recurrence from a zero state, and stores the
    ///   final conv + recurrent state.
    /// * Cached single-token decode (`seq_len == 1` after a prefill): a
    ///   one-step causal-conv update reusing the conv state and a single
    ///   recurrent step reusing the recurrent state.
    ///
    /// `attention_mask` (optional) is a `[B, T]` tensor of `0`/`1` that zeros
    /// the hidden states of padding positions, matching the reference.
    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let dtype = xs.dtype();

        let h = match attention_mask {
            Some(mask) => xs.broadcast_mul(&mask.unsqueeze(2)?)?,
            None => xs.clone(),
        };

        // Concatenated Q/K/V projections, transposed to `[B, conv_dim, T]`.
        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;
        let mixed = Tensor::cat(&[&q, &k, &v], 2)?.transpose(1, 2)?;

        // Short-conv: reuse the cached conv state for a single-token decode;
        // otherwise run the causal conv over the current (full) inputs.
        let conv_out = if let Some(state) = &self.conv_state {
            if seq_len == 1 {
                let (out, new_state) = causal_conv1d_step(
                    &mixed,
                    state,
                    &self.conv_weight,
                    self.conv_kernel,
                    self.conv_dim,
                )?;
                self.conv_state = Some(new_state);
                out
            } else {
                // Prefill continuation: concatenate cached context, conv, slice
                // the current positions, and store the latest window.
                let full = Tensor::cat(&[state, &mixed], 2)?;
                let out = causal_conv1d(
                    &full,
                    &self.conv_weight,
                    self.conv_kernel,
                    self.conv_dim,
                    seq_len,
                    true,
                )?;
                let flen = full.dim(2)?;
                self.conv_state =
                    Some(full.narrow(2, flen - (self.conv_kernel - 1), self.conv_kernel - 1)?);
                out
            }
        } else {
            let out = causal_conv1d(
                &mixed,
                &self.conv_weight,
                self.conv_kernel,
                self.conv_dim,
                seq_len,
                false,
            )?;
            self.conv_state =
                Some(mixed.narrow(2, seq_len - (self.conv_kernel - 1), self.conv_kernel - 1)?);
            out
        };
        let conv_out = self
            .activation
            .forward(&conv_out)?
            .transpose(1, 2)?
            .contiguous()?;

        // Split Q/K/V and reshape to `[B, T, num_heads, head_dim]`.
        let shape = (b, seq_len, self.num_heads, self.head_dim);
        let q = conv_out
            .narrow(2, 0, self.qkv_dim)?
            .reshape(shape.clone())?;
        let k = conv_out
            .narrow(2, self.qkv_dim, self.qkv_dim)?
            .reshape(shape.clone())?;
        let v = conv_out
            .narrow(2, 2 * self.qkv_dim, self.qkv_dim)?
            .reshape(shape)?;

        // Forget gate (log-space per-dim decay) and input gate.
        let g = self.forget_gate.forward(&h)?; // [B, T, H, D]
        let beta = candle_nn::ops::sigmoid(&self.b_proj.forward(&h)?)?; // [B, T, H]

        // Q/K L2-normalisation and query scale, matching the FLA kernel.
        let scale = Tensor::new((self.head_dim as f64).powf(-0.5) as f32, q.device())?;
        let q = l2norm(&q, 1e-6)?.broadcast_mul(&scale)?;
        let k = l2norm(&k, 1e-6)?;

        // Recurrent delta-rule attention.
        let use_decode_path = seq_len == 1 && self.recurrent_state.is_some();
        let (out, _final_state) = if use_decode_path {
            let (o, s) = recurrent_kda(&q, &k, &v, &g, &beta, self.recurrent_state.as_ref())?;
            self.recurrent_state = Some(s.clone());
            (o, s)
        } else {
            let init = if self.recurrent_state.is_some() {
                self.recurrent_state.clone()
            } else {
                None
            };
            let (o, s) = recurrent_kda(&q, &k, &v, &g, &beta, init.as_ref())?;
            self.recurrent_state = Some(s.clone());
            (o, s)
        };

        // Final gated RMSNorm and output projection.
        let gate = self
            .g_b_proj
            .forward(&self.g_a_proj.forward(&h)?)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let normed = self
            .o_norm
            .forward(&out, &gate)?
            .reshape((b, seq_len, self.qkv_dim))?;
        self.o_proj.forward(&normed)?.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    /// A tiny, deterministic KDA config for reference-parity tests.
    struct TinyCfg {
        hidden: usize,
        heads: usize,
        head_dim: usize,
        kernel: usize,
        lower_bound: Option<f64>,
        eps: f64,
    }

    impl KdaConfig for TinyCfg {
        fn hidden_size(&self) -> usize {
            self.hidden
        }
        fn linear_num_heads(&self) -> usize {
            self.heads
        }
        fn linear_head_dim(&self) -> usize {
            self.head_dim
        }
        fn linear_conv_kernel_dim(&self) -> usize {
            self.kernel
        }
        fn linear_lower_bound(&self) -> Option<f64> {
            self.lower_bound
        }
        fn rms_norm_eps(&self) -> f64 {
            self.eps
        }
        fn hidden_act(&self) -> Activation {
            Activation::Silu
        }
    }

    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const KERNEL: usize = 2;
    const QKVD: usize = HEADS * HEAD_DIM; // 8
    const CONV_DIM: usize = QKVD * 3; // 24

    fn cfg() -> TinyCfg {
        TinyCfg {
            hidden: HIDDEN,
            heads: HEADS,
            head_dim: HEAD_DIM,
            kernel: KERNEL,
            lower_bound: Some(-5.0),
            eps: 1e-5,
        }
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

    #[derive(Clone)]
    struct Weights {
        q: Vec<f32>, // [QKVD, HIDDEN]
        k: Vec<f32>,
        v: Vec<f32>,
        conv: Vec<f32>, // [CONV_DIM, 1, KERNEL]
        f_a: Vec<f32>,  // [HEAD_DIM, HIDDEN]
        f_b: Vec<f32>,  // [QKVD, HEAD_DIM]
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
        b: Vec<f32>,   // [HEADS, HIDDEN]
        g_a: Vec<f32>, // [HEAD_DIM, HIDDEN]
        g_b: Vec<f32>, // [QKVD, HEAD_DIM]
        o_norm: Vec<f32>,
        o: Vec<f32>, // [HIDDEN, QKVD]
    }

    fn gen_weights() -> Weights {
        let mut r = rng_seq();
        let mut w = |n: usize| (0..n).map(|_| r()).collect::<Vec<_>>();
        Weights {
            q: w(QKVD * HIDDEN),
            k: w(QKVD * HIDDEN),
            v: w(QKVD * HIDDEN),
            conv: w(CONV_DIM * KERNEL),
            f_a: w(HEAD_DIM * HIDDEN),
            f_b: w(QKVD * HEAD_DIM),
            dt_bias: w(QKVD),
            a_log: w(HEADS),
            b: w(HEADS * HIDDEN),
            g_a: w(HEAD_DIM * HIDDEN),
            g_b: w(QKVD * HEAD_DIM),
            o_norm: w(HEAD_DIM),
            o: w(HIDDEN * QKVD),
        }
    }

    fn build_block(ws: &Weights, dev: &Device) -> candle::Result<KdaLinearAttention> {
        let mut tensors = HashMap::new();
        let put = |t: &mut HashMap<String, Tensor>, name: &str, data: &[f32]| {
            let shp: candle::Shape = match name {
                "q_proj.weight" | "k_proj.weight" | "v_proj.weight" => (QKVD, HIDDEN).into(),
                "conv1d.weight" => (CONV_DIM, 1, KERNEL).into(),
                "forget_gate.f_a_proj.weight" | "g_a_proj.weight" => (HEAD_DIM, HIDDEN).into(),
                "forget_gate.f_b_proj.weight" | "g_b_proj.weight" => (QKVD, HEAD_DIM).into(),
                "forget_gate.dt_bias" => (QKVD,).into(),
                "forget_gate.A_log" => (HEADS,).into(),
                "b_proj.weight" => (HEADS, HIDDEN).into(),
                "o_norm.weight" => (HEAD_DIM,).into(),
                "o_proj.weight" => (HIDDEN, QKVD).into(),
                _ => unreachable!(),
            };
            t.insert(
                name.to_string(),
                Tensor::from_vec(data.to_vec(), shp, dev).unwrap(),
            );
        };
        put(&mut tensors, "q_proj.weight", &ws.q);
        put(&mut tensors, "k_proj.weight", &ws.k);
        put(&mut tensors, "v_proj.weight", &ws.v);
        put(&mut tensors, "conv1d.weight", &ws.conv);
        put(&mut tensors, "forget_gate.f_a_proj.weight", &ws.f_a);
        put(&mut tensors, "forget_gate.f_b_proj.weight", &ws.f_b);
        put(&mut tensors, "forget_gate.dt_bias", &ws.dt_bias);
        put(&mut tensors, "forget_gate.A_log", &ws.a_log);
        put(&mut tensors, "b_proj.weight", &ws.b);
        put(&mut tensors, "g_a_proj.weight", &ws.g_a);
        put(&mut tensors, "g_b_proj.weight", &ws.g_b);
        put(&mut tensors, "o_norm.weight", &ws.o_norm);
        put(&mut tensors, "o_proj.weight", &ws.o);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
        KdaLinearAttention::new(&cfg(), 0, vb)
    }

    // ---- Pure-f32 reference (independent of the candle block) ----

    fn lin(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0.0;
            for i in 0..in_dim {
                acc += w[o * in_dim + i] * x[i];
            }
            y[o] = acc;
        }
        y
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    /// Causal depthwise conv over `x` flattened `[CONV_DIM, T]` with weight
    /// `[CONV_DIM, KERNEL]` (the singleton `1` in-channel is implicit).
    fn causal_conv(x: &[f32], w: &[f32], t: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; CONV_DIM * t];
        for c in 0..CONV_DIM {
            for tt in 0..t {
                let mut acc = 0.0;
                for j in 0..KERNEL {
                    let src = tt as isize - (KERNEL as isize - 1) + j as isize;
                    if src >= 0 && (src as usize) < t {
                        acc += w[c * KERNEL + j] * x[c * t + src as usize];
                    }
                }
                out[c * t + tt] = acc;
            }
        }
        out
    }

    fn l2norm_dim(x: &[f32], d: usize) -> Vec<f32> {
        // x length divisible by d; normalize each d-slice.
        let n = x.len() / d;
        let mut out = x.to_vec();
        for i in 0..n {
            let mut sq = 0.0f64;
            for j in 0..d {
                sq += (x[i * d + j] as f64) * (x[i * d + j] as f64);
            }
            let inv = (sq + 1e-6).sqrt() as f32;
            for j in 0..d {
                out[i * d + j] = x[i * d + j] / inv;
            }
        }
        out
    }

    /// Reference forward for a single batch, sequence `h` flattened
    /// `[T, HIDDEN]`. Returns `(output [T, HIDDEN], state [HEADS, D, D])`.
    fn ref_forward(ws: &Weights, h: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = HEAD_DIM;
        // Concatenated Q/K/V per token.
        let mut mixed = vec![0.0f32; CONV_DIM * t];
        for tt in 0..t {
            let xt = &h[tt * HIDDEN..(tt + 1) * HIDDEN];
            let q = lin(xt, &ws.q, HIDDEN, QKVD);
            let k = lin(xt, &ws.k, HIDDEN, QKVD);
            let v = lin(xt, &ws.v, HIDDEN, QKVD);
            for c in 0..QKVD {
                mixed[c * t + tt] = q[c];
                mixed[(QKVD + c) * t + tt] = k[c];
                mixed[(2 * QKVD + c) * t + tt] = v[c];
            }
        }
        let conv = causal_conv(&mixed, &ws.conv, t);
        let conv: Vec<f32> = conv.iter().map(|&x| silu(x)).collect();
        // transpose back to [T, CONV_DIM]
        let mut qkv = vec![0.0f32; CONV_DIM * t];
        for tt in 0..t {
            for c in 0..CONV_DIM {
                qkv[tt * CONV_DIM + c] = conv[c * t + tt];
            }
        }

        let mut q = vec![0.0f32; t * HEADS * hd];
        let mut k = vec![0.0f32; t * HEADS * hd];
        let mut v = vec![0.0f32; t * HEADS * hd];
        for tt in 0..t {
            for hh in 0..HEADS {
                for d in 0..hd {
                    q[(tt * HEADS + hh) * hd + d] = qkv[tt * CONV_DIM + hh * hd + d];
                    k[(tt * HEADS + hh) * hd + d] = qkv[tt * CONV_DIM + QKVD + hh * hd + d];
                    v[(tt * HEADS + hh) * hd + d] = qkv[tt * CONV_DIM + 2 * QKVD + hh * hd + d];
                }
            }
        }
        q = l2norm_dim(&q, hd);
        k = l2norm_dim(&k, hd);
        let scale = 1.0 / (hd as f32).sqrt();
        for x in q.iter_mut() {
            *x *= scale;
        }

        // Forget gate g and input gate beta.
        let mut g = vec![0.0f32; t * HEADS * hd];
        let mut beta = vec![0.0f32; t * HEADS];
        let lower = cfg().lower_bound.unwrap() as f32;
        for tt in 0..t {
            let xt = &h[tt * HIDDEN..(tt + 1) * HIDDEN];
            let fg = lin(&lin(xt, &ws.f_a, HIDDEN, hd), &ws.f_b, hd, QKVD);
            let decay: Vec<f32> = ws.a_log.iter().map(|a| a.exp()).collect();
            for hh in 0..HEADS {
                beta[tt * HEADS + hh] = sigmoid(lin(xt, &ws.b, HIDDEN, HEADS)[hh]);
                for d in 0..hd {
                    let val = fg[hh * hd + d] + ws.dt_bias[hh * hd + d];
                    g[(tt * HEADS + hh) * hd + d] = lower * sigmoid(decay[hh] * val);
                }
            }
        }

        // Recurrent delta-rule.
        let mut state = vec![0.0f32; HEADS * hd * hd];
        let mut out = vec![0.0f32; t * HEADS * hd];
        for tt in 0..t {
            for hh in 0..HEADS {
                let kt = &k[(tt * HEADS + hh) * hd..(tt * HEADS + hh + 1) * hd];
                let vt = &v[(tt * HEADS + hh) * hd..(tt * HEADS + hh + 1) * hd];
                let qt = &q[(tt * HEADS + hh) * hd..(tt * HEADS + hh + 1) * hd];
                let gt = &g[(tt * HEADS + hh) * hd..(tt * HEADS + hh + 1) * hd];
                let bet = beta[tt * HEADS + hh];
                // decay state along the value dim, per-key decay g[d1]
                for d1 in 0..hd {
                    let dec = gt[d1].exp();
                    for d2 in 0..hd {
                        state[(hh * hd + d1) * hd + d2] *= dec;
                    }
                }
                // kv_mem[d2] = sum_d1 state[d1][d2] * k[d1]
                let mut kv_mem = [0.0f32; HEAD_DIM];
                for d2 in 0..hd {
                    let mut acc = 0.0;
                    for d1 in 0..hd {
                        acc += state[(hh * hd + d1) * hd + d2] * kt[d1];
                    }
                    kv_mem[d2] = acc;
                }
                // delta[d2] = (v[d2] - kv_mem[d2]) * beta
                let mut delta = [0.0f32; HEAD_DIM];
                for d2 in 0..hd {
                    delta[d2] = (vt[d2] - kv_mem[d2]) * bet;
                }
                // state[d1][d2] += k[d1] * delta[d2]
                for d1 in 0..hd {
                    for d2 in 0..hd {
                        state[(hh * hd + d1) * hd + d2] += kt[d1] * delta[d2];
                    }
                }
                // o[d2] = sum_d1 state[d1][d2] * q[d1]
                for d2 in 0..hd {
                    let mut acc = 0.0;
                    for d1 in 0..hd {
                        acc += state[(hh * hd + d1) * hd + d2] * qt[d1];
                    }
                    out[(tt * HEADS + hh) * hd + d2] = acc;
                }
            }
        }

        // Gated RMSNorm + output projection.
        let mut final_out = vec![0.0f32; t * HIDDEN];
        for tt in 0..t {
            let xt = &h[tt * HIDDEN..(tt + 1) * HIDDEN];
            let ga = lin(xt, &ws.g_a, HIDDEN, hd);
            let gate = lin(&ga, &ws.g_b, hd, QKVD);
            let mut flat = vec![0.0f32; QKVD];
            for hh in 0..HEADS {
                let mut var = 0.0f64;
                for d in 0..hd {
                    let o = out[(tt * HEADS + hh) * hd + d] as f64;
                    var += o * o;
                }
                let inv = (var / hd as f64 + cfg().eps).sqrt() as f32;
                for d in 0..hd {
                    let normed = out[(tt * HEADS + hh) * hd + d] / inv * ws.o_norm[d];
                    let gsig = sigmoid(gate[hh * hd + d]);
                    flat[hh * hd + d] = normed * gsig;
                }
            }
            let o = lin(&flat, &ws.o, QKVD, HIDDEN);
            final_out[tt * HIDDEN..(tt + 1) * HIDDEN].copy_from_slice(&o);
        }
        (final_out, state)
    }

    fn tens_to_f32(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            assert!(
                d <= tol,
                "mismatch at {i}: block={} ref={} (diff {})",
                a[i],
                b[i],
                d
            );
        }
    }

    #[test]
    fn prefill_matches_reference() {
        let dev = Device::Cpu;
        let ws = gen_weights();
        let mut block = build_block(&ws, &dev).unwrap();

        let t = 3;
        let mut r = rng_seq();
        let h: Vec<f32> = (0..t * HIDDEN).map(|_| r()).collect();
        let input = Tensor::from_vec(h.clone(), (1, t, HIDDEN), &dev).unwrap();

        let out = block.forward(&input, None).unwrap();
        let (ref_out, _state) = ref_forward(&ws, &h, t);
        close(&tens_to_f32(&out), &ref_out, 1e-4);
    }

    #[test]
    fn decode_reuses_prefill_state() {
        let dev = Device::Cpu;
        let ws = gen_weights();
        let mut r = rng_seq();
        let seq: Vec<f32> = (0..4 * HIDDEN).map(|_| r()).collect();
        let prefill = seq[..3 * HIDDEN].to_vec();
        let dec = seq[3 * HIDDEN..].to_vec();

        // Full-sequence forward (all 4 tokens at once).
        let mut full_block = build_block(&ws, &dev).unwrap();
        let full_in = Tensor::from_vec(seq.clone(), (1, 4, HIDDEN), &dev).unwrap();
        let full_out = full_block.forward(&full_in, None).unwrap();

        // Prefill then single-token decode.
        let mut block = build_block(&ws, &dev).unwrap();
        let pre_in = Tensor::from_vec(prefill.clone(), (1, 3, HIDDEN), &dev).unwrap();
        let pre_out = block.forward(&pre_in, None).unwrap();
        let dec_in = Tensor::from_vec(dec.clone(), (1, 1, HIDDEN), &dev).unwrap();
        let dec_out = block.forward(&dec_in, None).unwrap();

        // The decode token must equal the last token of the full forward, and
        // the prefill rows must match the full forward's first three rows.
        let pre = tens_to_f32(&pre_out);
        let dec = tens_to_f32(&dec_out);
        let full = tens_to_f32(&full_out);
        close(&pre, &full[..3 * HIDDEN], 1e-4);
        close(&dec, &full[3 * HIDDEN..], 1e-4);

        // And the decode output matches the scalar reference continuing from
        // the prefill state.
        let (_ref_pre, _state) = ref_forward(&ws, &prefill, 3);
        let (ref_dec, _s2) = ref_forward(&ws, &seq, 4);
        close(&dec, &ref_dec[3 * HIDDEN..], 1e-4);
    }
}
