//! GLM-5.3-Flash vision tower (story #5889).
//!
//! Implements the `glm5_next_vision` encoder from the official
//! `zai-org/GLM-5.3-Flash` checkpoint, mirroring `transformers`
//! `Glm5NextVisionModel`. The tower is a ViT-style encoder whose weight names
//! match `transformers` directly (`model.visual.{patch_embed,blocks,post_layernorm,
//! downsample,merger}`), so official checkpoints load without remapping.
//!
//! Pipeline (matching the reference):
//! 1. `patch_embed` — a Conv3d with kernel/stride `[temporal_patch_size, patch_size,
//!    patch_size]` implemented as a linear over flattened patch cubes (a full-stride
//!    conv over non-overlapping blocks is exactly an unfold + linear). It consumes the
//!    processor's flat `pixel_values` layout `[num_images, grid_h*grid_w, patch_dim]`
//!    and emits one embedding per spatial patch.
//! 2. `blocks` — `depth` pre-norm transformer blocks (RMSNorm pre-norm, QK-norm'd
//!    multi-head attention with per-axis rotary position embeddings, clamped-SiLU
//!    SwiGLU MLP). Attention is non-causal and block-diagonal per image.
//! 3. `post_layernorm` — final RMSNorm.
//! 4. `downsample` — a stride-`spatial_merge_size` Conv2d that spatially merges the
//!    `spatial_merge_size x spatial_merge_size` patch blocks into `out_hidden_size`.
//! 5. `merger` — a GELU SwiGLU MLP projecting `out_hidden_size -> out_hidden_size`
//!    (matching the text hidden size), whose output is the multimodal embedding
//!    injected at the text `image_token_id` positions.
//!
//! Non-goals (per the story): video inputs, training, FP8/NVFP4, tensor parallelism.

use crate::serde_default_fn;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{
    linear_no_bias, rms_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, RmsNorm, VarBuilder,
};

serde_default_fn!(usize, default_in_channels, 3);
serde_default_fn!(usize, default_temporal_patch_size, 2);
serde_default_fn!(f64, default_rms_norm_eps_vis, 1e-5);
serde_default_fn!(bool, default_attention_bias_vis, true);
serde_default_fn!(f64, default_attention_dropout_vis, 0.0);
serde_default_fn!(
    candle_nn::Activation,
    default_hidden_act_vis,
    candle_nn::Activation::Silu
);
serde_default_fn!(f64, default_swiglu_limit_vis, 10.0);
serde_default_fn!(f32, default_initializer_range_vis, 0.02);

/// `glm5_next_vision` configuration (mirrors `transformers` `Glm5NextVisionConfig`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Glm5NextVisionConfig {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub image_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub projection_intermediate_size: usize,
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_rms_norm_eps_vis")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_attention_bias_vis")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout_vis")]
    pub attention_dropout: f64,
    #[serde(default = "default_hidden_act_vis")]
    pub hidden_act: candle_nn::Activation,
    #[serde(default = "default_swiglu_limit_vis")]
    pub swiglu_limit: f64,
    #[serde(default = "default_initializer_range_vis")]
    pub initializer_range: f32,
}

/// Patch embedding as a linear over flattened patch cubes, equivalent to the
/// reference Conv3d (full-stride, non-overlapping) over
/// `[in_channels, temporal_patch_size, patch_size, patch_size]` blocks.
struct Glm5NextVisionPatchEmbed {
    proj: Linear,
    patch_dim: usize,
}

impl Glm5NextVisionPatchEmbed {
    fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_dim = cfg
            .in_channels
            .checked_mul(cfg.temporal_patch_size)
            .and_then(|v| v.checked_mul(cfg.patch_size))
            .and_then(|v| v.checked_mul(cfg.patch_size))
            .ok_or_else(|| candle::Error::Msg("vision patch_dim overflow".into()))?;
        // The checkpoint stores the conv weight as [embed, in, tps, ps, ps].
        let weight = vb
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "proj.weight",
            )?
            .flatten_from(1)?; // [hidden, patch_dim]
        let bias = vb.get((cfg.hidden_size,), "proj.bias")?;
        let proj = Linear::new(weight, Some(bias));
        Ok(Self { proj, patch_dim })
    }

    /// `pixel_values` is `[num_images, grid_h*grid_w, patch_dim]`; returns
    /// `[num_patches_total, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (num_images, patches_per, _) = pixel_values.dims3()?;
        let patches = pixel_values.reshape((num_images * patches_per, self.patch_dim))?;
        self.proj.forward(&patches)
    }
}

/// Per-axis rotary position embedding (`Glm5NextVisionRotaryEmbedding`).
struct Glm5NextVisionRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Glm5NextVisionRotaryEmbedding {
    fn new(head_dim: usize) -> Self {
        // Reference: dim = head_dim // 2, inv_freq over arange(0, dim, 2).
        let dim = head_dim / 2;
        let theta = 10000.0f32;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect();
        Self { inv_freq }
    }

    /// Returns `[N, head_dim]` pre-rotary `emb` (two interleaved axis copies).
    fn forward(&self, position_ids: &Tensor, dev: &Device) -> Result<Tensor> {
        let inv = Tensor::from_vec(self.inv_freq.clone(), (self.inv_freq.len(),), dev)?;
        // position_ids [N, 2] * inv_freq [k] -> [N, 2, k]
        let rotary = position_ids
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_mul(&inv)?
            .flatten(1, D::Minus1)?; // [N, 2*k] where 2*k = head_dim/2
                                     // Reference: emb = cat((rotary, rotary), -1) -> head_dim.
        Tensor::cat(&[&rotary, &rotary], D::Minus1)
    }
}

/// rotate_half: `cat((-x2, x1), -1)` over the trailing half split.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, d - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?; // [N, 1, head_dim]
    let sin = sin.unsqueeze(1)?;
    let qe = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let ke = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((qe, ke))
}

/// Block-diagonal (per-image, non-causal) attention mask, `0` inside each image
/// block and `-inf` across images, matching the reference packed attention.
fn block_diag_mask(n: usize, cu_seqlens: &[usize], dev: &Device) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; n * n];
    for w in cu_seqlens.windows(2) {
        let (s, e) = (w[0], w[1]);
        for i in s..e {
            for j in s..e {
                data[i * n + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(data, (n, n), dev)
}

/// QK-norm'd multi-head attention (`Glm5NextVisionAttention`).
struct Glm5NextVisionAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    head_dim: usize,
    scaling: f32,
}

impl Glm5NextVisionAttention {
    fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let qkv = Linear::new(
            vb.get((3 * cfg.hidden_size, cfg.hidden_size), "attn.qkv.weight")?,
            if cfg.attention_bias {
                Some(vb.get((3 * cfg.hidden_size,), "attn.qkv.bias")?)
            } else {
                None
            },
        );
        let proj = Linear::new(
            vb.get((cfg.hidden_size, cfg.hidden_size), "attn.proj.weight")?,
            if cfg.attention_bias {
                Some(vb.get((cfg.hidden_size,), "attn.proj.bias")?)
            } else {
                None
            },
        );
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("attn.q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("attn.k_norm"))?;
        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_heads,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x` is `[N, hidden]`; returns `[N, hidden]`.
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &[usize],
    ) -> Result<Tensor> {
        let n = x.dims()[0];
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((n, 3, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&qkv.narrow(1, 0, 1)?.squeeze(1)?)?; // [N, H, hd]
        let k = self.k_norm.forward(&qkv.narrow(1, 1, 1)?.squeeze(1)?)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        let (q, k) = apply_rotary(&q, &k, cos, sin)?;
        // Move heads to the batch axis for per-head non-causal attention.
        let q = q.transpose(0, 1)?; // [H, N, hd]
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        let mask = block_diag_mask(n, cu_seqlens, x.device())?.unsqueeze(0)?; // [1, N, N]
        let scores = q
            .matmul(&k.transpose(1, 2)?)? // [H, N, N]
            .affine(self.scaling as f64, 0.0)?
            .broadcast_add(&mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?; // [H, N, N]
        let out = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((n, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

/// Clamped-SiLU SwiGLU MLP (`Glm5NextVisionMLP`).
struct Glm5NextVisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    swiglu_limit: f64,
}

impl Glm5NextVisionMLP {
    fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let bias = cfg.attention_bias;
        let gate_proj = if bias {
            candle_nn::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?
        } else {
            linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?
        };
        let up_proj = if bias {
            candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))?
        };
        let down_proj = if bias {
            candle_nn::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.down_proj"),
            )?
        } else {
            linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.down_proj"),
            )?
        };
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            swiglu_limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(x)?
            .clamp(f64::NEG_INFINITY, self.swiglu_limit)?;
        let up = self
            .up_proj
            .forward(x)?
            .clamp(-self.swiglu_limit, self.swiglu_limit)?;
        self.down_proj
            .forward(&candle_nn::ops::silu(&gate)?.mul(&up)?)
    }
}

/// One vision transformer block (`Glm5NextVisionBlock`).
struct Glm5NextVisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: Glm5NextVisionAttention,
    mlp: Glm5NextVisionMLP,
}

impl Glm5NextVisionBlock {
    fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm1"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm2"))?;
        let attn = Glm5NextVisionAttention::new(cfg, vb.clone())?;
        let mlp = Glm5NextVisionMLP::new(cfg, vb)?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &[usize],
    ) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h, cos, sin, cu_seqlens)?;
        let x = x.add(&h)?;
        let h = self.mlp.forward(&self.norm2.forward(&x)?)?;
        x.add(&h)
    }
}

/// GELU SwiGLU merger (`Glm5NextVisionPatchMerger`) mapping the post-downsample
/// `out_hidden_size` embeddings into the text hidden space.
struct Glm5NextVisionMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    swiglu_limit: f64,
}

impl Glm5NextVisionMerger {
    fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.out_hidden_size;
        let context = cfg.projection_intermediate_size;
        let proj = linear_no_bias(dim, dim, vb.pp("proj"))?;
        let w = vb.get((dim,), "post_projection_norm.weight")?;
        let b = vb.get((dim,), "post_projection_norm.bias")?;
        let post_projection_norm = LayerNorm::new(w, b, 1e-5);
        let gate_proj = linear_no_bias(dim, context, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(dim, context, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(context, dim, vb.pp("down_proj"))?;
        Ok(Self {
            proj,
            post_projection_norm,
            gate_proj,
            up_proj,
            down_proj,
            swiglu_limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(x)?;
        let h = self.post_projection_norm.forward(&h)?.gelu_erf()?;
        let gate = self
            .gate_proj
            .forward(&h)?
            .clamp(f64::NEG_INFINITY, self.swiglu_limit)?;
        let up = self
            .up_proj
            .forward(&h)?
            .clamp(-self.swiglu_limit, self.swiglu_limit)?;
        self.down_proj
            .forward(&candle_nn::ops::silu(&gate)?.mul(&up)?)
    }
}

/// The GLM-5.3-Flash vision tower (`Glm5NextVisionModel`).
pub struct Glm5NextVisionModel {
    patch_embed: Glm5NextVisionPatchEmbed,
    rotary_pos_emb: Glm5NextVisionRotaryEmbedding,
    blocks: Vec<Glm5NextVisionBlock>,
    post_layernorm: RmsNorm,
    downsample: Conv2d,
    merger: Glm5NextVisionMerger,
    hidden_size: usize,
    out_hidden_size: usize,
    spatial_merge_size: usize,
}

impl Glm5NextVisionModel {
    /// Build the vision tower from `model.visual.*` weights.
    pub fn new(cfg: &Glm5NextVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = Glm5NextVisionPatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = Glm5NextVisionRotaryEmbedding::new(head_dim);
        let blocks = (0..cfg.depth)
            .map(|i| Glm5NextVisionBlock::new(cfg, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let post_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_layernorm"))?;
        let downsample = Conv2d::new(
            vb.get(
                (
                    cfg.out_hidden_size,
                    cfg.hidden_size,
                    cfg.spatial_merge_size,
                    cfg.spatial_merge_size,
                ),
                "downsample.weight",
            )?,
            Some(vb.get((cfg.out_hidden_size,), "downsample.bias")?),
            Conv2dConfig {
                stride: cfg.spatial_merge_size,
                ..Default::default()
            },
        );
        let merger = Glm5NextVisionMerger::new(cfg, vb.pp("merger"))?;
        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            post_layernorm,
            downsample,
            merger,
            hidden_size: cfg.hidden_size,
            out_hidden_size: cfg.out_hidden_size,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    /// Per-image cumulative sequence lengths in patch tokens (for the
    /// block-diagonal attention mask).
    fn cu_seqlens(grid_thw: &Tensor) -> Result<Vec<usize>> {
        let num_images = grid_thw.dims()[0];
        let mut cu = vec![0usize];
        for i in 0..num_images {
            let row = grid_thw.narrow(0, i, 1)?.squeeze(0)?.to_vec1::<i64>()?;
            let len = (row[0] * row[1] * row[2]) as usize;
            cu.push(cu.last().unwrap() + len);
        }
        Ok(cu)
    }

    /// Block-major per-patch position ids `[N, 2]` (h, w per image, repeated t
    /// times), mirroring `get_vision_position_ids`.
    fn position_ids(grid_thw: &Tensor, spatial_merge_size: usize, dev: &Device) -> Result<Tensor> {
        let num_images = grid_thw.dims()[0];
        let mut all: Vec<Tensor> = Vec::new();
        for i in 0..num_images {
            let row = grid_thw.narrow(0, i, 1)?.squeeze(0)?.to_vec1::<i64>()?;
            let (t, h, w) = (row[0] as usize, row[1] as usize, row[2] as usize);
            let m = spatial_merge_size;
            let hpos = Tensor::arange(0u32, h as u32, dev)?;
            let wpos = Tensor::arange(0u32, w as u32, dev)?;
            let grids = Tensor::meshgrid(&[hpos, wpos], false)?; // [h, w] each
            let hg = grids[0]
                .reshape((h / m, m, w / m, m))?
                .transpose(1, 2)?
                .flatten_all()?; // [h*w]
            let wg = grids[1]
                .reshape((h / m, m, w / m, m))?
                .transpose(1, 2)?
                .flatten_all()?;
            let hg = hg.repeat(t)?;
            let wg = wg.repeat(t)?;
            all.push(Tensor::stack(&[hg, wg], 1)?); // [t*h*w, 2]
        }
        Tensor::cat(&all, 0)
    }

    /// Runs the vision tower on the processor's flat patch layout.
    ///
    /// - `pixel_values`: `[num_images, grid_h*grid_w, patch_dim]`
    /// - `grid_thw`: `[num_images, 3]` with columns `[t, grid_h, grid_w]`
    ///
    /// Returns the merged multimodal embeddings `[num_merged_tokens, out_hidden_size]`
    /// ready to be injected at the text `image_token_id` positions.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let dev = pixel_values.device();
        let mut h = self.patch_embed.forward(pixel_values)?; // [N, hidden]
        let n = h.dims()[0];
        let cu = Self::cu_seqlens(grid_thw)?;
        if *cu.last().unwrap() != n {
            candle::bail!(
                "vision: grid_thw patch count ({}) != pixel patch count ({n})",
                cu.last().unwrap()
            );
        }
        let position_ids = Self::position_ids(grid_thw, self.spatial_merge_size, dev)?;
        let emb = self.rotary_pos_emb.forward(&position_ids, dev)?; // [N, head_dim]
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        for blk in &self.blocks {
            h = blk.forward(&h, &cos, &sin, &cu)?;
        }
        h = self.post_layernorm.forward(&h)?; // [N, hidden]
        let sm = self.spatial_merge_size;
        let n_blocks = n / (sm * sm);
        let h = h
            .reshape((n_blocks, sm, sm, self.hidden_size))?
            .permute((0, 3, 1, 2))?; // [n_blocks, hidden, sm, sm]
        let h = self
            .downsample
            .forward(&h)?
            .reshape((n_blocks, self.out_hidden_size))?;
        self.merger.forward(&h)
    }
}
