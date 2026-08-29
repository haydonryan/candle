//! DeepSeek-V4 fp8/fp4 quantized weight loading + CPU/disk offload.
//!
//! The real `DeepSeek-V4-Flash` checkpoint stores almost all weights in FP8
//! E4M3 with per-block unsigned E8M0 scales
//! (`quantization_config`: `fmt = "e4m3"`, `scale_fmt = "ue8m0"`,
//! `weight_block_size = [128, 128]`, `activation_scheme = "dynamic"`) and the
//! routed-experts weights in FP4 E2M1 with block scales (`expert_dtype =
//! "fp4"`). Because the checkpoint is ~148 GiB and the target GPU has 96 GB,
//! loading must also be offload-aware: tensors are mmap-backed on disk and
//! dequantized on CPU on demand, keeping only the active layer's weights in
//! device memory.
//!
//! Candle has a real `DType::F8E4M3`, but `DType::F8E8M0` is a placeholder
//! dtype with no arithmetic and FP4 has no dtype at all, so these helpers
//! dequantize straight from the raw safetensors bytes exposed by
//! [`candle_core::safetensors::MmapedSafetensors::get`].

use candle::{DType, Device, Error, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{rms_norm, Init, Module, VarBuilder};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use crate::models::deepseek_v4::{
    DeepseekV4Config, DeepseekV4DecoderLayer, DeepseekV4ForCausalLM, DeepseekV4HyperHead,
};
use candle::safetensors::MmapedSafetensors;

/// FP8 E4M3 -> f32: sign(1) exp(4) mantissa(3), exponent bias 7.
///
/// Mirrors the NVIDIA `cuda_fp8` E4M3 layout that `float8::F8E4M3` implements:
/// `(-1)^s * 2^(e-7) * (1 + m/8)` for `e` in `1..=15`, `(-1)^s * 2^-6 * m/8`
/// for the subnormal `e == 0`, and `NaN` for the single pattern `0x7F`
/// (`e == 15`, `m == 7`). E4M3 has no infinity; `e == 15` with `m < 7` is the
/// finite tail up to 448.0.
#[inline]
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 3) & 0x0F;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        return sign * man * 2f32.powi(-9);
    }
    if b & 0x7F == 0x7F {
        return f32::NAN;
    }
    sign * (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7)
}

/// FP8 E8M0 (ue8m0) scale byte -> f32.
///
/// The stored byte is the 8-bit exponent with bias 127 and no sign/mantissa
/// bits, so the scale value is `2^(b - 127)`.
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    2f32.powi(b as i32 - 127)
}

/// `(-1)^s * 2^(e-1) * (1 + m/2)` for `e` in `1..=3`, `(-1)^s * m/2` for the
/// subnormal `e == 0`. Unlike FP8 E4M3, E2M1 has **no** inf/NaN pattern: the
/// max exponent `e == 3` is finite, yielding `4.0` (`m == 0`) and `6.0`
/// (`m == 1`). This matches the canonical FP4 E2M1 value table
/// `(0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6)` used by
/// the transformers `finegrained_fp8` reference.
#[inline]
pub fn fp4_e2m1_to_f32(nib: u8) -> f32 {
    let sign = if nib & 0x8 != 0 { -1.0 } else { 1.0 };
    let exp = (nib >> 1) & 0x3;
    let man = (nib & 0x1) as f32;
    let abs = match exp {
        0 => man * 0.5,
        e => (1.0 + man / 2.0) * 2f32.powi(e as i32 - 1),
    };
    sign * abs
}

/// Dequantize a block-scaled FP8 E4M3 weight into f32 values.
///
/// `weight` is `n*m` raw FP8 bytes; `scale` is `(n/bn)*(m/bm)` raw E8M0 bytes,
/// one per `block` (`bn` x `bm`) tile. Every element is scaled by its tile's
/// scale: `out[i*m+j] = fp8(weight[i*m+j]) * e8m0(scale[i/bn*(m/bm)+j/bm])`.
/// Tiles must divide the weight shape evenly (checkpoints pad to 128).
/// Shared block-scale dequant driver: validates the `n x m` / `block`
/// tiling and `scale` layout, precomputes the per-tile E8M0 scales, and runs
/// the row-parallel per-element loop, calling `decode(idx)` for the raw
/// byte -> f32 value at absolute index `idx`. Rows are independent, so this is
/// parallelized across them (this dominates the ~146 GiB real checkpoint).
fn dequantize_block_scale(
    n: usize,
    m: usize,
    block: (usize, usize),
    scale: &[u8],
    decode: impl Fn(usize) -> f32 + Sync,
) -> Result<Vec<f32>> {
    let (bn, bm) = block;
    let gn = n / bn;
    let gm = m / bm;
    if bn == 0 || bm == 0 || gn == 0 || gm == 0 || !n.is_multiple_of(bn) || !m.is_multiple_of(bm) {
        candle::bail!("block {block:?} does not tile weight {n}x{m}");
    }
    if scale.len() != gn * gm {
        candle::bail!(
            "scale len {} != ({n}/{bn})*({m}/{bm}) = {}",
            scale.len(),
            gn * gm
        );
    }
    // Precompute every tile's scale once (ue8m0 decode is `2^(b-127)`), so the
    // hot per-element loop is just decode * scale.
    let scales: Vec<f32> = scale.iter().map(|&b| e8m0_to_f32(b)).collect();
    Ok((0..n)
        .into_par_iter()
        .flat_map(|i| {
            let srow = (i / bn) * gm;
            let mut row = Vec::with_capacity(m);
            for jb in 0..gm {
                let sc = scales[srow + jb];
                for j in jb * bm..(jb + 1) * bm {
                    row.push(decode(i * m + j) * sc);
                }
            }
            row
        })
        .collect())
}

/// Dequantize a block-scaled FP8 E4M3 weight into f32 values. `scale` is one
/// E8M0 byte per `block` tile.
pub fn dequantize_fp8_block_scale(
    weight: &[u8],
    scale: &[u8],
    n: usize,
    m: usize,
    block: (usize, usize),
) -> Result<Vec<f32>> {
    if weight.len() != n * m {
        candle::bail!("fp8 weight len {} != {n}*{m}", weight.len());
    }
    dequantize_block_scale(n, m, block, scale, |idx| fp8_e4m3_to_f32(weight[idx]))
}

/// Dequantize a block-scaled FP4 E2M1 weight into f32 values.
///
/// `packed` holds two 4-bit FP4 nibbles per byte (low nibble first) for an
/// `n` x `m` weight, so its length is `n*m/2`. `scale` is one E8M0 byte per
/// `block` tile, exactly as for fp8.
pub fn dequantize_fp4_block_scale(
    packed: &[u8],
    scale: &[u8],
    n: usize,
    m: usize,
    block: (usize, usize),
) -> Result<Vec<f32>> {
    if packed.len() != n * m / 2 {
        candle::bail!("fp4 packed len {} != {n}*{m}/2", packed.len());
    }
    dequantize_block_scale(n, m, block, scale, |idx| {
        let byte = packed[idx / 2];
        let nib = if idx.is_multiple_of(2) {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        fp4_e2m1_to_f32(nib)
    })
}

/// Load a block-scaled FP8 linear weight from mmap-backed shards and
/// dequantize it on CPU into a `(n, m)` f32 `Tensor` (caller may
/// `.to_dtype()`/`.to_device()` as needed).
pub fn dequantize_fp8_linear(
    st: &MmapedSafetensors,
    weight_name: &str,
    scale_name: &str,
    n: usize,
    m: usize,
    block: (usize, usize),
    dev: &Device,
) -> Result<Tensor> {
    let wbytes = st.get(weight_name)?.data();
    let sbytes = st.get(scale_name)?.data();
    let vals = dequantize_fp8_block_scale(wbytes, sbytes, n, m, block)?;
    Tensor::from_vec(vals, (n, m), dev)
}

/// Load a block-scaled FP4 linear weight (expert) from mmap-backed shards and
/// dequantize it on CPU into a `(n, m)` f32 `Tensor`.
pub fn dequantize_fp4_linear(
    st: &MmapedSafetensors,
    weight_name: &str,
    scale_name: &str,
    n: usize,
    m: usize,
    block: (usize, usize),
    dev: &Device,
) -> Result<Tensor> {
    let pbytes = st.get(weight_name)?.data();
    let sbytes = st.get(scale_name)?.data();
    let vals = dequantize_fp4_block_scale(pbytes, sbytes, n, m, block)?;
    Tensor::from_vec(vals, (n, m), dev)
}
/// are evicted (least-recently-used first) once `max_bytes` is exceeded, so a
/// ~148 GiB fp8 checkpoint only ever holds up to the configured budget
/// resident (the rest stays mmap-backed on disk).
#[derive(Default)]
struct WeightCache {
    tensors: HashMap<String, Tensor>,
    order: VecDeque<String>,
    bytes: usize,
    max_bytes: usize,
    evictions: usize,
}

impl WeightCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Default::default()
        }
    }

    fn get(&mut self, name: &str) -> Option<Tensor> {
        let t = self.tensors.get(name)?.clone();
        if let Some(pos) = self.order.iter().position(|n| n == name) {
            self.order.remove(pos);
        }
        self.order.push_back(name.to_string());
        Some(t)
    }

    fn insert(&mut self, name: &str, t: Tensor) {
        if let Some(prev) = self.tensors.remove(name) {
            self.bytes -= prev.elem_count() * prev.dtype().size_in_bytes();
            if let Some(pos) = self.order.iter().position(|n| n == name) {
                self.order.remove(pos);
            }
        }
        self.bytes += t.elem_count() * t.dtype().size_in_bytes();
        self.tensors.insert(name.to_string(), t);
        self.order.push_back(name.to_string());
        while self.bytes > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(t) = self.tensors.remove(&oldest) {
                self.bytes -= t.elem_count() * t.dtype().size_in_bytes();
                self.evictions += 1;
            }
        }
    }

    fn resident_bytes(&self) -> usize {
        self.bytes
    }

    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn evictions(&self) -> usize {
        self.evictions
    }
}

/// A resolved real-schema weight target for a candle model weight name.
///
/// The real `deepseek-ai/DeepSeek-V4-Flash` checkpoint stores flat names that
/// differ from the eager candle model's `VarBuilder` namespace, and stores the
/// routed MoE experts as per-expert `ffn.experts.N.w1/w2/w3` FP4 tensors that
/// must be dequantized and assembled into the stacked 3-D
#[derive(Debug)]
enum RealTarget {
    /// A single on-disk tensor (the real checkpoint tensor name).
    Single(String),
    /// Assemble `gate_up_proj [e, 2*inter, hidden]` from per-expert w1+w3.
    GateUp { layer: usize },
    /// Assemble `down_proj [e, hidden, inter]` from per-expert w2.
    Down { layer: usize },
}

/// Resolve a candle model weight name to its real `DeepSeek-V4-Flash` target.
///
/// The real checkpoint stores flat names (`embed.weight`, `layers.N.attn.*`,
/// `layers.N.ffn.*`, `hc_*`, `hc_head_*`) that differ from the eager candle
/// model's `VarBuilder` namespace, and the routed MoE experts are per-expert
/// `ffn.experts.N.w1/w2/w3` FP4 tensors that must be assembled into the stacked
/// 3-D `gate_up_proj` / `down_proj`. Names with no real mapping (e.g. the
/// optional `e_score_correction_bias`, absent from the real checkpoint) return
/// `None`; the model's `match vb.get(..)` falls back to zeros.
fn real_schema_target(name: &str) -> Option<RealTarget> {
    let root = match name {
        "model.embed_tokens.weight" => Some("embed.weight".to_string()),
        "model.norm.weight" => Some("norm.weight".to_string()),
        "lm_head.weight" => Some("head.weight".to_string()),
        "model.hc_head.hc_fn" => Some("hc_head_fn".to_string()),
        "model.hc_head.hc_base" => Some("hc_head_base".to_string()),
        "model.hc_head.hc_scale" => Some("hc_head_scale".to_string()),
        _ => None,
    };
    if let Some(t) = root {
        return Some(RealTarget::Single(t));
    }
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let li: usize = rest[..dot].parse().ok()?;
    let body = &rest[dot + 1..];
    let l = format!("layers.{li}");
    let single = |s: &str| RealTarget::Single(format!("{l}.{s}"));
    Some(match body {
        "self_attn.q_a_proj.weight" => single("attn.wq_a.weight"),
        "self_attn.q_a_norm.weight" => single("attn.q_norm.weight"),
        "self_attn.q_b_proj.weight" => single("attn.wq_b.weight"),
        "self_attn.kv_proj.weight" => single("attn.wkv.weight"),
        "self_attn.kv_norm.weight" => single("attn.kv_norm.weight"),
        "self_attn.o_a_proj.weight" => single("attn.wo_a.weight"),
        "self_attn.o_b_proj.weight" => single("attn.wo_b.weight"),
        "self_attn.sinks" => single("attn.attn_sink"),
        "self_attn.compressor.kv_proj.weight" => single("attn.compressor.wkv.weight"),
        "self_attn.compressor.gate_proj.weight" => single("attn.compressor.wgate.weight"),
        "self_attn.compressor.position_bias" => single("attn.compressor.ape"),
        "self_attn.compressor.kv_norm.weight" => single("attn.compressor.norm.weight"),
        "self_attn.compressor.indexer.kv_proj.weight" => {
            single("attn.indexer.compressor.wkv.weight")
        }
        "self_attn.compressor.indexer.gate_proj.weight" => {
            single("attn.indexer.compressor.wgate.weight")
        }
        "self_attn.compressor.indexer.position_bias" => single("attn.indexer.compressor.ape"),
        "self_attn.compressor.indexer.kv_norm.weight" => {
            single("attn.indexer.compressor.norm.weight")
        }
        "self_attn.compressor.indexer.q_b_proj.weight" => single("attn.indexer.wq_b.weight"),
        "self_attn.compressor.indexer.scorer.weights_proj.weight" => {
            single("attn.indexer.weights_proj.weight")
        }
        "attn_hc.fn" => single("hc_attn_fn"),
        "attn_hc.base" => single("hc_attn_base"),
        "attn_hc.scale" => single("hc_attn_scale"),
        "ffn_hc.fn" => single("hc_ffn_fn"),
        "ffn_hc.base" => single("hc_ffn_base"),
        "ffn_hc.scale" => single("hc_ffn_scale"),
        "input_layernorm.weight" => single("attn_norm.weight"),
        "post_attention_layernorm.weight" => single("ffn_norm.weight"),
        "mlp.gate.weight" => single("ffn.gate.weight"),
        "mlp.gate.tid2eid" => single("ffn.gate.tid2eid"),
        "mlp.shared_experts.gate_proj.weight" => single("ffn.shared_experts.w1.weight"),
        "mlp.shared_experts.up_proj.weight" => single("ffn.shared_experts.w3.weight"),
        "mlp.shared_experts.down_proj.weight" => single("ffn.shared_experts.w2.weight"),
        "mlp.experts.gate_up_proj" => RealTarget::GateUp { layer: li },
        "mlp.experts.down_proj" => RealTarget::Down { layer: li },
        _ => return None,
    })
}

/// A full DeepSeek-V4 checkpoint opened mmap-backed across its safetensors
/// shards, with on-demand fp8/fp4 dequantization and a size-bounded LRU cache
/// (CPU/disk offload). Thread-safe (`&self` loads) so it can back a
/// `VarBuilder`/[`SimpleBackend`] used to construct the eager
/// [`DeepseekV4ForCausalLM`].
///
/// Only the real `deepseek-ai/DeepSeek-V4-Flash` schema is supported: flat
/// names (`embed.weight`, `layers.N.attn.*`, `layers.N.ffn.*`, `hc_*`), fp8
/// E4M3 block-`[128,128]` ue8m0 scales named `<weight>.scale`, per-expert FP4
/// E2M1 routed experts (`ffn.experts.N.w1/w2/w3`) with a per-row `gran_k = 32`
/// ue8m0 block scale (`[n, m/32]`, I8-packed `[n, m/2]` weight), plain
/// BF16 compressor/gate/norm/embed/head weights, and MTP layers (which candle
/// does not model).
pub struct DeepseekV4Quantized {
    st: MmapedSafetensors,
    names: HashSet<String>,
    block: (usize, usize),
    fp4_block: (usize, usize),
    dev: Device,
    /// Where dequantized weights are materialized and cached (the offload
    /// cache). Always CPU: keeping weights off the GPU lets the real ~146 GiB
    /// checkpoint run on a 96 GB card, because candle's stream-ordered CUDA
    /// allocator pools deallocated blocks and would otherwise retain every
    /// layer's dequantized memory, growing VRAM monotonically toward the full
    /// ~500 GiB bf16 model. Per-layer weights are copied to `dev` (the compute
    /// device) only while that layer is active and dropped afterwards.
    cache_dev: Device,
    out_dtype: DType,
    cache: Mutex<WeightCache>,
    /// GPU-resident weight cache (size-bounded LRU on `dev`). Keeps recently
    /// used layers' weights on the GPU (up to `gpu_max_bytes`, sized toward the
    /// available VRAM) so they are not re-transferred/re-dequantized on reuse,
    /// instead of dropping every layer's GPU tensors after each forward step.
    gpu_cache: Mutex<WeightCache>,
}

impl DeepseekV4Quantized {
    /// `block` is the fp8 dequant block (real V4-Flash `(128, 128)`); the FP4
    /// expert dequant block is fixed at the real per-row `gran_k = 32`
    /// `(1, 32)` layout. The real `deepseek-ai/DeepSeek-V4-Flash` flat-name
    /// layout (name remap + per-expert FP4 assembly) is the only supported
    /// schema.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from mmap'ing the shard files.
    pub unsafe fn new(
        paths: &[impl AsRef<Path>],
        block: (usize, usize),
        dev: Device,
        out_dtype: DType,
        max_bytes: usize,
        gpu_max_bytes: usize,
    ) -> Result<Self> {
        let st = MmapedSafetensors::multi(paths)?;
        let names = st.tensors().into_iter().map(|(n, _)| n).collect();
        Ok(Self {
            st,
            names,
            block,
            fp4_block: (1, 32),
            dev,
            cache_dev: Device::Cpu,
            out_dtype,
            cache: Mutex::new(WeightCache::new(max_bytes)),
            gpu_cache: Mutex::new(WeightCache::new(gpu_max_bytes)),
        })
    }

    /// Real `deepseek-ai/DeepSeek-V4-Flash` constructor: fp8 block `(128, 128)`
    /// (per `weight_block_size`), per-row FP4 expert block `(1, 32)`
    /// (`gran_k = 32`), `.scale` scale suffix, and `.ffn.experts.` FP4 names.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from mmap'ing the shard files.
    pub unsafe fn new_real(
        paths: &[impl AsRef<Path>],
        dev: Device,
        out_dtype: DType,
        max_bytes: usize,
        gpu_max_bytes: usize,
    ) -> Result<Self> {
        Self::new(
            paths,
            (128, 128),
            dev,
            out_dtype,
            max_bytes,
            gpu_max_bytes,
        )
    }

    /// True if `name` is a tensor present in the checkpoint.
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    fn is_fp4(&self, name: &str) -> bool {
        name.contains(".ffn.experts.")
    }

    /// Scale tensor name for a real-schema weight: `<basename>.scale`, where
    /// `basename` drops a trailing `.weight` (e.g. `wq_a.weight` -> `wq_a.scale`,
    /// `shared_experts.w1` -> `shared_experts.w1.scale`).
    fn real_scale_name(&self, name: &str) -> String {
        if let Some(base) = name.strip_suffix(".weight") {
            format!("{base}.scale")
        } else {
            format!("{name}.scale")
        }
    }

    /// Real-schema load: route an assembled expert tensor or a single on-disk
    /// tensor. Names with no real mapping (e.g. `e_score_correction_bias`)
    /// fall through to `dequantize`, which errors cleanly when the tensor is
    /// absent (the model handles that as an optional weight).
    fn load_real(&self, name: &str, dims: &[usize]) -> Result<Tensor> {
        match real_schema_target(name) {
            Some(RealTarget::GateUp { layer }) => self.assemble_gate_up(layer, dims),
            Some(RealTarget::Down { layer }) => self.assemble_down(layer, dims),
            Some(RealTarget::Single(real)) => self.dequantize(&real, dims),
            None => self.dequantize(name, dims),
        }
    }

    /// Assemble `gate_up_proj [e, 2*inter, hidden]` from the per-expert FP4
    /// `w1` (gate) and `w3` (up) weights concatenated along dim 0.
    fn assemble_gate_up(&self, layer: usize, dims: &[usize]) -> Result<Tensor> {
        let e = dims[0];
        let inter = dims[1] / 2;
        let hidden = dims[2];
        let mut rows = Vec::with_capacity(e);
        for i in 0..e {
            let p = format!("layers.{layer}.ffn.experts.{i}");
            let w1 = self.dequantize(&format!("{p}.w1.weight"), &[inter, hidden])?;
            let w3 = self.dequantize(&format!("{p}.w3.weight"), &[inter, hidden])?;
            rows.push(Tensor::cat(&[&w1, &w3], 0)?); // [2*inter, hidden]
        }
        Tensor::stack(&rows, 0)
    }

    /// Assemble `down_proj [e, hidden, inter]` from the per-expert FP4 `w2`.
    fn assemble_down(&self, layer: usize, dims: &[usize]) -> Result<Tensor> {
        let e = dims[0];
        let hidden = dims[1];
        let inter = dims[2];
        let mut rows = Vec::with_capacity(e);
        for i in 0..e {
            let p = format!("layers.{layer}.ffn.experts.{i}");
            rows.push(self.dequantize(&format!("{p}.w2.weight"), &[hidden, inter])?);
        }
        Tensor::stack(&rows, 0)
    }

    /// `dims` is the logical (model-requested) shape — for FP4 experts the
    /// on-disk tensor is a packed 1-D byte array, so the logical shape is
    /// required to know `n`/`m` and to reshape.
    pub fn load_weight(&self, name: &str, dims: &[usize]) -> Result<Tensor> {
        if let Some(t) = self.cache.lock().get(name) {
            return Ok(t);
        }
        let t = self.load_real(name, dims)?;
        self.cache.lock().insert(name, t.clone());
        Ok(t)
    }

    /// Finalize dtype: integer tensors (e.g. the hash router's `tid2eid`) are
    /// preserved verbatim; float tensors are converted to `out_dtype`.
    fn finalize_dtype(&self, t: Tensor, dims: &[usize]) -> Result<Tensor> {
        let t = t.reshape(dims)?;
        match t.dtype() {
            DType::I64 | DType::I32 | DType::I16 | DType::U32 | DType::U8 => Ok(t),
            _ => t.to_dtype(self.out_dtype),
        }
    }

    fn dequantize(&self, name: &str, dims: &[usize]) -> Result<Tensor> {
        let rank = dims.len();
        if rank < 2 {
            // 1-D plain weights (norms, sinks, hc base/scale, ...).
            let t = self.st.load(name, &self.cache_dev)?;
            return self.finalize_dtype(t, dims);
        }
        let rows: usize = dims[..rank - 1].iter().product();
        let cols = dims[rank - 1];
        let is_fp4 = self.is_fp4(name);
        let scale = self.real_scale_name(name);
        let block = if is_fp4 { self.fp4_block } else { self.block };
        let t = if is_fp4 {
            dequantize_fp4_linear(&self.st, name, &scale, rows, cols, block, &self.cache_dev)?
        } else if self.names.contains(&scale) {
            dequantize_fp8_linear(&self.st, name, &scale, rows, cols, block, &self.cache_dev)?
        } else {
            // Plain multi-dim weight (e.g. bf16/fp32 projection or norm).
            self.st.load(name, &self.cache_dev)?
        };
        self.finalize_dtype(t, dims)
    }

    /// Resident dequantized bytes and cached-tensor count (for tuning/tests).
    pub fn resident_bytes(&self) -> usize {
        self.cache.lock().resident_bytes()
    }

    /// Number of offload-cache evictions since construction (LRU budget drops).
    pub fn evictions(&self) -> usize {
        self.cache.lock().evictions()
    }

    pub fn cached_len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Resolve a weight onto the compute device (`dev`), keeping it resident in
    /// the GPU-bounded LRU so recently used layers stay on the GPU instead of
    /// being re-transferred/re-dequantized on reuse. Falls back to the CPU
    /// offload cache (dequantize on demand) for a cold miss.
    fn gpu_weight(&self, name: &str, dims: &[usize], dev: &Device, dtype: DType) -> Result<Tensor> {
        if let Some(t) = self.gpu_cache.lock().get(name) {
            return t.to_dtype(dtype);
        }
        let t = self
            .load_weight(name, dims)?
            .to_device(dev)?
            .to_dtype(dtype)?;
        self.gpu_cache.lock().insert(name, t.clone());
        Ok(t)
    }

    /// Resident bytes held on the compute device (GPU) by the layer cache.
    pub fn gpu_resident_bytes(&self) -> usize {
        self.gpu_cache.lock().resident_bytes()
    }

    /// Number of distinct GPU-resident tensors currently cached.
    pub fn gpu_cached_len(&self) -> usize {
        self.gpu_cache.lock().len()
    }

    /// Number of GPU-cache evictions since construction (LRU budget drops).
    pub fn gpu_evictions(&self) -> usize {
        self.gpu_cache.lock().evictions()
    }

    /// Build the eager [`DeepseekV4ForCausalLM`] from this loader's on-demand
    /// dequantized weights (through a `VarBuilder`/`SimpleBackend` adapter).
    pub fn load_model(
        &self,
        cfg: &DeepseekV4Config,
        use_flash_attn: bool,
    ) -> Result<DeepseekV4ForCausalLM> {
        let backend = QuantizedBackend { loader: self };
        let vb = VarBuilder::new_with_args(Box::new(backend), self.out_dtype, &self.dev);
        DeepseekV4ForCausalLM::new(cfg, use_flash_attn, vb)
    }

    /// Streaming prefill: build only the shared root weights (embedding, final
    /// norm, mHC head, lm_head) plus exactly ONE decoder layer at a time, run
    /// that layer's forward, then drop its GPU tensors before the next layer's
    /// weights are loaded. This keeps peak VRAM to ~one layer's weights + the
    /// activations (instead of all 43 layers materialized eagerly), which is
    /// what lets the ~146 GiB real `DeepSeek-V4-Flash` checkpoint run on a 96 GB
    /// GPU. All layer types (Sliding / CSA / HCA) and the per-layer DSA
    /// indexer/compressor state are handled by reusing the same
    /// [`DeepseekV4DecoderLayer`] the eager forward uses, so semantics match the
    /// eager path exactly. Returns `[B, S, vocab]` logits.
    pub fn forward_real(
        &self,
        cfg: &DeepseekV4Config,
        use_flash_attn: bool,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dev = &self.dev;
        let out_dtype = self.out_dtype;

        // Shared root weights — resident for the whole forward (small).
        let root =
            VarBuilder::new_with_args(Box::new(QuantizedBackend { loader: self }), out_dtype, dev);
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            root.pp("model").pp("embed_tokens"),
        )?;
        let norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            root.pp("model").pp("norm"),
        )?;
        let hc_head = DeepseekV4HyperHead::new(cfg, root.pp("model").pp("hc_head"))?;

        let (bs, seq) = input_ids.dims2()?;
        let emb = embed_tokens.forward(input_ids)?; // [B,S,D]
        let mut hidden = emb
            .unsqueeze(2)?
            .broadcast_as((bs, seq, cfg.hc_mult, cfg.hidden_size))?
            .contiguous()?; // [B,S,hc_mult,D]

        for i in 0..cfg.num_hidden_layers {
            let layer_vb = VarBuilder::new_with_args(
                Box::new(LayerBackend {
                    loader: self,
                    layer: i,
                }),
                out_dtype,
                dev,
            );
            let mut layer = DeepseekV4DecoderLayer::new(cfg, i, use_flash_attn, layer_vb)?;
            hidden = layer.forward(&hidden, Some(input_ids), seqlen_offset)?;
            // Layer GPU weights are kept in the size-bounded GPU LRU
            // (`gpu_cache`, up to `gpu_max_bytes`) so recently used layers stay
            // resident on the GPU instead of being re-transferred on reuse;
            // tensors beyond the budget are evicted (LRU), freeing VRAM. The
            // CPU dequantized weights stay in the size-bounded offload cache.
            drop(layer);
        }

        let collapsed = hc_head.forward(&hidden)?; // [B,S,D]
        let out = norm.forward(&collapsed)?;

        let head_vb =
            VarBuilder::new_with_args(Box::new(QuantizedBackend { loader: self }), out_dtype, dev);
        let lm_head =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, head_vb.pp("lm_head"))?;
        lm_head.forward(&out)
    }
}

/// `SimpleBackend` adapter that lets the eager [`DeepseekV4ForCausalLM`] build
/// itself from on-demand dequantized weights through a plain `VarBuilder`.
struct QuantizedBackend<'a> {
    loader: &'a DeepseekV4Quantized,
}

impl SimpleBackend for QuantizedBackend<'_> {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let dims = s.dims().to_vec();
        let t = self.loader.gpu_weight(name, &dims, dev, dtype)?;
        if t.shape() != &s {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        // No requested shape (used for plain tensors such as the hash-router's
        // `tid2eid`); fall back to the on-disk shape.
        let dims = match real_schema_target(name) {
            Some(RealTarget::Single(real)) => self.loader.st.get(&real)?.shape().to_vec(),
            _ => self.loader.st.get(name)?.shape().to_vec(),
        };
        self.loader.gpu_weight(name, &dims, dev, dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.loader.contains(name)
    }
}

/// Per-layer `SimpleBackend`: serves only `model.layers.<i>.<name>` through the
/// real loader, so a streaming forward can construct exactly one decoder layer
/// at a time (its GPU weights are dropped before the next layer is loaded),
/// keeping peak VRAM to ~one layer + activations instead of all 43 layers.
struct LayerBackend<'a> {
    loader: &'a DeepseekV4Quantized,
    layer: usize,
}

impl LayerBackend<'_> {
    fn full_name(&self, name: &str) -> String {
        format!("model.layers.{}.{}", self.layer, name)
    }
}

impl SimpleBackend for LayerBackend<'_> {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let full = self.full_name(name);
        let dims = s.dims().to_vec();
        let t = self.loader.gpu_weight(&full, &dims, dev, dtype)?;
        if t.shape() != &s {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {full}"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let full = self.full_name(name);
        let dims = match real_schema_target(&full) {
            Some(RealTarget::Single(real)) => self.loader.st.get(&real)?.shape().to_vec(),
            _ => self.loader.st.get(&full)?.shape().to_vec(),
        };
        self.loader.gpu_weight(&full, &dims, dev, dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.loader.contains(&self.full_name(name))
    }
}

/// `block` is the fp8 dequant block (real V4-Flash `(128, 128)`); the FP4
/// expert dequant block is fixed at the real per-row `gran_k = 32` `(1, 32)`
/// layout. Only the real `deepseek-ai/DeepSeek-V4-Flash` flat-name schema is
/// supported (name remap + per-expert FP4 assembly). `offload_budget_bytes`
/// caps the resident dequantized weight cache; the rest of the checkpoint
/// stays mmap-backed on disk.
///
/// # Safety
///
/// Inherited from mmap'ing the shard files.
#[allow(clippy::too_many_arguments)]
pub unsafe fn load_quantized_for_causal_lm(
    cfg: &DeepseekV4Config,
    use_flash_attn: bool,
    paths: &[impl AsRef<Path>],
    dev: &Device,
    out_dtype: DType,
    block: (usize, usize),
    offload_budget_bytes: usize,
    gpu_max_bytes: usize,
) -> Result<DeepseekV4ForCausalLM> {
    let loader = DeepseekV4Quantized::new(
        paths,
        block,
        dev.clone(),
        out_dtype,
        offload_budget_bytes,
        gpu_max_bytes,
    )?;
    loader.load_model(cfg, use_flash_attn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deepseek_v4::{DeepseekV4Config, DeepseekV4ForCausalLM};

    fn close(a: f32, b: f32, tol: f32, label: &str) {
        let ok = (a.is_nan() && b.is_nan())
            || (a == b)
            || (a - b).abs() < tol
            || (a.is_infinite() && a == b);
        assert!(ok, "{label}: got {a}, expected {b}");
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len(), "{label}: length");
        for (x, y) in a.iter().zip(b.iter()) {
            close(*x, *y, tol, label);
        }
    }

    #[test]
    fn e8m0_scale_values() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(134), 128.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
    }

    #[test]
    fn fp8_e4m3_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5); // exp 7, man 4
        assert_eq!(fp8_e4m3_to_f32(0x3F), 1.875); // exp 7, man 7
        assert_eq!(fp8_e4m3_to_f32(0x78), 256.0); // exp 15, man 0 (finite tail)
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0); // exp 15, man 6 (E4M3 max)
        assert!(fp8_e4m3_to_f32(0x7F).is_nan()); // only NaN pattern
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9)); // subnormal
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0); // negative zero
        assert_eq!(fp8_e4m3_to_f32(0xC0), -2.0); // negative normal
    }

    #[test]
    fn fp8_e4m3_matches_float8_for_all_bits() {
        // Authoritative cross-check: every one of the 256 E4M3 bit patterns must
        // decode identically to the `float8` crate that candle itself uses.
        for b in 0u8..=255 {
            let mine = fp8_e4m3_to_f32(b);
            let refv = float8::F8E4M3::from_bits(b).to_f64() as f32;
            close(mine, refv, 1e-6, &format!("fp8 {b:#04x}"));
        }
    }

    #[test]
    fn fp4_e2m1_values() {
        // Canonical FP4 E2M1 value table (matches the transformers
        // `finegrained_fp8._FP4_E2M1_LUT` reference):
        //   (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6).
        // E2M1 has no inf/NaN pattern: the max exponent e=3 is finite (4/6).
        let lut = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, //
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (nib, &want) in lut.iter().enumerate() {
            let got = fp4_e2m1_to_f32(nib as u8);
            assert_eq!(got, want, "fp4 nibble 0x{nib:X}");
            assert!(got.is_finite(), "fp4 nibble 0x{nib:X} must be finite");
        }
    }

    #[test]
    fn fp8_block_dequant() {
        let weight = [
            0x3C, 0x3F, 0x80, 0xC0, //
            0x3C, 0x3F, 0x80, 0xC0, //
            0x00, 0x01, 0x78, 0x7E, //
            0x3C, 0x3F, 0x80, 0xC0,
        ];
        let scale = [127, 128, 126, 134]; // 2x2 grid: 1.0, 2.0, 0.5, 128.0
        let out = dequantize_fp8_block_scale(&weight, &scale, 4, 4, (2, 2)).unwrap();
        let expected = [
            1.5,
            1.875,
            -0.0,
            -4.0, //
            1.5,
            1.875,
            -0.0,
            -4.0, //
            0.0,
            0.0009765625,
            32768.0,
            57344.0, //
            0.75,
            0.9375,
            -0.0,
            -256.0,
        ];
        assert_eq!(out.len(), 16);
        for (i, (a, b)) in out.iter().zip(expected).enumerate() {
            close(*a, b, 1e-5, &format!("fp8 dequant[{i}]"));
        }
    }

    #[test]
    fn fp4_block_dequant() {
        // 16 nibbles row-major: idx = i*4+j.
        // [0x2,0x3,0x4,0x5, 0xA,0xB,0x6,0x7, 0x1,0x2,0x3,0x4, 0x5,0x6,0x0,0x0]
        let packed = [0x32, 0x54, 0xBA, 0x76, 0x21, 0x43, 0x65, 0x00];
        let scale = [127, 128, 126, 134]; // 1.0, 2.0, 0.5, 128.0
        let out = dequantize_fp4_block_scale(&packed, &scale, 4, 4, (2, 2)).unwrap();
        let expected = [
            1.0, 1.5, 4.0, 6.0, //
            -1.0, -1.5, 8.0, 12.0, //
            0.25, 0.5, 192.0, 256.0, //
            1.5, 2.0, 0.0, 0.0,
        ];
        assert_eq!(out.len(), 16);
        for (i, (a, b)) in out.iter().zip(expected).enumerate() {
            close(*a, b, 1e-5, &format!("fp4 dequant[{i}]"));
        }
    }

    #[test]
    fn dequant_rejects_mismatched_sizes() {
        let w = [0x3C; 8];
        let s = [127; 1];
        assert!(dequantize_fp8_block_scale(&w, &s, 4, 4, (2, 2)).is_err());
        assert!(dequantize_fp8_block_scale(&w, &[127; 4], 4, 4, (3, 3)).is_err());
        let p = [0x32; 4];
        assert!(dequantize_fp4_block_scale(&p, &[127; 4], 4, 4, (2, 2)).is_err());
    }

    /// Deterministic pseudo-random f32 values in [-1, 1].
    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Quantize f32 values to FP8 E4M3 with `block`-sized ue8m0 scales.
    /// Returns (e4m3 bytes, ue8m0 scale bytes) laid out as the loader expects.
    fn quant_fp8(vals: &[f32], n: usize, m: usize, block: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
        let (bn, bm) = block;
        let (gn, gm) = (n / bn, m / bm);
        let mut w = vec![0u8; n * m];
        let mut s = vec![0u8; gn * gm];
        for gi in 0..gn {
            for gj in 0..gm {
                let mut maxabs = 0f32;
                for i in gi * bn..(gi + 1) * bn {
                    for j in gj * bm..(gj + 1) * bm {
                        maxabs = maxabs.max(vals[i * m + j].abs());
                    }
                }
                let exp = if maxabs == 0.0 {
                    0
                } else {
                    (maxabs / 448.0).log2().ceil() as i32
                };
                let exp = exp.clamp(-127, 127);
                let sf = 2f32.powi(exp);
                s[gi * gm + gj] = (exp + 127) as u8;
                for i in gi * bn..(gi + 1) * bn {
                    for j in gj * bm..(gj + 1) * bm {
                        w[i * m + j] =
                            float8::F8E4M3::from_f64((vals[i * m + j] / sf) as f64).to_bits();
                    }
                }
            }
        }
        (w, s)
    }

    /// Round a positive magnitude to the nearest FP4 E2M1 representable value.
    fn fp4_nibble(x: f32) -> u8 {
        // magnitudes: 0, 0.5, 1, 1.5, 2, 3
        let reps = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0];
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for (k, &r) in reps.iter().enumerate() {
            let d = (x - r).abs();
            if d < best_d {
                best_d = d;
                best = k as u8;
            }
        }
        best
    }

    /// Quantize f32 values to FP4 E2M1 with `block` scales, packing 2 nibbles/byte
    /// (low nibble first), matching the loader's `dequantize_fp4_block_scale`.
    fn quant_fp4(vals: &[f32], n: usize, m: usize, block: (usize, usize)) -> (Vec<u8>, Vec<u8>) {
        let (bn, bm) = block;
        let (gn, gm) = (n / bn, m / bm);
        let mut w = vec![0u8; n * m / 2];
        let mut s = vec![0u8; gn * gm];
        for gi in 0..gn {
            for gj in 0..gm {
                let mut maxabs = 0f32;
                for i in gi * bn..(gi + 1) * bn {
                    for j in gj * bm..(gj + 1) * bm {
                        maxabs = maxabs.max(vals[i * m + j].abs());
                    }
                }
                let exp = if maxabs == 0.0 {
                    0
                } else {
                    (maxabs / 3.0).log2().ceil() as i32
                };
                let exp = exp.clamp(-127, 127);
                let sf = 2f32.powi(exp);
                s[gi * gm + gj] = (exp + 127) as u8;
                for i in gi * bn..(gi + 1) * bn {
                    for j in gj * bm..(gj + 1) * bm {
                        let idx = i * m + j;
                        let x = vals[idx] / sf;
                        let mag = fp4_nibble(x.abs());
                        let sign = if x < 0.0 { 0x8 } else { 0x0 };
                        let nib = (mag & 0x7) | sign;
                        if idx.is_multiple_of(2) {
                            w[idx / 2] |= nib;
                        } else {
                            w[idx / 2] |= nib << 4;
                        }
                    }
                }
            }
        }
        (w, s)
    }

    /// Write a safetensors file from (name, dtype, shape, data) entries.
    fn write_st(path: &Path, tensors: &[(String, String, Vec<usize>, Vec<u8>)]) {
        let mut sorted: Vec<_> = tensors.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut obj = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in sorted {
            let start = data.len();
            let end = start + bytes.len();
            let mut t = serde_json::Map::new();
            t.insert("dtype".into(), serde_json::Value::String(dtype.clone()));
            t.insert(
                "shape".into(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|d| serde_json::Value::from(*d as u64))
                        .collect(),
                ),
            );
            t.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![
                    serde_json::Value::from(start as u64),
                    serde_json::Value::from(end as u64),
                ]),
            );
            obj.insert(name.clone(), serde_json::Value::Object(t));
            data.extend_from_slice(bytes);
        }
        let header = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap();
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&data);
        std::fs::write(path, &buf).unwrap();
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ---- Real-schema (deepseek-ai/DeepSeek-V4-Flash) tests ----

    /// Real V4-Flash name remap: candle model weight names -> on-disk flat
    /// names (mirrors the checkpoint's `model.safetensors.index.json`).
    #[test]
    fn real_schema_name_remap() {
        fn single(name: &str, expected: &str) {
            match real_schema_target(name) {
                Some(RealTarget::Single(t)) => assert_eq!(t, expected, "{name}"),
                other => panic!("{name}: expected Single({expected}), got {other:?}"),
            }
        }
        single("model.embed_tokens.weight", "embed.weight");
        single("model.norm.weight", "norm.weight");
        single("lm_head.weight", "head.weight");
        single("model.hc_head.hc_fn", "hc_head_fn");
        single("model.hc_head.hc_base", "hc_head_base");
        single("model.hc_head.hc_scale", "hc_head_scale");
        single(
            "model.layers.0.self_attn.q_a_proj.weight",
            "layers.0.attn.wq_a.weight",
        );
        single(
            "model.layers.7.self_attn.kv_proj.weight",
            "layers.7.attn.wkv.weight",
        );
        single(
            "model.layers.42.self_attn.o_b_proj.weight",
            "layers.42.attn.wo_b.weight",
        );
        single("model.layers.3.self_attn.sinks", "layers.3.attn.attn_sink");
        single(
            "model.layers.5.input_layernorm.weight",
            "layers.5.attn_norm.weight",
        );
        single(
            "model.layers.5.post_attention_layernorm.weight",
            "layers.5.ffn_norm.weight",
        );
        single("model.layers.1.mlp.gate.weight", "layers.1.ffn.gate.weight");
        single(
            "model.layers.1.mlp.gate.tid2eid",
            "layers.1.ffn.gate.tid2eid",
        );
        single(
            "model.layers.9.mlp.shared_experts.gate_proj.weight",
            "layers.9.ffn.shared_experts.w1.weight",
        );
        single(
            "model.layers.9.mlp.shared_experts.up_proj.weight",
            "layers.9.ffn.shared_experts.w3.weight",
        );
        single(
            "model.layers.9.mlp.shared_experts.down_proj.weight",
            "layers.9.ffn.shared_experts.w2.weight",
        );
        single("model.layers.2.attn_hc.fn", "layers.2.hc_attn_fn");
        single("model.layers.2.ffn_hc.scale", "layers.2.hc_ffn_scale");
        // Compressor / indexer (CSA / HCA layers).
        single(
            "model.layers.10.self_attn.compressor.kv_proj.weight",
            "layers.10.attn.compressor.wkv.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.gate_proj.weight",
            "layers.10.attn.compressor.wgate.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.position_bias",
            "layers.10.attn.compressor.ape",
        );
        single(
            "model.layers.10.self_attn.compressor.kv_norm.weight",
            "layers.10.attn.compressor.norm.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.kv_proj.weight",
            "layers.10.attn.indexer.compressor.wkv.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.gate_proj.weight",
            "layers.10.attn.indexer.compressor.wgate.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.position_bias",
            "layers.10.attn.indexer.compressor.ape",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.kv_norm.weight",
            "layers.10.attn.indexer.compressor.norm.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.q_b_proj.weight",
            "layers.10.attn.indexer.wq_b.weight",
        );
        single(
            "model.layers.10.self_attn.compressor.indexer.scorer.weights_proj.weight",
            "layers.10.attn.indexer.weights_proj.weight",
        );
        // Assembled experts.
        assert!(matches!(
            real_schema_target("model.layers.4.mlp.experts.gate_up_proj"),
            Some(RealTarget::GateUp { layer: 4 })
        ));
        assert!(matches!(
            real_schema_target("model.layers.4.mlp.experts.down_proj"),
            Some(RealTarget::Down { layer: 4 })
        ));
        // Optional weight absent from the real checkpoint -> None.
        assert!(real_schema_target("model.layers.4.mlp.gate.e_score_correction_bias").is_none());
    }

    /// Real V4-Flash fp4 expert dequant at the per-row `gran_k = 32` block scale
    /// (`[n, m/32]`), hand-computed from the checkpoint layout.
    #[test]
    fn real_fp4_per_row_gran32_dequant() {
        // 2 rows x 8 columns with block (1,8): one ue8m0 scale per row covering
        // all 8 columns (mirrors the real gran_k=32 mapping on a small tile).
        // packed = 8 bytes, low-nibble-first along columns.
        let packed = [0x32, 0x54, 0xBA, 0x21, 0x21, 0x43, 0x15, 0x32];
        let scale = [127u8, 128u8]; // row0: 1.0, row1: 2.0
        let out = dequantize_fp4_block_scale(&packed, &scale, 2, 8, (1, 8)).unwrap();
        let expected = [
            // row0 (scale 1.0)
            1.0, 1.5, 2.0, 3.0, -1.0, -1.5, 0.5, 1.0, // row1 (scale 2.0)
            1.0, 2.0, 3.0, 4.0, 6.0, 1.0, 2.0, 3.0,
        ];
        assert_eq!(out.len(), 16);
        for (i, (a, b)) in out.iter().zip(expected).enumerate() {
            close(*a, b, 1e-5, &format!("real fp4[{i}]"));
        }
    }

    /// A real-schema-shaped synthetic checkpoint (flat names) loaded through
    /// `load_quantized_for_causal_lm`: exercises the name remap, per-expert FP4
    /// w1/w2/w3 assembly into stacked 3-D gate_up/down_proj, fp8 block dequant,
    /// the hash router's I64 `tid2eid`, shared experts and mHC tensors.
    fn real_synth_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 256, "hidden_size": 128, "moe_intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 128, "q_lora_rank": 128, "o_lora_rank": 128, "qk_rope_head_dim": 64,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 1, "num_hash_layers": 1,
                "index_head_dim": 64, "index_n_heads": 2, "index_topk": 8, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "sliding_window": 128,
                "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "compress_rope_theta": 160000.0, "attention_bias": false, "attention_dropout": 0.0,
                "swiglu_limit": 10.0, "initializer_range": 0.02, "use_cache": true,
                "bos_token_id": 0, "eos_token_id": 1, "tie_word_embeddings": false,
                "compress_ratios": [0, 0],
                "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                                 "original_max_position_embeddings": 65536, "type": "yarn"}
            }"#,
        )
        .unwrap()
    }

    fn i64_bytes(vals: &[i64]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Build the real-schema flat-named shards for `cfg` on CPU. Returns the
    /// shard paths (and an unquantized name->Tensor map for cross-checks).
    fn write_real_synth_shards(cfg: &DeepseekV4Config) -> Vec<std::path::PathBuf> {
        let h = cfg.hidden_size;
        let d = cfg.head_dim;
        let inter = cfg.moe_intermediate_size;
        let e = cfg.n_routed_experts;
        let mix = (2 + cfg.hc_mult) * cfg.hc_mult;
        let mut shards: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        let seed = 0xBEEF_1234u64;
        // (name, kind, logical shape). kind: 0=plain f32/bf16, 1=fp8, 2=fp4-packed, 3=i64
        let mut plan: Vec<(String, u8, Vec<usize>)> = Vec::new();
        let push = |name: String, kind: u8, shape: Vec<usize>, s: &mut Vec<_>| {
            s.push((name, kind, shape));
        };
        push("embed.weight".into(), 0, vec![cfg.vocab_size, h], &mut plan);
        push("head.weight".into(), 0, vec![cfg.vocab_size, h], &mut plan);
        push("norm.weight".into(), 0, vec![h], &mut plan);
        push(
            "hc_head_fn".into(),
            0,
            vec![cfg.hc_mult, cfg.hc_mult * h],
            &mut plan,
        );
        push("hc_head_base".into(), 0, vec![cfg.hc_mult], &mut plan);
        push("hc_head_scale".into(), 0, vec![1], &mut plan);
        for li in 0..cfg.num_hidden_layers {
            let l = format!("layers.{li}");
            // attention (fp8)
            for (k, shape) in [
                ("attn.wq_a.weight", vec![cfg.q_lora_rank, h]),
                (
                    "attn.wq_b.weight",
                    vec![cfg.num_attention_heads * d, cfg.q_lora_rank],
                ),
                ("attn.wkv.weight", vec![d, h]),
                (
                    "attn.wo_a.weight",
                    vec![
                        cfg.o_groups * cfg.o_lora_rank,
                        cfg.num_attention_heads * d / cfg.o_groups,
                    ],
                ),
                ("attn.wo_b.weight", vec![h, cfg.o_groups * cfg.o_lora_rank]),
            ] {
                push(format!("{l}.{k}"), 1, shape, &mut plan);
            }
            push(
                format!("{l}.attn.q_norm.weight"),
                0,
                vec![cfg.q_lora_rank],
                &mut plan,
            );
            push(format!("{l}.attn.kv_norm.weight"), 0, vec![d], &mut plan);
            push(
                format!("{l}.attn.attn_sink"),
                0,
                vec![cfg.num_attention_heads],
                &mut plan,
            );
            push(format!("{l}.attn_norm.weight"), 0, vec![h], &mut plan);
            push(format!("{l}.ffn_norm.weight"), 0, vec![h], &mut plan);
            push(format!("{l}.ffn.gate.weight"), 0, vec![e, h], &mut plan);
            push(
                format!("{l}.ffn.gate.tid2eid"),
                3,
                vec![cfg.vocab_size, cfg.num_experts_per_tok],
                &mut plan,
            );
            for ei in 0..e {
                for w in ["w1", "w3"] {
                    push(
                        format!("{l}.ffn.experts.{ei}.{w}.weight"),
                        2,
                        vec![inter, h],
                        &mut plan,
                    );
                }
                push(
                    format!("{l}.ffn.experts.{ei}.w2.weight"),
                    2,
                    vec![h, inter],
                    &mut plan,
                );
            }
            push(
                format!("{l}.ffn.shared_experts.w1.weight"),
                1,
                vec![inter, h],
                &mut plan,
            );
            push(
                format!("{l}.ffn.shared_experts.w3.weight"),
                1,
                vec![inter, h],
                &mut plan,
            );
            push(
                format!("{l}.ffn.shared_experts.w2.weight"),
                1,
                vec![h, inter],
                &mut plan,
            );
            push(
                format!("{l}.hc_attn_fn"),
                0,
                vec![mix, cfg.hc_mult * h],
                &mut plan,
            );
            push(format!("{l}.hc_attn_base"), 0, vec![mix], &mut plan);
            push(format!("{l}.hc_attn_scale"), 0, vec![3], &mut plan);
            push(
                format!("{l}.hc_ffn_fn"),
                0,
                vec![mix, cfg.hc_mult * h],
                &mut plan,
            );
            push(format!("{l}.hc_ffn_base"), 0, vec![mix], &mut plan);
            push(format!("{l}.hc_ffn_scale"), 0, vec![3], &mut plan);
        }

        for (wi, (name, kind, shape)) in plan.into_iter().enumerate() {
            let n_elems: usize = shape.iter().product();
            let vals = lcg(seed.wrapping_add(wi as u64 * 104729), n_elems);
            let rank = shape.len();
            let (rows, cols) = (shape[..rank - 1].iter().product(), shape[rank - 1]);
            match kind {
                0 => {
                    // plain weights (embed/head/norm/gate/hc/...): store as F32 so
                    // the byte length matches the f32 payload; the loader reads the
                    // header dtype and converts to out_dtype.
                    let b = f32_bytes(&vals);
                    shards.push((name, "F32".to_string(), shape, b));
                }
                1 => {
                    // fp8 E4M3 + ue8m0 scale at (128,128)
                    let (wb, sb) = quant_fp8(&vals, rows, cols, (128, 128));
                    let gn = rows / 128 * (cols / 128);
                    shards.push((name.clone(), "F8_E4M3".to_string(), shape, wb));
                    let sname = if let Some(b) = name.strip_suffix(".weight") {
                        format!("{b}.scale")
                    } else {
                        format!("{name}.scale")
                    };
                    shards.push((sname, "F8_E8M0".to_string(), vec![gn], sb));
                }
                2 => {
                    // fp4 E2M1 per-row gran_k=32: packed [n, m/2], scale [n, m/32]
                    let (wb, sb) = quant_fp4(&vals, rows, cols, (1, 32));
                    shards.push((name.clone(), "I8".to_string(), vec![rows * cols / 2], wb));
                    let sname = if let Some(b) = name.strip_suffix(".weight") {
                        format!("{b}.scale")
                    } else {
                        format!("{name}.scale")
                    };
                    shards.push((sname, "F8_E8M0".to_string(), vec![rows * (cols / 32)], sb));
                }
                3 => {
                    // I64 tid2eid: indices into [0, e)
                    let idx: Vec<i64> = vals
                        .iter()
                        .map(|x| ((x.abs() * 1e3) as usize % e) as i64)
                        .collect();
                    shards.push((name, "I64".to_string(), shape, i64_bytes(&idx)));
                }
                _ => unreachable!(),
            }
        }

        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("v4q_real_{}_{n}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("model.safetensors");
        write_st(&p, &shards);
        vec![p]
    }

    #[test]
    fn deepseek_v4_real_schema_load_forward() -> candle::Result<()> {
        let cfg = real_synth_config();
        let dev = Device::Cpu;
        let paths = write_real_synth_shards(&cfg);
        let budget = 1 << 28; // 256 MiB, enough to hold the tiny model resident

        let ids = Tensor::new(&[1u32, 3, 7, 2, 5][..], &dev)?.unsqueeze(0)?;
        let block = (128, 128);
        let mut model = unsafe {
            load_quantized_for_causal_lm(
                &cfg,
                false,
                &paths,
                &dev,
                DType::F32,
                block,
                budget,
                budget,
            )?
        };
        let logits = model.forward(&ids, 0)?;
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size], "logits shape");
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "real-schema logits finite"
        );

        // Determinism: a second load + forward matches exactly.
        let mut m2 = unsafe {
            load_quantized_for_causal_lm(
                &cfg,
                false,
                &paths,
                &dev,
                DType::F32,
                block,
                budget,
                budget,
            )?
        };
        let logits2 = m2.forward(&ids, 0)?;
        assert_close(&logits, &logits2, 0.0, "real-schema deterministic");

        let _ = std::fs::remove_dir_all(paths[0].parent().unwrap());
        Ok(())
    }

    /// The per-layer streaming forward (`forward_real`) must produce exactly the
    /// same logits as the eager `load_model` path (same dequant + same forward,
    /// one layer at a time). This is the CPU-level guarantee that streaming does
    /// not change semantics; the GPU run then just changes *where* the weights
    /// live, not what is computed.
    #[test]
    fn deepseek_v4_streaming_forward_matches_eager() -> candle::Result<()> {
        let cfg = real_synth_config();
        let dev = Device::Cpu;
        let paths = write_real_synth_shards(&cfg);
        let budget = 1 << 28;

        let ids = Tensor::new(&[1u32, 3, 7, 2, 5][..], &dev)?.unsqueeze(0)?;

        let loader = unsafe {
            DeepseekV4Quantized::new_real(&paths, dev.clone(), DType::F32, budget, budget)?
        };
        // Streaming path: only one layer's weights resident at a time.
        let logits_stream = loader.forward_real(&cfg, false, &ids, 0)?;
        assert_eq!(
            logits_stream.dims(),
            &[1, 5, cfg.vocab_size],
            "streaming logits shape"
        );
        let flat = logits_stream.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "streaming logits finite"
        );

        // Eager path: all layers resident at once.
        let mut eager = loader.load_model(&cfg, false)?;
        let logits_eager = eager.forward(&ids, 0)?;

        assert_close(
            &logits_stream,
            &logits_eager,
            0.0,
            "streaming vs eager identical",
        );

        // Determinism: a second streaming forward matches bit-for-bit.
        let logits_stream2 = loader.forward_real(&cfg, false, &ids, 0)?;
        assert_close(
            &logits_stream,
            &logits_stream2,
            0.0,
            "streaming deterministic",
        );

        let _ = std::fs::remove_dir_all(paths[0].parent().unwrap());
        Ok(())
    }
}
