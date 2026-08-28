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
use candle_nn::{Init, VarBuilder};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Mutex;

use super::{DeepseekV4Config, DeepseekV4ForCausalLM};
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

/// FP4 E2M1 nibble -> f32: sign(1) exp(2) mantissa(1), exponent bias 1.
///
/// `(-1)^s * 2^(e-1) * (1 + m/2)` for `e` in `1..=2`, `(-1)^s * m/2` for the
/// subnormal `e == 0`, and `inf`/`NaN` for `e == 3`.
#[inline]
pub fn fp4_e2m1_to_f32(nib: u8) -> f32 {
    let sign = if nib & 0x8 != 0 { -1.0 } else { 1.0 };
    let exp = (nib >> 1) & 0x3;
    let man = (nib & 0x1) as f32;
    let abs = match exp {
        0 => man * 0.5,
        3 => {
            if man == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
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
pub fn dequantize_fp8_block_scale(
    weight: &[u8],
    scale: &[u8],
    n: usize,
    m: usize,
    block: (usize, usize),
) -> Result<Vec<f32>> {
    let (bn, bm) = block;
    let gn = n / bn;
    let gm = m / bm;
    if bn == 0 || bm == 0 || gn == 0 || gm == 0 || !n.is_multiple_of(bn) || !m.is_multiple_of(bm) {
        candle::bail!("fp8 block {block:?} does not tile weight {n}x{m}");
    }
    if weight.len() != n * m {
        candle::bail!("fp8 weight len {} != {n}*{m}", weight.len());
    }
    if scale.len() != gn * gm {
        candle::bail!(
            "fp8 scale len {} != ({n}/{bn})*({m}/{bm}) = {}",
            scale.len(),
            gn * gm
        );
    }
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        let srow = (i / bn) * gm;
        for j in 0..m {
            let sc = e8m0_to_f32(scale[srow + j / bm]);
            out.push(fp8_e4m3_to_f32(weight[i * m + j]) * sc);
        }
    }
    Ok(out)
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
    let (bn, bm) = block;
    let gn = n / bn;
    let gm = m / bm;
    if bn == 0 || bm == 0 || gn == 0 || gm == 0 || !n.is_multiple_of(bn) || !m.is_multiple_of(bm) {
        candle::bail!("fp4 block {block:?} does not tile weight {n}x{m}");
    }
    if packed.len() != n * m / 2 {
        candle::bail!("fp4 packed len {} != {n}*{m}/2", packed.len());
    }
    if scale.len() != gn * gm {
        candle::bail!(
            "fp4 scale len {} != ({n}/{bn})*({m}/{bm}) = {}",
            scale.len(),
            gn * gm
        );
    }
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        let srow = (i / bn) * gm;
        for j in 0..m {
            let idx = i * m + j;
            let byte = packed[idx / 2];
            let nib = if idx.is_multiple_of(2) {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let sc = e8m0_to_f32(scale[srow + j / bm]);
            out.push(fp4_e2m1_to_f32(nib) * sc);
        }
    }
    Ok(out)
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

/// On-demand, size-bounded loader for offload.
///
/// Backs tensors with an mmap (disk-backed, no RAM cost until touched) and
/// materializes them one at a time on CPU/device, caching only the most recent
/// layers and evicting the least-recently-used tensors once the byte budget is
/// exceeded. This lets a ~148 GiB fp8 checkpoint run on a 96 GB GPU by keeping
/// just the active layer's dequantized weights resident.
pub struct OffloadCache<'a> {
    st: &'a MmapedSafetensors,
    dev: Device,
    cache: HashMap<String, Tensor>,
    order: VecDeque<String>,
    current_bytes: usize,
    max_bytes: usize,
}

impl<'a> OffloadCache<'a> {
    /// Wrap the mmap shards; `max_bytes` caps the resident (dequantized) size.
    pub fn new(st: &'a MmapedSafetensors, dev: Device, max_bytes: usize) -> Self {
        Self {
            st,
            dev,
            cache: HashMap::new(),
            order: VecDeque::new(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn bytes_of(t: &Tensor) -> Result<usize> {
        Ok(t.elem_count() * t.dtype().size_in_bytes())
    }

    fn evict_lru(&mut self) {
        while self.current_bytes > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(t) = self.cache.remove(&oldest) {
                self.current_bytes -= Self::bytes_of(&t).unwrap_or(0);
            }
        }
    }

    fn touch(&mut self, name: &str) {
        if let Some(pos) = self.order.iter().position(|n| n == name) {
            self.order.remove(pos);
        }
        self.order.push_back(name.to_string());
    }

    /// Cache a materialized tensor under `name`, evicting LRU entries over budget.
    pub fn insert(&mut self, name: &str, t: Tensor) -> Result<()> {
        if let Some(prev) = self.cache.remove(name) {
            self.current_bytes -= Self::bytes_of(&prev)?;
            if let Some(pos) = self.order.iter().position(|n| n == name) {
                self.order.remove(pos);
            }
        }
        self.current_bytes += Self::bytes_of(&t)?;
        self.cache.insert(name.to_string(), t);
        self.order.push_back(name.to_string());
        self.evict_lru();
        Ok(())
    }

    /// Return the cached tensor for `name`, if still resident.
    pub fn get_cached(&self, name: &str) -> Option<&Tensor> {
        self.cache.get(name)
    }

    /// Load a raw (already-fp32/bf16) tensor on demand, caching it.
    pub fn load(&mut self, name: &str, dtype: DType) -> Result<Tensor> {
        if self.cache.contains_key(name) {
            let t = self.cache[name].clone();
            self.touch(name);
            return Ok(t);
        }
        let t = self.st.load(name, &self.dev)?.to_dtype(dtype)?;
        self.insert(name, t)?;
        Ok(self.cache[name].clone())
    }

    /// Load and dequantize a block-scaled FP8 linear, caching the f32 result.
    pub fn fp8_linear(
        &mut self,
        weight_name: &str,
        scale_name: &str,
        n: usize,
        m: usize,
        block: (usize, usize),
        out_dtype: DType,
    ) -> Result<Tensor> {
        if self.cache.contains_key(weight_name) {
            let t = self.cache[weight_name].clone();
            self.touch(weight_name);
            return Ok(t);
        }
        let t = dequantize_fp8_linear(self.st, weight_name, scale_name, n, m, block, &self.dev)?
            .to_dtype(out_dtype)?;
        self.insert(weight_name, t)?;
        Ok(self.cache[weight_name].clone())
    }

    /// Load and dequantize a block-scaled FP4 linear (expert), caching the result.
    pub fn fp4_linear(
        &mut self,
        weight_name: &str,
        scale_name: &str,
        n: usize,
        m: usize,
        block: (usize, usize),
        out_dtype: DType,
    ) -> Result<Tensor> {
        if self.cache.contains_key(weight_name) {
            let t = self.cache[weight_name].clone();
            self.touch(weight_name);
            return Ok(t);
        }
        let t = dequantize_fp4_linear(self.st, weight_name, scale_name, n, m, block, &self.dev)?
            .to_dtype(out_dtype)?;
        self.insert(weight_name, t)?;
        Ok(self.cache[weight_name].clone())
    }

    /// Resident size in bytes and number of cached tensors (for tests/tuning).
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Drop everything to free device memory between layers.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
        self.current_bytes = 0;
    }
}

/// Size-bounded LRU of dequantized/materialized weights — the runtime half of
/// [`OffloadCache`], made `&self`-safe so it can back a `VarBuilder`. Tensors
/// are evicted (least-recently-used first) once `max_bytes` is exceeded, so a
/// ~148 GiB fp8 checkpoint only ever holds up to the configured budget
/// resident (the rest stays mmap-backed on disk).
#[derive(Default)]
struct WeightCache {
    tensors: HashMap<String, Tensor>,
    order: VecDeque<String>,
    bytes: usize,
    max_bytes: usize,
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
            }
        }
    }

    fn resident_bytes(&self) -> usize {
        self.bytes
    }

    fn len(&self) -> usize {
        self.tensors.len()
    }
}

/// A full DeepSeek-V4 checkpoint opened mmap-backed across its safetensors
/// shards, with on-demand fp8/fp4 dequantization and a size-bounded LRU cache
/// (CPU/disk offload). Thread-safe (`&self` loads) so it can back a
/// `VarBuilder`/[`SimpleBackend`] used to construct the eager
/// [`DeepseekV4ForCausalLM`].
///
/// Real V4-Flash convention: fp8 E4M3 weights with ue8m0 block scales at
/// `weight_block_size = [128, 128]` (scale tensors named `<weight>_scale_inv`)
/// and fp4 E2M1 expert weights (`experts.gate_up_proj` / `experts.down_proj`).
pub struct DeepseekV4Quantized {
    st: MmapedSafetensors,
    names: HashSet<String>,
    block: (usize, usize),
    scale_suffix: String,
    fp4_prefixes: Vec<String>,
    dev: Device,
    out_dtype: DType,
    cache: Mutex<WeightCache>,
}

impl DeepseekV4Quantized {
    /// `paths` are the safetensors shards (any order; `multi` merges by name).
    /// `block` is the fp8/fp4 dequant block (real V4-Flash: `(128, 128)`).
    /// `scale_suffix` is appended to a weight name to find its scale tensor
    /// (real V4-Flash: `"_scale_inv"`). `fp4_prefixes` name the expert tensors
    /// stored as FP4 E2M1 (real V4-Flash: `experts.gate_up_proj`/`down_proj`).
    /// `max_bytes` caps the resident dequantized weight cache.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from mmap'ing the shard files.
    pub unsafe fn new(
        paths: &[impl AsRef<Path>],
        block: (usize, usize),
        scale_suffix: &str,
        fp4_prefixes: &[String],
        dev: Device,
        out_dtype: DType,
        max_bytes: usize,
    ) -> Result<Self> {
        let st = MmapedSafetensors::multi(paths)?;
        let names = st.tensors().into_iter().map(|(n, _)| n).collect();
        Ok(Self {
            st,
            names,
            block,
            scale_suffix: scale_suffix.to_string(),
            fp4_prefixes: fp4_prefixes.to_vec(),
            dev,
            out_dtype,
            cache: Mutex::new(WeightCache::new(max_bytes)),
        })
    }

    /// True if `name` is a tensor present in the checkpoint.
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    fn is_fp4(&self, name: &str) -> bool {
        self.fp4_prefixes.iter().any(|p| name.contains(p.as_str()))
    }

    fn scale_name(&self, name: &str) -> String {
        format!("{name}{}", self.scale_suffix)
    }

    /// `dims` is the logical (model-requested) shape — for FP4 experts the
    /// on-disk tensor is a packed 1-D byte array, so the logical shape is
    /// required to know `n`/`m` and to reshape.
    pub fn load_weight(&self, name: &str, dims: &[usize]) -> Result<Tensor> {
        if let Some(t) = self.cache.lock().unwrap().get(name) {
            return Ok(t);
        }
        let t = self.dequantize(name, dims)?;
        self.cache.lock().unwrap().insert(name, t.clone());
        Ok(t)
    }

    fn dequantize(&self, name: &str, dims: &[usize]) -> Result<Tensor> {
        let rank = dims.len();
        if rank < 2 {
            // 1-D plain weights (norms, sinks, hc base/scale, ...).
            return self.st.load(name, &self.dev)?.to_dtype(self.out_dtype);
        }
        let rows: usize = dims[..rank - 1].iter().product();
        let cols = dims[rank - 1];
        let scale = self.scale_name(name);
        let t = if self.is_fp4(name) {
            dequantize_fp4_linear(&self.st, name, &scale, rows, cols, self.block, &self.dev)?
        } else if self.names.contains(&scale) {
            dequantize_fp8_linear(&self.st, name, &scale, rows, cols, self.block, &self.dev)?
        } else {
            // Plain multi-dim weight (e.g. bf16/fp32 projection or norm).
            self.st.load(name, &self.dev)?
        };
        t.reshape(dims)?.to_dtype(self.out_dtype)
    }

    /// Resident dequantized bytes and cached-tensor count (for tuning/tests).
    pub fn resident_bytes(&self) -> usize {
        self.cache.lock().unwrap().resident_bytes()
    }

    pub fn cached_len(&self) -> usize {
        self.cache.lock().unwrap().len()
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
        super::DeepseekV4ForCausalLM::new(cfg, use_flash_attn, vb)
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
        let t = self.loader.load_weight(name, &dims)?;
        if t.shape() != &s {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt());
        }
        t.to_device(dev)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        // No requested shape (used for plain tensors such as the hash-router's
        // `tid2eid`); fall back to the on-disk shape.
        let dims = self.loader.st.get(name)?.shape().to_vec();
        self.loader
            .load_weight(name, &dims)?
            .to_device(dev)?
            .to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.loader.contains(name)
    }
}

/// Load a full [`DeepseekV4ForCausalLM`] from fp8/fp4 safetensors shards, using
/// [`DeepseekV4Quantized`] (mmap + on-demand dequant + bounded LRU offload) as
/// the weight source for the eager V4 assembly.
///
/// `block` is the fp8/fp4 dequant block (real V4-Flash `(128, 128)`),
/// `scale_suffix` the fp8 scale-tensor suffix (`"_scale_inv"`), and
/// `fp4_prefixes` the expert FP4 weight name prefixes.
/// `offload_budget_bytes` caps the resident dequantized weight cache; the rest
/// of the checkpoint stays mmap-backed on disk.
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
    scale_suffix: &str,
    fp4_prefixes: &[String],
    offload_budget_bytes: usize,
) -> Result<DeepseekV4ForCausalLM> {
    let loader = DeepseekV4Quantized::new(
        paths,
        block,
        scale_suffix,
        fp4_prefixes,
        dev.clone(),
        out_dtype,
        offload_budget_bytes,
    )?;
    loader.load_model(cfg, use_flash_attn)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(fp4_e2m1_to_f32(0x0), 0.0);
        assert_eq!(fp4_e2m1_to_f32(0x1), 0.5); // subnormal
        assert_eq!(fp4_e2m1_to_f32(0x2), 1.0);
        assert_eq!(fp4_e2m1_to_f32(0x3), 1.5);
        assert_eq!(fp4_e2m1_to_f32(0x4), 2.0);
        assert_eq!(fp4_e2m1_to_f32(0x5), 3.0);
        assert!(fp4_e2m1_to_f32(0x6).is_infinite());
        assert!(fp4_e2m1_to_f32(0x7).is_nan());
        assert_eq!(fp4_e2m1_to_f32(0xA), -1.0);
        assert!(fp4_e2m1_to_f32(0xE).is_infinite());
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
            1.0,
            1.5,
            4.0,
            6.0, //
            -1.0,
            -1.5,
            f32::INFINITY,
            f32::NAN, //
            0.25,
            0.5,
            192.0,
            256.0, //
            1.5,
            f32::INFINITY,
            0.0,
            0.0,
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

    #[test]
    fn offload_cache_evicts_lru_over_budget() {
        let dev = Device::Cpu;
        // Stand-in source: an empty mmap-backed store is only needed by the
        // load path, so exercise insert/eviction directly with a real store
        // built from an in-memory buffer below via OffloadCache::insert.
        let path = std::env::temp_dir().join("offload_cache_test.safetensors");
        // Minimal single-tensor safetensors: { "w": {"dtype":"F32","shape":[2,2],"data_offsets":[0,16]} }
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, &buf).unwrap();
        let st = unsafe { MmapedSafetensors::new(&path).unwrap() };
        let mut cache = OffloadCache::new(&st, dev, 40);
        let t = |v: f32| Tensor::from_vec(vec![v; 4], (2, 2), &Device::Cpu).unwrap();
        cache.insert("a", t(1.0)).unwrap(); // 16 bytes
        cache.insert("b", t(2.0)).unwrap(); // 32
        cache.insert("c", t(3.0)).unwrap(); // 48 > 40 -> evicts a -> 32
        assert!(cache.get_cached("a").is_none());
        assert!(cache.get_cached("b").is_some());
        assert!(cache.get_cached("c").is_some());
        assert_eq!(cache.current_bytes(), 32);
        cache.insert("d", t(4.0)).unwrap(); // 48 > 40 -> evicts b -> 32
        assert!(cache.get_cached("b").is_none());
        assert!(cache.get_cached("c").is_some());
        assert!(cache.get_cached("d").is_some());
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
        let _ = std::fs::remove_file(&path);
    }

    // ---- Synthetic V4-Flash-shaped fp8/fp4 checkpoint + full-model load ----

    /// V4-Flash-shaped but small enough for CPU: head_dim 512, 64 heads,
    /// sliding-window attention, fp8 linear weights + fp4 experts.
    fn synth_config() -> DeepseekV4Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 64, "hidden_size": 16, "moe_intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 64, "num_key_value_heads": 1,
                "head_dim": 512, "q_lora_rank": 8, "o_lora_rank": 8, "qk_rope_head_dim": 256,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "num_nextn_predict_layers": 0, "o_groups": 8, "num_hash_layers": 0,
                "index_head_dim": 4, "index_n_heads": 1, "index_topk": 4, "hc_mult": 2,
                "hc_sinkhorn_iters": 5, "hc_eps": 1e-6, "partial_rotary_factor": 0.5,
                "sliding_window": 128, "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
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

    /// Weight spec for the synthetic checkpoint.
    enum WKind {
        Fp8,
        Fp4,
        Plain,
    }
    struct W {
        name: String,
        kind: WKind,
        shape: Vec<usize>,
    }

    /// The full V4-Flash-shaped weight plan (matches the eager model's
    /// `VarBuilder` namespace). `seed` per weight for reproducibility.
    fn synth_weight_plan(cfg: &DeepseekV4Config) -> Vec<W> {
        let h = cfg.num_attention_heads;
        let d = cfg.head_dim;
        let mut ws = Vec::new();
        ws.push(W {
            name: "model.embed_tokens.weight".into(),
            kind: WKind::Fp8,
            shape: vec![cfg.vocab_size, cfg.hidden_size],
        });
        ws.push(W {
            name: "lm_head.weight".into(),
            kind: WKind::Fp8,
            shape: vec![cfg.vocab_size, cfg.hidden_size],
        });
        ws.push(W {
            name: "model.norm.weight".into(),
            kind: WKind::Plain,
            shape: vec![cfg.hidden_size],
        });
        let mix = (2 + cfg.hc_mult) * cfg.hc_mult;
        ws.push(W {
            name: "model.hc_head.hc_fn".into(),
            kind: WKind::Plain,
            shape: vec![cfg.hc_mult, cfg.hc_mult * cfg.hidden_size],
        });
        ws.push(W {
            name: "model.hc_head.hc_base".into(),
            kind: WKind::Plain,
            shape: vec![cfg.hc_mult],
        });
        ws.push(W {
            name: "model.hc_head.hc_scale".into(),
            kind: WKind::Plain,
            shape: vec![1],
        });
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            ws.push(W {
                name: format!("{p}.self_attn.q_a_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.q_lora_rank, cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.self_attn.q_a_norm.weight"),
                kind: WKind::Plain,
                shape: vec![cfg.q_lora_rank],
            });
            ws.push(W {
                name: format!("{p}.self_attn.q_b_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![h * d, cfg.q_lora_rank],
            });
            ws.push(W {
                name: format!("{p}.self_attn.kv_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![d, cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.self_attn.kv_norm.weight"),
                kind: WKind::Plain,
                shape: vec![d],
            });
            ws.push(W {
                name: format!("{p}.self_attn.o_a_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.o_groups * cfg.o_lora_rank, h * d / cfg.o_groups],
            });
            ws.push(W {
                name: format!("{p}.self_attn.o_b_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank],
            });
            ws.push(W {
                name: format!("{p}.self_attn.sinks"),
                kind: WKind::Plain,
                shape: vec![h],
            });
            ws.push(W {
                name: format!("{p}.attn_hc.fn"),
                kind: WKind::Plain,
                shape: vec![mix, cfg.hc_mult * cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.attn_hc.base"),
                kind: WKind::Plain,
                shape: vec![mix],
            });
            ws.push(W {
                name: format!("{p}.attn_hc.scale"),
                kind: WKind::Plain,
                shape: vec![3],
            });
            ws.push(W {
                name: format!("{p}.ffn_hc.fn"),
                kind: WKind::Plain,
                shape: vec![mix, cfg.hc_mult * cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.ffn_hc.base"),
                kind: WKind::Plain,
                shape: vec![mix],
            });
            ws.push(W {
                name: format!("{p}.ffn_hc.scale"),
                kind: WKind::Plain,
                shape: vec![3],
            });
            ws.push(W {
                name: format!("{p}.input_layernorm.weight"),
                kind: WKind::Plain,
                shape: vec![cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.post_attention_layernorm.weight"),
                kind: WKind::Plain,
                shape: vec![cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.mlp.gate.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.n_routed_experts, cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.mlp.experts.gate_up_proj"),
                kind: WKind::Fp4,
                shape: vec![
                    cfg.n_routed_experts,
                    2 * cfg.moe_intermediate_size,
                    cfg.hidden_size,
                ],
            });
            ws.push(W {
                name: format!("{p}.mlp.experts.down_proj"),
                kind: WKind::Fp4,
                shape: vec![
                    cfg.n_routed_experts,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                ],
            });
            ws.push(W {
                name: format!("{p}.mlp.shared_experts.gate_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.moe_intermediate_size, cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.mlp.shared_experts.up_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.moe_intermediate_size, cfg.hidden_size],
            });
            ws.push(W {
                name: format!("{p}.mlp.shared_experts.down_proj.weight"),
                kind: WKind::Fp8,
                shape: vec![cfg.hidden_size, cfg.moe_intermediate_size],
            });
        }
        ws
    }

    #[test]
    fn deepseek_v4_quantized_load_forward_synthetic() -> candle::Result<()> {
        let cfg = synth_config();
        let dev = Device::Cpu;
        let block = (4, 4);
        let scale_suffix = "_scale_inv";
        let fp4_prefixes: Vec<String> = vec![
            "mlp.experts.gate_up_proj".to_string(),
            "mlp.experts.down_proj".to_string(),
        ];

        // Build the weight data (quantized on disk + unquantized reference).
        let plan = synth_weight_plan(&cfg);
        let mut shard0: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut shard1: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut shard2: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut unquant: HashMap<String, Tensor> = HashMap::new();
        let seed = 0x1234_5678u64;
        for (wi, w) in plan.iter().enumerate() {
            let n_elems: usize = w.shape.iter().product();
            let vals = lcg(seed.wrapping_add(wi as u64 * 7919), n_elems);
            let rank = w.shape.len();
            let (rows, cols) = (w.shape[..rank - 1].iter().product(), w.shape[rank - 1]);
            let (mut wbytes, mut scale_bytes): (Vec<u8>, Vec<u8>) = match w.kind {
                WKind::Fp8 => {
                    let (wb, sb) = quant_fp8(&vals, rows, cols, block);
                    (wb, sb)
                }
                WKind::Fp4 => {
                    let (wb, sb) = quant_fp4(&vals, rows, cols, block);
                    (wb, sb)
                }
                WKind::Plain => (f32_bytes(&vals), Vec::new()),
            };
            unquant.insert(
                w.name.clone(),
                Tensor::from_vec(vals, w.shape.clone(), &dev)?,
            );
            // Which shard? Root weights + layer 0 -> shard0/1, layer 1 -> shard2.
            let target = if w.name.starts_with("model.layers.1") {
                &mut shard2
            } else if w.name.starts_with("model.layers.0") {
                &mut shard1
            } else {
                &mut shard0
            };
            match w.kind {
                WKind::Fp8 => {
                    target.push((
                        w.name.clone(),
                        "F8_E4M3".to_string(),
                        w.shape.clone(),
                        std::mem::take(&mut wbytes),
                    ));
                    let gn = rows / block.0 * (cols / block.1);
                    target.push((
                        format!("{}{}", w.name, scale_suffix),
                        "F8_E8M0".to_string(),
                        vec![gn],
                        std::mem::take(&mut scale_bytes),
                    ));
                }
                WKind::Fp4 => {
                    target.push((
                        w.name.clone(),
                        "U8".to_string(),
                        vec![w.shape.iter().product::<usize>() / 2],
                        std::mem::take(&mut wbytes),
                    ));
                    let gn = rows / block.0 * (cols / block.1);
                    target.push((
                        format!("{}{}", w.name, scale_suffix),
                        "F8_E8M0".to_string(),
                        vec![gn],
                        std::mem::take(&mut scale_bytes),
                    ));
                }
                WKind::Plain => {
                    target.push((
                        w.name.clone(),
                        "F32".to_string(),
                        w.shape.clone(),
                        std::mem::take(&mut wbytes),
                    ));
                }
            }
        }

        let dir = std::env::temp_dir().join(format!("v4q_synth_{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let p0 = dir.join("model-00001-of-00003.safetensors");
        let p1 = dir.join("model-00002-of-00003.safetensors");
        let p2 = dir.join("model-00003-of-00003.safetensors");
        write_st(&p0, &shard0);
        write_st(&p1, &shard1);
        write_st(&p2, &shard2);

        // Tiny offload budget -> the loader must evict while materializing all
        // weights (proves the size-bounded offload cache is exercised).
        let budget = 512 * 1024usize; // < total checkpoint size -> evicts while keeping some resident
        let paths = [p0.as_path(), p1.as_path(), p2.as_path()];
        let loader = unsafe {
            DeepseekV4Quantized::new(
                &paths,
                block,
                scale_suffix,
                &fp4_prefixes,
                dev.clone(),
                DType::F32,
                budget,
            )?
        };
        // Every weight requested by the model must be present or intentionally
        // fall back (e_score_correction_bias is absent -> zeros in the model).
        assert!(loader.contains("model.embed_tokens.weight"));
        assert!(loader.contains("model.layers.0.mlp.experts.gate_up_proj"));
        assert!(loader.contains("model.layers.0.self_attn.q_a_proj.weight_scale_inv"));

        let mut model = loader.load_model(&cfg, false)?;

        // The on-demand offload cache was exercised and stayed within budget.
        assert!(
            loader.resident_bytes() <= budget,
            "offload cache over budget"
        );
        assert!(loader.cached_len() > 0, "offload cache was never populated");

        let ids = Tensor::new(&[1u32, 3, 7, 2, 5][..], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&ids, 0)?;
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size], "logits shape");
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|v| v.is_finite()), "logits must be finite");

        // Determinism: a second forward (fresh KV cache) matches exactly.
        let logits2 = {
            let mut m = unsafe {
                load_quantized_for_causal_lm(
                    &cfg,
                    false,
                    &paths,
                    &dev,
                    DType::F32,
                    block,
                    scale_suffix,
                    &fp4_prefixes,
                    budget,
                )?
            };
            m.forward(&ids, 0)?
        };
        assert_close(&logits, &logits2, 0.0, "deterministic");

        // Cross-check against the exact (unquantized) model: the fp8/fp4
        // pipeline must be close to the full-precision forward.
        let vb = VarBuilder::from_tensors(unquant, DType::F32, &dev);
        let mut ref_model = DeepseekV4ForCausalLM::new(&cfg, false, vb)?;
        let ref_logits = ref_model.forward(&ids, 0)?;
        assert_eq!(ref_logits.dims(), logits.dims(), "ref shape");
        let l = logits.flatten_all()?.to_vec1::<f32>()?;
        let r = ref_logits.flatten_all()?.to_vec1::<f32>()?;
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, b) in l.iter().zip(r.iter()) {
            max_abs = max_abs.max((a - b).abs());
            max_rel = max_rel.max((a - b).abs() / (b.abs() + 1e-3));
        }
        // 4x4-block fp8/fp4 on random weights: expect a few percent max error.
        assert!(max_abs < 0.5, "quantized vs full logits max abs {max_abs}");
        assert!(max_rel < 0.1, "quantized vs full logits max rel {max_rel}");

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
