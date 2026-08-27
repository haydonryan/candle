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

use candle::safetensors::MmapedSafetensors;
use candle::{DType, Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

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
}
