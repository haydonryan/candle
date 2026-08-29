//! Mixed-precision fp8/fp4 GEMM on NVIDIA tensor cores via cuBLASLt.
//!
//! This is the fp8/fp4 compute path for the real DeepSeek-V4-Flash weights on
//! Blackwell (sm_120a): quantized weights stay fp8/fp4 on the GPU (no bf16
//! upcast) and the matmul runs on the fp8 tensor cores with fp32 accumulation.
//!
//! # Survey finding (story #4332, task 15768)
//!
//! cuBLASLt on this Blackwell (CUDA 13.3, RTX PRO 6000) accepts **only
//! scalar-scaled fp8 x fp8** (`CUDA_R_8F_E4M3` on both inputs, fp32
//! accumulate). Every vector/block matrix-scale mode needed for the raw
//! block-scaled checkpoint format (`VEC32_UE8M0`, `VEC128_32F`,
//! `BLK128x128_32F`, ...) is rejected with `CUBLAS_STATUS_INVALID_VALUE` at the
//! heuristic stage (verified empirically). candle-flash-attn has no fp8 GEMM
//! kernels.
//!
//! The selected path therefore keeps weights fp8/fp4 on the GPU and **folds the
//! ue8m0 block/vec scale directly into the fp8 value** before the GEMM:
//! * fp8 e4m3 attention weights (`[N, K]`, `[128,128]` ue8m0 block scale):
//!   multiplying an fp8 value by a ue8m0 (power-of-two) scale only shifts the
//!   exponent, so `w8 * s` is exactly representable in fp8 — the fold is
//!   **lossless**.
//! * fp4 e2m1 MoE expert weights (`[N, K/2]` packed, per-row `gran_k=32` ue8m0
//!   scale): dequantizing `fp4 * s` into fp8 e4m3 is lossless because fp8 has
//!   strictly more mantissa bits than fp4 and the scale is a power of two.
//!
//! Activations are quantized to fp8 (per-tensor scale folded into `alpha`) and
//! the GEMM runs as a plain scalar-scaled fp8 x fp8 with fp32 accumulation.
//! Weights are never upcast to bf16.

use candle::cuda::{CudaDevice, CudaStorage};
use candle::{DType, Error, Result, Storage, Tensor};
use cudarc::cublaslt::{result, sys};
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;

const WORKSPACE_BYTES: usize = 32 << 20;

/// cuBLASLt matrix layout handle (RAII).
struct Layout(sys::cublasLtMatrixLayout_t);
impl Layout {
    fn new(dtype: sys::cudaDataType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let h = result::create_matrix_layout(dtype, rows, cols, ld)
            .map_err(|e| Error::msg(format!("create_matrix_layout: {e:?}")))?;
        // cuBLASLt defaults to column-major; our tensors are row-major.
        let row: i32 = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW as i32;
        unsafe {
            result::set_matrix_layout_attribute(
                h,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                (&row as *const i32).cast(),
                4,
            )
            .map_err(|e| Error::msg(format!("set layout order: {e:?}")))?;
        }
        Ok(Self(h))
    }
}
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matrix_layout(self.0);
        }
    }
}

/// cuBLASLt matmul descriptor (RAII).
struct Desc(sys::cublasLtMatmulDesc_t);
impl Desc {
    fn new(compute_type: sys::cublasComputeType_t, scale_type: sys::cudaDataType) -> Result<Self> {
        let h = result::create_matmul_desc(compute_type, scale_type)
            .map_err(|e| Error::msg(format!("create_matmul_desc: {e:?}")))?;
        Ok(Self(h))
    }

    fn set_transpose(&self, transpose: bool, matrix: Matrix) -> Result<()> {
        let v: i32 = transpose as i32;
        let attr = match matrix {
            Matrix::A => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            Matrix::B => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        };
        unsafe {
            result::set_matmul_desc_attribute(self.0, attr, (&v as *const i32).cast(), 4)
                .map_err(|e| Error::msg(format!("set transpose: {e:?}")))?;
        }
        Ok(())
    }
}
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_desc(self.0);
        }
    }
}

enum Matrix {
    A,
    B,
}

/// cuBLASLt matmul preference (RAII).
struct Pref(sys::cublasLtMatmulPreference_t);
impl Pref {
    fn new() -> Result<Self> {
        let h = result::create_matmul_pref()
            .map_err(|e| Error::msg(format!("create_matmul_pref: {e:?}")))?;
        Ok(Self(h))
    }

    fn set_workspace(&self, bytes: usize) -> Result<()> {
        unsafe {
            result::set_matmul_pref_attribute(
                self.0,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&bytes as *const usize).cast(),
                std::mem::size_of::<usize>(),
            )
            .map_err(|e| Error::msg(format!("set workspace: {e:?}")))?;
        }
        Ok(())
    }
}
impl Drop for Pref {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_pref(self.0);
        }
    }
}

/// A raw cuBLASLt handle bound to a CUDA stream, used for fp8 x fp8 GEMMs.
pub struct BlasLt {
    handle: sys::cublasLtHandle_t,
    workspace: cudarc::driver::CudaSlice<u8>,
    stream: Arc<CudaStream>,
}

impl BlasLt {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let handle =
            result::create_handle().map_err(|e| Error::msg(format!("cublasLtCreate: {e:?}")))?;
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_BYTES) }
            .map_err(|e| Error::msg(format!("workspace alloc: {e:?}")))?;
        Ok(Self {
            handle,
            workspace,
            stream,
        })
    }

    /// Plain scalar-scaled fp8 x fp8 -> fp32 GEMM: `d = alpha * (a @ b)`.
    /// `a` is `(m, k)` fp8 (row-major), `b` is `(n, k)` fp8 with `transb` (so
    /// the product uses `b^T`), `d` is `(m, n)` fp32. `beta = 0` so `c` is the
    /// same buffer as `d`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn fp8x8_gemm(
        &mut self,
        cfg: &GemmConfig,
        a: *const u8,
        b: *const u8,
        alpha: f32,
        d: *mut u8,
    ) -> Result<()> {
        let (a_rows, a_cols) = (cfg.m, cfg.k);
        let (b_rows, b_cols) = if cfg.transb {
            (cfg.n, cfg.k)
        } else {
            (cfg.k, cfg.n)
        };
        let a_layout = Layout::new(sys::cudaDataType_t::CUDA_R_8F_E4M3, a_rows, a_cols, cfg.lda)?;
        let b_layout = Layout::new(sys::cudaDataType_t::CUDA_R_8F_E4M3, b_rows, b_cols, cfg.ldb)?;
        let c_layout = Layout::new(sys::cudaDataType_t::CUDA_R_32F, cfg.m, cfg.n, cfg.ldc)?;
        let d_layout = Layout::new(sys::cudaDataType_t::CUDA_R_32F, cfg.m, cfg.n, cfg.ldc)?;

        let desc = Desc::new(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;
        desc.set_transpose(cfg.transa, Matrix::A)?;
        desc.set_transpose(cfg.transb, Matrix::B)?;

        let pref = Pref::new()?;
        pref.set_workspace(WORKSPACE_BYTES)?;
        let heuristic = result::get_matmul_algo_heuristic(
            self.handle,
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            d_layout.0,
            pref.0,
        )
        .map_err(|e| Error::msg(format!("get_matmul_algo_heuristic: {e:?}")))?;

        let beta: f32 = 0.0;
        let (ws, _) = self.workspace.device_ptr_mut(&self.stream);
        let ws = ws as *mut std::ffi::c_void;
        let stream = self.stream.cu_stream() as *mut _;
        result::matmul(
            self.handle,
            desc.0,
            (&alpha as *const f32).cast(),
            (&beta as *const f32).cast(),
            a.cast(),
            a_layout.0,
            b.cast(),
            b_layout.0,
            d.cast(),
            c_layout.0,
            d as *mut std::ffi::c_void,
            d_layout.0,
            &heuristic.algo,
            ws,
            WORKSPACE_BYTES,
            stream,
        )
        .map_err(|e| Error::msg(format!("cublasLtMatmul: {e:?}")))?;
        Ok(())
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_handle(self.handle);
        }
    }
}

/// GEMM shape config for [`BlasLt::fp8x8_gemm`].
struct GemmConfig {
    transa: bool,
    transb: bool,
    m: u64,
    n: u64,
    k: u64,
    lda: i64,
    ldb: i64,
    ldc: i64,
}

impl GemmConfig {
    /// `act` is `(M, K)`; weight is on-disk `(N, K)` (candle Linear
    /// orientation, so the right operand is its transpose).
    fn linear(act: &Tensor, out: usize, inp: usize, transb: bool) -> Result<GemmConfig> {
        let m = act.dim(0)? as u64;
        let k = inp as u64;
        let n = out as u64;
        Ok(GemmConfig {
            transa: false,
            transb,
            m,
            n,
            k,
            lda: k as i64,
            ldb: k as i64,
            ldc: n as i64,
        })
    }
}

fn tensor_ptr<T: candle::cuda::CudaDType>(
    t: &Tensor,
    stream: &Arc<CudaStream>,
) -> Result<*const u8> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(s) = &*storage else {
        candle::bail!("expected a CUDA tensor, got {:?}", t.device())
    };
    let slice: &cudarc::driver::CudaSlice<T> = s.as_cuda_slice()?;
    let (ptr, _) = slice.device_ptr(stream);
    // The tensor may be a view (e.g. per-group/per-expert narrowed rows) whose
    // data starts at `start_offset` elements into the storage.
    let byte_off = layout.start_offset() * std::mem::size_of::<T>();
    Ok(unsafe { (ptr as *const u8).add(byte_off) })
}

fn alloc_f32(dev: &CudaDevice, len: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
    dev.alloc_zeros::<f32>(len)
        .map_err(|e| Error::msg(format!("alloc f32 output: {e:?}")))
}

/// Wrap a raw fp32 CUDA buffer as a contiguous `(m, n)` tensor.
fn wrap_f32(slice: cudarc::driver::CudaSlice<f32>, dev: CudaDevice, m: usize, n: usize) -> Tensor {
    let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev));
    Tensor::from_storage(storage, (m, n), candle::op::BackpropOp::none(), false)
}

/// fp8 x fp8 matmul on the tensor cores: `out = alpha * (act_fp8 @ w8)`.
///
/// `act_fp8` is a `(M, K)` fp8 tensor (already scaled so its values are the
/// actual activations), `w8` is the `(N, K)` fp8 weight with the ue8m0 scale
/// already folded in (see [`fold_fp8_block_scale`]/[`fp4_to_fp8`]). `alpha`
/// carries any residual activation scale. Returns `(M, N)` in `out_dtype`
/// (fp32 accumulation, downcast on exit).
pub fn fp8_matmul(
    blas: &mut BlasLt,
    act_fp8: &Tensor,
    w8: &Tensor,
    alpha: f32,
    out_dtype: DType,
) -> Result<Tensor> {
    let (n, k) = w8.dims2()?;
    let dev = act_fp8.device().as_cuda_device()?.clone();
    let stream = get_stream(act_fp8)?;
    let cfg = GemmConfig::linear(act_fp8, n, k, true)?;
    let mut out_slice = alloc_f32(&dev, (cfg.m * cfg.n) as usize)?;
    let (d_ptr, _) = out_slice.device_ptr_mut(&stream);
    let d_ptr = d_ptr as *mut u8;

    let a = tensor_ptr::<u8>(act_fp8, &stream)?;
    let b = tensor_ptr::<u8>(w8, &stream)?;

    unsafe {
        blas.fp8x8_gemm(&cfg, a, b, alpha, d_ptr)?;
    }
    let out = wrap_f32(out_slice, dev, cfg.m as usize, cfg.n as usize);
    if out_dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(out_dtype)
    }
}

fn get_stream(t: &Tensor) -> Result<Arc<CudaStream>> {
    Ok(t.device().as_cuda_device()?.cuda_stream())
}

/// An fp8-quantized linear weight: `bytes` is the `(out, in)` fp8 tensor
/// (value = `fp8_e4m3_to_f32`), `scale` is the per-tensor weight scale folded
/// into `alpha` at GEMM time so that `w ≈ fp8(bytes) / scale`.
///
/// Scaling the raw weight by `scale` (≈ `448 / max|w|`) before rounding keeps
/// the fp8 values near the top of the e4m3 range for maximum precision; the
/// scale is cancelled by `alpha` in [`fp8_linear`] together with the
/// activation scale, so the GEMM returns the true `act @ w` product.
pub struct Fp8Weight {
    pub bytes: Tensor,
    pub scale: f32,
}

impl Fp8Weight {
    /// Quantize an f32 `(out, in)` weight tensor to fp8.
    pub fn from_tensor(w: &Tensor) -> Result<Self> {
        let (out, in_) = w.dims2()?;
        let v = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (bytes, scale) = quantize_fp8(&v);
        let t = Tensor::from_vec(bytes, (out, in_), w.device())?;
        Ok(Self { bytes: t, scale })
    }
}

/// fp8 linear projection: quantize `act` (`(..., in)`) to fp8, then run
/// `fp8 x fp8` on the tensor cores against the fp8 weight `w` (`(out, in)`,
/// scales folded in via [`Fp8Weight`]). Both the activation scale and the
/// weight scale fold into `alpha`, so the result is `act @ w` (up to fp8
/// quantization) in `out_dtype` (`(..., out)`).
///
/// Activations are quantized with per-block ue8m0 scales (story #4280):
/// the `in` dimension is split into `bk = 128`-wide slices, each folded into
/// the fp8 value, and the reduction is run as one scalar-scaled GEMM per slice
/// (slice scale folded into that slice's scalar `alpha`) whose fp32 results
/// are summed. This keeps the per-block accuracy of the checkpoint's dynamic
/// activation scheme while staying within cuBLASLt's scalar-only fp8x8 mode.
pub fn fp8_linear(
    blas: &mut BlasLt,
    act: &Tensor,
    w: &Fp8Weight,
    out_dtype: DType,
) -> Result<Tensor> {
    const BK: usize = 128;
    let rank = act.rank();
    let in_ = act.dim(rank - 1)?;
    let out = w.bytes.dim(0)?;
    let lead: usize = act.dims()[..rank - 1].iter().product();
    let flat = act
        .reshape((lead, in_))?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let v = flat.flatten_all()?.to_vec1::<f32>()?;
    let (a8, block_scales) = quantize_fp8_block(&v, lead, in_, BK);
    let a8t = Tensor::from_vec(a8, (lead, in_), act.device())?;
    let nblocks = in_.div_ceil(BK);
    let mut acc: Option<Tensor> = None;
    for b in 0..nblocks {
        let k0 = b * BK;
        let k1 = (k0 + BK).min(in_);
        // fp8_matmul requires contiguous rows (lda = k_block); the narrowed
        // views are strided by the full `in_`, so materialize each slice.
        let act_b = a8t.narrow(1, k0, k1 - k0)?.contiguous()?;
        let w_b = w.bytes.narrow(1, k0, k1 - k0)?.contiguous()?;
        // Slice scale folded into alpha: out += (act_b * 2^s_b) @ w_b / (2^s_b * w.scale)
        // = act_b @ w_b (per-slice), summed across slices = act @ w.
        let alpha = 1.0 / (block_scales[b] * w.scale);
        let part = fp8_matmul(blas, &act_b, &w_b, alpha, out_dtype)?;
        acc = Some(match acc {
            None => part,
            Some(a) => a.add(&part)?,
        });
    }
    let out_t = acc.unwrap();
    let mut dims = act.dims()[..rank - 1].to_vec();
    dims.push(out);
    out_t.reshape(dims)
}
/// fp8 grouped low-rank linear (block-diagonal over `n_groups`): `act` is
/// `(..., n_groups, in_per_group)`, `w` is the `(n_groups * out_per_group,
/// in_per_group)` fp8 weight (single per-tensor scale). Returns
/// `(..., n_groups, out_per_group)`.
pub fn fp8_grouped_linear(
    blas: &mut BlasLt,
    act: &Tensor,
    w: &Fp8Weight,
    n_groups: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let ndim = act.rank();
    let in_per_group = w.bytes.dim(1)?;
    let out_per_group = w.bytes.dim(0)? / n_groups;
    let batch: usize = act.dims()[..ndim - 2].iter().product();
    let xr = act.reshape((batch, n_groups, in_per_group))?;
    let mut outs = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let act_g = xr.narrow(1, g, 1)?.squeeze(1)?;
        let sub = Fp8Weight {
            bytes: w.bytes.narrow(0, g * out_per_group, out_per_group)?,
            scale: w.scale,
        };
        let o_g = fp8_linear(blas, &act_g, &sub, out_dtype)?; // (batch, out_per_group)
        outs.push(o_g.unsqueeze(1)?);
    }
    let y = Tensor::cat(&outs, 1)?; // (batch, n_groups, out_per_group)
    let mut out_dims = act.dims()[..ndim - 2].to_vec();
    out_dims.push(n_groups);
    out_dims.push(out_per_group);
    y.reshape(out_dims)
}

/// Fold a `[bn, bk]` ue8m0 block scale into a raw fp8 weight, exactly.
///
/// `w8` is the `(n, k)` fp8 e4m3 weight, `scale` is the `(n/bn, k/bk)` ue8m0
/// scale. Because ue8m0 is a power of two, `w8 * scale` only shifts the fp8
/// exponent — the result is exactly representable in fp8. Out-of-range values
/// are clamped to the fp8 max (448.0).
pub fn fold_fp8_block_scale(
    w8: &[u8],
    scale: &[u8],
    n: usize,
    k: usize,
    bn: usize,
    bk: usize,
) -> Vec<u8> {
    debug_assert_eq!(w8.len(), n * k);
    debug_assert_eq!(scale.len(), (n / bn) * (k / bk));
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        let srow = &scale[(i / bn) * (k / bk)..((i / bn) + 1) * (k / bk)];
        for j in 0..k {
            let shift = srow[j / bk] as i32 - 127; // ue8m0 exponent delta
            let b = w8[i * k + j];
            let sign = b & 0x80;
            let exp = ((b >> 3) & 0x0F) as i32;
            let man = b & 0x07;
            // fp8 e4m3: value = (-1)^s * 2^(exp-7) * (1 + man/8) (or subnormal).
            // Multiply by 2^shift => exp += shift.
            let nexp = exp + shift;
            let nb = sign | ((nexp.clamp(0, 14) as u8) << 3) | man;
            out.push(nb);
        }
    }
    out
}

/// Convert a packed fp4 e2m1 weight with a per-row `gran_k=32` ue8m0 scale to
/// fp8, losslessly.
///
/// `wpack` is the `(n, k/2)` packed fp4 weight (low nibble = even k), `scale`
/// is the `(n, k/32)` ue8m0 vec32 scale. `fp4 * scale` is exactly representable
/// in fp8 e4m3 (fp8 has more mantissa bits than fp4; scale is a power of two).
pub fn fp4_to_fp8(wpack: &[u8], scale: &[u8], n: usize, k: usize) -> Vec<u8> {
    debug_assert_eq!(wpack.len(), n * k / 2);
    debug_assert_eq!(scale.len(), n * (k / 32));
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            let nib = if j % 2 == 0 {
                wpack[i * k / 2 + j / 2] & 0x0F
            } else {
                wpack[i * k / 2 + j / 2] >> 4
            };
            let sc = scale[i * (k / 32) + j / 32] as i32 - 127;
            let v = fp4_e2m1_to_f32(nib) * 2f32.powi(sc);
            out.push(f32_to_fp8(v));
        }
    }
    out
}

/// Quantize f32 values to fp8 e4m3 bytes (nearest, with a per-tensor scale).
/// Returns `(bytes, scale)` such that `value ≈ fp8(bytes) * scale`.
pub fn quantize_fp8(vals: &[f32]) -> (Vec<u8>, f32) {
    let max = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-30);
    let scale = 448.0 / max;
    let bytes = vals
        .iter()
        .map(|&v| f32_to_fp8(v * scale))
        .collect::<Vec<_>>();
    (bytes, scale)
}

/// Quantize a contiguous row-major `(m, k)` f32 activation array to fp8 e4m3
/// with per-block ue8m0 (F8E8M0) scales over `bk`-wide slices of the `k`
/// dimension (matching the checkpoint's dynamic activation scheme, `[1, bk]`
/// blocks like the weights).
///
/// Each `k`-slice gets one power-of-two scale `2^s` (with `s` chosen so the
/// slice's max magnitude maps into `(224, 448]`), shared across all `m` rows.
/// Because ue8m0 is a power of two, `v * 2^s` only shifts the fp8 exponent —
/// the scale is folded directly into the fp8 value (exact, like the weight
/// folding in [`fold_fp8_block_scale`]). The GEMM then splits the reduction by
/// slice and carries each slice's scale in a scalar `alpha` (see
/// [`fp8_linear`]), which is exactly what the scalar-scaled cuBLASLt fp8x8
/// kernels accept.
///
/// Returns the folded fp8 bytes (`m * k`) and the per-slice scales `2^s`
/// (length `ceil(k / bk)`).
pub fn quantize_fp8_block(vals: &[f32], m: usize, k: usize, bk: usize) -> (Vec<u8>, Vec<f32>) {
    let nblocks = k.div_ceil(bk);
    let mut out = vec![0u8; m * k];
    let mut scales = vec![1.0f32; nblocks];
    for b in 0..nblocks {
        let k0 = b * bk;
        let k1 = (k0 + bk).min(k);
        let mut mx = 1e-30f32;
        for i in 0..m {
            for j in k0..k1 {
                mx = mx.max(vals[i * k + j].abs());
            }
        }
        // ue8m0 power-of-two scale: floor so the block max maps to <= 448.
        let s = (448.0 / mx).log2().floor() as i32;
        let scale = 2f32.powi(s);
        scales[b] = scale;
        for i in 0..m {
            for j in k0..k1 {
                out[i * k + j] = f32_to_fp8(vals[i * k + j] * scale);
            }
        }
    }
    (out, scales)
}

/// Round an f32 to the nearest fp8 e4m3 byte (finite, clamp to range).
pub fn f32_to_fp8(v: f32) -> u8 {
    // Brute-force nearest over the 256 representable values is small and exact.
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for b in 0u16..=255 {
        let x = fp8_e4m3_to_f32(b as u8);
        if x.is_nan() {
            continue;
        }
        let d = (v - x).abs();
        if d < best_d {
            best_d = d;
            best = b as u8;
        }
    }
    best
}

/// fp8 e4m3 byte -> f32 (mirrors `float8::F8E4M3`).
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

/// fp4 e2m1 nibble -> f32 (sign 1, exp 2, mantissa 1, bias 1).
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    fn encode_ue8m0(v: f32) -> u8 {
        if v == 0.0 {
            return 0;
        }
        let e = v.abs().log2().round() as i32;
        (e.clamp(-126, 126) + 127) as u8
    }

    fn encode_fp4(v: f32) -> u8 {
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for nib in 0u8..16 {
            let x = fp4_e2m1_to_f32(nib);
            if x.is_nan() {
                continue;
            }
            let d = (v - x).abs();
            if d < best_d {
                best_d = d;
                best = nib;
            }
        }
        best
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let denom = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        max_abs_diff(a, b) / denom
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8_block_gemm_ref(
        act: &[f32],
        m: usize,
        k: usize,
        w: &[u8],
        n: usize,
        scale_f32: &[f32],
        bn: usize,
        bk: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let s = scale_f32[(j / bn) * (k / bk) + (kk / bk)];
                    acc += act[i * k + kk] * (fp8_e4m3_to_f32(w[j * k + kk]) * s);
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn fp4_block_gemm_ref(
        act: &[f32],
        m: usize,
        k: usize,
        wpack: &[u8],
        n: usize,
        scale_f32: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let s = scale_f32[j * (k / 32) + (kk / 32)];
                    let nib = if kk % 2 == 0 {
                        wpack[j * k / 2 + kk / 2] & 0x0F
                    } else {
                        wpack[j * k / 2 + kk / 2] >> 4
                    };
                    acc += act[i * k + kk] * (fp4_e2m1_to_f32(nib) * s);
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn test_fp8_attention_parity() {
        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let stream = dev.as_cuda_device().unwrap().cuda_stream();
        let mut blas = BlasLt::new(stream).unwrap();

        let (m, k, n) = (64usize, 512usize, 512usize);
        let bn = 128usize;
        let bk = 128usize;
        // Activations and fp8 weights in fp32, block-scaled.
        let act: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 7919.0) % 200.0 - 100.0) / 100.0)
            .collect();
        let wf: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 104729.0) % 400.0 - 200.0) / 400.0)
            .collect();
        let w8: Vec<u8> = wf.iter().map(|&v| f32_to_fp8(v)).collect();
        let scale_f32: Vec<f32> = (0..(n / bn) * (k / bk))
            .map(|i| 0.25 * 2f32.powi((i % 5) as i32))
            .collect();
        let scale_ue: Vec<u8> = scale_f32.iter().map(|&v| encode_ue8m0(v)).collect();

        // Fold scale into fp8 weight (exact), quantize act to fp8.
        let w_folded = fold_fp8_block_scale(&w8, &scale_ue, n, k, bn, bk);
        let (act8, act_scale) = quantize_fp8(&act);
        // fp8 reference: same fp8-quantized activation, dequant fp8 * block scale, in fp32
        // (isolates the exact weight-fold path from the fp8 activation-quantization noise).
        let act_f: Vec<f32> = act8
            .iter()
            .map(|&b| fp8_e4m3_to_f32(b) / act_scale)
            .collect();
        let ref_out = fp8_block_gemm_ref(&act_f, m, k, &w8, n, &scale_f32, bn, bk);

        let act_t = Tensor::from_vec(act8, (m, k), &dev).unwrap();
        let w_t = Tensor::from_vec(w_folded, (n, k), &dev).unwrap();
        let got = fp8_matmul(&mut blas, &act_t, &w_t, 1.0 / act_scale, DType::F32).unwrap();
        let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();

        println!(
            "fp8 attention rel_err={} max_diff={}",
            rel_err(&got_v, &ref_out),
            max_abs_diff(&got_v, &ref_out)
        );
        assert!(rel_err(&got_v, &ref_out) < 5e-2, "fp8 parity failed");
    }

    #[test]
    fn test_fp4_moe_parity() {
        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let stream = dev.as_cuda_device().unwrap().cuda_stream();
        let mut blas = BlasLt::new(stream).unwrap();

        let (m, k, n) = (64usize, 512usize, 256usize);
        let act: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 7919.0) % 200.0 - 100.0) / 100.0)
            .collect();
        let wf: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 104729.0) % 200.0 - 100.0) / 200.0)
            .collect();
        let mut wpack = vec![0u8; n * k / 2];
        for j in 0..n {
            for kk in 0..k {
                let nib = encode_fp4(wf[j * k + kk]);
                if kk % 2 == 0 {
                    wpack[j * k / 2 + kk / 2] |= nib;
                } else {
                    wpack[j * k / 2 + kk / 2] |= nib << 4;
                }
            }
        }
        let scale_f32: Vec<f32> = (0..n * (k / 32))
            .map(|i| 0.25 * 2f32.powi((i % 5) as i32))
            .collect();
        let scale_ue: Vec<u8> = scale_f32.iter().map(|&v| encode_ue8m0(v)).collect();

        let w_fp8 = fp4_to_fp8(&wpack, &scale_ue, n, k);
        let (act8, act_scale) = quantize_fp8(&act);
        let act_f: Vec<f32> = act8
            .iter()
            .map(|&b| fp8_e4m3_to_f32(b) / act_scale)
            .collect();
        let ref_out = fp4_block_gemm_ref(&act_f, m, k, &wpack, n, &scale_f32);

        let act_t = Tensor::from_vec(act8, (m, k), &dev).unwrap();
        let w_t = Tensor::from_vec(w_fp8, (n, k), &dev).unwrap();
        let got = fp8_matmul(&mut blas, &act_t, &w_t, 1.0 / act_scale, DType::F32).unwrap();
        let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();

        println!(
            "fp4 MoE rel_err={} max_diff={}",
            rel_err(&got_v, &ref_out),
            max_abs_diff(&got_v, &ref_out)
        );
        assert!(rel_err(&got_v, &ref_out) < 5e-2, "fp4 parity failed");
    }

    #[test]
    fn benchmark_fp8_vs_bf16() {
        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let stream = dev.as_cuda_device().unwrap().cuda_stream();
        let mut blas = BlasLt::new(stream).unwrap();

        // Representative attention-projection shape (M=seq, K=hidden, N=proj out).
        let (m, k, n) = (2048usize, 4096usize, 4096usize);
        let act: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 7919.0) % 200.0 - 100.0) / 100.0)
            .collect();
        let wf: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 104729.0) % 400.0 - 200.0) / 400.0)
            .collect();

        // fp8 path
        let (act8, act_scale) = quantize_fp8(&act);
        let w8: Vec<u8> = wf.iter().map(|&v| f32_to_fp8(v)).collect();
        let act_t = Tensor::from_vec(act8, (m, k), &dev).unwrap();
        let w_t = Tensor::from_vec(w8, (n, k), &dev).unwrap();

        // bf16 baseline: upcast both to bf16 (the slow path we're replacing).
        let act_bf = Tensor::from_vec(act.clone(), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_bf = Tensor::from_vec(wf.clone(), (n, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_bf_t = w_bf.t().unwrap().contiguous().unwrap();

        // Warm up + time fp8 gemm (on GPU).
        for _ in 0..3 {
            let _ = fp8_matmul(&mut blas, &act_t, &w_t, act_scale, DType::F32).unwrap();
        }
        dev.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        let reps = 20;
        for _ in 0..reps {
            let _ = fp8_matmul(&mut blas, &act_t, &w_t, act_scale, DType::F32).unwrap();
        }
        dev.synchronize().unwrap();
        let fp8_us = t0.elapsed().as_micros() as f64 / reps as f64;

        // Time bf16 gemm (candle matmul on GPU).
        for _ in 0..3 {
            let _ = act_bf.matmul(&w_bf_t).unwrap();
        }
        dev.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            let _ = act_bf.matmul(&w_bf_t).unwrap();
        }
        dev.synchronize().unwrap();
        let bf16_us = t1.elapsed().as_micros() as f64 / reps as f64;

        println!(
            "BENCH fp8_us={fp8_us:.1} bf16_us={bf16_us:.1} speedup={:.2}x (M={m} K={k} N={n})",
            bf16_us / fp8_us
        );
    }
}
