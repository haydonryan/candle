// Custom DeepSeek V4 DSA flash kernel: QK^T*scale + heterogeneous additive mask
// (sliding-window local prefix | per-query block_bias compressed suffix) + per-head
// sink logit column + max-subtract + softmax + drop-sink + @V.
//
// Replicates the transformers eager op (modeling_deepseek_v4.py eager_attention_forward):
//   attn = (Q @ K^T) * scale + mask                 // [B,H,Sq,Skv], mask additive 0/-inf
//   combined = cat([attn, sink_logit[h]], dim=-1)   // sink appended as extra logit column
//   combined -= max(combined)                       // max-subtract INCLUDES sink
//   p = softmax(combined)
//   scores = p[..., :-1]                            // drop sink column (probability leak)
//   out = scores @ V
//
// The sink guarantees a finite max even when every KV column is masked (rows never NaN).
//
// Block-sparse execution: the KV axis is processed in BK=64-column blocks. Any block whose
// columns are all masked (`-inf`) is skipped entirely - no QK dot, no K/V load, no softmax
// fold. Because an -inf column contributes exactly 0 to the running max (which starts at the
// finite sink logit), the denominator, and the accumulator, this is bit-identical to the
// dense reference while QK/V traffic scales with the attended blocks only (sliding window +
// selected top-k compressed blocks), not the full KV length.
//
// Layouts (contiguous):
//   q:   [B, Sq, H,  D]   (BF16)
//   k:   [B, Skv, Hk, D]  (BF16; MQA: Hk divides H, head kh = h / (H/Hk))
//   v:   [B, Skv, Hk, D]  (BF16; K == V in DeepSeek V4, treated independently here)
//   mask: [B, 1, Sq, Skv] (f32 additive, 0 / -inf)
//   sink: [H]             (f32 per-head logit)
//   out: [B, Sq, H, D]    (BF16)
//
// Constraints: D is a power of two and <= 512 (128 for the tiny/CI target, 512 for
// V4-Flash 64-head target). num_heads % num_heads_k == 0. BF16 only (DeepSeek V4 dtype).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16(x); }

// One thread per head-dim element; nt == D (power of two, <= 512). BK columns processed
// per online-softmax block to amortise the D-element accumulator rescale.
template <int BK>
__global__ void flash_attn_windowed_blockmask_kernel(
    const __nv_bfloat16* __restrict__ q,    // [B, Sq, H, D]
    const __nv_bfloat16* __restrict__ k,    // [B, Skv, Hk, D]
    const __nv_bfloat16* __restrict__ v,    // [B, Skv, Hk, D]
    const float* __restrict__ mask,         // [B, 1, Sq, Skv] additive 0/-inf
    const float* __restrict__ sink,         // [H] per-head logit
    __nv_bfloat16* __restrict__ out,        // [B, Sq, H, D]
    const int Sq, const int Skv, const int H, const int Hk, const int D,
    const float scale) {
    const int nt = D;
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int qrow = blockIdx.y;
    const int t = threadIdx.x;            // owns head-dim element d == t
    const int kh = h / (H / Hk);          // kv head (MQA broadcast)

    const long long q_base = ((long long)b * Sq + qrow) * (long long)H * D + (long long)h * D;
    const long long kv_base = ((long long)b * Skv) * (long long)Hk * D + (long long)kh * D;
    const long long mask_base = ((long long)b * Sq + qrow) * Skv;

    __shared__ float sm_scores[BK];
    __shared__ float sm_red[512];         // reduction scratch (one slot per thread, D <= 512)
    __shared__ float sm_m, sm_l, sm_rescale;
    __shared__ bool sm_skip;              // block-sparse: 1 if this BK-block is fully masked

    if (t == 0) {
        sm_m = sink[h];                   // running max starts at the sink logit (finite)
        sm_l = 1.0f;                      // exp(sink - m) with m == sink
    }
    __syncthreads();

    const float qv = to_float(q[q_base + t]);
    float acc = 0.0f;

    for (int col_base = 0; col_base < Skv; col_base += BK) {
        // --- BLOCK-SPARSE SKIP: if every column in this BK-block is fully masked
        // (-inf), skip it entirely - no QK dot, no K/V load, no softmax fold. An
        // -inf column contributes exactly 0 to the running max (which starts at the
        // finite sink logit), the denominator, and the accumulator, so skipping is
        // bit-identical to the dense reference while QK/V traffic is proportional
        // only to the attended blocks (sliding window + selected compressed blocks).
        if (t == 0) {
            bool skip = true;
            const int end = col_base + BK < Skv ? col_base + BK : Skv;
            for (int c = col_base; c < end; c++) {
                if (isfinite(mask[mask_base + c])) { skip = false; break; }
            }
            sm_skip = skip;
        }
        __syncthreads();
        if (sm_skip) continue;
        // --- compute the BK column scores (dot + scale + additive mask) ---
        for (int c = 0; c < BK; c++) {
            const int col = col_base + c;
            if (col < Skv) {
                // block reduction of per-thread partial dot products -> sm_scores[c]
                sm_red[t] = qv * to_float(k[kv_base + (long long)col * (long long)Hk * D + t]);
                __syncthreads();
                for (int s = nt / 2; s > 0; s >>= 1) {
                    if (t < s) sm_red[t] += sm_red[t + s];
                    __syncthreads();
                }
                if (t == 0) sm_scores[c] = sm_red[0] * scale + mask[mask_base + col];
            } else if (t == 0) {
                sm_scores[c] = -INFINITY;   // pad columns fully masked
            }
            __syncthreads();
        }

        // --- thread 0: block max, rescale running state, fold block into denominator ---
        if (t == 0) {
            float bm = sm_scores[0];
            for (int c = 1; c < BK; c++) bm = fmaxf(bm, sm_scores[c]);
            const float m_new = fmaxf(sm_m, bm);
            const float rescale = __expf(sm_m - m_new);
            sm_m = m_new;
            sm_rescale = rescale;
            sm_l = sm_l * rescale;
            float bl = 0.0f;
            for (int c = 0; c < BK; c++) bl += __expf(sm_scores[c] - m_new);
            sm_l += bl;
        }
        __syncthreads();

        // --- all threads: rescale accumulator, accumulate weighted V ---
        acc *= sm_rescale;
        for (int c = 0; c < BK; c++) {
            const int col = col_base + c;
            if (col < Skv) {
                const float w = __expf(sm_scores[c] - sm_m);
                acc += w * to_float(v[kv_base + (long long)col * (long long)Hk * D + t]);
            }
        }
        __syncthreads();
    }

    out[q_base + t] = to_bf16(acc / sm_l);
}

}  // namespace

// Host launcher. Layouts as documented above. All pointers must be device pointers.
// `stream_ptr` is the opaque CUDA stream (reinterpret_cast<cudaStream_t>).
extern "C" void flash_attn_windowed_blockmask(
    const void* q, const void* k, const void* v,
    const float* mask, const float* sink, void* out,
    const int B, const int Sq, const int Skv, const int H, const int Hk, const int D,
    const float softmax_scale, void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int BK = 64;
    dim3 grid(H * B, Sq);
    flash_attn_windowed_blockmask_kernel<BK><<<grid, D, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), mask, sink, static_cast<__nv_bfloat16*>(out),
        Sq, Skv, H, Hk, D, softmax_scale);
}
