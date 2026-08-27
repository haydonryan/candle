// Custom DeepSeek V4 DSA flash kernel for variable-length (continuous-batching)
// batches with a paged K/V cache. This is the decode/serving analogue of
// `flash_attn_windowed_blockmask`: it fuses the exact eager op
// (QK^T*scale + heterogeneous additive mask + per-head sink logit +
// max-subtract + softmax + drop-sink + @V), but the KV axis lives in physical
// pages addressed through a per-sequence `block_table` and the batch is
// variable-length (each query row knows its own sequence and KV length).
//
// Block-sparse execution is preserved: the KV axis is processed in BK=64-column
// blocks and any block whose columns are all masked (`-inf`) is skipped, so QK/V
// traffic scales only with the attended blocks (sliding window + selected
// compressed blocks), not the full max KV length.
//
// Layouts (contiguous):
//   q:   [total_q, H,  D]   (BF16)          - one row per query (varlen)
//   k:   [num_blocks, page_block_size, Hk, D] (BF16, paged)
//   v:   [num_blocks, page_block_size, Hk, D] (BF16, paged; K == V in DeepSeek V4)
//   cu_seqlens_q: [B + 1] (i32) cumulative query lengths
//   cu_seqlens_k: [B + 1] (i32) cumulative KV lengths
//   block_table: [B, max_blocks] (i32) physical block index per (seq, block-in-seq);
//                trailing entries unused (padded) and never dereferenced past kvlen.
//   mask: [total_q, max_kv] (f32 additive, 0 / -inf) one row per query; columns
//         >= that query's kvlen are -inf (padding).
//   sink: [H] (f32 per-head logit)
//   out: [total_q, H, D] (BF16)
//
// Constraints: D power of two <= 512; num_heads % num_heads_k == 0;
// page_block_size a multiple of 64; BF16 only (DeepSeek V4 dtype).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16(x); }

template <int BK>
__global__ void flash_attn_varlen_paged_blockmask_kernel(
    const __nv_bfloat16* __restrict__ q,    // [total_q, H, D]
    const __nv_bfloat16* __restrict__ k,    // [num_blocks, page_block_size, Hk, D]
    const __nv_bfloat16* __restrict__ v,    // [num_blocks, page_block_size, Hk, D]
    const int* __restrict__ cu_seqlens_q,   // [B+1]
    const int* __restrict__ cu_seqlens_k,   // [B+1]
    const int* __restrict__ block_table,    // [B, max_blocks]
    const float* __restrict__ mask,         // [total_q, max_kv]
    const float* __restrict__ sink,         // [H]
    __nv_bfloat16* __restrict__ out,        // [total_q, H, D]
    const int B, const int total_q, const int max_kv,
    const int H, const int Hk, const int D,
    const int page_block_size, const int max_blocks,
    const float scale) {
    const int nt = D;
    const int row = blockIdx.x;          // total_q row
    const int h = blockIdx.y;            // head
    const int t = threadIdx.x;           // owns head-dim element d == t

    // Map the query row to its sequence: find b with
    // cu_seqlens_q[b] <= row < cu_seqlens_q[b+1].
    int b = 0;
    for (int i = 0; i < B; i++) {
        if (row < cu_seqlens_q[i + 1]) { b = i; break; }
    }
    const int kh = h / (H / Hk);         // kv head (MQA broadcast)
    const int kvlen = cu_seqlens_k[b + 1] - cu_seqlens_k[b];
    const int* __restrict__ bt = block_table + (long long)b * max_blocks;

    const long long q_base = (long long)row * H * D + (long long)h * D;
    const long long mask_base = (long long)row * max_kv;

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

    for (int col_base = 0; col_base < max_kv; col_base += BK) {
        // --- BLOCK-SPARSE SKIP: if every column in this BK-block is fully masked
        // (-inf, including columns past kvlen), skip it entirely.
        if (t == 0) {
            bool skip = true;
            const int end = col_base + BK < max_kv ? col_base + BK : max_kv;
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
            if (col < kvlen) {
                // Physical paged K index for (sequence col -> block, offset).
                const int blk_in_seq = col / page_block_size;
                const int off = col % page_block_size;
                const int phys = bt[blk_in_seq];
                const long long kv_idx =
                    ((long long)phys * page_block_size + off) * (long long)Hk * D +
                    (long long)kh * D + t;
                // block reduction of per-thread partial dot products -> sm_scores[c]
                sm_red[t] = qv * to_float(k[kv_idx]);
                __syncthreads();
                for (int s = nt / 2; s > 0; s >>= 1) {
                    if (t < s) sm_red[t] += sm_red[t + s];
                    __syncthreads();
                }
                if (t == 0) sm_scores[c] = sm_red[0] * scale + mask[mask_base + col];
            } else if (t == 0) {
                sm_scores[c] = -INFINITY;   // columns past this sequence's kvlen: masked
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
            if (col < kvlen) {
                const int blk_in_seq = col / page_block_size;
                const int off = col % page_block_size;
                const int phys = bt[blk_in_seq];
                const long long kv_idx =
                    ((long long)phys * page_block_size + off) * (long long)Hk * D +
                    (long long)kh * D + t;
                const float w = __expf(sm_scores[c] - sm_m);
                acc += w * to_float(v[kv_idx]);
            }
        }
        __syncthreads();
    }

    out[q_base + t] = to_bf16(acc / sm_l);
}

}  // namespace

// Host launcher. Layouts as documented above. All pointers must be device pointers.
// `stream_ptr` is the opaque CUDA stream (reinterpret_cast<cudaStream_t>).
extern "C" void flash_attn_varlen_paged_blockmask(
    const void* q, const void* k, const void* v,
    const int* cu_seqlens_q, const int* cu_seqlens_k, const int* block_table,
    const float* mask, const float* sink, void* out,
    const int B, const int total_q, const int max_kv,
    const int H, const int Hk, const int D,
    const int page_block_size, const int max_blocks,
    const float softmax_scale, void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int BK = 64;
    dim3 grid(total_q, H);
    flash_attn_varlen_paged_blockmask_kernel<BK><<<grid, D, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), cu_seqlens_q, cu_seqlens_k, block_table,
        mask, sink, static_cast<__nv_bfloat16*>(out),
        B, total_q, max_kv, H, Hk, D, page_block_size, max_blocks, softmax_scale);
}
