use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,
        total_q: u32,

        is_bf16: c_int,
        is_causal: c_int,
        unpadded_lse: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,

        block_table_ptr: *const i32,
        block_table_batch_stride: u32,
        page_block_size: c_int,

        mm_prefix_ranges_ptr: *const i32,
        mm_prefix_range_batch_stride: u32,
        max_mm_prefix_ranges: c_int,
        stream_ptr: *mut c_void,
    );

    /// Custom DeepSeek V4 DSA flash kernel: QK^T*scale + heterogeneous additive mask
    /// (sliding-window local prefix | per-query block_bias compressed suffix) + per-head
    /// sink logit + max-subtract + softmax + drop-sink + @V. BF16 only.
    ///
    /// Layouts (contiguous): q/k/v [B, Sq|Skv, H|Hk, D] bf16; mask [B, 1, Sq, Skv] f32
    /// (0/-inf); sink [H] f32. Output [B, Sq, H, D] bf16.
    pub(crate) fn flash_attn_windowed_blockmask(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        mask_ptr: *const c_void,
        sink_ptr: *const c_void,
        out_ptr: *const c_void,
        b: c_int,
        sq: c_int,
        skv: c_int,
        h: c_int,
        hk: c_int,
        d: c_int,
        softmax_scale: f32,
        stream_ptr: *mut c_void,
    );

    /// Custom DeepSeek V4 DSA flash kernel for variable-length (continuous-batching)
    /// batches with a paged K/V cache. Layouts (contiguous): q/out [total_q, H, D]
    /// bf16; k/v [num_blocks, page_block_size, Hk, D] bf16; cu_seqlens_q/k [B+1] i32;
    /// block_table [B, max_blocks] i32; mask [total_q, max_kv] f32 (0/-inf); sink [H] f32.
    pub(crate) fn flash_attn_varlen_paged_blockmask(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        cu_seqlens_q_ptr: *const c_int,
        cu_seqlens_k_ptr: *const c_int,
        block_table_ptr: *const c_int,
        mask_ptr: *const c_void,
        sink_ptr: *const c_void,
        out_ptr: *const c_void,
        b: c_int,
        total_q: c_int,
        max_kv: c_int,
        h: c_int,
        hk: c_int,
        d: c_int,
        page_block_size: c_int,
        max_blocks: c_int,
        softmax_scale: f32,
        stream_ptr: *mut c_void,
    );

}
