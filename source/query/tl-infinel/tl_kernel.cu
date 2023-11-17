// ReSharper disable CppLocalVariableMightNotBeInitialized
// ReSharper disable CppClangTidyClangDiagnosticSometimesUninitialized
// ReSharper disable CppClangTidyClangDiagnosticConditionalUninitialized
#include "tl_device_defines.cuh"
#include "tl_defines.h"
#include <gstream/grid_format/sparse_block.h>
#include <gstream/grid_format/detail/shard_tree.h>
#include <gstream/cuda_proxy.h>
#include <ash/utility/dbg_log.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

GSTREAM_DEVICE_ONLY
bool get_bit(uint32_t const vertex_id, uint32_t const* bitmap_lev0, uint32_t const* bitmap_lev1) {
    if (bitmap_lev0[vertex_id >> 12UL] & (1 << ((vertex_id >> (12UL - 5UL) & 31))))
        return bitmap_lev1[vertex_id >> 5UL] & (1 << (vertex_id & 31));
    return false;
}

GSTREAM_DEVICE_ONLY void set_bit(uint32_t const vertex_id, uint32_t* bitmap_lev0, uint32_t* bitmap_lev1) {
    CUDA_atomicOr(
        &bitmap_lev1[vertex_id >> 5UL],
        1 << (vertex_id & 31)
    );
    CUDA_atomicOr(
        &bitmap_lev0[vertex_id >> 12UL],
        1 << ((vertex_id >> (12UL - 5UL)) & 31)
    );
}

GSTREAM_HOST_ONLY
kernel_buffer make_kernel_buffer(
    sleaf_t const* G_sleaf,
    unsigned num_blocks) {

    kernel_buffer kbuf;
    uint32_t GV_len, max_vectorize_len, lv1bitmap_size, lv0bitmap_size;
    sleaf_t const* GV_sleaf = G_sleaf;

    // data length
    {
        laddr_t const GVmin = GV_sleaf->rtaux.min_row;
        laddr_t const GVmax = GV_sleaf->rtaux.max_row;
        GV_len = GVmax - GVmin + 2; 
        max_vectorize_len = std::max(GV_len, GV_len);

        laddr_t const bmin = G_sleaf->rtaux.min_col;
        laddr_t const bmax = G_sleaf->rtaux.max_col;
        lv1bitmap_size = ash::aligned_size(ash::aligned_size(bmax - bmin + 1, 8) / 8, 4);
        lv0bitmap_size = ash::aligned_size(ash::aligned_size(lv1bitmap_size, 128) / 128, 4);
        assert(lv1bitmap_size > 0);
        assert(lv0bitmap_size > 0);

        kbuf.GV_min_vid = GVmin;
        kbuf.GV_length = GV_len;
        kbuf.G_min_col = bmin;
        kbuf.G_max_col = bmax;
        kbuf.G_lv1bitmap_size = lv1bitmap_size;
        kbuf.G_lv0bitmap_size = lv0bitmap_size;
        kbuf.max_vectorize_length = max_vectorize_len;
    }
    
    uint64_t offset = 0;
    {
        {
            uint64_t const arr_size = max_vectorize_len * sizeof(colptr_t);
            kbuf.p.physical.u32temp = offset;
            offset += ash::aligned_size(arr_size, gstream::CudaMallocAlignment);
        }
        {
            uint64_t const arr_size = GV_len * sizeof(colptr_t);
            kbuf.p.physical.GV_colptr = offset;
            offset += ash::aligned_size(arr_size, gstream::CudaMallocAlignment);
        }
    }

    // Bitmap
    {
        kbuf.p.physical.bitmap_lv1 = offset;
        uint64_t const lev1_size = static_cast<uint64_t>(lv1bitmap_size) * num_blocks;
        offset += ash::aligned_size(lev1_size, gstream::CudaMallocAlignment);

        kbuf.p.physical.bitmap_lv0 = offset;
        uint64_t const lev0_size = static_cast<uint64_t>(lv0bitmap_size) * num_blocks;
        offset += ash::aligned_size(lev0_size, gstream::CudaMallocAlignment);
    }

    // CUB buffer
    {
        kbuf.p.physical.cub.buffer = offset;
        size_t const cub_bufsize = tl_CUB_buffer_size(static_cast<int>(max_vectorize_len));
        kbuf.p.physical.cub.bufsize = cub_bufsize;
        offset += cub_bufsize;
    }

    kbuf.p.physical.bufsize = offset;
    return kbuf;
}

GSTREAM_HOST_ONLY
void tl_kernel_proxy(
    tl_kernel_binder::kernel_fnptr const& kern,
    kernel_arguments const& kargs,
    auxiliary_t const& aux,
    sleaf_t const* const* sleaf_v,
    gstream::cuda::stream_type const& stream) {
#ifndef NULL_GPU_KERNEL
    kernel_launch_parameters kparams;
    {
        sleaf_t const& pivot = *sleaf_v[1];
        kparams.num_threads_per_block = NumThreads;
        kparams.num_blocks_per_grid = std::min(pivot.num_rows, MaxBlocks);
        kparams.shared_mem_size = 0;
    }

    sparse_block const* G = kargs.shard[0];
    if (!gstream::cuda::device_memset_async(kargs.usrbuf, 0, kargs.usrbuf_size, stream)) {
        ASH_FATAL("CUDA device memset failure!");
    }

    kernel_buffer* kbuf = static_cast<kernel_buffer*>(kargs.r_attr_host);
    vectorize(G, kbuf, &kparams, stream);

    void* ifn_kbuf_dev = _seek_pointer(kargs.r_attr_dev, sizeof(kernel_buffer));
    void* ifn_kbuf_host = _seek_pointer(kargs.r_attr_host, sizeof(kernel_buffer));
    
    if (tl_aux->double_buffering_mode)
        ifn->infinel_double_buffering_exec(kern, kargs, kparams, stream, ifn_kbuf_host, ifn_kbuf_dev, tl_output_host);
    else 
        ifn->infinel_exec(kern, kargs, kparams, stream, ifn_kbuf_host, ifn_kbuf_dev, tl_output_host);

    return;
#endif
}

GSTREAM_CUDA_KERNEL
void tl_kernel(kernel_arguments args) {
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WarpSize;
    uint32_t warp_local_id = threadIdx.x / WarpSize;
    uint32_t chunk_index = 0;
    cuda_uint64_t thread_local_counter = 0;
    
    kernel_buffer const* kbuf = static_cast<kernel_buffer*>(args.r_attr_dev);
    void* ifn_kbuf = static_cast<kernel_buffer*>(_seek_pointer(args.r_attr_dev, sizeof(kernel_buffer)));
    
    args.kernel_sched.device_init(ifn_kbuf);
    __syncthreads();

    if (args.kernel_sched.return_thread_block_state() == infinel::thread_block_state_type::COMPLETED) return;
    
    void* buffer;
    tl_warp_context wc_state;
    bool load_wc = !args.is_first_kernel_call;

    __shared__ uint32_t* bitmap_lv1;
    __shared__ uint32_t* bitmap_lv0;
    __shared__ laddr_t GV_min_vid;
    __shared__ laddr_t GV_max_vid;
    __shared__ laddr_t G_bitmap_min_vid;
    __shared__ laddr_t G_bitmap_max_vid;

    __shared__ colptr_t* GV_colptr;
    __shared__ tl_warp_context* wc;
    __shared__ infinel::chunk_scheduler<tl_warp_context>* kernel_sched;
    __shared__ infinel::thread_block_kernel_state_type thread_block_kernel_state;
    __shared__ infinel::warp_state_type warp_state[NumThreads / WarpSize];
    
    if (threadIdx.x == 0) {
        bitmap_lv1 = _seek_pointer(kbuf->p.logical.bitmap_lv1, static_cast<uint64_t>(kbuf->G_lv1bitmap_size) * blockIdx.x);
        bitmap_lv0 = _seek_pointer(kbuf->p.logical.bitmap_lv0, static_cast<uint64_t>(kbuf->G_lv0bitmap_size) * blockIdx.x);
        GV_min_vid = kbuf->GV_min_vid;
        GV_max_vid = kbuf->GV_length + kbuf->GV_min_vid - 2;
        G_bitmap_min_vid = kbuf->G_min_col;
        G_bitmap_max_vid = kbuf->G_max_col;
        GV_colptr = kbuf->p.logical.GV_colptr;
        kernel_sched = &args.kernel_sched;
        wc = kernel_sched->load_warp_context();
        thread_block_kernel_state = infinel::thread_block_kernel_state_type::RUNNING;
        for (uint32_t i = 0; i < NumThreads / WarpSize; i++)
            warp_state[i] = infinel::warp_state_type::RUNNING;
    }
    __syncthreads();
    
    kernel_sched->allocate_init(buffer);
    wc_state = {(wc + warp_id)->i, (wc + warp_id)->j, (wc + warp_id)->k, (wc + warp_id)->i_end};

    sparse_block const* G = args.shard[0];

#define GVPTR_OFF(x) ((x) - GV_min_vid)
#define BMAP_OFF(x) ((x) - G_bitmap_min_vid)

    uint32_t const num_rows_of_G = G->num_lists();

    if (args.is_first_kernel_call) {    
        wc_state.i_end = num_rows_of_G;
    }

    for (uint32_t iter1 = (load_wc ? wc_state.i : blockIdx.x); iter1 < wc_state.i_end; iter1 += gridDim.x) {
        sparse_block::adj_list_t const G_row = G->adj_list(iter1);
    
        if (G_row.src_vertex < GV_min_vid)
            continue;
        if (G_row.src_vertex > GV_max_vid)
            break;
        uint32_t const col_beg_1 = GV_colptr[GVPTR_OFF(G_row.src_vertex)];
        uint32_t const col_end_1 = GV_colptr[GVPTR_OFF(G_row.src_vertex + 1)];
        if (col_beg_1 == col_end_1)
            continue;

        if (!load_wc)
            for (uint32_t i = threadIdx.x; i < G_row.length; i += blockDim.x) {
                set_bit(BMAP_OFF(G_row[i]), bitmap_lv0, bitmap_lv1);
            }

        __syncthreads();

        gstream::v32_element const* G_colv = G->colv();

        for (laddr_t iter2 = (load_wc ? wc_state.j : col_beg_1); iter2 < col_end_1; iter2 += 1) {
            laddr_t const G_col = read_32v_element(G_colv, iter2);

            if (G_col < GV_min_vid)
                continue;
            if (G_col > GV_max_vid)
                break;
            uint32_t const col_beg_2 = GV_colptr[GVPTR_OFF(G_col)];
            uint32_t const col_end_2 = GV_colptr[GVPTR_OFF(G_col + 1)];

            if (col_beg_2 == col_end_2)
                continue;

            for (uint32_t V_col_idx = (load_wc ? wc_state.k : col_beg_2) + threadIdx.x; V_col_idx < col_end_2; V_col_idx += blockDim.x) {
                uint32_t const target = read_32v_element(G_colv, V_col_idx);

                if (target < G_bitmap_min_vid)
                    continue;
                if (target > G_bitmap_max_vid)
                    break;

                if (get_bit(BMAP_OFF(target), bitmap_lv0, bitmap_lv1)) {
                    uint32_t A = G_row.src_vertex;
                    uint32_t B = G_col;
                    uint32_t C = target;
                    kernel_sched->write_4byte_three_element(buffer, chunk_index, A, B, C, warp_id, thread_block_kernel_state, warp_state[warp_local_id]);
                    thread_local_counter += 1;
                }
                if (warp_state[warp_local_id] == infinel::warp_state_type::STOPPED) {
                    if ((threadIdx.x & 31) == 0) {
                        wc_state = {iter1, iter2, V_col_idx - threadIdx.x + blockDim.x, wc_state.i_end};
                        kernel_sched->store_warp_context(warp_id, wc_state);
                    }
                    CUDA_atomicAdd(static_cast<cuda_uint64_t*>(args.w_attr), thread_local_counter);
                    kernel_sched->add_element_counter(thread_local_counter);
                    return;
                }
            }
            __syncwarp();
            if (warp_state[warp_local_id] == infinel::warp_state_type::STOPPED) {
                CUDA_atomicAdd(static_cast<cuda_uint64_t*>(args.w_attr), thread_local_counter);
                kernel_sched->add_element_counter(thread_local_counter);
                return;
            }
            load_wc = false;
        }
        __syncthreads();

        if (thread_block_kernel_state == infinel::thread_block_kernel_state_type::STOPPED) {
            if ((threadIdx.x & 31) == 0) {
                wc_state = {iter1, UINT32_MAX, 0, wc_state.i_end};
                kernel_sched->store_warp_context(warp_id, wc_state);
            }
                CUDA_atomicAdd(static_cast<cuda_uint64_t*>(args.w_attr), thread_local_counter);
                kernel_sched->add_element_counter(thread_local_counter);
            return;
        }

        for (uint32_t i = threadIdx.x; i < G_row.length; i += blockDim.x) {
            uint32_t const c = BMAP_OFF(G_row[i]);
            bitmap_lv0[c >> 12UL] = 0;
            bitmap_lv1[c >> 5UL]  = 0;
        }
        __syncthreads();

        load_wc = false;
    }

    kernel_sched->complete_thread_block();

    if ((threadIdx.x & 31) == 0) {
        wc_state = {UINT32_MAX, 0, 0, 0};
        kernel_sched->store_warp_context(warp_id, wc_state);
    }

    CUDA_atomicAdd(static_cast<cuda_uint64_t*>(args.w_attr), thread_local_counter);
    kernel_sched->add_element_counter(thread_local_counter);
}
