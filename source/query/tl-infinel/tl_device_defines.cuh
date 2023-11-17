#pragma once
#include "tl_defines.h"
#include "tl_kernel_type.h"
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_stream_defines.h>
#include <infinel/infinel_system.h>
#include <gstream/cuda_env.h>

using gstream::grid_format::laddr_t;
using gstream::grid_format::colptr_t;
using gstream::grid_format::sparse_block;
using gstream::grid_format::sleaf_t;
using gstream::grid_format::GridBlockWidth;
using gstream::kernel_launch_parameters;
using gstream::superstep_result;

using infinel_system = infinel::infinel_system<tl_warp_context, tl_kernel_binder>;
extern infinel_system* ifn;

template <typename T>
GSTREAM_DEVICE_COMPATIBLE
T* _seek_pointer(T* p, int64_t offset) {
    static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
    char* const p2 = reinterpret_cast<char*>(p);
    return reinterpret_cast<T*>(p2 + offset);
}

template <typename T>
GSTREAM_DEVICE_COMPATIBLE
T* _seek_pointer(T* p, uint64_t offset) {
    return _seek_pointer(p, static_cast<int64_t>(offset));
}

struct kernel_buffer {
    laddr_t  GV_min_vid;
    uint32_t GV_length;
    laddr_t  G_min_col;
    laddr_t  G_max_col;
    uint32_t G_lv1bitmap_size;
    uint32_t G_lv0bitmap_size;
    uint64_t max_vectorize_length;
    
    union _pointer_t {
        struct physical_t {
            uint64_t bufsize;
            uint64_t u32temp;
            uint64_t GV_colptr;
            uint64_t bitmap_lv0;
            uint64_t bitmap_lv1;
            struct {
                uint64_t buffer;
                uint64_t bufsize;
            } cub;
        } physical;

        struct logical_t {
            uint64_t  bufsize;
            colptr_t* u32temp;
            colptr_t* GV_colptr;
            uint32_t* bitmap_lv0;
            uint32_t* bitmap_lv1;
            struct {
                void* buffer;
                uint64_t bufsize;
            } cub;
        } logical;
    } p;

    static_assert(sizeof(_pointer_t::physical_t) == sizeof(_pointer_t::logical_t), "Size mismatch!");

    GSTREAM_DEVICE_COMPATIBLE void physical_to_logical(void* base) {
        p.logical.u32temp    = static_cast<laddr_t*>(base);
        p.logical.GV_colptr     = static_cast<laddr_t*>(_seek_pointer(base, p.physical.GV_colptr));
        p.logical.bitmap_lv0 = static_cast<uint32_t*>(_seek_pointer(base, p.physical.bitmap_lv0));
        p.logical.bitmap_lv1 = static_cast<uint32_t*>(_seek_pointer(base, p.physical.bitmap_lv1));
        p.logical.cub.buffer = _seek_pointer(base, p.physical.cub.buffer);
    }
};

GSTREAM_HOST_ONLY void vectorize(
    sparse_block const* GV_dev,
    kernel_buffer* kbuf,
    kernel_launch_parameters const* kparams,
    gstream::cuda::stream_type const& stream);
GSTREAM_HOST_ONLY kernel_buffer make_kernel_buffer(
    sleaf_t const* G_sleaf, unsigned num_blocks);
GSTREAM_HOST_ONLY void tl_kernel_proxy(
    tl_kernel_binder::kernel_fnptr const& kern,
    kernel_arguments const& kargs,
    auxiliary_t const& aux,
    sleaf_t const* const* sleaf_v,
    gstream::cuda::stream_type const& stream
);
GSTREAM_CUDA_KERNEL void tl_kernel(kernel_arguments args);

GSTREAM_HOST_ONLY
size_t tl_CUB_buffer_size(
    int num_items
);

GSTREAM_HOST_ONLY
void tl_CUB_execlusive_sum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    colptr_t* d_in,
    colptr_t* d_out,
    int num_items,
    gstream::cuda::stream_type const& stream,
    bool debug_synchronous
);