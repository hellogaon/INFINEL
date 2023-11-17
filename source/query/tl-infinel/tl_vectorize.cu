#include "tl_device_defines.cuh"
#include <gstream/grid_format/sparse_block.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

GSTREAM_CUDA_KERNEL
void tl_generate_adj_length(laddr_t* outbuf, sparse_block const* shard, laddr_t const bias) {
    using namespace gstream;

    uint32_t const thr_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const num_threads_in_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t const num_blocks_in_grid = gridDim.x * gridDim.y * gridDim.z;
    uint32_t const num_threads = num_threads_in_block * num_blocks_in_grid;
    uint32_t const num_rows = shard->num_lists();

    assert(outbuf != nullptr);
    for (uint32_t i = thr_id; i < num_rows; i += num_threads) {
        assert(bias <= shard->source_vertex(i));
        outbuf[shard->source_vertex(i) - bias] = shard->colptr(i + 1) - shard->colptr(i);
    }
}

GSTREAM_CUDA_KERNEL
void tl_set_zero_by_source_vertex(uint32_t* outbuf, sparse_block const* shard, laddr_t const bias) {
    using namespace gstream;

    uint32_t const thr_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const num_threads_in_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t const num_blocks_in_grid = gridDim.x * gridDim.y * gridDim.z;
    uint32_t const num_threads = num_threads_in_block * num_blocks_in_grid;
    uint32_t const num_rows = shard->num_lists();

    assert(outbuf != nullptr);
    for (uint32_t i = thr_id; i < num_rows; i += num_threads) {
        assert(bias <= shard->source_vertex(i));
        outbuf[shard->source_vertex(i) - bias] = 0;
    }
}

GSTREAM_HOST_ONLY
void vectorize(
    sparse_block const* GV_dev,
    kernel_buffer* kbuf,
    kernel_launch_parameters const* kparams,
    gstream::cuda::stream_type const& stream) {

    // vectorize GV
    GSTREAM_CUDA_KERNEL_CALL(
        tl_generate_adj_length,
        kparams->num_blocks_per_grid,
        kparams->num_threads_per_block,
        0,
        static_cast<cudaStream_t>(stream)
    ) (kbuf->p.logical.u32temp, GV_dev, kbuf->GV_min_vid);
    assert(kbuf->GV_length < INT_MAX);
    tl_CUB_execlusive_sum(
        kbuf->p.logical.cub.buffer,
        kbuf->p.logical.cub.bufsize,
        kbuf->p.logical.u32temp,
        kbuf->p.logical.GV_colptr,
        static_cast<int>(kbuf->GV_length),
        stream,
        false
    );
    GSTREAM_CUDA_KERNEL_CALL(
        tl_set_zero_by_source_vertex,
        kparams->num_blocks_per_grid,
        kparams->num_threads_per_block,
        0,
        static_cast<cudaStream_t>(stream)
    ) (kbuf->p.logical.u32temp, GV_dev, kbuf->GV_min_vid);
}