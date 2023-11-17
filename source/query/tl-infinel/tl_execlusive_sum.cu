#include "tl_device_defines.cuh"
#include <cub/cub.cuh>
#include <ash/utility/dbg_log.h>

GSTREAM_HOST_ONLY
size_t tl_CUB_buffer_size(int num_items) {
    size_t bufsize = 0;
    colptr_t* dummy = nullptr;
    cudaError_t const err = cub::DeviceScan::ExclusiveSum(
        nullptr,
        bufsize,
        dummy,
        dummy,
        num_items
    );
    if (err != cudaError_t::cudaSuccess)
        ASH_FATAL("CUB Error; Failed to get a buffer size for CUB");
    bufsize = ash::aligned_size(bufsize, gstream::CudaMallocAlignment);
    return bufsize;
}

GSTREAM_HOST_ONLY
void tl_CUB_execlusive_sum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    colptr_t* d_in,
    colptr_t* d_out,
    int num_items,
    gstream::cuda::stream_type const& stream,
    bool debug_synchronous) {
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        (cudaStream_t)(stream),
        debug_synchronous
    );
}