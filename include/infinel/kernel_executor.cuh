#ifndef INFINEL_KERNEL_EXECUTOR_CUH
#define INFINEL_KERNEL_EXECUTOR_CUH
#include <gstream/cuda_proxy.h>

namespace infinel {

template <typename WarpContext, typename KernelBinder>
class kernel_executor {
public:
    using kernel_binder = KernelBinder;
    using kernel_arguments = typename kernel_binder::kernel_arguments;
    using kernel_fnptr = typename kernel_binder::kernel_fnptr;

    ~kernel_executor() noexcept {}


GSTREAM_HOST_ONLY
    void init() {}

GSTREAM_HOST_ONLY
    void kernel_call(
        kernel_fnptr const& kern,
        kernel_arguments const& kargs,
        gstream::kernel_launch_parameters const& kparams,
        gstream::cuda::stream_type const& stream) {
        GSTREAM_CUDA_KERNEL_CALL(
            kern,
            kparams.num_blocks_per_grid,
            kparams.num_threads_per_block,
            kparams.shared_mem_size,
            static_cast<cudaStream_t>(stream)
        ) (kargs);
    } 
};

} //namespace infinel

#endif // !INFINEL_KERNEL_EXECUTOR_CUH

