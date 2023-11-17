#ifndef INFINEL_INFINEL_KERNEL_DEFINES_CUH
#define INFINEL_INFINEL_KERNEL_DEFINES_CUH
#include <gstream/cuda_proxy.h>

namespace infinel {

enum class thread_block_state_type : bool {
    RUNNING = false,
    COMPLETED = true
};

enum class thread_block_kernel_state_type : bool {
    RUNNING = false,
    STOPPED = true
};

enum class warp_state_type : bool {
    RUNNING = false,
    STOPPED = true
};

template <typename WarpContext> 
struct kernel_buffer {
    cuda_uint64_t completed_counter;
    cuda_uint64_t alloc_counter;
    cuda_uint64_t element_counter;
    
    union _pointer_t {
        struct physical_t {
            uint64_t bufsize;
            uint64_t warp_cxt;
            uint64_t thread_block_state;
            uint64_t output;
        } physical;

        struct logical_t {
            uint64_t bufsize;
            WarpContext* warp_cxt;
            thread_block_state_type* thread_block_state;
            void* output;
        } logical;
    } p;
};

} //namespace infinel

#endif // !INFINEL_INFINEL_KERNEL_DEFINES_CUH
