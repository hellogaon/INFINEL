#ifndef INFINEL_CHUNK_SCHEDULER_CUH
#define INFINEL_CHUNK_SCHEDULER_CUH
#include <infinel/infinel_kernel_defines.cuh>
#include <gstream/cuda_proxy.h>

namespace infinel {

template <typename WarpContext>
class chunk_scheduler {
public:
    using kernel_buffer = kernel_buffer<WarpContext>;

    struct config_t {
        kernel_buffer* kb;
        unsigned num_total_threads;
        uint64_t chunk_num;
        uint64_t chunk_size;
        uint64_t output_offset;
    };

    ~chunk_scheduler() noexcept {}

GSTREAM_HOST_ONLY
    void host_init(
            unsigned const num_total_threads, 
            uint64_t const chunk_num, 
            uint64_t const chunk_size,
            uint64_t const output_offset) {
        _cfg.num_total_threads = num_total_threads;
        _cfg.chunk_num = chunk_num - _cfg.num_total_threads;
        _cfg.chunk_size = chunk_size;
        _cfg.output_offset = output_offset;
        assert(chunk_num >= _cfg.num_total_threads);
    }

GSTREAM_DEVICE_ONLY
    void device_init(void* kb) {
        _cfg.kb = static_cast<kernel_buffer*>(kb);
    }

GSTREAM_DEVICE_ONLY
    void complete_thread_block() {
        if (threadIdx.x == 0) {
            _cfg.kb->p.logical.thread_block_state[blockIdx.x] = thread_block_state_type::COMPLETED;
            atomicAdd(&_cfg.kb->completed_counter, 1);
        }
    }

GSTREAM_DEVICE_ONLY
    thread_block_state_type return_thread_block_state() {
        return _cfg.kb->p.logical.thread_block_state[blockIdx.x];
    }

GSTREAM_DEVICE_ONLY
    WarpContext* load_warp_context() {
        return _cfg.kb->p.logical.warp_cxt;
    }

GSTREAM_DEVICE_ONLY
    void store_warp_context(uint32_t const warp_id, WarpContext const& rhs) {
        _cfg.kb->p.logical.warp_cxt[warp_id] = rhs;
    }

GSTREAM_DEVICE_ONLY
    void allocate_init(void*& buf) {
        uint32_t thread_unique_id = (blockIdx.x * blockDim.x) + threadIdx.x;
        buf = static_cast<char*>(_cfg.kb->p.logical.output) + _cfg.output_offset + (thread_unique_id * _cfg.chunk_size);
    }

GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
    void write_4byte_three_element(
            void*& buf, uint32_t& index, 
            uint32_t const A, uint32_t const B, uint32_t const C, 
            uint32_t const warp_id, 
            thread_block_kernel_state_type& thread_block_kernel_state, warp_state_type& warp_state) {

        *(static_cast<uint32_t*>(buf) + index) = A;
        *(static_cast<uint32_t*>(buf) + index + 1) = B;
        *(static_cast<uint32_t*>(buf) + index + 2) = C;
        index += 3;
        if ((index << 2) >= _cfg.chunk_size) {
            buf = _acquire_block();
            if (buf == nullptr) {
                thread_block_kernel_state = thread_block_kernel_state_type::STOPPED;
                warp_state = warp_state_type::STOPPED;
            }
            else index = 0;
        }
    }

GSTREAM_DEVICE_ONLY
    void add_element_counter(cuda_uint64_t const value) {
        CUDA_atomicAdd(&_cfg.kb->element_counter, value);
    }


private:
    config_t _cfg;

GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
    void* _acquire_block() const {
        cuda_uint64_t chunk_id = atomicAdd(&_cfg.kb->alloc_counter, 1);
        if (chunk_id < _cfg.chunk_num) {
            return static_cast<char*>(_cfg.kb->p.logical.output) + _cfg.output_offset + (( _cfg.num_total_threads + chunk_id) * _cfg.chunk_size);
        }
        return nullptr;
    }
};

} //namespace infinel

#endif // !INFINEL_CHUNK_SCHEDULER_CUH
