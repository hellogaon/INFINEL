#ifndef INFINEL_KERNEL_BUFFER_MANAGER_CUH
#define INFINEL_KERNEL_BUFFER_MANAGER_CUH
#include <infinel/infinel_kernel_defines.cuh>
#include <gstream/cuda_proxy.h>
#include <ash/size.h>
#include <ash/pointer.h>

namespace infinel {

template <typename WarpContext>
class kernel_buffer_manager {
public:
    struct config_t {
        unsigned num_threads_per_block;
        unsigned num_blocks_per_grid;
        unsigned warp_size;
        uint64_t chunk_num;
        uint64_t chunk_size;
    };

    ~kernel_buffer_manager() noexcept {}

    using kernel_buffer = kernel_buffer<WarpContext>;

GSTREAM_HOST_ONLY
    void init(
            unsigned const num_threads_per_block, 
            unsigned const num_blocks_per_grid, 
            unsigned const warp_size, 
            uint64_t const chunk_num,
            uint64_t const chunk_size) {
        _cfg.num_threads_per_block = num_threads_per_block;
        _cfg.num_blocks_per_grid = num_blocks_per_grid;
        _cfg.warp_size = warp_size;
        _cfg.chunk_num = chunk_num;
        _cfg.chunk_size = chunk_size;
    }

GSTREAM_HOST_ONLY
    uint64_t get_RA_size() {
        return sizeof(kernel_buffer);
    }

GSTREAM_HOST_ONLY
    uint64_t get_usrbuf_size(void* RA_buffer) {
        kernel_buffer const* kbuf = static_cast<kernel_buffer*>(RA_buffer);
        return kbuf->p.physical.bufsize;
    }

GSTREAM_HOST_ONLY
    uint64_t get_warp_cxt_size() {
        return sizeof(WarpContext) * _cfg.num_blocks_per_grid *  _cfg.num_threads_per_block / _cfg.warp_size;
    }

GSTREAM_HOST_ONLY
    uint64_t get_thread_block_state_size() {
        return sizeof(thread_block_state_type) * _cfg.num_blocks_per_grid;
    }

GSTREAM_HOST_ONLY
    uint64_t get_output_size() {
        return _cfg.chunk_num * _cfg.chunk_size;
    }

GSTREAM_HOST_ONLY
    void make_kernel_buffer(void* RA_buffer) {
        kernel_buffer* kbuf = static_cast<kernel_buffer*>(RA_buffer);

        uint64_t offset = 0;
        {
            uint64_t const vec_size = get_warp_cxt_size();
            kbuf->p.physical.warp_cxt = offset;
            offset += vec_size;
        }
        {
            uint64_t const vec_size = get_thread_block_state_size();
            kbuf->p.physical.thread_block_state = offset;
            offset += vec_size;
        }
        {
            uint64_t const output_size = get_output_size();
            kbuf->p.physical.output = offset;
            offset += output_size;
        }
        kbuf->p.physical.bufsize = offset;
    }

GSTREAM_HOST_ONLY
    void physical_to_logical(void* RA_buffer, void* base) {
        kernel_buffer* kbuf                 = static_cast<kernel_buffer*>(RA_buffer);
        kbuf->p.logical.warp_cxt            = static_cast<WarpContext*>(ash::seek_pointer(base, kbuf->p.physical.warp_cxt));
        kbuf->p.logical.thread_block_state   = static_cast<thread_block_state_type*>(ash::seek_pointer(base, kbuf->p.physical.thread_block_state));
        kbuf->p.logical.output           = ash::seek_pointer(base, kbuf->p.physical.output);
    }

private:
    config_t _cfg;
};

} //namespace infinel

#endif // !INFINEL_KERNEL_BUFFER_MANAGER_CUH
