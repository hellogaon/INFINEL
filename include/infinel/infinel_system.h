#ifndef INFINEL_INFINEL_SYSTEM_H
#define INFINEL_INFINEL_SYSTEM_H
#include <infinel/chunk_scheduler.cuh>
#include <infinel/kernel_executor.cuh>
#include <infinel/kernel_buffer_manager.cuh>
#include <queue>

namespace infinel {

constexpr static uint64_t D2HBatchSize = 1000000000;

template <typename WarpContext, typename KernelBinder>
class infinel_system {
public:
    using warp_context = WarpContext;
    using kernel_binder = KernelBinder;
    using kernel_arguments = typename kernel_binder::kernel_arguments;
    using kernel_fnptr = typename kernel_binder::kernel_fnptr;
    using user_lb_fnptr = void(*)(warp_context const& /*wc*/, 
                          tl_warp_context& /*wc_part1*/, 
                          tl_warp_context& /*wc_part2*/, 
                          unsigned /*num_blocks_per_grid*/);

    struct config_t {
        unsigned num_threads_per_block;
        unsigned num_blocks_per_grid;
        unsigned warp_size;
        uint64_t chunk_num;
        uint64_t chunk_size;
        uint64_t output_offset;
        bool load_balance_mode;
        bool double_buffering_mode;
        bool only_kernel_mode;
        bool verification_mode;
        user_lb_fnptr user_lb_fn;
    };

    ~infinel_system() noexcept {
        if (_cfg.load_balance_mode) {
            free(_lb_bef.warp_cxt);
            free(_lb_aft.warp_cxt);
            free(_lb_bef.thread_block_state);
            free(_lb_aft.thread_block_state);
        }
    }

    void init(config_t const& cfg) {
        _cfg = cfg;

        _kern_exec.init();
        _kb_mngr.init(_cfg.num_threads_per_block, _cfg.num_blocks_per_grid, _cfg.warp_size, _cfg.chunk_num, _cfg.chunk_size);

        if (_cfg.load_balance_mode) {
            _lb_bef.warp_cxt = static_cast<warp_context*>(malloc(_kb_mngr.get_warp_cxt_size()));
            _lb_aft.warp_cxt = static_cast<warp_context*>(malloc(_kb_mngr.get_warp_cxt_size()));
            _lb_bef.thread_block_state = static_cast<thread_block_state_type*>(malloc(_kb_mngr.get_thread_block_state_size()));
            _lb_aft.thread_block_state = static_cast<thread_block_state_type*>(malloc(_kb_mngr.get_thread_block_state_size()));
        }
    }
    
    uint64_t get_RA_size() {
        return _kb_mngr.get_RA_size();
    }

    uint64_t get_usrbuf_size(void* RA_buffer) {
        return _kb_mngr.get_usrbuf_size(RA_buffer);
    }

    void make_kernel_buffer(void* RA_buffer) {
        return _kb_mngr.make_kernel_buffer(RA_buffer);
    }

    void physical_to_logical(void* RA_buffer, void* base) {
        return _kb_mngr.physical_to_logical(RA_buffer, base);
    }

    void infinel_exec(
            kernel_fnptr const& kern,
            kernel_arguments kargs,
            gstream::kernel_launch_parameters const& kparams,
            gstream::cuda::stream_type const& stream,
            void* ifn_kbuf_host,
            void* ifn_kbuf_dev,
            void* output) { 
        
        uint64_t completed_counter;
        uint64_t element_counter;
        uint64_t alloc_counter;
        uint64_t const total_completed_counter = kparams.num_blocks_per_grid;
        kernel_buffer<warp_context>* kbuf_host = static_cast<kernel_buffer<warp_context>*>(ifn_kbuf_host);
        kernel_buffer<warp_context>* kbuf_dev = static_cast<kernel_buffer<warp_context>*>(ifn_kbuf_dev);
        bool is_first_kernel_call = true;
        
        uint64_t kernel_iteration = 1;
        uint64_t num_total_threads = kparams.num_threads_per_block * kparams.num_blocks_per_grid;
        _chunk_sched[0].host_init(num_total_threads, _cfg.chunk_num, _cfg.chunk_size, 0);
        kargs.kernel_sched = _chunk_sched[0];
        
        while(true) {
            // init
            kargs.is_first_kernel_call = is_first_kernel_call;
            _kernel_device_init(kbuf_host, kbuf_dev, stream, 0);

            // kernel call
            _kern_exec.kernel_call(kern, kargs, kparams, stream);
            gstream::cuda::stream_synchronize(stream);

            // D2H
            _kernel_D2H_primitive(kbuf_host, kbuf_dev, completed_counter, element_counter, alloc_counter, stream, 0);
            _kernel_D2H_reference(kbuf_host, kbuf_dev, output, alloc_counter, num_total_threads, stream, 0);
            gstream::cuda::stream_synchronize(stream);
            
            if (_cfg.load_balance_mode) {
                gstream::cuda::stream_synchronize(stream);
                _load_balance(kbuf_host, kbuf_dev, completed_counter, total_completed_counter, kparams, stream);
            }

            // sync
            gstream::cuda::stream_synchronize(stream);

            printf("kernel_iteration: %lu element_counter: %lu completed_block: %lu/%lu\n", kernel_iteration, element_counter, completed_counter, total_completed_counter);

            if (_cfg.verification_mode)
                _triangle_listing_verification(output, element_counter, alloc_counter, num_total_threads, 0);

            if (completed_counter == total_completed_counter)
                break;

            is_first_kernel_call = false;
            kernel_iteration += 1;
        }
    }

    void infinel_double_buffering_exec(
            kernel_fnptr const& kern,
            kernel_arguments kargs,
            gstream::kernel_launch_parameters const& kparams,
            gstream::cuda::stream_type const& stream,
            void* ifn_kbuf_host,
            void* ifn_kbuf_dev,
            void* output) { 
        
        uint64_t completed_counter;
        uint64_t element_counter;
        uint64_t alloc_counter;
        uint64_t const total_completed_counter = kparams.num_blocks_per_grid;
        kernel_buffer<warp_context>* kbuf_host = static_cast<kernel_buffer<warp_context>*>(ifn_kbuf_host);
        kernel_buffer<warp_context>* kbuf_dev = static_cast<kernel_buffer<warp_context>*>(ifn_kbuf_dev);
        bool is_first_kernel_call = true;
        cudaStream_t stream2;
        cudaStreamCreate(&stream2);

        uint64_t kernel_iteration = 1;
        uint64_t num_total_threads = kparams.num_threads_per_block * kparams.num_blocks_per_grid;
        _chunk_sched[0].host_init(num_total_threads, _cfg.chunk_num / 2, _cfg.chunk_size, 0);
        _chunk_sched[1].host_init(num_total_threads, _cfg.chunk_num / 2, _cfg.chunk_size, _cfg.output_offset);
        kargs.kernel_sched = _chunk_sched[0];
        kargs.is_first_kernel_call = is_first_kernel_call;

        // stream init
        _kernel_device_init(kbuf_host, kbuf_dev, stream, 0);

        // stream kernel call
        _kern_exec.kernel_call(kern, kargs, kparams, stream);

        gstream::cuda::stream_synchronize(stream);
        kargs.is_first_kernel_call = false;

        while(true) {
            // stream D2H & stream2 init, kernel call
            kargs.kernel_sched = _chunk_sched[1];
            _kernel_D2H_primitive(kbuf_host, kbuf_dev, completed_counter, element_counter, alloc_counter, stream2, 1);
            gstream::cuda::stream_synchronize(stream2);
            if (_cfg.load_balance_mode) {
                _load_balance(kbuf_host, kbuf_dev, completed_counter, total_completed_counter, kparams, stream2);
            }
            _kernel_D2H_reference(kbuf_host, kbuf_dev, output, alloc_counter, num_total_threads, stream, 0);
            _kernel_device_init(kbuf_host, kbuf_dev, stream2, 1);
            _kern_exec.kernel_call(kern, kargs, kparams, stream2);

            // stream sync
            gstream::cuda::stream_synchronize(stream);
            gstream::cuda::stream_synchronize(stream2);

            printf("kernel_iteration: %lu element_counter: %lu completed_block: %lu/%lu\n", kernel_iteration, element_counter, completed_counter, total_completed_counter);

            if (_cfg.verification_mode)
                _triangle_listing_verification(output, element_counter, alloc_counter, num_total_threads, 0);

            if (completed_counter == total_completed_counter)
                break;

            kernel_iteration += 1;

            // stream init, kernel call & stream2 D2H
            kargs.kernel_sched = _chunk_sched[0];
            _kernel_D2H_primitive(kbuf_host, kbuf_dev, completed_counter, element_counter, alloc_counter, stream, 0);
            gstream::cuda::stream_synchronize(stream);
            if (_cfg.load_balance_mode) {
                _load_balance(kbuf_host, kbuf_dev, completed_counter, total_completed_counter, kparams, stream);
            }
            _kernel_D2H_reference(kbuf_host, kbuf_dev, output, alloc_counter, num_total_threads, stream2, 1);
            _kernel_device_init(kbuf_host, kbuf_dev, stream, 0);
            _kern_exec.kernel_call(kern, kargs, kparams, stream);

            // stream sync
            gstream::cuda::stream_synchronize(stream);
            gstream::cuda::stream_synchronize(stream2);

            printf("kernel_iteration: %lu element_counter: %lu completed_block: %lu/%lu\n", kernel_iteration, element_counter, completed_counter, total_completed_counter);

            if (_cfg.verification_mode)
                _triangle_listing_verification(output, element_counter, alloc_counter, num_total_threads, 1);

            if (completed_counter == total_completed_counter)
                break;

            kernel_iteration += 1;
        }

        cudaStreamDestroy(stream2);
    }

private:
    config_t _cfg;
    chunk_scheduler<warp_context> _chunk_sched[2];
    kernel_executor<warp_context, kernel_binder> _kern_exec;
    kernel_buffer_manager<warp_context> _kb_mngr;

    struct load_balance_pointer {
        warp_context* warp_cxt;
        thread_block_state_type* thread_block_state;
    } _lb_bef, _lb_aft;

    void _kernel_device_init(
            kernel_buffer<warp_context>* kbuf_host, 
            kernel_buffer<warp_context>* kbuf_dev, 
            gstream::cuda::stream_type const& stream,
            uint32_t const stream_id) {

        gstream::cuda::device_memset_async(
            &kbuf_dev->alloc_counter,
            0, 
            sizeof(cuda_uint64_t), 
            stream
        );

        gstream::cuda::device_memset_async(
            &kbuf_dev->element_counter,
            0, 
            sizeof(cuda_uint64_t), 
            stream
        );

        if (_cfg.double_buffering_mode) {
            gstream::cuda::device_memset_async(
                static_cast<char*>(kbuf_host->p.logical.output) + stream_id * _cfg.output_offset, 
                0, 
                _cfg.output_offset,
                stream
            );
        }
        else {
            gstream::cuda::device_memset_async(
                kbuf_host->p.logical.output, 
                0, 
                _kb_mngr.get_output_size(),
                stream
            );
        }
    }

   void _kernel_D2H_primitive( 
        kernel_buffer<warp_context>* kbuf_host, 
        kernel_buffer<warp_context>* kbuf_dev, 
        uint64_t& completed_counter,
        uint64_t& element_counter,
        uint64_t& alloc_counter,
        gstream::cuda::stream_type const& stream,
        uint32_t const stream_id) {
        
        gstream::cuda::d2hcpy_async(
            &completed_counter,
            &(kbuf_dev->completed_counter),
            sizeof(uint64_t),
            static_cast<cudaStream_t>(stream)
        );

        gstream::cuda::d2hcpy_async(
            &element_counter,
            &(kbuf_dev->element_counter),
            sizeof(uint64_t),
            static_cast<cudaStream_t>(stream)
        );

        gstream::cuda::d2hcpy_async(
            &alloc_counter,
            &(kbuf_dev->alloc_counter),
            sizeof(uint64_t),
            static_cast<cudaStream_t>(stream)
        );
    }

    void _kernel_D2H_reference( 
        kernel_buffer<warp_context>* kbuf_host, 
        kernel_buffer<warp_context>* kbuf_dev, 
        void* output,
        uint64_t& alloc_counter,
        uint64_t& active_chunks,
        gstream::cuda::stream_type const& stream,
        uint32_t const stream_id) {
        if (!_cfg.only_kernel_mode) {
            if (_cfg.double_buffering_mode) {
                uint64_t total_size = min(alloc_counter + active_chunks, _cfg.chunk_num / 2) * _cfg.chunk_size;
                uint64_t batch_size = D2HBatchSize;
                uint32_t iter = (total_size + batch_size - 1) / batch_size;

                for (uint32_t i = 0; i < iter; i++) {
                    uint64_t start_addr = batch_size * i;
                    gstream::cuda::d2hcpy_async(
                        static_cast<char*>(output) + stream_id * _cfg.output_offset + start_addr,
                        static_cast<char*>(kbuf_host->p.logical.output) + stream_id * _cfg.output_offset + start_addr,
                        min(total_size, batch_size),
                        static_cast<cudaStream_t>(stream)
                    );
                    total_size -= batch_size;
                }
            }
            else {
                uint64_t total_size = min(alloc_counter + active_chunks, _cfg.chunk_num) * _cfg.chunk_size;
                uint64_t batch_size = D2HBatchSize;
                uint32_t iter = (total_size + batch_size - 1) / batch_size;
                
                for (uint32_t i = 0; i < iter; i++) {
                    uint64_t start_addr = batch_size * i;
                    gstream::cuda::d2hcpy_async(
                        static_cast<char*>(output) + start_addr,
                        static_cast<char*>(kbuf_host->p.logical.output) + start_addr,
                        min(total_size, batch_size),
                        static_cast<cudaStream_t>(stream)
                    );
                    total_size -= batch_size;
                }
            }
        }
    }
    
    void _triangle_listing_verification(void* output, uint64_t const total_num_triangles, uint64_t alloc_counter, uint64_t active_chunks, uint32_t stream_id) {
        uint64_t num_triangles = 0;
        uint64_t const tl_output_size = min(alloc_counter + active_chunks, _cfg.chunk_num / (_cfg.double_buffering_mode ? 2 : 1)) * _cfg.chunk_size;
        uint64_t const len = tl_output_size / sizeof(uint32_t);
        void* tl_output = static_cast<void*>(static_cast<char*>(output) + stream_id * _cfg.output_offset);

        printf("\t triangle_num |        triangle        \n");
        printf("\t--------------+------------------------\n");

        for (uint64_t i = 0; i < len; i += 3) {
            uint32_t A = *(static_cast<uint32_t const*>(tl_output) + i    ); 
            uint32_t B = *(static_cast<uint32_t const*>(tl_output) + i + 1); 
            uint32_t C = *(static_cast<uint32_t const*>(tl_output) + i + 2); 

            if(A || B || C) {
                num_triangles++;
                printf("\t%13lu | [%u, %u, %u]\n", num_triangles, A, B, C);
            }
        }
        assert(total_num_triangles == num_triangles);
    }

    void _load_balance_D2H(
            kernel_buffer<warp_context>* kbuf_host,
            kernel_buffer<warp_context>* /*kbuf_dev*/,
            gstream::cuda::stream_type const& stream) {

        gstream::cuda::d2hcpy_async(
            _lb_bef.warp_cxt,
            kbuf_host->p.logical.warp_cxt,
            _kb_mngr.get_warp_cxt_size(),
            stream
        );

        gstream::cuda::d2hcpy_async(
            _lb_bef.thread_block_state,
            kbuf_host->p.logical.thread_block_state,
            _kb_mngr.get_thread_block_state_size(),
            stream
        );
    }

    void _load_balance_H2D(
            kernel_buffer<warp_context>* kbuf_host,
            kernel_buffer<warp_context>* kbuf_dev,
            uint64_t& completed_counter,
            gstream::cuda::stream_type const& stream) {

        gstream::cuda::h2dcpy_async(
            &(kbuf_dev->completed_counter),
            &completed_counter,
            sizeof(uint64_t),
            stream
        );

        gstream::cuda::h2dcpy_async(
            kbuf_host->p.logical.warp_cxt,
            _lb_aft.warp_cxt,
            _kb_mngr.get_warp_cxt_size(),
            stream
        );

        gstream::cuda::h2dcpy_async(
            kbuf_host->p.logical.thread_block_state,
            _lb_aft.thread_block_state,
            _kb_mngr.get_thread_block_state_size(),
            stream
        );

    }

    void _load_balance(
            kernel_buffer<warp_context>* kbuf_host, 
            kernel_buffer<warp_context>* kbuf_dev, 
            uint64_t& completed_counter,
            uint64_t const& total_completed_counter,
            gstream::kernel_launch_parameters const& kparams,
            gstream::cuda::stream_type const& stream) {
        
        if (!(completed_counter != total_completed_counter && completed_counter > total_completed_counter / 2)) return;

        uint32_t const num_warps_per_block = kparams.num_threads_per_block / _cfg.warp_size;

        _load_balance_D2H(kbuf_host, kbuf_dev, stream);
        gstream::cuda::stream_synchronize(stream);
        
        std::queue<uint32_t> thread_block_id_pool;
        for (uint32_t i = 0; i < kparams.num_blocks_per_grid; i++)
            if (_lb_bef.thread_block_state[i] == thread_block_state_type::COMPLETED) {
                _lb_aft.thread_block_state[i] = thread_block_state_type::COMPLETED;
                thread_block_id_pool.push(i);
            }

        for (uint32_t i = 0; i < kparams.num_blocks_per_grid; i++) {
            if (_lb_bef.thread_block_state[i] == thread_block_state_type::RUNNING) {
                uint32_t balance_block_id = thread_block_id_pool.front();
                thread_block_id_pool.pop();
                for (uint32_t j = 0; j < num_warps_per_block; j++) {
                    _cfg.user_lb_fn(
                        _lb_bef.warp_cxt[i * num_warps_per_block + j], 
                        _lb_aft.warp_cxt[i * num_warps_per_block + j], 
                        _lb_aft.warp_cxt[balance_block_id * num_warps_per_block + j],
                        kparams.num_blocks_per_grid);
                }
                _lb_aft.thread_block_state[i] = thread_block_state_type::RUNNING;
                _lb_aft.thread_block_state[balance_block_id] = thread_block_state_type::RUNNING;
            }
        }
        
        completed_counter = thread_block_id_pool.size();
        _load_balance_H2D(kbuf_host, kbuf_dev, completed_counter, stream);
    }

};

} //namespace infinel

#endif // !INFINEL_INFINEL_SYSTEM_H
