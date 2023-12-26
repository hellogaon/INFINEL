#include "tl_defines.h"
#include "tl_device_defines.cuh"
#include <gstream/qrydef.h>
#include <gstream/grid_format/grid_stream2.h>
#include <ash/size.h>
#include <ash/utility/dbg_log.h>
#include <gstream/cuda_proxy.h>
#include <vector>
#include <algorithm>

struct tl_context {
    uint64_t num_triangles = 0;
    tl_kernel_binder kbind;
    struct {
        uint64_t total_kbuf_size = 0;
    } bench;
} *TL = nullptr;

tl_auxiliary_t* tl_aux;
void* tl_output_host;
infinel_system* ifn;

using gstream::grid_format::sparse_block;

/* Begin: Procedures */
uint64_t tl_set_RA_size(sleaf_t const* const* /*sleaf_arr*/, unsigned /*num_shards*/) {
    return sizeof(kernel_buffer) + ifn->get_RA_size();
}

void tl_embed_RA(sleaf_t const* const* sleaf_arr, sparse_block const* const* shard_arr, unsigned num_shards, void* RA_buffer, gstream::gridtx_id_t /*tx_id*/) {
    kernel_buffer* kbuf = static_cast<kernel_buffer*>(RA_buffer);
    void* ifn_kbuf = _seek_pointer(RA_buffer, sizeof(kernel_buffer));
    *kbuf = make_kernel_buffer(sleaf_arr[0], std::min(sleaf_arr[1]->num_rows, MaxBlocks));
    ifn->make_kernel_buffer(ifn_kbuf);
}

uint64_t tl_set_usrbuf_size(sleaf_t const* const*/*sleaf_arr*/, unsigned /*num_shards*/, void* RA_buffer) {
    kernel_buffer const* kbuf = static_cast<kernel_buffer*>(RA_buffer);
    void* ifn_kbuf = _seek_pointer(RA_buffer, sizeof(kernel_buffer));
    return kbuf->p.physical.bufsize + ifn->get_usrbuf_size(ifn_kbuf);
}

void tl_on_WA_init(int /*device_id*/, void* initial_WA, uint64_t const WA_size) {
    assert(WA_size == sizeof(uint64_t));
    memset(initial_WA, 0, WA_size);
}

void tl_on_WA_sync(int device_id, void* WA, uint64_t const /*WA_size*/) {
    uint64_t const num_trinagles = *static_cast<uint64_t*>(WA);
    ASH_DMESG("Sync device [%d]: %zu triangles", device_id, static_cast<size_t>(num_trinagles));
    TL->num_triangles += num_trinagles;
}
/* End of procedures */

/* Hooks */
void tl_on_h2dcpy_RA(void* RA_src_host, void* /*RA_dst_device*/, void* usrbuf_device) {
    kernel_buffer* kbuf = static_cast<kernel_buffer*>(RA_src_host);
    void* ifn_kbuf = _seek_pointer(RA_src_host, sizeof(kernel_buffer));
    void* ifn_usrbuf = _seek_pointer(usrbuf_device, kbuf->p.physical.bufsize);
    kbuf->physical_to_logical(usrbuf_device);
    ifn->physical_to_logical(ifn_kbuf, ifn_usrbuf);
}

/* infinel load balance function */

void tl_load_balance(
        tl_warp_context const& wc,
        tl_warp_context& wc_part1, 
        tl_warp_context& wc_part2, 
        unsigned num_blocks_per_grid) {

    uint32_t left = (wc.i_end - wc.i) / num_blocks_per_grid + 1;
    uint32_t mid = wc.i + ((left + 1) >> 1) * num_blocks_per_grid;
    assert(mid > num_blocks_per_grid);
    wc_part1 = {wc.i, wc.j, wc.k, mid};
    wc_part2 = {mid - num_blocks_per_grid, UINT32_MAX, 0, wc.i_end};
}

/* Query */
void init(program_args const* args) {
    // init_tl_context
    {
        assert(TL == nullptr);
        TL = new tl_context;
        TL->num_triangles = 0;
        TL->kbind.kern = tl_kernel;
        TL->kbind.pxy = tl_kernel_proxy;
    }

    // init_tl_aux
    {
        assert(tl_aux == nullptr);
        tl_aux = new tl_auxiliary_t;
        tl_aux->tl_buffer_size = args->chunk_num * args->chunk_size;
        tl_aux->double_buffering_mode = args->double_buffering_mode;
    }

    // init_infinel
    {   
        infinel_system::config_t cfg;
        cfg.num_blocks_per_grid = MaxBlocks;
        cfg.num_threads_per_block = NumThreads;
        cfg.warp_size = WarpSize;
        cfg.chunk_num = args->chunk_num;
        cfg.chunk_size = args->chunk_size;
        cfg.output_offset = args->chunk_num * args->chunk_size / 2;
        cfg.load_balance_mode = args->load_balance_mode;
        cfg.double_buffering_mode = args->double_buffering_mode;
        cfg.only_kernel_mode = args->only_kernel_mode;
        cfg.verification_mode = args->verification_mode;
        cfg.user_lb_fn = tl_load_balance;
        assert(cfg.chunk_size % sizeof(tl_output_format) == 0);
        ifn = new infinel_system;
        ifn->init(cfg);
    }

    // init_tl_output
    {
        tl_output_host = nullptr;
        uint64_t const aligned_size = ash::aligned_size(tl_aux->tl_buffer_size, DiskIOSectorSize);
        tl_output_host = gstream::cuda::pinned_malloc(aligned_size);
        memset(tl_output_host, 0, aligned_size);

    }
}

void cleanup() {
    // cleanup_tl_context
    assert(TL != nullptr);
    delete TL;
    TL = nullptr;

    // cleanup_tl_aux
    assert(tl_aux != nullptr);
    delete tl_aux;
    tl_aux = nullptr;

    // cleanup_infinel
    assert(ifn != nullptr);
    delete ifn;
    ifn = nullptr;

    // cleanup_tl_output
    gstream::cuda::pinned_free(tl_output_host);
}

void init_tl_query(
    gstream::gstream_query* q, 
    gstream::grid_format::grid_stream2* gs, 
    program_args const* args) {
    q->grid_stream = gs;
    q->kbind = &TL->kbind;
    q->total_iteration = 1;

    q->qryopt.num_operands = 3;
    q->qryopt.RA_opt.enabled = true;
    q->qryopt.RA_opt.use_in_degree = false;
    q->qryopt.RA_opt.use_out_degree = false;
    q->qryopt.RA_opt.use_usrdefRA = true;
    q->qryopt.RA_opt.fixed_size = true;
    q->qryopt.WA_opt.mode = gstream::WA_mode::DeviceLocal;
    q->qryopt.WA_opt.WA_size_per_device = sizeof(uint64_t);
    q->qryopt.enable_usrbuf = true;
    q->qryopt.num_streams = 1;
    q->qryopt.pinned_buffer_size = ash::GiB(20);
    q->qryopt.device_buffer_size = args->device_buffer_size;

    q->qryproc.set_RA_size = tl_set_RA_size;
    q->qryproc.embed_RA = tl_embed_RA;
    q->qryproc.set_usrbuf_size = tl_set_usrbuf_size;
    q->qryproc.on_WA_init = tl_on_WA_init;
    q->qryproc.on_WA_sync = tl_on_WA_sync;
    
    q->hook.h2dcpy_RA = tl_on_h2dcpy_RA;
}

void begin_tl(gstream::gstream_framework* fw) {
    using namespace gstream;
    using namespace framework3;
    superstep_result const rslt = begin_superstep(fw);
    assert(rslt.success);
    ASH_LOG("Result: %zu", TL->num_triangles);
}
