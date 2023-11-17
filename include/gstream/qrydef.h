#ifndef GSTREAM_QRYDEF_H
#define GSTREAM_QRYDEF_H
#include <gstream/detail/kernel_binder.h>
#include <gstream/grid_format/grid_stream_defines.h>
#include <functional>
#include <ash/stop_watch.h>

namespace gstream {

struct superstep_state {
    uint64_t num_transactions = 0;
    uint64_t num_scheduled = 0;
    uint64_t num_processed = 0;
    uint64_t num_processing = 0;
    ash::stop_watch watch;
};

struct superstep_result {
    bool success = false;
    uint64_t num_transactions = 0;
    ash::stop_watch watch;
};

namespace procedure {

template <typename FnSig>
using qryproc = std::function<FnSig>;

using set_RA_size = qryproc<uint64_t(sleaf_t const* const */*sleaf_arr*/, unsigned /*num_shards*/)>;
using on_embed_RA = qryproc<void(sleaf_t const* const*/*sleaf_arr*/, grid_format::sparse_block const* const* /*shard_arr*/, unsigned /*num_shards*/, void* /*RA_buffer*/, gridtx_id_t /*tx_id*/)>;
using set_kernel_launch_params = qryproc<kernel_launch_parameters(sleaf_t const* const* /*sleaf_arr*/, unsigned /*num_shards*/, void* /*RA_buffer*/)>;
using set_usrbuf_size = qryproc<uint64_t(sleaf_t const* const*/*sleaf_arr*/, unsigned /*num_shards*/, void* /*RA_buffer*/)>;
using on_WA_init = qryproc<void(int /*device_id*/, void* /*initial_WA*/, uint64_t /*WA_size*/)>;
using on_WA_unit_init = qryproc<void(int /*device_id*/, unsigned /*column_idx*/, void* /*initial_WA_unit*/, uint64_t /*WA_size*/)>;
using on_WA_sync = qryproc<void(int /*device_id*/, void* /*WA*/, uint64_t /*WA_size*/)>;
using on_WA_unit_sync = qryproc<void(int /*device_id*/, unsigned/*column_idx*/,  void* /*WA*/, uint64_t /*WA_size*/)>;

// hooks
using h2dcpy_RA_hook = qryproc<void(void* /*RA_src_host*/, void* /*RA_dst_device*/, void* /*usrbuf_device*/)>;

} // namespace procedure

class gstream_query {
public:
    grid_format::grid_stream2* grid_stream;
    unsigned                   total_iteration;
    detail::kernel_binder*     kbind;

    struct qryopt_t {
        uint8_t num_operands;
        RA_option RA_opt;
        WA_option WA_opt;
        bool enable_usrbuf : 1;
        unsigned num_streams;
        uint64_t pinned_buffer_size;
        uint64_t device_buffer_size;
    } qryopt;

    struct qryproc_t {
        procedure::set_RA_size set_RA_size;
        procedure::on_embed_RA embed_RA;
        procedure::set_usrbuf_size set_usrbuf_size;
        procedure::on_WA_init on_WA_init;
        procedure::on_WA_unit_init on_WA_unit_init;
        procedure::on_WA_sync on_WA_sync;
        procedure::on_WA_unit_sync on_WA_unit_sync;
    } qryproc;

    struct hook_t {
        procedure::h2dcpy_RA_hook h2dcpy_RA;
    } hook;

    struct auxiliary_t {
        bool is_twostage = false;
        uint64_t total_twostage_tx;
        uint32_t* num_partiton_tx;
    } auxiliary;
};

namespace detail {
template <unsigned NumOperands, typename KernelScheduler>
class kernel_binder_template: public kernel_binder {
public:
    static constexpr unsigned num_operands = NumOperands;
    using kernel_scheduler = KernelScheduler;

    struct kernel_arguments {
        grid_format::sparse_block* shard[num_operands];
        void* indeg[num_operands];
        void* outdeg[num_operands];
        void* r_attr_host; // pinned memory (host)
        void* r_attr_dev;  // device memory
        void* w_attr; // device memory
        void* usrbuf; // device memory
        uint64_t usrbuf_size;
        gridtx_id_t tx_id;

        // for infinel
        kernel_scheduler kernel_sched;
        bool is_first_kernel_call;
    };

    struct auxiliary_t {
        uint32_t device_id;
        uint32_t stream_id;
    };

    using kernel_fnsig = void(kernel_arguments);
    using kernel_fnptr = std::add_pointer_t<kernel_fnsig>;
    using kernel_proxy = void(*)(kernel_fnptr const&, kernel_arguments const&, auxiliary_t const&, sleaf_t const* const*/*sleafs*/  , cuda::stream_type const& /*stream*/);

    kernel_binder_template() {
        memset(&args, 0, sizeof(kernel_arguments));
        kern = nullptr;
        pxy = nullptr;
    }

    kernel_binder_template(kernel_binder_template const& other) {
        memcpy(&args, &other.args, sizeof(kernel_arguments));
        kern = other.kern;
        pxy = other.pxy;
    }

    ~kernel_binder_template() noexcept override {
    }

    void call(sleaf_t const* const* sleaf_v, cuda::stream_type* stream) override {
        pxy(kern, args, aux, sleaf_v, *stream);
    }

    void bind_topology_pointers(unsigned index, grid_format::sparse_block* shard, void* indeg, void* outdeg) override {
        args.shard[index] = shard;
        args.indeg[index] = indeg;
        args.outdeg[index] = outdeg;
    }

    void bind_attributes(gridtx_id_t tx_id, void* r_attr_host, void* r_attr_dev, void* w_attr, void* usrbuf, uint64_t usrbuf_size) override {
        args.tx_id = tx_id;
        args.r_attr_host = r_attr_host;
        args.r_attr_dev = r_attr_dev;
        args.w_attr = w_attr;
        args.usrbuf = usrbuf;
        args.usrbuf_size = usrbuf_size;
    }

    void bind_auxiliary(uint32_t device_id, uint32_t stream_id) override {
        aux.device_id = device_id;
        aux.stream_id = stream_id;
    }

    uint64_t object_size() const override {
        return sizeof(kernel_binder_template);
    }

    void make_copy(kernel_binder* p) const override {
        kernel_binder_template* p2 = static_cast<kernel_binder_template*>(p);
        new (p2) kernel_binder_template(*this);
    }


    kernel_arguments args;
    auxiliary_t aux;
    kernel_fnptr kern;
    kernel_proxy pxy;
};

} // namespace detail

using detail::kernel_binder_template;

} // namespace gstream

#endif // GSTREAM_QRYDEF_H
