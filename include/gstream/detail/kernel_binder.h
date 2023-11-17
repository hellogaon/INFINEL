#ifndef GSTREAM_DETAIL_KERNEL_BINDER_H
#define GSTREAM_DETAIL_KERNEL_BINDER_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_stream_defines.h>

namespace gstream {
namespace detail {

class kernel_binder {
public:
    virtual ~kernel_binder() noexcept = default;
    virtual void bind_topology_pointers(unsigned index, grid_format::sparse_block* shard, void* indeg, void* outdeg) = 0;
    virtual void bind_attributes(gridtx_id_t tx_id, void* r_attr_host, void* r_attr_dev, void* w_attr, void* usrbuf, uint64_t usrbuf_size) = 0;
    virtual void bind_auxiliary(uint32_t device_id, uint32_t stream_id) = 0;
    virtual void call(sleaf_t const* const* sleaf_v, cuda::stream_type* stream) = 0;
    virtual uint64_t object_size() const = 0;
    virtual void make_copy(kernel_binder*) const = 0;
    
};

} // namespace detail
} // namespace gstream

#endif // !GSTREAM_DETAIL_KERNEL_BINDER_H
