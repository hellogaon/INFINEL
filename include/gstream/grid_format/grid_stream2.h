#ifndef GSTREAM_GRID_FORMAT_GRID_STREAM2_H
#define GSTREAM_GRID_FORMAT_GRID_STREAM2_H
#include <gstream/grid_format/grid_stream_defines.h>
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/detail/shard_tree.h>
#include <gstream/gstream_defines.h>
#include <ash/io/binary_file_stream.h>
#include <ash/detail/noncopyable.h>
#include <deque>

namespace gstream {
namespace grid_format {

using grid_stream_opt = unsigned;
class grid_stream2 final: ash::noncopyable {
public:
    struct config_t {
        char const*     graph_name;
        char const*     graph_dir;
        grid_stream_opt stream_opt;
        uint64_t        extent_size;
    };

    struct iterator;

    grid_stream2();
    ~grid_stream2() noexcept;

    stree_t const* stree(gbid_t const& gbid) const {
        return &_buffer.stree[columnar_offset(gbid, _info.dim)];
    }

    stree_t const* stree(grid_off_t const row, grid_off_t const col) const {
        return stree(make_gbid(row, col));
    }

    sleaf_t** sleaf(suid_type const suid) const {
        assert(_buffer.suid_index_v[suid] != nullptr);
        return &_buffer.suid_index_v[suid];
    }

    config_t const& config() const {
        return _cfg;
    }

    extent_load_info const&  ext_ld_info(extent_id_t const& ext_id) const;
    degree_block_info const& indeg_info(grid_off_t off) const;
    degree_block_info const& outdeg_info(grid_off_t off) const;
    grid_data_info const&    grid_info() const;

    bool open(config_t const& cfg);
    void close();

    iterator begin() const;
    iterator end() const;
    bool     load_outdeg(void* buf);
    bool     load_indeg(void* buf);
    bool     load_extent(void* buf, extent_load_info const& ld_info) const;
    bool     load_extent(void* buf, extent_id_t ext_id) const;

    uint64_t max_shard_size() const;
    uint64_t max_extent_size() const;
    uint64_t max_contigous_shards() const;
    uint64_t num_shards() const;
    uint64_t num_extents() const;

    bool is_open() const {
        return _stream.grid != nullptr;
    }

private:
    void _cleanup();
    void _error_log(gstream_retcode err);
    bool _load_file_buffered(void* buf, char const* path);

    bool _load_metadata();
    bool _open_grid_stream();
    bool _mapping_extents();

    config_t _cfg;
    grid_data_info _info;

    struct _buffer_struct {
        stree_t* stree;
        stree_node_block* stree_blocks;
        sleaf_t** suid_index_v;
        degree_block_info* degidx;
    } _buffer;

    struct _stream_struct {
        ash::binary_file_stream* grid;
        std::ifstream* indeg;
        std::ifstream* outdeg;
    } _stream;

    degree_block_info*     _indeg_info;
    degree_block_info*     _outdeg_info;
    detail::extent_mapper* _ext_mapper;
    std::deque<gstream_retcode>     _err_stk;
};

struct grid_stream2::iterator final {
    friend class grid_stream2;
    using end_type = std::nullptr_t;

    explicit iterator(grid_stream2 const& gs);
    iterator(grid_stream2 const& gs, end_type const&);
    iterator  operator++();
    iterator  operator++(int);
    iterator& operator=(end_type const&);
    bool operator==(iterator const& other) const;
    bool operator==(end_type const&) const;
    bool operator!=(iterator const& other) const;
    bool operator!=(end_type const&) const;
    stree_t const& operator*() const;

private:
    grid_dim_t _dim;
    stree_t const* _stree;
    laddr_t _row;
    laddr_t _col;
};

void print_grid_info2(grid_data_info const& info) noexcept;

}
}

#endif // GSTREAM_GRID_FORMAT_GRID_STREAM2_H
