#ifndef GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
#define GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_stream_defines.h>
#include <ash/detail/noncopyable.h>
#include <assert.h>
#include <functional>

namespace gstream {
namespace grid_format {

struct stree_node_block;
using stree_node_ptr = stree_node_block*;
using stree_const_node_ptr = stree_node_block const*;

using stree_intn_ptr   = stree_intn_t*;
using stree_leaf_ptr   = stree_leaf_t*;
using stree_parent_ptr = stree_intn_ptr;

#pragma pack(push, 8)
struct stree_node_header {
    stree_parent_ptr parent;        // 8-byte
    gbid_t   gbid;                  // 8-byte
    uint16_t level;                 // 2-byte ---+
    qtree_node_type node_type;      // 1-byte    |--> 8-byte
    qtree_location  location;       // 1-byte    |
    // ********************** padding: 4-byte ---+
    laddr_range_t   row_range;      // 8-byte
    laddr_range_t   col_range;      // 8-byte
};

struct stree_intn_t: stree_node_header {
/// Begin of Index tree (xtree) data layout:
    //! [Mark: qtree-traversal-order
    stree_node_ptr nw;         // 8-byte
    stree_node_ptr sw;         // 8-byte
    stree_node_ptr ne;         // 8-byte
    stree_node_ptr se;         // 8-byte
    qtree_node_type nw_type;   // 1-byte --+
    qtree_node_type sw_type;   // 1-byte   |
    qtree_node_type ne_type;   // 1-byte   |--> 8-byte
    qtree_node_type se_type;   // 1-byte   |
    int32_t         _unused;   // 4-byte --+
/// End of xtree data layout

    stree_node_ptr& child_ptr(unsigned row, unsigned col) {
        //! [Mark: qtree-traversal-order]
        assert(row < 2 && "index overflow error.");
        assert(col < 2 && "index overflow error.");
        col <<= 1;
        row |= col;
        return reinterpret_cast<stree_node_ptr*>(&nw)[row];
    }

    qtree_node_type& child_type(unsigned row, unsigned col) {
        //! [Mark: qtree-traversal-order]
        assert(row < 2 && "index overflow error.");
        assert(col < 2 && "index overflow error.");
        col <<= 1;
        row |= col;
        return reinterpret_cast<qtree_node_type*>(&nw_type)[row];
    }

};

struct stree_leaf_t: stree_node_header {
/// Begin of Index tree (xtree) data layout:
    suid_type unique_id;       // 8-byte
    uint64_t  physical_offset; // 8-byte
    // uint64_t  shard_size;   // REMOVED
    uint64_t  num_edges;       // 8-byte
    uint32_t  num_rows;        // 4-byte
    uint32_t  _reserved;       // 4-byte, runtime
    struct {
        uint64_t sparse_size;
        uint64_t dense_size;
    } size_info;
/// End of xtree data layout

    struct {
        extent_id_t extent_id;
        uint32_t    record_id;
    } ext_info;

    struct {
        degree_element_type indeg_elm_ty;
        uint64_t indeg_size;
        degree_element_type outdeg_elm_ty;
        uint64_t outdeg_size;
    } deg_info;

    struct {
        laddr_t min_row;
        laddr_t max_row;
        laddr_t min_col;
        laddr_t max_col;
    } rtaux; // run-time auxiliary information
};
#pragma pack(pop)

inline bool operator<(stree_leaf_t const& l, stree_leaf_t const& r) {
    return l.unique_id < r.unique_id;
}

inline bool operator>(stree_leaf_t const& l, stree_leaf_t const& r) {
    return l.unique_id > r.unique_id;
}

inline bool operator==(stree_leaf_t const& l, stree_leaf_t const& r) {
    return l.unique_id == r.unique_id;
}

inline bool is_same_sleaf(stree_leaf_t const& l, stree_leaf_t const& r) {
    return l == r;
}

void print_sleaf_info(stree_leaf_t const& sleaf);

struct stree_node_block {
    union {
        stree_node_header header;
        stree_intn_t      intn;
        stree_leaf_t      leaf;
    };

    operator stree_node_header&() {
        return header;
    }

    operator stree_intn_t&() {
        assert(header.node_type == qtree_node_type::InternalNode);
        return intn;
    }

    operator stree_leaf_t&() {
        assert(header.node_type == qtree_node_type::LeafNode);
        return leaf;
    }
};

class shard_tree: ash::noncopyable {
public:
    struct config_t {
        gbid_t gbid;
        stree_node_ptr root_node;
        qtree_node_type root_type;
        uint64_t num_nodes;
    };

    using traversal_callback = std::function<void(stree_node_ptr)>;

    shard_tree();
    shard_tree(config_t const& cfg);
    ~shard_tree() noexcept;

    void init(config_t const& cfg);
    void dfs(traversal_callback const& callback) const;
    void suid_order_search(traversal_callback const& callback) const;

    qtree_node_type root_type() const;
    stree_node_ptr  root_node() const;
    uint64_t        num_nodes() const;
    gbid_t const&   gbid() const;
    bool            empty() const;
    laddr_range_t   row_range() const;
    laddr_range_t   col_range() const;

private:
    config_t _cfg;
};

inline qtree_node_type shard_tree::root_type() const {
    return _cfg.root_type;
}

inline stree_node_ptr shard_tree::root_node() const {
    return _cfg.root_node;
}

inline uint64_t shard_tree::num_nodes() const {
    return _cfg.num_nodes;
}

inline gbid_t const& shard_tree::gbid() const {
    return _cfg.gbid;
}

inline bool shard_tree::empty() const {
    return _cfg.root_node == nullptr;
}

inline laddr_range_t shard_tree::row_range() const {
    return root_node()->header.row_range;
}

inline laddr_range_t shard_tree::col_range() const {
    return root_node()->header.col_range;
}

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
