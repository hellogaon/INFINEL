#ifndef GSTREAM_GRID_FORMAT_DEFINES_H
#define GSTREAM_GRID_FORMAT_DEFINES_H
#include <ash/size.h>
#include <gstream/gstream_defines.h>
#include <ash/vectorization.h>
#include <limits>

namespace gstream {
namespace grid_format {

using gaddr_t = uint64_t;
using laddr_t = uint32_t;
using colptr_t = uint32_t;
using grid_off_t = uint32_t;
using grid_dim_t = ash::dim2<uint32_t>;

constexpr unsigned LocalAddrBits = 30;
constexpr unsigned GridBlockWidth = 1u << LocalAddrBits;  

constexpr grid_off_t InvalidGridOffset = std::numeric_limits<grid_off_t>::max();

using suid_type = uint64_t;
constexpr suid_type InvalidShardUnqiueID = std::numeric_limits<suid_type>::max();
using ioid_type = uint32_t;

class  index_tree;
struct xtree_disk_node;
struct xtree_intn_t;
struct xtree_leaf_t;

using xtree_t = index_tree;
using xintn_t = xtree_intn_t;
using xleaf_t = xtree_leaf_t;

typedef struct gbid_struct {
    using compressed_t = uint64_t;
    grid_off_t row;
    grid_off_t col;

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        compressed_t compressed() const {
        return *reinterpret_cast<uint64_t const*>(this);
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        operator compressed_t() const {
        return compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>(gbid_struct const& rhs) const {
        return compressed() > rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator<=(gbid_struct const& rhs) const {
        return compressed() <= rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>=(gbid_struct const& rhs) const {
        return compressed() >= rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator==(gbid_struct const& rhs) const {
        return compressed() == rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator!=(gbid_struct const& rhs) const {
        return compressed() != rhs.compressed();
    }

    ASH_FORCEINLINE void set(grid_off_t const row_, grid_off_t const col_) {
        row = row_;
        col = col_;
    }

    ASH_FORCEINLINE void set(compressed_t const compressed) {
        *reinterpret_cast<compressed_t*>(this) = compressed;
    }

} gbid_t;
static_assert(sizeof(gbid_t) == 8, "a size of gbid_struct is not 8");

ASH_FORCEINLINE gbid_t make_gbid(grid_off_t const row, grid_off_t const col) {
    return gbid_t{ row, col };
}

ASH_FORCEINLINE uint64_t columnar_offset(
    grid_off_t const row, grid_off_t const col, grid_dim_t const& dim) {
    uint64_t r = col;
    r *= dim.y;
    r += row;
    return r;
}

ASH_FORCEINLINE uint64_t columnar_offset(gbid_t const& gbid, grid_dim_t const& dim) {
    return columnar_offset(gbid.row, gbid.col, dim);
}

struct global_edge_t {
    gaddr_t src;
    gaddr_t dst;
};

struct e32_t {
    laddr_t u;
    laddr_t v;
};

inline bool e32_compare(e32_t const& l, e32_t const& r) {
    if (l.u != r.u)
        return l.u < r.u;
    return l.v < r.v;
}

inline bool e32_equal(e32_t const& l, e32_t const& r) {
    return l.u == r.u && l.v == r.v;
}

inline bool e32_is_diagonal(e32_t const& e) {
    return e.u == e.v;
}

inline bool e32_is_not_diagonal(e32_t const& e) {
    return e.u != e.v;
}

struct segmented_block_header;
class segmented_block;

struct sparse_block_header;
class sparse_block;

template <typename T = void*>
struct unified_pointer {
    static_assert(std::is_pointer<T>::value, "A template argument T is must be pointer type");
    union {
        uint64_t physical;
        T logical;
    };
};

enum class qtree_node_type : char {
    NullNode = 0,
    InternalNode = 1,
    LeafNode = 2,
};

enum class qtree_location : char {
    NW = 0,
    SW = 1,
    NE = 2,
    SE = 3
};

using grid_opt_t = unsigned;

struct grid_data_info {
    char name[128];
    grid_dim_t dim;
    uint64_t vertex_range;
    uint64_t num_ext_vertices;
    uint64_t grid_size;
    uint64_t num_edges;
    uint64_t xtree_size;
    uint64_t num_shards;
    uint64_t indeg_size;
    uint64_t outdeg_size;
    unsigned max_indeg_unit_size;
    unsigned max_outdeg_unit_size;
    uint64_t base_shard_size;
    grid_opt_t grid_option;
};

enum class degree_type: char {
    NIL = 0,
    InDegree = 1,
    OutDegree
};

enum class degree_element_type: char {
    NIL = 0,
    U8 = 1,
    U16 = 2,
    U24 = 3,
    U32 = 4,
    U64 = 8
};

inline uint64_t degree_buffer_size_fixed_ver(unsigned const num_rows, degree_element_type /*elem_ty*/) {
    uint64_t size = num_rows * FixedDegreeUnitSize;
    return ash::aligned_size(size, CudaMallocAlignment);
}

template <typename T>
struct degree_block {
    using value_type = T;
    static constexpr value_type limit = std::numeric_limits<value_type>::max();

    T data[GridBlockWidth];

    value_type operator[](laddr_t const off) const {
        return get(off);
    }
    value_type get(laddr_t off) const {
        return data[off];
    }
    void set(laddr_t off, value_type val) {
        data[off] = val;
    }
};

struct degree_block_info {
    degree_element_type unit_type;
    uint64_t offset;
};

using degree_block_u32 = degree_block<uint32_t>;

template <typename T>
struct range_template {
    using value_type = T;
    value_type lo;
    value_type hi;
    value_type width() const {
        return hi - lo + 1;
    }
};

using laddr_range_t = range_template<laddr_t>;
using gaddr_range_t = range_template<gaddr_t>;

} // namespace grid_format

using grid_format::xtree_t;
using grid_format::xleaf_t;

} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DEFINES_H
