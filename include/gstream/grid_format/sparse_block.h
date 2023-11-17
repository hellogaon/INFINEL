#ifndef GSTREAM_GRID_FORMAT_SPARSE_BLOCK_H
#define GSTREAM_GRID_FORMAT_SPARSE_BLOCK_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/gstream_defines.h>
#include <gstream/bit32_vector.h>
#include <ash/size.h>
#include <assert.h>

namespace gstream {
namespace grid_format {

template <typename T>
GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
static constexpr T _cuda_aligned_size(T const size) {
    return ash::aligned_size(size, InternalDataSectionAlignment);
}

struct sparse_block_header {
    gbid_t gbid;
    struct {
        uint64_t rowv;
        uint64_t colv;
    } section_offset;
    uint32_t num_cols;
};

class sparse_block {
public:
    struct adj_list_t {
        laddr_t src_vertex;
        uint32_t length;
        v32_element const* _colv;
        uint32_t _colptr;

        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        laddr_t operator[](uint32_t const i) const {
            assert(i < length);
            assert(_is_aligned_address(_colv, 4));
            return read_32v_element(_colv, static_cast<uint64_t>(i) + _colptr);
        }
    };

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t num_lists() const {
        return ((header.section_offset.rowv - sizeof(sparse_block_header)) >> 2) - 1;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t num_edges() const {
        return header.num_cols;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint64_t block_size() const {
        return _aligned_size(sizeof(v32_element) * header.num_cols, InternalDataSectionAlignment) + header.section_offset.colv;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    gbid_t gbid() const {
        return header.gbid;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    adj_list_t adj_list(laddr_t const i) const {
        adj_list_t list;
        list.src_vertex = source_vertex(i);
        list.length = _list_size(i);
        list._colv = colv();
        list._colptr = colptr(i);
        return list;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    laddr_t source_vertex(laddr_t const i) const {
        laddr_t const src = read_32v_element(rowv(), i);
        return src;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t _list_size(laddr_t const i) const {
        return colptr(i + 1) - colptr(i);
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    void const* _data() const {
        return &header + 1;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    v32_element const* rowv() const {
        v32_element const* v = reinterpret_cast<v32_element const*>(_seek_pointer(this, header.section_offset.rowv));
        return v;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        v32_element const* colv() const {
        v32_element const* v = reinterpret_cast<v32_element const*>(_seek_pointer(this, header.section_offset.colv));
        return v;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    colptr_t colptr(laddr_t const row_off = 0) const {
        colptr_t const* v = static_cast<colptr_t const*>(_data());
        return v[row_off];
    }

    sparse_block_header header;

private:
    GSTREAM_DEVICE_COMPATIBLE static bool _is_aligned_address(void const* addr, uint64_t const alignment) {
    static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
        auto const addr2 = reinterpret_cast<uint64_t>(addr);
        return addr2 % alignment == 0;
    }

    template <typename T1, typename T2>
    GSTREAM_DEVICE_COMPATIBLE
    static constexpr T1 _aligned_size(const T1 size, const T2 align) {
        return align * ((size + align - 1) / align);
    }

    template <typename T>
    GSTREAM_DEVICE_COMPATIBLE
    static T* _seek_pointer(T* p, int64_t offset) {
        static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
        auto const p2 = (char*)p;
        return reinterpret_cast<T*>(p2 + offset);
    }
};

GSTREAM_HOST_ONLY void print_shard_info(sparse_block const* shard);

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_SPARSE_BLOCK_H
