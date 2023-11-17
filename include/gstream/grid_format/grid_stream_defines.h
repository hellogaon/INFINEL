#ifndef GSTREAM_GRID_FORMAT_GRID_STREAM_DEFINES_H
#define GSTREAM_GRID_FORMAT_GRID_STREAM_DEFINES_H
#include <stdint.h>

namespace gstream {
namespace grid_format {

class grid_stream2;

struct shard_load_info;
struct extent_load_info;
struct extent_config;
struct extent_t;

using extent_id_t = uint32_t;
constexpr static extent_id_t InvalidExtentID = UINT32_MAX;

class  shard_tree;
struct stree_intn_t;
struct stree_leaf_t;
struct stree_node_block;

using stree_t = shard_tree;
using sintn_t = stree_intn_t;
using sleaf_t = stree_leaf_t;
using snode_t = stree_node_block;

struct stree_node_header;

namespace detail {

class extent_mapper;
class extent_map_generator;

} // ns detail

} // ns grid_format

using grid_format::shard_tree;
using grid_format::stree_t;
using grid_format::stree_intn_t;
using grid_format::sintn_t;
using grid_format::stree_leaf_t;
using grid_format::sleaf_t;

} // ns gstream

#endif // GSTREAM_GRID_FORMAT_GRID_STREAM_DEFINES_H
