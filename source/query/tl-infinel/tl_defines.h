#pragma once
#include <gstream/gstream_runtime.h>
#include <gstream/grid_format/grid_stream_defines.h>

constexpr unsigned WarpSize = 32;
constexpr int NumThreads = 128;
constexpr unsigned MaxBlocks = 1280;
constexpr unsigned DiskIOSectorSize = 4096;

struct program_args {
    char const* graph_name;
    char const* graph_dir;
    uint64_t device_buffer_size;
    uint64_t chunk_num;
    uint64_t chunk_size;
    bool load_balance_mode;
    bool double_buffering_mode;
    bool only_kernel_mode;
    bool verification_mode;
};

struct tl_context;
struct tl_auxiliary_t {
    uint64_t tl_buffer_size;
    bool double_buffering_mode;
};
struct tl_output_t {
    char V[4];
};
struct tl_output_format {
    tl_output_t A, B, C;
};
struct tl_warp_context {
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t i_end;
};

extern tl_context* TL;
extern tl_auxiliary_t* tl_aux;
extern void* tl_output_host;

void init(program_args const* args);
void cleanup();
void init_tl_query(
    gstream::gstream_query* q,
    gstream::grid_format::grid_stream2* gs,
    program_args const* args
);
void begin_tl(gstream::gstream_framework* fw);
