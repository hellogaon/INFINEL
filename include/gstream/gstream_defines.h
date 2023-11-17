#ifndef GSTREAM_DEFINES_H
#define GSTREAM_DEFINES_H
#include <stdint.h>

namespace gstream {

enum class gstream_return_code : unsigned int {
    Success,
    CudaError,
    RuntimeError,
    LogicError,
    InvalidCall,
    InvalidArgument,
    BadAlloc,
    BadFree,
    CppStdException,
    CppStdUnknownException,
    BufferOverflow,
    FileNotFound,
    FileOpenError,
    FileReadError,
    FileWriteError,
    InitFailure,
    ChannelBroken,
    DataLoadingError,
};

using gstream_retcode = gstream_return_code;

constexpr unsigned InternalDataSectionAlignment = 4;
constexpr unsigned CudaMallocAlignment = 256;
constexpr unsigned DiskIoSectorSize = 4096;
constexpr unsigned MaxArity = 16;

struct kernel_launch_parameters {
    unsigned num_threads_per_block;
    unsigned num_blocks_per_grid;
    unsigned shared_mem_size;
};

#pragma pack(push, 1)
struct RA_option {
    bool enabled : 1;
    bool use_in_degree : 1;
    bool use_out_degree : 1;
    bool use_usrdefRA : 1;
    bool fixed_size : 1;
};
#pragma pack(pop)

enum class WA_mode: unsigned char {
    Null,
    Columnar,
    DeviceLocal,
    PerKernel,
};
#pragma pack(push, 1)
struct WA_option {
    WA_mode mode;
    uint64_t WA_size_per_device;
    uint64_t WA_columnar_unit_size;
};
#pragma pack(pop)

struct superstep_state;
struct superstep_result;
class gstream_query;

using device_id_t = int;
constexpr device_id_t InvalidDeviceID = -1;
using gridtx_id_t = int64_t;
constexpr gridtx_id_t InvalidGridTxID = -1;

namespace detail {

class kernel_binder;

} // !namespace detail

using FixedDegreeUnitType = uint32_t;
constexpr unsigned FixedDegreeUnitSize = sizeof(FixedDegreeUnitType); //TODO: Remove it, this value is temporary setting

namespace framework3 {

struct framework3_context;

} // !namespace framework3

using gstream_framework = framework3::framework3_context;

} // !namespace gstream

#endif // GSTREAM_DEFINES_H
