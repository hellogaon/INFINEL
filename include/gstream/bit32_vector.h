#ifndef GSTREAM_BIT32_VECTOR_H
#define GSTREAM_BIT32_VECTOR_H
#include <stdint.h>
#include <gstream/cuda_env.h>

namespace gstream {

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
uint32_t read_32v_element(void const* arr_, uint64_t const& index) {
    return *(static_cast<uint32_t const*>(arr_) + index);
}

struct v32_element {
    char data[4];
};

} // !namespace gstream

#endif // GSTREAM_FLIP24_VECTOR_H
