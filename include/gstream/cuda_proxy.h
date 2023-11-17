#ifndef GSTREAM_CUDA_PROXY_H
#define GSTREAM_CUDA_PROXY_H
#include <gstream/cuda_env.h>
#include <ash/config/osdetect.h>
#include <stdint.h>
#include <assert.h>
#include <memory>
#include <functional>
#include<gstream/cuda_proxy.h>

namespace gstream {
namespace cuda {

bool  device_count(int* out);
bool  set_device(int device_id) noexcept;
void* device_malloc(size_t size);
bool  device_free(void* p) noexcept;
stream_type* create_streams(size_t count);
bool destory_streams(stream_type* streams, size_t count) noexcept;

constexpr int UnexpectedCudaReturnValue = -1;
constexpr int StreamStateIdle = 1;
constexpr int StreamStateBusy = 2;
int is_idle_stream(stream_type& stream);

bool stream_synchronize(stream_type const& stream);
bool device_synchronize() noexcept;

bool h2dcpy(void* dst, void* src, size_t size);
bool h2dcpy_async(void* dst, void* src, size_t size, stream_type const& stream);
bool d2hcpy(void* dst, void* src, size_t size);
bool d2hcpy_async(void* dst, void* src, size_t size, stream_type const& stream);

bool device_memset(void* dst, int value, size_t count);
bool device_memset_async(void* dst, int value, size_t count, stream_type const& stream);

void* pinned_malloc(size_t size);
bool  pinned_free(void* p) noexcept;

#ifdef ASH_ENV_WINDOWS
#define GSTREAM_CUDART_CB __stdcall
#else
#define GSTREAM_CUDART_CB 
#endif // PLATFORM
using stream_callback_t = void(GSTREAM_CUDART_CB*)(void* /*usr*/);
bool add_host_function(stream_type const& stream, stream_callback_t fn, void* usr_data);

using legacy_callback_t = std::function<void(bool/* error */, int /* cudaErrorCode */)>;
bool add_stream_callback(stream_type const& stream, legacy_callback_t* fn);

} // namespace cuda
} // namespace gstream

#endif // GSTREAM_CUDA_PROXY_H
