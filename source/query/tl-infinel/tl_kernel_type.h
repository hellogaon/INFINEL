#pragma once
#include "tl_defines.h"
#include <infinel/chunk_scheduler.cuh>
#include <gstream/qrydef.h>

using tl_kernel_binder = gstream::kernel_binder_template<3, infinel::chunk_scheduler<tl_warp_context>>;
using kernel_arguments = tl_kernel_binder::kernel_arguments;
using auxiliary_t = tl_kernel_binder::auxiliary_t;
