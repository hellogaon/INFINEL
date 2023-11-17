#ifndef GSTREAM_RUNTIME_H
#define GSTREAM_RUNTIME_H
#include <gstream/gstream_defines.h>

namespace gstream {

gstream_framework* init_framework(gstream_query* query);
void               cleanup_framework(gstream_framework const* fw);

superstep_result begin_superstep(gstream_framework* fw);

} // namespace gstream

#endif // !GSTREAM_RUNTIME_H
