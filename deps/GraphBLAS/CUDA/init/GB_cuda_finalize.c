
#include "GB.h"

GrB_Info GB_cuda_finalize (void)
{
    GB_cuda_stream_pool_finalize () ;

    return GrB_SUCCESS ;
}
