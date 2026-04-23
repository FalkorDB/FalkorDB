//------------------------------------------------------------------------------
// GraphBLAS/CUDA/device/GB_cuda_pointer_ok
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

//------------------------------------------------------------------------------

// Determine if the pointer can be used by the GPU

#include "GB_cuda.hpp"

bool GB_cuda_pointer_ok
(
    const void *p,
    const char *name
)
{

    if (p == NULL)
    {
//      printf ("\n%s: %s (%p) is NULL\n", __FILE__, name, p) ;
        return (true) ;
    }

    struct cudaPointerAttributes attr ;
    cudaError_t status = cudaPointerGetAttributes (&attr, p) ;

    if (status != cudaSuccess)
    {
//      printf ("\n%s: %s (%p) cudaPointerGetAttributes failed\n", __FILE__, name, p) ;
        return (false) ;
    }
    if (attr.type == cudaMemoryTypeHost)
    {
//      printf ("\n%s: %s (%p) belongs to the host only\n", __FILE__, name, p) ;
        return (false) ;
    }
    else if (attr.type == cudaMemoryTypeDevice)
    {
//      printf ("\n%s: %s (%p) belongs to the GPU only\n", __FILE__, name, p) ;
        return (true) ;
    }
    else if (attr.type == cudaMemoryTypeManaged)
    {
//      printf ("\n%s: %s (%p) belongs is managed memory (CPU and GPU)\n", __FILE__, name, p) ;
        return (true) ;
    }
    else
    {
//      printf ("\n%s: %s (%p) is unknown memory\n", __FILE__, name, p) ;
        return (false) ;
    }
}

