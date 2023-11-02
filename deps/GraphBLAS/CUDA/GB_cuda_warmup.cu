//------------------------------------------------------------------------------
// GB_cuda_warmup.cu: warmup the GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.h"

bool GB_cuda_warmup (int device)
{
    // allocate 'nothing' just to load the drivers.
    // No need to free the result.
    bool ok = GB_cuda_set_device( device );
    if (!ok)
    {
        printf ("invalid GPU: %d\n", device) ;
        return (false) ;
    }

    double gpu_memory_size = GB_Global_gpu_memorysize_get (device);

    size_t size = 0 ;
    void *p = GB_malloc_memory (1, 1, &size) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da memory???\n") ;
        return (false) ;
    }
//    printf ("oooo nice block of memory of size %lu\n", size) ;
    GB_free_memory ( &p, size) ;
//    printf ("be free, block of memory of size %lu\n", size) ;

//    printf ("good ol' cudaMalloc just to be sure\n");
    cudaMalloc ( &p, size ) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da GPU???\n") ;
        return (false) ;
    }
    cudaFree (p) ;

//    printf ("GPU %d nice and toasty now\n", device) ;

    // TODO check for jit cache? or in GB_init?

    return  true; //(err == cudaSuccess) ;
}

