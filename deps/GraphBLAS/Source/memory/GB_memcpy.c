//------------------------------------------------------------------------------
// GB_memcpy: parallel memcpy
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Note that this function uses its own hard-coded chunk size.

#include "GB.h"

#define GB_MEM_CHUNK (1024*1024)

void GB_memcpy                  // parallel memcpy
(
    void *dest,                 // destination
    const void *src,            // source
    size_t n,                   // # of bytes to copy
    int nthreads                // max # of threads to use
)
{

    // FIXME for CUDA: do (cpu,gpu) <= (cpu,gpu) memcpy's
    //  cpu <- cpu (already done below)
    //  cpu <- gpu (effectively done below but could be better)
    //  gpu <- cpu (need this)
    //  gpu <- gpu (need this)

    if (nthreads <= 1 || n <= GB_MEM_CHUNK)
    { 

        //----------------------------------------------------------------------
        // memcpy using a single thread
        //----------------------------------------------------------------------

        memcpy (dest, src, n) ;
    }
    else
    {

        //----------------------------------------------------------------------
        // memcpy using multiple threads
        //----------------------------------------------------------------------

        size_t nchunks = 1 + (n / GB_MEM_CHUNK) ;
        if (((size_t) nthreads) > nchunks)
        { 
            nthreads = (int) nchunks ;
        }
        GB_void *pdest = (GB_void *) dest ;
        const GB_void *psrc = (GB_void *) src ;

        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (k = 0 ; k < nchunks ; k++)
        {
            size_t start = k * GB_MEM_CHUNK ;
            if (start < n)
            { 
                size_t chunk = GB_IMIN (n - start, GB_MEM_CHUNK) ;
                memcpy (pdest + start, psrc + start, chunk) ;
            }
        }
    }
}

