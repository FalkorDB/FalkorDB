//------------------------------------------------------------------------------
// GB_serialize_to_blob: copy a set of blocks to the blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

#include "GB.h"
#include "GB_serialize.h"

void GB_serialize_to_blob
(
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    size_t *s_handle,       // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks+1
    int64_t *Sblocks,       // array of size nblocks
    int32_t nblocks,        // # of blocks
    int nthreads_max        // # of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (blob != NULL) ;
    ASSERT (s_handle != NULL) ;
    ASSERT (nblocks >= 0) ;
    ASSERT ((nblocks > 0) == (Blocks != NULL)) ;
    ASSERT (nthreads_max > 0) ;

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (nblocks == 0)
    { 
        // no blocks for this array
        return ;
    }

    //--------------------------------------------------------------------------
    // copy the blocks into the blob
    //--------------------------------------------------------------------------

    size_t s = (*s_handle) ;

    if (nblocks == 1)
    { 
        // copy a single block into the blob in parallel
        GB_memcpy (blob + s, Blocks [0].p, Sblocks [0], nthreads_max) ;
    }
    else
    {
        // copy each block with a single task
        int nthreads = GB_IMIN (nthreads_max, nblocks) ;
        int32_t blockid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
        for (blockid = 0 ; blockid < nblocks ; blockid++)
        { 
            // copy the compressed block of size s_size into the blob
            size_t s_start = (blockid == 0) ? 0 : Sblocks [blockid-1] ;
            size_t s_end   = Sblocks [blockid] ;
            size_t s_size  = s_end - s_start ;
            memcpy (blob + s + s_start, Blocks [blockid].p, s_size) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the updated index into the blob
    //--------------------------------------------------------------------------

    s += Sblocks [nblocks-1] ;
    (*s_handle) = s ;
}

