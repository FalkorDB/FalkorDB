//------------------------------------------------------------------------------
// GB_serialize_free_blocks: free the set of blocks used to compress an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Free the Blocks constructed by GB_serialize_array.

#include "GB.h"
#include "serialize/GB_serialize.h"

void GB_serialize_free_blocks
(
    GB_blocks **Blocks_handle,      // array of size nblocks
    uint64_t Blocks_mem,            // memsize and memlane of Blocks
    int32_t nblocks                 // # of blocks, or zero if no blocks
)
{

    ASSERT (Blocks_handle != NULL) ;
    GB_blocks *Blocks = (*Blocks_handle) ;
    if (Blocks != NULL)
    {
        // free all blocks
        for (int32_t blockid = 0 ; blockid < nblocks ; blockid++)
        {
            uint64_t p_mem = Blocks [blockid].p_mem ;
            if (GB_memsize (p_mem) > 0)
            { 
                // free the block
                GB_void *p = (GB_void *) Blocks [blockid].p ;
                GB_FREE_MEMORY (&p, p_mem) ;
            }
        }
        // free the Blocks array itself
        GB_FREE_MEMORY (Blocks_handle, Blocks_mem) ;
    }
}

