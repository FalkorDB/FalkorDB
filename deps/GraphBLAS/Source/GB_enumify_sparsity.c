//------------------------------------------------------------------------------
// GB_enumify_sparsity: enumerate the sparsity structure of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_sparsity    // enumerate the sparsity structure of a matrix
(
    // output:
    int *ecode,             // enumerated sparsity structure:
                            // 0:hyper, 1:sparse, 2:bitmap, 3:full
    // input:
    int sparsity            // 0:no matrix, 1:GxB_HYPERSPARSE, 2:GxB_SPARSE,
                            // 4:GxB_BITMAP, 8:GxB_FULL
)
{

    int e ;
    if (sparsity == GxB_HYPERSPARSE || sparsity == 0)
    { 
        // hypersparse, or no matrix
        e = 0 ;
    }
    else if (sparsity == GxB_BITMAP)
    { 
        e = 2 ;
    }
    else if (sparsity == GxB_FULL)
    { 
        e = 3 ;
    }
    else // sparse
    { 
        e = 1 ;
    }
    (*ecode) = e ;
}

