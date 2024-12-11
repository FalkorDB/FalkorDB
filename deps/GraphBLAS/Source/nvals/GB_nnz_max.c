//------------------------------------------------------------------------------
// GB_nnz_max.c: max number of entries that can be held in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_nnz_max(A) is the maximum number of entries that can be held in the data
// structure, including zombies and all entries in a bitmap, but not including
// pending tuples.  For iso full matrices, GB_nnz_max (A) can be less than
// GB_nnz_full (A), and is typically 1.

#include "GB.h"

int64_t GB_nnz_max
(
    GrB_Matrix A
)
{

    if (A == NULL || A->x == NULL || A->type == NULL)
    { 
        // A is empty
        return (0) ;
    }
    int64_t nnz_max ;
    int64_t xmax = A->x_size / A->type->size ;
    if (A->p != NULL)
    {
        // A is sparse (p,i,x) or hypersparse (p,h,i,x):
        nnz_max = (A->i == NULL) ? 0 : (A->i_size / sizeof (int64_t)) ;
        if (!A->iso)
        { 
            nnz_max = GB_IMIN (nnz_max, xmax) ;
        }
    }
    else if (A->b != NULL)
    {
        // A is bitmap (b,x):
        nnz_max = A->b_size / sizeof (bool) ;
        if (!A->iso)
        { 
            nnz_max = GB_IMIN (nnz_max, xmax) ;
        }
    }
    else
    { 
        // A is full (x only):
        nnz_max = xmax ;
    }
    return (nnz_max) ;
}

