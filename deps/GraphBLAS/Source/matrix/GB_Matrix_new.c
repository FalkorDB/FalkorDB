//------------------------------------------------------------------------------
// GB_Matrix_new: create a new matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new matrix is nrows-by-ncols, with no entries in it.  Default format for
// an empty matrix is hypersparse CSC: A->p is size 2 and all zero, A->h is
// size 1, A->plen is 1, and contents A->x and A->i are NULL.  If this method
// fails, *A is set to NULL.
//
// A->p: row/column 'pointers' (offsets).  If 32 bit, the nvals(A) < UINT32_MAX
//  is required.  If 64 bit, nvals(A) < UINT64_MAX.
//
// A->i, A->h, and A->Y:  indices. Let N = max(m,n).  If 32-bit, then
//  N < INT32_MAX is required.  Otherwise N < 2^60.
//
// For Ap and Ai, independently:
//
//  Global settings: no matrix is converted if this is changed
//  ----------------
//
//  32 : use 32 (or 64 if required) for new or recomputed
//                      matrices; any prior 64 ok (will be the default; but use
//                      64-bit for now)
//
//  64 : use 64 by default (this is the default for now)
//
//  per-matrix settings:
//  -------------------
//
//  0:  default : rely on the global settings
//
//  32 : use 32 bits if possible, but allow 64 bit if needed.
//
//  64 : use 64 bits, convert now is already 32 bits.
//                      Sometimes the matrix may become 32-bit in the future,
//                      if data is transplanted from a matrix with 32-bit
//                      content.
//
// Changing the global settings has no impact on the block/non-blocking status
// of any existing matrix.  If the per-matrix setting is changed, it may cause
// future pending work that will be finalized by GrB_wait on that matrix.  If
// GrB_wait is called to materialize the matrix, and the matrix is not modified
// afterwards, it remains materialized and is not changed.

#define GB_FREE_ALL GB_Matrix_free (A)

#include "GB.h"

GrB_Info GB_Matrix_new          // create a new matrix with no entries
(
    GrB_Matrix *A,              // handle of matrix to create
    GrB_Type type,              // type of matrix to create
    uint64_t nrows,             // matrix dimension is nrows-by-ncols
    uint64_t ncols
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (A) ;
    (*A) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;

    if (nrows > GB_NMAX || ncols > GB_NMAX)
    { 
        // problem too large
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create the matrix
    //--------------------------------------------------------------------------

    int64_t vlen, vdim ;
    bool A_is_csc ;
    if (ncols == 1)
    { 
        // n-by-1 matrices are always held by column, including 1-by-1
        A_is_csc = true ;
    }
    else if (nrows == 1)
    { 
        // 1-by-n matrices (except 1-by-1) are always held by row
        A_is_csc = false ;
    }
    else
    { 
        // m-by-n (including 0-by-0) with m != 1 or n != 1 use global setting
        A_is_csc = GB_Global_is_csc_get ( ) ;
    }

    if (A_is_csc)
    { 
        vlen = (int64_t) nrows ;
        vdim = (int64_t) ncols ;
    }
    else
    { 
        vlen = (int64_t) ncols ;
        vdim = (int64_t) nrows ;
    }

    // determine the p_is_32, j_is_32 and i_is_32 settings for the new matrix
    bool Ap_is_32, Aj_is_32, Ai_is_32 ;
    GB_determine_pji_is_32 (&Ap_is_32, &Aj_is_32, &Ai_is_32,
        GxB_AUTO_SPARSITY, 1, vlen, vdim, NULL) ;

    // create the matrix
    GB_OK (GB_new (A, // auto sparsity (sparse/hyper), new header
        type, vlen, vdim, GB_ph_calloc, A_is_csc, GxB_AUTO_SPARSITY,
        GB_Global_hyper_switch_get ( ), 1, Ap_is_32, Aj_is_32, Ai_is_32)) ;

    return (GrB_SUCCESS) ;
}

