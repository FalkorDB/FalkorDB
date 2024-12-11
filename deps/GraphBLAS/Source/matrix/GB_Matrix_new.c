//------------------------------------------------------------------------------
// GB_Matrix_new: create a new matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new matrix is nrows-by-ncols, with no entries in it.  Default format for
// an empty matrix is hypersparse CSC: A->p is size 2 and all zero, A->h is
// size 1, A->plen is 1, and contents A->x and A->i are NULL.  If this method
// fails, *A is set to NULL.

#include "GB.h"

GrB_Info GB_Matrix_new          // create a new matrix with no entries
(
    GrB_Matrix *A,              // handle of matrix to create
    GrB_Type type,              // type of matrix to create
    GrB_Index nrows,            // matrix dimension is nrows-by-ncols
    GrB_Index ncols
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

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
        // m-by-n (including 0-by-0) with m != and n != use the global setting
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

    return (GB_new (A, // auto sparsity, new header
        type, vlen, vdim, GB_Ap_calloc, A_is_csc, GxB_AUTO_SPARSITY,
        GB_Global_hyper_switch_get ( ), 1)) ;
}

