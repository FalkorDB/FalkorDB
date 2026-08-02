//------------------------------------------------------------------------------
// gb_is_all: check two matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Applies a binary operator to two matrices A and B, and returns result = true
// if the pattern of A and B are identical, and if the result of C = op(A,B) is
// true for all entries in C.

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&C) ;

GrB_Info gb_is_all          // check two matrices for equality, given an op
(
    // output:
    bool *result,           // true if op (A,B) is all true, false otherwise
    // input:
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op,
    const int arena,
    char err [ERRLEN]
)
{

    GrB_Matrix C = NULL ;
    (*result) = true ;

    uint64_t nrows1, ncols1, nrows2, ncols2, nvals, nvals1, nvals2 ;

    //--------------------------------------------------------------------------
    // check the size of A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nrows (&nrows1, A)) ;
    OK (GrB_Matrix_nrows (&nrows2, B)) ;
    if (nrows1 != nrows2)
    { 
        // # of rows differ
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    OK (GrB_Matrix_ncols (&ncols1, A)) ;
    OK (GrB_Matrix_ncols (&ncols2, B)) ;
    if (ncols1 != ncols2)
    { 
        // # of cols differ
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // check the # entries in A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nvals (&nvals1, A)) ;
    OK (GrB_Matrix_nvals (&nvals2, B)) ;
    if (nvals1 != nvals2)
    { 
        // # of entries differ
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    // check if A and B both have no entries
    if (nvals1 == 0)
    { 
        // A and B are empty matrices of the same size and type
        (*result) = true ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C = A .* B, where the pattern of C is the intersection of A and B
    //--------------------------------------------------------------------------

    int fmt ;
    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
    int sparsity = 0 ;
    OK (gb_get_sparsity (A, B, &sparsity, err)) ;
    OK (gb_new (&C, GrB_BOOL, nrows1, ncols1, fmt, sparsity, arena, err)) ;
    OK1 (C, GrB_Matrix_eWiseMult_BinaryOp (C, NULL, NULL, op, A, B, NULL)) ;

    //--------------------------------------------------------------------------
    // ensure C has the same number of entries as A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nvals (&nvals, C)) ;
    if (nvals != nvals1)
    { 
        // pattern of A and B are different
        FREE_ALL ;
        (*result) = false ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // result = and (C)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_reduce_BOOL (result, NULL, GrB_LAND_MONOID_BOOL, C, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

