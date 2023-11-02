//------------------------------------------------------------------------------
// gb_is_all: check two matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Applies a binary operator to two matrices A and B, and returns result = true
// if the pattern of A and B are identical, and if the result of C = op(A,B) is
// true for all entries in C.

#include "gb_interface.h"

bool gb_is_all              // true if op (A,B) is all true, false otherwise
(
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op
)
{

    GrB_Index nrows1, ncols1, nrows2, ncols2, nvals, nvals1, nvals2 ;

    //--------------------------------------------------------------------------
    // check the size of A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nrows (&nrows1, A)) ;
    OK (GrB_Matrix_nrows (&nrows2, B)) ;
    if (nrows1 != nrows2)
    { 
        // # of rows differ
        return (false) ;
    }

    OK (GrB_Matrix_ncols (&ncols1, A)) ;
    OK (GrB_Matrix_ncols (&ncols2, B)) ;
    if (ncols1 != ncols2)
    { 
        // # of cols differ
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // check the # entries in A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nvals (&nvals1, A)) ;
    OK (GrB_Matrix_nvals (&nvals2, B)) ;
    if (nvals1 != nvals2)
    { 
        // # of entries differ
        return (false) ;
    }

    // check if A and B both have no entries
    if (nvals1 == 0)
    { 
        // A and B are empty matrices of the same size and type
        return (true) ;
    }

    //--------------------------------------------------------------------------
    // C = A .* B, where the pattern of C is the intersection of A and B
    //--------------------------------------------------------------------------

    GxB_Format_Value fmt ;
    OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;
    int sparsity = gb_get_sparsity (A, B, 0) ;
    GrB_Matrix C = gb_new (GrB_BOOL, nrows1, ncols1, fmt, sparsity) ;
    OK1 (C, GrB_Matrix_eWiseMult_BinaryOp (C, NULL, NULL, op, A, B, NULL)) ;

    //--------------------------------------------------------------------------
    // ensure C has the same number of entries as A and B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nvals (&nvals, C)) ;
    if (nvals != nvals1)
    { 
        // pattern of A and B are different
        GrB_Matrix_free (&C) ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // result = and (C)
    //--------------------------------------------------------------------------

    bool result = true ;
    OK (GrB_Matrix_reduce_BOOL (&result, NULL, GrB_LAND_MONOID_BOOL, C, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GrB_Matrix_free (&C) ;
    return (result) ;
}

