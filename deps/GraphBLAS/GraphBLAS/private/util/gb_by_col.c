//------------------------------------------------------------------------------
// gb_by_col: ensure a matrix is stored by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The return value A is set to either the input matrix A_input, or the A_copy
// matrix.

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_copy) ;

GrB_Info gb_by_col
(
    // output
    GrB_Matrix *A_handle,       // return the matrix by column
    GrB_Matrix *A_copy_handle,  // copy made of A, stored by column, or NULL
    // input
    GrB_Matrix A_input,         // input matrix, by row or column
    const int arena,
    char err [ERRLEN]
)
{
    GrB_Matrix A_copy = NULL, A = NULL ;
    (*A_handle) = NULL ;
    (*A_copy_handle) = NULL ;

    // get the format of A_input
    int fmt ;
    OK (GrB_Matrix_get_INT32 (A_input, &fmt, GxB_FORMAT)) ;

    if (fmt == GxB_BY_ROW)
    { 
        // make a deep copy of A_input and change it to be stored by column
        OK (gb_dup (&A_copy, A_input, arena, err)) ;
        OK1 (A_copy, GrB_Matrix_set_INT32 (A_copy, GxB_BY_COL, GxB_FORMAT)) ;
        OK1 (A_copy, GrB_Matrix_wait (A_copy, GrB_MATERIALIZE)) ;
        A = A_copy ;
    }
    else
    { 
        // A is just A_input, with no change
        A = A_input ;
    }

    // return results
    (*A_handle) = A ;
    (*A_copy_handle) = A_copy ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

