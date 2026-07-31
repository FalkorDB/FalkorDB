//------------------------------------------------------------------------------
// gb_get_matrix: get a matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A = gb_get_matrix (matrix) constructs a GrB_Matrix from a MATLAB mxArray,
// which can either be a MATLAB sparse matrix (double, complex, or logical) or
// a GraphBLAS (GrB or GyB) object.  The input is a gb_matrix constructed by
// gbmx_get_matrix.

// The input matrix must not be NULL, but it can be an empty matrix, as matrix
// = [ ].  In this case, A is returned as NULL.  This is not an error here,
// since the caller might be getting an optional input matrix, such as Cin or
// the Mask.

// If A_to_free is returned as non-NULL, it contains a pointer to a newly
// allocated GrB_Matrix that contains readonly content from a MATLAB matrix.
// The A_to_free matrix must be freed by the caller (which does not free the
// readonly MATLAB content).

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A) ;

GrB_Info gb_get_matrix
(
    // output
    GrB_Matrix *A_handle,   // output matrix
    GrB_Matrix *A_to_free,  // must be freed by the caller if not NULL
    // input
    gb_matrix matrix,       // input MATLAB, GhB, or GrB matrix
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;
    CHECK_ERROR (A_handle == NULL, "matrix missing") ;
    CHECK_ERROR (A_to_free == NULL, "matrix missing") ;
    CHECK_ERROR (matrix == NULL, "matrix missing") ;

    //--------------------------------------------------------------------------
    // construct the GrB_Matrix
    //--------------------------------------------------------------------------

    if (matrix->G != NULL)
    { 
        // matrix is a GhB handle object
        (*A_handle) = matrix->G ;
        (*A_to_free) = NULL ;           // no shallow copy to free when done
    }
    else if (matrix->is_empty)
    { 
        // matrix is a 0-by-0 MATLAB matrix.  Create a new 0-by-0 matrix of the
        // same type as matrix, with the default format.  The new matrix must
        // be freed by the caller when done.
        OK (GxB_Matrix_new_arena (&A, matrix->type, 0, 0, arena, arena)) ;
        (*A_handle) = A ;
        (*A_to_free) = A ;
    }
    else
    { 
        // construct a shallow GrB_Matrix copy of a built-in MATLAB matrix or
        // a GrB value object, which must be freed by the caller when done.
        OK (gb_get_matlab_or_grb_matrix (&A, matrix, arena, err)) ;
        (*A_handle) = A ;
        (*A_to_free) = A ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

