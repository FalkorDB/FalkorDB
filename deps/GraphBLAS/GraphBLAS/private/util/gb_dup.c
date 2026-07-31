//------------------------------------------------------------------------------
// gb_dup: copy a GrB matrix, perhaps using smaller integer sizes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is almost the same as GrB_Matrix_dup, except that it allows
// the output matrix C to have different integer sizes than Cin.  The matrix
// Cin might be a shallow GrB matrix constructed from a MATLAB/Octave sparse
// matrix, which always uses 64-bit integers.

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&C) ;

GrB_Info gb_dup             // copy a matrix
(
    // output:
    GrB_Matrix *C_handle,   // copy of the input matrix
    // input:
    GrB_Matrix Cin,         // matrix to copy
    const int arena,
    char err [ERRLEN]
)
{ 

    GrB_Matrix C = NULL ;
    int fmt ;   // by row or by column
    OK (GrB_Matrix_get_INT32 (Cin, &fmt, GxB_FORMAT)) ;
    OK (gb_typecast (&C, Cin, NULL, fmt, 0, arena, err)) ;
    (*C_handle) = C ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

