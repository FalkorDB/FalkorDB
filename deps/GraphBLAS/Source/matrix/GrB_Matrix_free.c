//------------------------------------------------------------------------------
// GrB_Matrix_free: free a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// free all the content of a matrix.  After GrB_Matrix_free (&A), A is set
// to NULL.  A may have readonly-content; it is simple removed from A and not
// modified.  See also GrB_Matrix_clear.

#include "GB.h"

GrB_Info GrB_Matrix_free        // free a matrix
(
    GrB_Matrix *A               // handle of matrix to free
)
{ 

    GB_Matrix_free (A) ;
    return (GrB_SUCCESS) ;
}

