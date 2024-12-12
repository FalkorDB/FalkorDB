//------------------------------------------------------------------------------
// GrB_Matrix_free: free a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// free all the content of a matrix.  After GrB_Matrix_free (&A), A is set
// to NULL

#include "GB.h"

GrB_Info GrB_Matrix_free        // free a matrix
(
    GrB_Matrix *A               // handle of matrix to free
)
{ 

    GB_Matrix_free (A) ;
    return (GrB_SUCCESS) ;
}

