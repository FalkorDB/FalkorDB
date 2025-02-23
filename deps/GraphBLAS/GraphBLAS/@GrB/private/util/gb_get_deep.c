//------------------------------------------------------------------------------
// gb_get_deep: create a deep GrB_Matrix copy of a built-in X
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a built-in X
(
    const mxArray *X        // input built-in matrix (sparse or struct)
)
{ 

    GrB_Matrix S = gb_get_shallow (X) ;
    int fmt ;
    OK (GrB_Matrix_get_INT32 (S, &fmt, GxB_FORMAT)) ;
    GrB_Matrix A = gb_typecast (S, NULL, fmt, 0) ;
    OK (GrB_Matrix_free (&S)) ;
    return (A) ;
}

