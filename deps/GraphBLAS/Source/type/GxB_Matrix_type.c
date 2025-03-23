//------------------------------------------------------------------------------
// GxB_Matrix_type: return the type of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Matrix_type    // get the type of a matrix
(
    GrB_Type *type,         // returns the type of the matrix
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_type (&type, A)") ;

    //--------------------------------------------------------------------------
    // get the type
    //--------------------------------------------------------------------------

    return (GB_matvec_type (type, A, Werk)) ;
}

