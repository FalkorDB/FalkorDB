//------------------------------------------------------------------------------
// GB_matvec_type: return the type of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_matvec_type            // get the type of a matrix
(
    GrB_Type *type,         // returns the type of the matrix
    const GrB_Matrix A,     // matrix to query
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (type) ;
    ASSERT_MATRIX_OK (A, "A for type", GB0) ;

    //--------------------------------------------------------------------------
    // return the type
    //--------------------------------------------------------------------------

    (*type) = A->type ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

