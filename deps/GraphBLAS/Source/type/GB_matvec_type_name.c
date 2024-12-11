//------------------------------------------------------------------------------
// GB_matvec_type_name: return the name of the type of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_matvec_type_name  // return the name of the type of a matrix
(
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    const GrB_Matrix A,     // matrix to query
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (type_name) ;
    ASSERT_MATRIX_OK (A, "A for type_name", GB0) ;

    //--------------------------------------------------------------------------
    // return the type
    //--------------------------------------------------------------------------

    memcpy (type_name, A->type->name, GxB_MAX_NAME_LEN) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

