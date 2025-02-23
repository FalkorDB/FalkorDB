//------------------------------------------------------------------------------
// GB_matvec_name_size_get: get max size of a matrix/vector/scalar name or type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

GrB_Info GB_matvec_name_size_get (GrB_Matrix A, size_t *value, int field)
{
    const char *name = NULL ;

    switch (field)
    {

        case GrB_NAME :     
            name = A->user_name ;
            break ;

        case GxB_JIT_C_NAME : 
            name = A->type->name ;
            break ;

        case GrB_EL_TYPE_STRING : 
            name = GB_type_name_get (A->type) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (name == NULL) ? 1 : (strlen (name) + 1) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

