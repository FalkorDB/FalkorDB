//------------------------------------------------------------------------------
// GB_matvec_name_get: get a name of a matrix or its type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_name_get (GrB_Matrix A, char *name, int field)
{
    const char *typename ;
    (*name) = '\0' ;

    switch (field)
    {

        case GrB_NAME : 
            if (A->user_name != NULL)
            {
                strcpy (name, A->user_name) ;
            }
            break ;

        case GxB_JIT_C_NAME : 
            strcpy (name, A->type->name) ;
            break ;

        case GrB_EL_TYPE_STRING : 
            typename = GB_type_name_get (A->type) ;
            if (typename != NULL)
            {
                strcpy (name, typename) ;
            }
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

