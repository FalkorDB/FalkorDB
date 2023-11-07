//------------------------------------------------------------------------------
// GB_type_name_get: get the user_name of a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

const char *GB_type_name_get (GrB_Type type)
{

    if (type == NULL)
    { 
        return (NULL) ;
    }

    return (GB_code_name_get (type->code, type->user_name)) ;
}

