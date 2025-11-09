//------------------------------------------------------------------------------
// GrB_Type_free:  free a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Type_free          // free a user-defined type
(
    GrB_Type *type              // handle of user-defined type to free
)
{

    if (type != NULL)
    {
        // only free a dynamically-allocated type, which have header_size > 0
        GrB_Type t = *type ;
        if (t != NULL)
        {
            GB_FREE_MEMORY (&(t->user_name), t->user_name_size) ;
            size_t defn_size = t->defn_size ;
            if (defn_size > 0)
            { 
                GB_FREE_MEMORY (&(t->defn), defn_size) ;
            }
            size_t header_size = t->header_size ;
            if (header_size > 0)
            {
                t->magic = GB_FREED ;  // to help detect dangling pointers
                t->header_size = 0 ;
                GB_FREE_MEMORY (type, header_size) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

