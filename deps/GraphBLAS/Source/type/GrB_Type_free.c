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
        // only free a dynamically-allocated type, which have
        // GB_memsize (header_mem) > 0
        GrB_Type t = *type ;
        if (t != NULL)
        {
            GB_FREE_MEMORY (&(t->user_name), t->user_name_mem) ;
            uint64_t defn_mem = t->defn_mem ;
            if (GB_memsize (defn_mem) > 0)
            { 
                GB_FREE_MEMORY (&(t->defn), defn_mem) ;
            }
            uint64_t header_mem = t->header_mem ;
            if (GB_memsize (header_mem) > 0)
            {
                t->magic = GB_FREED ;   // to help detect dangling pointers
                t->header_mem = 0 ;     // header is freed
                GB_FREE_MEMORY (type, header_mem) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

