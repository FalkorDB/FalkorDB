//------------------------------------------------------------------------------
// GrB_Monoid_free:  free a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Monoid_free            // free a user-created monoid
(
    GrB_Monoid *monoid              // handle of monoid to free
)
{

    if (monoid != NULL)
    {
        // only free a dynamically-allocated monoid
        GrB_Monoid mon = *monoid ;
        if (mon != NULL)
        {
            uint64_t header_mem = mon->header_mem ;
            // free the monoid user_name
            GB_FREE_MEMORY (&(mon->user_name), mon->user_name_mem) ;
            if (GB_memsize (header_mem) > 0)
            { 
                mon->magic = GB_FREED ; // to help detect dangling pointers
                mon->header_mem = 0 ;   // header is freed
                GB_FREE_MEMORY (&(mon->identity), mon->identity_mem) ;
                GB_FREE_MEMORY (&(mon->terminal), mon->terminal_mem) ;
                GB_FREE_MEMORY (monoid, header_mem) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

