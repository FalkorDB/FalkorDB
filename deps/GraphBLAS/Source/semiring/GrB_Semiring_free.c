//------------------------------------------------------------------------------
// GrB_Semiring_free: free a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Semiring_free          // free a user-created semiring
(
    GrB_Semiring *semiring          // handle of semiring to free
)
{

    if (semiring != NULL)
    {
        // only free a dynamically-allocated semiring
        GrB_Semiring s = *semiring ;
        if (s != NULL)
        {
            // free the semiring name
            GB_FREE_MEMORY (&(s->name), s->name_mem) ;
            // free the semiring user_name
            GB_FREE_MEMORY (&(s->user_name), s->user_name_mem) ;
            // free the semiring header
            uint64_t header_mem = s->header_mem ;
            if (GB_memsize (header_mem) > 0)
            { 
                s->magic = GB_FREED ;   // to help detect dangling pointers
                s->header_mem = 0 ;     // header is freed
                GB_FREE_MEMORY (semiring, header_mem) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

