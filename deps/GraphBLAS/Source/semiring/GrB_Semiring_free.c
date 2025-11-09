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
            GB_FREE_MEMORY (&(s->name), s->name_size) ;
            // free the semiring user_name
            GB_FREE_MEMORY (&(s->user_name), s->user_name_size) ;
            // free the semiring header
            size_t header_size = s->header_size ;
            if (header_size > 0)
            { 
                s->magic = GB_FREED ;  // to help detect dangling pointers
                s->header_size = 0 ;
                GB_FREE_MEMORY (semiring, header_size) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

