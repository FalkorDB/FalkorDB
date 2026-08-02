//------------------------------------------------------------------------------
// GrB_Descriptor_free: free a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Predefined descriptors are not freed.  Attempts to do so are silently
// ignored.

#include "GB.h"

GrB_Info GrB_Descriptor_free            // free a descriptor
(
    GrB_Descriptor *descriptor          // handle of descriptor to free
)
{

    if (descriptor != NULL)
    {
        // only free a dynamically-allocated descriptor
        GrB_Descriptor desc = *descriptor ;
        if (desc != NULL)
        {
            uint64_t header_mem = desc->header_mem ;
            // free the Descriptor user_name
            GB_FREE_MEMORY (&(desc->user_name), desc->user_name_mem) ;
            if (GB_memsize (header_mem) > 0)
            { 
                GB_FREE_MEMORY (&(desc->logger), desc->logger_mem) ;
                desc->logger_mem = 0 ;      // logger is freed
                desc->magic = GB_FREED ;    // to help detect dangling pointers
                desc->header_mem = 0 ;      // header is freed
                GB_FREE_MEMORY (descriptor, header_mem) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

