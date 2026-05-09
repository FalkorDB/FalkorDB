//------------------------------------------------------------------------------
// GB_Op_free: free any operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_Op_free             // free a user-created op
(
    GB_Operator *op_handle      // handle of operator to free
)
{

    if (op_handle != NULL)
    {
        // only free a dynamically-allocated operator
        GB_Operator op = *op_handle ;
        if (op != NULL)
        {
            GB_FREE_MEMORY (&(op->user_name), op->user_name_mem) ;
            size_t defn_mem = op->defn_mem ;
            if (GB_memsize (defn_mem) > 0)
            { 
                GB_FREE_MEMORY (&(op->defn), defn_mem) ;
            }
            size_t theta_mem = op->theta_mem ;
            if (GB_memsize (theta_mem) > 0)
            { 
                GB_FREE_MEMORY (&(op->theta), theta_mem) ;
            }
            uint64_t header_mem = op->header_mem ;
            if (GB_memsize (header_mem) > 0)
            { 
                op->magic = GB_FREED ;  // to help detect dangling pointers
                op->header_mem = 0 ;    // header is free
                GB_FREE_MEMORY (op_handle, header_mem) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

