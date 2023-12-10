//------------------------------------------------------------------------------
// GB_Op_free: free any operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
            GB_FREE (&(op->user_name), op->user_name_size) ;
            size_t defn_size = op->defn_size ;
            if (defn_size > 0)
            { 
                GB_FREE (&(op->defn), defn_size) ;
            }
            size_t header_size = op->header_size ;
            if (header_size > 0)
            { 
                op->magic = GB_FREED ;  // to help detect dangling pointers
                op->header_size = 0 ;
                GB_FREE (op_handle, header_size) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

