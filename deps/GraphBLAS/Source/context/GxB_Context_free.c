//------------------------------------------------------------------------------
// GxB_Context_free: free a Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Predefined Contexts (GxB_CONTEXT_WORLD in particular) are not freed.
// Attempts to do so are silently ignored.

// If the Context to be freed is also currently the Context any user thread,
// then results are undefined.  Before freeing a Context, first disengage it
// from any user thread that might have it engaged as their Context object,
// via any of the following:
//
// GxB_Context_disengage (Context) ;   // disengages a particular Context
// GxB_Context_disengage (NULL) ;               // disengages any Context
// GxB_Context_disengage (GxB_CONTEXT_WORLD) ;  // disengages any Context

#include "GB.h"

GrB_Info GxB_Context_free           // free a Context
(
    GxB_Context *Context_handle     // handle of Context to free
)
{

    if (Context_handle != NULL)
    {
        // only free a dynamically-allocated Context
        GxB_Context Context = *Context_handle ;
        if (Context != NULL)
        {
            size_t header_size = Context->header_size ;
            // free the Context user_name
            GB_FREE_MEMORY (&(Context->user_name), Context->user_name_size) ;
            if (header_size > 0)
            { 
                Context->magic = GB_FREED ;  // to help detect dangling pointers
                Context->header_size = 0 ;
                GB_FREE_MEMORY (Context_handle, header_size) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

