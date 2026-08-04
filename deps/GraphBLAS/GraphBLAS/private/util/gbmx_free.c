//------------------------------------------------------------------------------
// gbmx_free: mxFree wrapper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

//  void *p = mxMalloc ( ... ) ;
//  gbmx_free (&p) ;                // frees p and sets p to NULL

void gbmx_free                  // mxFree wrapper
(
    void **p_handle             // handle to pointer to be freed
)
{

    if (p_handle != NULL)
    {
        if (*p_handle != NULL)
        { 
            mxFree (*p_handle) ;
        }
        (*p_handle) = NULL ;
    }
}

