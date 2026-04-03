//------------------------------------------------------------------------------
// GxB_Context_new: create a new Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Default values are set to the current GxB_CONTEXT_WORLD settings.

#include "GB.h"

GrB_Info GxB_Context_new            // create a new Context
(
    GxB_Context *Context_handle     // handle of Context to create
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (Context_handle) ;
    (*Context_handle) = NULL ;
    GxB_Context Context = NULL ;

    //--------------------------------------------------------------------------
    // create the Context
    //--------------------------------------------------------------------------

    // allocate the Context
    size_t header_size ;
    Context = GB_CALLOC_MEMORY (1, sizeof (struct GB_Context_opaque),
        &header_size);
    if (Context == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    Context->magic = GB_MAGIC ;
    Context->header_size = header_size ;
    Context->user_name = NULL ;             // user_name for GrB_get/GrB_set
    Context->user_name_size = 0 ;

    // initialize the Context with the same settings as GxB_CONTEXT_WORLD
    Context->nthreads_max = GB_Context_nthreads_max_get (NULL) ;
    Context->chunk = GB_Context_chunk_get (NULL) ;
    int32_t gpu_ids [GB_MAX_NGPUS] ;
    int32_t ngpus = GB_Context_gpu_ids_get (NULL, gpu_ids) ;
    GrB_Info info = GB_Context_gpu_ids_set (Context, gpu_ids, ngpus) ;
    if (info != GrB_SUCCESS)
    {
        // This "cannot" fail since the global settings have already been
        // checked, so the inputs to the call to GB_Context_gpu_ids_set will
        // always be valid.  As a result, the test coverage cannot test this
        // case.
        GxB_Context_free (&Context) ;
        return (info) ;
    }

    // return the result
    (*Context_handle) = Context ;
    return (GrB_SUCCESS) ;
}

