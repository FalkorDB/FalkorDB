//------------------------------------------------------------------------------
// GxB_Context_new_arena: create a new Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Default values are set to the current GxB_CONTEXT_WORLD settings.

#include "GB.h"

#define GB_FREE_ALL GxB_Context_free (&Context) ;

GrB_Info GxB_Context_new_arena      // create a new Context in given arena
(
    GxB_Context *Context_handle,    // handle of Context to create
    const int header_arena
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (Context_handle) ;
    (*Context_handle) = NULL ;
    GxB_Context Context = NULL ;
    GB_OK (GB_check_arena (header_arena)) ;

    //--------------------------------------------------------------------------
    // create the Context
    //--------------------------------------------------------------------------

    uint64_t mem = GB_mem (header_arena, 0) ;

    // allocate the Context
    uint64_t header_mem = mem ;
    Context = GB_CALLOC_MEMORY (1, sizeof (struct GB_Context_opaque),
        &header_mem);
    if (Context == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    Context->magic = GB_MAGIC ;
    Context->header_mem = header_mem ;
    // user_name for GrB_get/GrB_set:
    Context->user_name = NULL ; Context->user_name_mem = 0 ;

    // initialize the Context with the same settings as GxB_CONTEXT_WORLD
    Context->nthreads_max = GB_Context_nthreads_max_get (NULL) ;
    Context->chunk = GB_Context_chunk_get (NULL) ;
    int32_t gpu_ids [GB_MAX_NGPUS] ;
    int32_t ngpus = GB_Context_gpu_ids_get (NULL, gpu_ids) ;
    GB_OK (GB_Context_gpu_ids_set (Context, gpu_ids, ngpus)) ;

    // return the result
    (*Context_handle) = Context ;
    return (GrB_SUCCESS) ;
}

