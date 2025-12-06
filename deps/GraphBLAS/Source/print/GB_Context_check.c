//------------------------------------------------------------------------------
// GB_Context_check: check and print a Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_Context_check       // check a GraphBLAS Context
(
    const GxB_Context Context,  // GraphBLAS Context to print and check
    const char *name,           // name of the Context, optional
    int pr,                     // print level
    FILE *f                     // file for output
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GBPR0 ("\n    GraphBLAS Context: %s ", ((name != NULL) ? name : "")) ;

    if (Context == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (Context) ;

    GBPR0 ("\n") ;

    // name given by GrB_set, or 'GrB_*' name for built-in objects
    char *given_name = Context->user_name ;
    if (Context->user_name_size > 0 && given_name != NULL)
    { 
        GBPR0 ("    Context given name: [%s]\n", given_name) ;
    }

    int nthreads_max = GB_Context_nthreads_max_get (Context) ;
    GBPR0 ("    Context.nthreads: %d\n", nthreads_max) ;

    double chunk = GB_Context_chunk_get (Context) ;
    GBPR0 ("    Context.chunk:    %g\n", chunk) ;

    int32_t ngpus, gpu_ids [GB_MAX_NGPUS] ;
    ngpus = GB_Context_gpu_ids_get (Context, gpu_ids) ;
    if (ngpus >= 0)
    {
        GBPR0 ("    Context.ngpus:    %d\n", ngpus) ;
        GBPR0 ("    Context.gpu_ids:  [") ;
        for (int k = 0 ; k < ngpus ; k++)
        { 
            GBPR0 (" %d ", gpu_ids [k]) ;
        }
        GBPR0 ("]\n") ;
    }

    return (GrB_SUCCESS) ;
}

