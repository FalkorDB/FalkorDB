//------------------------------------------------------------------------------
// GB_Context_check: check and print a Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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

    int nthreads_max = GB_Context_nthreads_max_get (Context) ;
    GBPR0 ("    Context.nthreads: %d\n", nthreads_max) ;

    double chunk = GB_Context_chunk_get (Context) ;
    GBPR0 ("    Context.chunk:    %g\n", chunk) ;

    int gpu_id = GB_Context_gpu_id_get (Context) ;
    if (gpu_id >= 0) GBPR0 ("    Context.gpu_id:   %d\n", gpu_id) ;

    return (GrB_SUCCESS) ;
}

