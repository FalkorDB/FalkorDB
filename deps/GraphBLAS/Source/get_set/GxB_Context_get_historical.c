//------------------------------------------------------------------------------
// GxB_Context_get: get a field of Context (HISTORICAL; do not use for new code)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Context_get_INT32:  get a Context option (int32_t)
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_INT32      // get a parameter of a Context
(
    GxB_Context Context,            // Context to query
    int field,                      // parameter to query
    int32_t *value                  // return value from the Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            (*value) = GB_Context_nthreads_max_get (Context) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_FP64: get a Context option (double scalar)
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_FP64       // get a parameter in a Context
(
    GxB_Context Context,            // Context to query
    int field,                      // parameter to query
    double *value                   // return value from the Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            (*value) = GB_Context_chunk_get (Context) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get: get a Context option (va_arg variant)
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get            // get a parameter in a Context
(
    GxB_Context Context,            // Context to query
    int field,                      // parameter to query
    ...                             // return value of the descriptor
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // get the parameter
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            {
                va_start (ap, field) ;
                int *value = va_arg (ap, int *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (value) ;
                (*value) = GB_Context_nthreads_max_get (Context) ;
            }
            break ;

        case GxB_CONTEXT_CHUNK :            // same as GxB_CHUNK

            {
                va_start (ap, field) ;
                double *value = va_arg (ap, double *) ;
                va_end (ap) ;
                GB_RETURN_IF_NULL (value) ;
                (*value) = GB_Context_chunk_get (Context) ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

