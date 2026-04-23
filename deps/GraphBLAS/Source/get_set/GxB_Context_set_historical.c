//------------------------------------------------------------------------------
// GxB_Context_set: set a field in Context (HISTORICAL; do not use for new code)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Context_set_INT32:  set a Context option (int32_t)
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_INT32      // set a parameter in a Context
(
    GxB_Context Context,            // Context to modify
    int field,                      // parameter to change
    int32_t value                   // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // set the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            GB_Context_nthreads_max_set (Context, value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_set_FP64: set a Context option (double scalar)
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_FP64       // set a parameter in a Context
(
    GxB_Context Context,            // Context to modify
    int field,                      // parameter to change
    double value                    // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // set the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            GB_Context_chunk_set (Context, value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_set: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set            // set a parameter in a Context
(
    GxB_Context Context,            // Context to modify
    int field,                      // parameter to change
    ...                             // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // set the parameter
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            {
                va_start (ap, field) ;
                int value = va_arg (ap, int) ;
                GB_Context_nthreads_max_set (Context, value) ;
                va_end (ap) ;
            }
            break ;

        case GxB_CONTEXT_CHUNK :            // same as GxB_CHUNK

            {
                va_start (ap, field) ;
                double value = va_arg (ap, double) ;
                GB_Context_chunk_set (Context, value) ;
                va_end (ap) ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

