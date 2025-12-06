//------------------------------------------------------------------------------
// GxB_Context_get_*: get a field in a context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GxB_Context_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_Scalar
(
    GxB_Context Context,
    GrB_Scalar scalar,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_1 (scalar, "GxB_Context_get_Scalar (Context, scalar, field)") ;

    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int32_t ivalue = 0 ;

    switch (field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            dvalue = GB_Context_chunk_get (Context) ;
            break ;

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            ivalue = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_NGPUS : 

            ivalue = GB_Context_gpu_ids_get (Context, NULL) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    switch (field)
    {

        case GxB_CONTEXT_CHUNK :         // same as GxB_CHUNK

            info = GB_setElement ((GrB_Matrix) scalar, NULL, &dvalue, 0, 0,
                GB_FP64_code, Werk) ;
            break ;

        default : 
            info = GB_setElement ((GrB_Matrix) scalar, NULL, &ivalue, 0, 0,
                GB_INT32_code, Werk) ;
            break ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_String
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_String
(
    GxB_Context Context,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field != GrB_NAME)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    (*value) = '\0' ;
    if (Context == GxB_CONTEXT_WORLD)
    { 
        // built-in Context
        strcpy (value, "GxB_CONTEXT_WORLD") ;
    }
    else if (Context->user_name_size > 0)
    { 
        // user-defined Context, with name defined by GrB_set
        strcpy (value, Context->user_name) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_INT
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_INT
(
    GxB_Context Context,
    int32_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            (*value) = GB_Context_nthreads_max_get (Context) ;
            break ;

        case GxB_CONTEXT_NGPUS : 

            (*value) = GB_Context_gpu_ids_get (Context, NULL) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_SIZE
(
    GxB_Context Context,
    size_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_CONTEXT_OK (Context, "context for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field == GxB_CONTEXT_GPU_IDS)
    {
        (*value) = sizeof (int32_t) * GB_MAX_NGPUS ;
        return (GrB_SUCCESS) ;
    }
    else if (field == GrB_NAME)
    { 
        if (Context->user_name != NULL)
        { 
            (*value) = Context->user_name_size ;
        }
        else
        { 
            (*value) = GxB_MAX_NAME_LEN ;
        }
        return (GrB_SUCCESS) ;
    }
    else
    { 
        return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Context_get_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_Context_get_VOID
(
    GxB_Context Context,
    void * value,
    int field
)
{ 
    if (field == GxB_CONTEXT_GPU_IDS)
    {
        return (GB_Context_gpu_ids_get (Context, (int32_t *) value)) ;
    }
    else
    { 
        return (GrB_INVALID_VALUE) ;
    }
}

