//------------------------------------------------------------------------------
// GxB_Context_set_*: set a field in a Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GxB_Context_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_Scalar
(
    GxB_Context Context,
    GrB_Scalar scalar,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    GB_RETURN_IF_NULL_OR_INVALID (scalar) ;
    ASSERT_CONTEXT_OK (Context, "Context to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    int32_t ivalue = 0 ;
    double dvalue = 0 ;

    switch ((int) field)
    {

        case GxB_CONTEXT_NGPUS : 
        case GxB_CONTEXT_NTHREADS : 
            info = GrB_Scalar_extractElement_INT32 (&ivalue, scalar) ;
            break ;

        case GxB_CONTEXT_CHUNK : 
            info = GrB_Scalar_extractElement_FP64 (&dvalue, scalar) ;
            break ;

        default : 
            info = GrB_INVALID_VALUE ;
            break ;
    }

    if (info != GrB_SUCCESS)
    { 
        return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
    }

    switch ((int) field)
    {

        default:
        case GxB_CONTEXT_NTHREADS : 

            GB_Context_nthreads_max_set (Context, ivalue) ;
            break ;

        case GxB_CONTEXT_NGPUS : 

            // set # of gpus to the given ivalue, and GPU ids to 0:ivalue-1
            return (GB_Context_gpu_ids_set (Context, NULL, ivalue)) ;

        case GxB_CONTEXT_CHUNK :            // same as GxB_CHUNK

            GB_Context_chunk_set (Context, dvalue) ;
            break ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_set_String
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_String
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
    ASSERT_CONTEXT_OK (Context, "Context to get option", GB0) ;

    if (Context == GxB_CONTEXT_WORLD || field != GrB_NAME)
    { 
        // built-in GxB_CONTEXT_WORLD may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_user_name_set (&(Context->user_name),
        &(Context->user_name_size), value, false)) ;
}

//------------------------------------------------------------------------------
// GxB_Context_set_INT
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_INT
(
    GxB_Context Context,
    int32_t value,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;
    ASSERT_CONTEXT_OK (Context, "Context to set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_CONTEXT_NTHREADS :         // same as GxB_NTHREADS

            GB_Context_nthreads_max_set (Context, value) ;
            break ;

        case GxB_CONTEXT_NGPUS : 

            // set # of gpus to the given value, and GPU ids to 0:value-1
            return (GB_Context_gpu_ids_set (Context, NULL, value)) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Context_set_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_Context_set_VOID
(
    GxB_Context Context,
    void * value,
    int field,
    size_t size
)
{ 
    if (field == GxB_CONTEXT_GPU_IDS)
    {
        int32_t ngpus = GB_Context_gpu_ids_get (Context, NULL) ;
        if (size < ngpus * sizeof (int32_t))
        { 
            return (GrB_INVALID_VALUE) ;
        }
        else
        { 
            return (GB_Context_gpu_ids_set (Context, (int32_t *) value,
                ngpus)) ;
        }
    }
    else
    { 
        return (GrB_INVALID_VALUE) ;
    }
}

