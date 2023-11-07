//------------------------------------------------------------------------------
// GrB_Global_set_*: set a global option
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include "GB_jitifyer.h"

//------------------------------------------------------------------------------
// GB_global_enum_set: get an enum value from the global state
//------------------------------------------------------------------------------

static GrB_Info GB_global_enum_set (int32_t value, int field)
{

    switch (field)
    {

        case GrB_STORAGE_ORIENTATION_HINT : 

            switch (value)
            {
                case GrB_ROWMAJOR : value = GxB_BY_ROW ; break ;
                case GrB_COLMAJOR : value = GxB_BY_COL ; break ;
                case GrB_BOTH     : value = GxB_BY_ROW ; break ;
                case GrB_UNKNOWN  : value = GxB_BY_ROW ; break ;
                default : return (GrB_INVALID_VALUE) ;
            }
            // fall through to the GxB_FORMAT case

        case GxB_FORMAT : 

            if (! (value == GxB_BY_ROW || value == GxB_BY_COL))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            GB_Global_is_csc_set (value != (int) GxB_BY_ROW) ; 
            break ;

        case GxB_GLOBAL_NTHREADS :          // same as GxB_NTHREADS

            GB_Context_nthreads_max_set (NULL, value) ;
            break ;

        case GxB_GLOBAL_GPU_ID :            // same as GxB_GPU_ID

            GB_Context_gpu_id_set (NULL, value) ;
            break ;

        case GxB_BURBLE : 

            GB_Global_burble_set ((bool) value) ;
            break ;

        case GxB_PRINT_1BASED : 

            GB_Global_print_one_based_set ((bool) value) ;
            break ;

        case GxB_JIT_USE_CMAKE : 

            GB_jitifyer_set_use_cmake ((bool) value) ;
            break ;

        case GxB_JIT_C_CONTROL : 

            GB_jitifyer_set_control (value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_Scalar
(
    GrB_Global g,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_set_Scalar (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int32_t ivalue = 0 ;
    int64_t i64value = 0 ;
    GrB_Info info ;

    switch ((int) field)
    {

        case GxB_HYPER_SWITCH : 

            info = GrB_Scalar_extractElement_FP64 (&dvalue, value) ;
            if (info == GrB_SUCCESS)
            {
                GB_Global_hyper_switch_set ((float) dvalue) ;
            }
            break ;

        case GxB_GLOBAL_CHUNK :             // same as GxB_CHUNK

            info = GrB_Scalar_extractElement_FP64 (&dvalue, value) ;
            if (info == GrB_SUCCESS)
            {
                GB_Context_chunk_set (NULL, dvalue) ;
            }
            break ;

        case GxB_HYPER_HASH : 

            info = GrB_Scalar_extractElement_INT64 (&i64value, value) ;
            if (info == GrB_SUCCESS)
            {
                GB_Global_hyper_hash_set (i64value) ;
            }
            break ;

        default : 

            info = GrB_Scalar_extractElement_INT32 (&ivalue, value) ;
            if (info == GrB_SUCCESS)
            {
                info = GB_global_enum_set (ivalue, field) ;
            }
            break ;
    }

    return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_String
(
    GrB_Global g,
    char * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_set_String (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_JIT_C_COMPILER_NAME : 

            return (GB_jitifyer_set_C_compiler (value)) ;

        case GxB_JIT_C_COMPILER_FLAGS : 

            return (GB_jitifyer_set_C_flags (value)) ;

        case GxB_JIT_C_LINKER_FLAGS : 

            return (GB_jitifyer_set_C_link_flags (value)) ;

        case GxB_JIT_C_LIBRARIES : 

            return (GB_jitifyer_set_C_libraries (value)) ;

        case GxB_JIT_C_CMAKE_LIBS : 

            return (GB_jitifyer_set_C_cmake_libs (value)) ;

        case GxB_JIT_C_PREFACE : 

            return (GB_jitifyer_set_C_preface (value)) ;

        case GxB_JIT_ERROR_LOG : 

            return (GB_jitifyer_set_error_log (value)) ;

        case GxB_JIT_CACHE_PATH : 

            return (GB_jitifyer_set_cache_path (value)) ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Global_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_INT32
(
    GrB_Global g,
    int32_t value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_set_INT32 (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_global_enum_set (value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_VOID
(
    GrB_Global g,
    void * value,
    GrB_Field field,
    size_t size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_set_VOID (g, value, field, size)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_BITMAP_SWITCH : 

            if (value == NULL)
            { 
                // set all switches to their default
                GB_Global_bitmap_switch_default ( ) ;
            }
            else
            {
                if (size < sizeof (double) * GxB_NBITMAP_SWITCH)
                { 
                    return (GrB_INVALID_VALUE) ;
                }
                double *dvalue = (double *) value ;
                for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                { 
                    float b = (float) dvalue [k] ;
                    GB_Global_bitmap_switch_set (k, b) ;
                }
            }
            break ;

        case GxB_PRINTF : 

            if (size != sizeof (GB_printf_function_t))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            GB_Global_printf_set ((GB_printf_function_t) value) ;
            break ;

        case GxB_FLUSH : 

            if (size != sizeof (GB_flush_function_t))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            GB_Global_flush_set ((GB_flush_function_t) value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

