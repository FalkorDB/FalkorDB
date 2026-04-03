//------------------------------------------------------------------------------
// GrB_Global_set_*: set a global option
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"
#include "jitifyer/GB_jitifyer.h"

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

        case GxB_OFFSET_INTEGER_HINT : 

            if (!(value == 32 || value == 64))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            GB_Global_p_control_set (value) ;
            break ;

        case GxB_COLINDEX_INTEGER_HINT : 

            if (!(value == 32 || value == 64))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            if (GB_Global_is_csc_get ( ))
            { 
                GB_Global_j_control_set (value) ;
            }
            else
            { 
                GB_Global_i_control_set (value) ;
            }
            break ;

        case GxB_ROWINDEX_INTEGER_HINT : 

            if (!(value == 32 || value == 64))
            { 
                return (GrB_INVALID_VALUE) ;
            }
            if (GB_Global_is_csc_get ( ))
            { 
                GB_Global_i_control_set (value) ;
            }
            else
            { 
                GB_Global_j_control_set (value) ;
            }
            break ;

        case GxB_GLOBAL_NTHREADS : 

            GB_Context_nthreads_max_set (NULL, value) ;
            break ;

        case GxB_BURBLE : 

            GB_Global_burble_set ((bool) value) ;
            break ;

        case GxB_PRINT_1BASED : 

            GB_Global_print_one_based_set ((bool) value) ;
            break ;

        case GxB_INCLUDE_READONLY_STATISTICS : 

            GB_Global_stats_mem_shallow_set ((bool) value) ;
            break ;

        case GxB_JIT_USE_CMAKE : 

            GB_jitifyer_set_use_cmake ((bool) value) ;
            break ;

        case GxB_JIT_C_CONTROL : 

            GB_jitifyer_set_control (value) ;
            break ;

        case GxB_GLOBAL_NGPUS : 

            // set # of gpus to the given value, and the GPU ids to 0:value-1
            return (GB_Context_gpu_ids_set (NULL, NULL, value)) ;

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
    GrB_Scalar scalar,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL_OR_INVALID (scalar) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (0)
    {
        double dvalue = 0 ;
        int32_t ivalue = 0 ;
        int64_t i64value = 0 ;
        switch ((int) field)
        {

            case GxB_HYPER_SWITCH : 

                info = GrB_Scalar_extractElement_FP64 (&dvalue, scalar) ;
                if (info == GrB_SUCCESS)
                { 
                    GB_Global_hyper_switch_set ((float) dvalue) ;
                }
                break ;

            case GxB_GLOBAL_CHUNK : 

                info = GrB_Scalar_extractElement_FP64 (&dvalue, scalar) ;
                if (info == GrB_SUCCESS)
                { 
                    GB_Context_chunk_set (NULL, dvalue) ;
                }
                break ;

            case GxB_HYPER_HASH : 

                info = GrB_Scalar_extractElement_INT64 (&i64value, scalar) ;
                if (info == GrB_SUCCESS)
                { 
                    GB_Global_hyper_hash_set (i64value) ;
                }
                break ;

            default : 

                info = GrB_Scalar_extractElement_INT32 (&ivalue, scalar) ;
                if (info == GrB_SUCCESS)
                { 
                    info = GB_global_enum_set (ivalue, field) ;
                }
                break ;
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    return ((info == GrB_NO_VALUE) ? GrB_EMPTY_OBJECT : info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_String
(
    GrB_Global g,
    char * value,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (0)
    {
        switch ((int) field)
        {

            case GxB_JIT_C_COMPILER_NAME : 

                info = GB_jitifyer_set_C_compiler (value) ;
                break ;

            case GxB_JIT_C_COMPILER_FLAGS : 

                info = GB_jitifyer_set_C_flags (value) ;
                break ;

            case GxB_JIT_C_LINKER_FLAGS : 

                info = GB_jitifyer_set_C_link_flags (value) ;
                break ;

            case GxB_JIT_C_LIBRARIES : 

                info = GB_jitifyer_set_C_libraries (value) ;
                break ;

            case GxB_JIT_C_CMAKE_LIBS : 

                info = GB_jitifyer_set_C_cmake_libs (value) ;
                break ;

            case GxB_JIT_C_PREFACE : 

                info = GB_jitifyer_set_C_preface (value) ;
                break ;

            case GxB_JIT_CUDA_PREFACE : 

                info = GB_jitifyer_set_CUDA_preface (value) ;
                break ;

            case GxB_JIT_ERROR_LOG : 

                info = GB_jitifyer_set_error_log (value) ;
                break ;

            case GxB_JIT_CACHE_PATH : 

                info = GB_jitifyer_set_cache_path (value) ;
                break ;

            default : 

                info = GrB_INVALID_VALUE ;
                break ;
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Global_set_INT32
(
    GrB_Global g,
    int32_t value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (0)
    {
        info = GB_global_enum_set (value, field) ;
    }
    GB_OPENMP_LOCK_UNSET (0)

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_set_VOID
//------------------------------------------------------------------------------

#include "include/GB_pedantic_disable.h"

GrB_Info GrB_Global_set_VOID
(
    GrB_Global g,
    void * value,
    int field,
    size_t size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (0)
    {
        switch (field)
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
                        info = GrB_INVALID_VALUE ;
                    }
                    else
                    {
                        double *dvalue = (double *) value ;
                        for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                        { 
                            float b = (float) dvalue [k] ;
                            GB_Global_bitmap_switch_set (k, b) ;
                        }
                    }
                }
                break ;

            case GxB_PRINTF : 

                if (size != sizeof (GB_printf_function_t))
                { 
                    info = GrB_INVALID_VALUE ;
                }
                else
                { 
                    GB_Global_printf_set ((GB_printf_function_t) value) ;
                }
                break ;

            case GxB_FLUSH : 

                if (size != sizeof (GB_flush_function_t))
                { 
                    info = GrB_INVALID_VALUE ;
                }
                else
                { 
                    GB_Global_flush_set ((GB_flush_function_t) value) ;
                }
                break ;

            case GxB_GLOBAL_GPU_IDS : 

                {
                    int32_t ngpus = GB_Context_gpu_ids_get (NULL, NULL) ;
                    if (size < ngpus * sizeof (int32_t))
                    { 
                        info = GrB_INVALID_VALUE ;
                    }
                    else
                    { 
                        info = GB_Context_gpu_ids_set (NULL, (int32_t *) value, ngpus) ;
                    }
                }
                break ;

            default : 

                info = GrB_INVALID_VALUE ;
                break ;
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    #pragma omp flush
    return (info) ;
}

