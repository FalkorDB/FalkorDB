//------------------------------------------------------------------------------
// GrB_Global_get_*: get a global option
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"
#include "jitifyer/GB_jitifyer.h"

//------------------------------------------------------------------------------
// GrB_Global: an object defining the global state
//------------------------------------------------------------------------------

struct GB_Global_opaque GB_OPAQUE (WORLD_OBJECT) =
{
    GB_MAGIC,                       // magic: initialized
    0,                              // header_size: statically allocated
} ;

const GrB_Global GrB_GLOBAL = & GB_OPAQUE (WORLD_OBJECT) ;

//------------------------------------------------------------------------------
// GB_global_enum_get: get an enum value from the global state
//------------------------------------------------------------------------------

static GrB_Info GB_global_enum_get (int32_t *value, int field)
{

    switch (field)
    {

        case GrB_LIBRARY_VER_MAJOR : 

            (*value) = GxB_IMPLEMENTATION_MAJOR ;
            break ;

        case GrB_LIBRARY_VER_MINOR : 

            (*value) = GxB_IMPLEMENTATION_MINOR ;
            break ;

        case GrB_LIBRARY_VER_PATCH : 

            (*value) = GxB_IMPLEMENTATION_SUB ;
            break ;

        case GrB_API_VER_MAJOR : 

            (*value) = GxB_SPEC_MAJOR ;
            break ;

        case GrB_API_VER_MINOR : 

            (*value) = GxB_SPEC_MINOR ;
            break ;

        case GrB_API_VER_PATCH : 

            (*value) = GxB_SPEC_SUB ;
            break ;

        case GrB_BLOCKING_MODE : 

            // return just the GrB modes
            (*value) = (int) GB_Global_mode_get ( )  ;
            if ((*value) == GxB_NONBLOCKING_GPU) (*value) = GrB_NONBLOCKING ;
            if ((*value) == GxB_BLOCKING_GPU) (*value) = GrB_BLOCKING ;
            break ;

        case GxB_MODE : 

            // return all 4 possible modes (GrB and GxB)
            (*value) = (int) GB_Global_mode_get ( )  ;
            break ;

        case GrB_STORAGE_ORIENTATION_HINT : 

            (*value) = (int) (GB_Global_is_csc_get ( )) ?
                    GrB_COLMAJOR : GrB_ROWMAJOR ;
            break ;

        case GxB_FORMAT : 

            (*value) = (int) (GB_Global_is_csc_get ( )) ?
                    GxB_BY_COL : GxB_BY_ROW ;
            break ;

        case GxB_OFFSET_INTEGER_HINT : 

            (*value) = (int) GB_Global_p_control_get ( ) ;
            break ;

        case GxB_COLINDEX_INTEGER_HINT : 

            (*value) = (int) (GB_Global_is_csc_get ( ) ?
                GB_Global_j_control_get ( ) :
                GB_Global_i_control_get ( )) ;
            break ;

        case GxB_ROWINDEX_INTEGER_HINT : 

            (*value) = (int) (GB_Global_is_csc_get ( ) ?
                GB_Global_i_control_get ( ) :
                GB_Global_j_control_get ( )) ;
            break ;

        case GxB_GLOBAL_NTHREADS : 

            (*value) = (int) GB_Context_nthreads_max_get (NULL) ;
            break ;

        case GxB_GLOBAL_NGPUS : 

            (*value) = GB_Context_gpu_ids_get (NULL, NULL) ;
            break ;

        case GxB_NGPUS_MAX : 

            (*value) = GB_Global_gpu_count_get ( ) ;
            break ;

        case GxB_BURBLE : 

            (*value) = (int) GB_Global_burble_get ( ) ;
            break ;

        case GxB_LIBRARY_OPENMP : 

            #ifdef _OPENMP
            (*value) = (int) true ;
            #else
            (*value) = (int) false ;
            #endif
            break ;

        case GxB_PRINT_1BASED : 

            (*value) = (int) GB_Global_print_one_based_get ( ) ;
            break ;

        case GxB_INCLUDE_READONLY_STATISTICS : 

            (*value) = (int) GB_Global_stats_mem_shallow_get ( ) ;
            break ;

        case GxB_JIT_C_CONTROL : 

            (*value) = (int) GB_jitifyer_get_control ( ) ;
            break ;

        case GxB_JIT_USE_CMAKE : 

            (*value) = (int) GB_jitifyer_get_use_cmake ( ) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_Scalar
(
    GrB_Global g,
    GrB_Scalar scalar,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (g) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_1 (scalar, "GrB_Global_get_Scalar (g, scalar, field)") ;
    ASSERT_SCALAR_OK (scalar, "input Scalar for GrB_Global_get_Scalar", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    GB_OPENMP_LOCK_SET (0)
    {
        int32_t i ;
        info = GB_global_enum_get (&i, field) ;
        if (info == GrB_SUCCESS)
        { 
            // field specifies an int: assign it to the scalar
            info = GB_setElement ((GrB_Matrix) scalar, NULL, &i, 0, 0,
                GB_INT32_code, Werk) ;
        }
        else
        { 
            double x ;
            int64_t i64 ;
            switch (field)
            {

                case GxB_HYPER_SWITCH : 

                    x = (double) GB_Global_hyper_switch_get ( ) ;
                    info = GB_setElement ((GrB_Matrix) scalar, NULL, &x, 0, 0,
                        GB_FP64_code, Werk) ;

                    break ;

                case GxB_GLOBAL_CHUNK : 

                    x = GB_Context_chunk_get (NULL) ;
                    info = GB_setElement ((GrB_Matrix) scalar, NULL, &x, 0, 0,
                        GB_FP64_code, Werk) ;
                    break ;

                case GxB_HYPER_HASH : 

                    i64 = GB_Global_hyper_hash_get ( ) ;
                    info = GB_setElement ((GrB_Matrix) scalar, NULL, &i64, 0, 0,
                        GB_INT64_code, Werk) ;
                    break ;

                default : 

                    info = GrB_INVALID_VALUE ;
            }
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    ASSERT_SCALAR_OK (scalar, "output Scalar for GrB_Global_get_Scalar", GB0) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_global_string_get: get a string from the global state
//------------------------------------------------------------------------------

static GrB_Info GB_global_string_get (const char **value, int field)
{

    switch ((int) field)
    {

        case GrB_NAME : 
        case GxB_LIBRARY_NAME : 

            (*value) = GxB_IMPLEMENTATION_NAME ;
            break ;

        case GxB_LIBRARY_DATE : 

            (*value) = GxB_IMPLEMENTATION_DATE ;
            break ;

        case GxB_LIBRARY_ABOUT : 

            (*value) = GxB_IMPLEMENTATION_ABOUT ;
            break ;

        case GxB_LIBRARY_LICENSE : 

            (*value) = GxB_IMPLEMENTATION_LICENSE ;
            break ;

        case GxB_LIBRARY_COMPILE_DATE : 

            (*value) = __DATE__ ;
            break ;

        case GxB_LIBRARY_COMPILE_TIME : 

            (*value) = __TIME__ ;
            break ;

        case GxB_LIBRARY_URL : 

            (*value) = "http://faculty.cse.tamu.edu/davis/GraphBLAS" ;
            break ;

        case GxB_API_DATE : 

            (*value) = GxB_SPEC_DATE ;
            break ;

        case GxB_API_ABOUT : 

            (*value) = GxB_SPEC_ABOUT ;
            break ;

        case GxB_API_URL : 

            (*value) = "http://graphblas.org" ;
            break ;

        case GxB_COMPILER_NAME : 

            (*value) = GB_COMPILER_NAME ;
            break ;

        //----------------------------------------------------------------------
        // JIT configuration:
        //----------------------------------------------------------------------

        case GxB_JIT_C_COMPILER_NAME : 

            (*value) = GB_jitifyer_get_C_compiler ( ) ;
            break ;

        case GxB_JIT_C_COMPILER_FLAGS : 

            (*value) = GB_jitifyer_get_C_flags ( ) ;
            break ;

        case GxB_JIT_C_LINKER_FLAGS : 

            (*value) = GB_jitifyer_get_C_link_flags ( ) ;
            break ;

        case GxB_JIT_C_LIBRARIES : 

            (*value) = GB_jitifyer_get_C_libraries ( ) ;
            break ;

        case GxB_JIT_C_CMAKE_LIBS : 

            (*value) = GB_jitifyer_get_C_cmake_libs ( ) ;
            break ;

        case GxB_JIT_C_PREFACE : 

            (*value) = GB_jitifyer_get_C_preface ( ) ;
            break ;

        case GxB_JIT_CUDA_PREFACE : 

            (*value) = GB_jitifyer_get_CUDA_preface ( ) ;
            break ;

        case GxB_JIT_ERROR_LOG : 

            (*value) = GB_jitifyer_get_error_log ( ) ;
            break ;

        case GxB_JIT_CACHE_PATH : 

            (*value) = GB_jitifyer_get_cache_path ( ) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_String
(
    GrB_Global g,
    char * value,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;
    (*value) = '\0' ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    GB_OPENMP_LOCK_SET (0)
    {
        const char *s ;
        info = GB_global_string_get (&s, field) ;
        if (info == GrB_SUCCESS)
        { 
            strcpy (value, s) ;
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_INT32
(
    GrB_Global g,
    int32_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    GB_OPENMP_LOCK_SET (0)
    {
        info = GB_global_enum_get (value, field) ;
    }
    GB_OPENMP_LOCK_UNSET (0)

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_SIZE
(
    GrB_Global g,
    size_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;
    (*value) = 0 ;

    //--------------------------------------------------------------------------
    // get the size of the field
    //--------------------------------------------------------------------------

    const char *s ;
    GrB_Info info = GrB_NO_VALUE ;

    GB_OPENMP_LOCK_SET (0)
    {
        info = GB_global_string_get (&s, field) ;
        if (info == GrB_SUCCESS)
        { 
            (*value) = GB_STRLEN (s) + 1 ;
        }
        else
        { 
            switch ((int) field)
            {

                case GxB_BITMAP_SWITCH : 

                    (*value) = sizeof (double) * GxB_NBITMAP_SWITCH ;
                    info = GrB_SUCCESS ;
                    break ;

                case GxB_COMPILER_VERSION : 

                    (*value) = sizeof (int32_t) * 3 ;
                    info = GrB_SUCCESS ;
                    break ;

                case GxB_MALLOC_FUNCTION : 
                case GxB_CALLOC_FUNCTION : 
                case GxB_REALLOC_FUNCTION : 
                case GxB_FREE_FUNCTION : 

                    (*value) = sizeof (void *) ;
                    info = GrB_SUCCESS ;
                    break ;

                case GxB_GLOBAL_GPU_IDS : 

                    (*value) = sizeof (int32_t) * GB_MAX_NGPUS ;
                    info = GrB_SUCCESS ;
                    break ;

                default : 

                    info = GrB_INVALID_VALUE ;
            }
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_VOID
(
    GrB_Global g,
    void * value,
    int field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    GB_OPENMP_LOCK_SET (0)
    {
        switch (field)
        {

            case GxB_BITMAP_SWITCH : 

                {
                    double *dvalue = (double *) value ;
                    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                    {
                        dvalue [k] = (double) GB_Global_bitmap_switch_get (k) ;
                    }
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_COMPILER_VERSION : 

                {
                    int32_t *ivalue = (int32_t *) value ;
                    ivalue [0] = GB_COMPILER_MAJOR ;
                    ivalue [1] = GB_COMPILER_MINOR ;
                    ivalue [2] = GB_COMPILER_SUB ;
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_MALLOC_FUNCTION : 
                {
                    void **func = (void **) value ;
                    (*func) = GB_Global_malloc_function_get ( ) ;
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_CALLOC_FUNCTION : 
                {
                    void **func = (void **) value ;
                    (*func) = GB_Global_calloc_function_get ( ) ;
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_REALLOC_FUNCTION : 
                {
                    void **func = (void **) value ;
                    (*func) = GB_Global_realloc_function_get ( ) ;
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_FREE_FUNCTION : 
                {
                    void **func = (void **) value ;
                    (*func) = GB_Global_free_function_get ( ) ;
                }
                info = GrB_SUCCESS ;
                break ;

            case GxB_GLOBAL_GPU_IDS : 

                GB_Context_gpu_ids_get (NULL, (int32_t *) value) ;
                info = GrB_SUCCESS ;
                break ;

            default : 

                info = GrB_INVALID_VALUE ;
        }
    }
    GB_OPENMP_LOCK_UNSET (0)

    #pragma omp flush
    return (info) ;
}

