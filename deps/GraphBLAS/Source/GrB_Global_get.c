//------------------------------------------------------------------------------
// GrB_Global_get_*: get a global option
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include "GB_jitifyer.h"

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

        case GxB_GLOBAL_NTHREADS :      // same as GxB_NTHREADS

            (*value) = (int) GB_Context_nthreads_max_get (NULL) ;
            break ;

        case GxB_GLOBAL_GPU_ID :            // same as GxB_GPU_ID

            (*value) = (int) GB_Context_gpu_id_get (NULL) ;
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
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_get_Scalar (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    GrB_Info info = GB_global_enum_get (&i, field) ;
    if (info == GrB_SUCCESS)
    { 
        // field specifies an int: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,
            GB_INT32_code, Werk) ;
    }
    else
    { 
        double x ;
        int64_t i64 ; 
        switch ((int) field)
        {

            case GxB_HYPER_SWITCH : 

                x = (double) GB_Global_hyper_switch_get ( ) ;
                info = GB_setElement ((GrB_Matrix) value, NULL, &x, 0, 0,
                    GB_FP64_code, Werk) ;

                break ;

            case GxB_GLOBAL_CHUNK :         // same as GxB_CHUNK

                x = GB_Context_chunk_get (NULL) ;
                info = GB_setElement ((GrB_Matrix) value, NULL, &x, 0, 0,
                    GB_FP64_code, Werk) ;
                break ;

            case GxB_HYPER_HASH : 

                i64 = GB_Global_hyper_hash_get ( ) ;
                info = GB_setElement ((GrB_Matrix) value, NULL, &i64, 0, 0,
                    GB_INT64_code, Werk) ;
                break ;

            default : 

                return (GrB_INVALID_VALUE) ;
        }
    }

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
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_get_String (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;
    (*value) = '\0' ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *s ;
    GrB_Info info = GB_global_string_get (&s, field) ;
    if (info == GrB_SUCCESS)
    { 
        strcpy (value, s) ;
    }
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
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_get_INT32 (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_global_enum_get (value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_SIZE
(
    GrB_Global g,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_get_SIZE (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;
    (*value) = 0 ;

    //--------------------------------------------------------------------------
    // get the size of the field
    //--------------------------------------------------------------------------

    const char *s ;
    GrB_Info info = GB_global_string_get (&s, field) ;
    if (info == GrB_SUCCESS)
    { 
        (*value) = strlen (s) + 1 ;
    }
    else
    { 
        switch ((int) field)
        {

            case GxB_BITMAP_SWITCH : 

                (*value) = sizeof (double) * GxB_NBITMAP_SWITCH ;
                break ;

            case GxB_COMPILER_VERSION : 

                (*value) = sizeof (int32_t) * 3 ;
                break ;

            case GxB_MALLOC_FUNCTION : 
            case GxB_CALLOC_FUNCTION : 
            case GxB_REALLOC_FUNCTION : 
            case GxB_FREE_FUNCTION : 

                (*value) = sizeof (void *) ;
                break ;

            default : 

                return (GrB_INVALID_VALUE) ;
        }
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Global_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Global_get_VOID
(
    GrB_Global g,
    void * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Global_get_VOID (g, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (g) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_BITMAP_SWITCH : 

            {
                double *dvalue = (double *) value ;
                for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                {
                    dvalue [k] = (double) GB_Global_bitmap_switch_get (k) ;
                }
            }
            break ;

        case GxB_COMPILER_VERSION : 

            {
                int32_t *ivalue = (int32_t *) value ;
                ivalue [0] = GB_COMPILER_MAJOR ;
                ivalue [1] = GB_COMPILER_MINOR ;
                ivalue [2] = GB_COMPILER_SUB ;
            }
            break ;

        case GxB_MALLOC_FUNCTION : 
            {
                void **func = (void **) value ;
                (*func) = GB_Global_malloc_function_get ( ) ;
            }
            break ;

        case GxB_CALLOC_FUNCTION : 
            {
                void **func = (void **) value ;
                (*func) = GB_Global_calloc_function_get ( ) ;
            }
            break ;

        case GxB_REALLOC_FUNCTION : 
            {
                void **func = (void **) value ;
                (*func) = GB_Global_realloc_function_get ( ) ;
            }
            break ;

        case GxB_FREE_FUNCTION : 
            {
                void **func = (void **) value ;
                (*func) = GB_Global_free_function_get ( ) ;
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

