//------------------------------------------------------------------------------
// GxB_Global_Option_set: set a global default option for all future matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Global_Option_set is a single va_arg-based method for any global option,
// of any type.  The following functions are non-va_arg-based methods
// (useful for compilers and interfaces that do not support va_arg):
//
//  GxB_Global_Option_set_INT32         int32_t scalars
//  GxB_Global_Option_set_FP64          double scalars
//  GxB_Global_Option_set_FP64_ARRAY    double arrays
//  GxB_Global_Option_set_INT64_ARRAY   int64_t arrays
//  GxB_Global_Option_set_CHAR          strings
//  GxB_Global_Option_set_FUNCTION      function pointers (as void *)

#include "GB.h"
#include "GB_jitifyer.h"

//------------------------------------------------------------------------------
// GxB_Global_Option_set_INT32: set a global option (int32_t)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_INT32      // set a global default option
(
    GxB_Option_Field field,         // option to change
    int32_t value                   // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_INT32 (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    switch (field)
    {

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

            GB_jitifyer_set_control ((int) value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Global_Option_set_FP64: set a global option (double)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_FP64      // set a global default option
(
    GxB_Option_Field field,         // option to change
    double value                    // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_FP64 (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_HYPER_SWITCH : 

            GB_Global_hyper_switch_set ((float) value) ;
            break ;

        case GxB_GLOBAL_CHUNK :             // same as GxB_CHUNK

            GB_Context_chunk_set (NULL, value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Global_Option_set_FP64_ARRAY: set a global option (double array)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_FP64_ARRAY      // set a global default option
(
    GxB_Option_Field field,         // option to change
    double *value                   // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_FP64_ARRAY (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

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
                for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                { 
                    GB_Global_bitmap_switch_set (k, (float) (value [k])) ;
                }
            }
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Global_Option_set_INT64_ARRAY: set a global option (int64_t array)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_INT64_ARRAY      // set a global default option
(
    GxB_Option_Field field,         // option to change
    int64_t *value                  // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_INT64_ARRAY (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_MEMORY_POOL : 

            // nothing to do: no longer used
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Global_Option_set_CHAR: set a global option (string)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_CHAR      // set a global default option
(
    GxB_Option_Field field,         // option to change
    const char *value               // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_CHAR (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    switch (field)
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
// GxB_Global_Option_set_FUNCTION: set a global option (function pointer)
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set_FUNCTION      // set a global default option
(
    GxB_Option_Field field,         // option to change
    void *value                     // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set_FUNCTION (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    switch (field)
    {

        case GxB_PRINTF : 

            GB_Global_printf_set ((GB_printf_function_t) value) ;
            break ;

        case GxB_FLUSH : 

            GB_Global_flush_set ((GB_flush_function_t) value) ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Global_Option_set: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Global_Option_set      // set a global default option
(
    GxB_Option_Field field,         // option to change
    ...                             // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Global_Option_set (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        //----------------------------------------------------------------------
        // matrix format
        //----------------------------------------------------------------------

        case GxB_HYPER_SWITCH : 

            {
                va_start (ap, field) ;
                double hyper_switch = va_arg (ap, double) ;
                va_end (ap) ;
                GB_Global_hyper_switch_set ((float) hyper_switch) ;
            }
            break ;

        case GxB_BITMAP_SWITCH : 

            {
                va_start (ap, field) ;
                double *bitmap_switch = va_arg (ap, double *) ;
                va_end (ap) ;
                if (bitmap_switch == NULL)
                { 
                    // set all switches to their default
                    GB_Global_bitmap_switch_default ( ) ;
                }
                else
                {
                    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
                    { 
                        float b = (float) (bitmap_switch [k]) ;
                        GB_Global_bitmap_switch_set (k, b) ;
                    }
                }
            }
            break ;

        case GxB_FORMAT : 

            {
                va_start (ap, field) ;
                int format = va_arg (ap, int) ;
                va_end (ap) ;
                if (! (format == GxB_BY_ROW || format == GxB_BY_COL))
                { 
                    return (GrB_INVALID_VALUE) ;
                }
                GB_Global_is_csc_set (format != (int) GxB_BY_ROW) ; 
            }
            break ;

        //----------------------------------------------------------------------
        // GxB_CONTEXT_WORLD
        //----------------------------------------------------------------------

        case GxB_GLOBAL_NTHREADS :          // same as GxB_NTHREADS

            {
                va_start (ap, field) ;
                int value = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Context_nthreads_max_set (NULL, value) ;
            }
            break ;

        case GxB_GLOBAL_GPU_ID :            // same as GxB_GPU_ID

            {
                va_start (ap, field) ;
                int value = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Context_gpu_id_set (NULL, value) ;
            }
            break ;

        case GxB_GLOBAL_CHUNK :             // same as GxB_CHUNK

            {
                va_start (ap, field) ;
                double value = va_arg (ap, double) ;
                va_end (ap) ;
                GB_Context_chunk_set (NULL, value) ;
            }
            break ;

        //----------------------------------------------------------------------
        // memory pool control
        //----------------------------------------------------------------------

        case GxB_MEMORY_POOL : 

            // nothing to do: no longer used
            break ;

        //----------------------------------------------------------------------
        // diagnostics
        //----------------------------------------------------------------------

        case GxB_BURBLE : 

            {
                va_start (ap, field) ;
                int burble = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Global_burble_set ((bool) burble) ;
            }
            break ;

        case GxB_PRINTF : 

            {
                va_start (ap, field) ;
                void *printf_func = va_arg (ap, void *) ;
                va_end (ap) ;
                GB_Global_printf_set ((GB_printf_function_t) printf_func) ;
            }
            break ;

        case GxB_FLUSH : 

            {
                va_start (ap, field) ;
                void *flush_func = va_arg (ap, void *) ;
                va_end (ap) ;
                GB_Global_flush_set ((GB_flush_function_t) flush_func) ;
            }
            break ;

        case GxB_PRINT_1BASED : 

            {
                va_start (ap, field) ;
                int onebased = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Global_print_one_based_set ((bool) onebased) ;
            }
            break ;

        //----------------------------------------------------------------------
        // JIT configuruation
        //----------------------------------------------------------------------

        case GxB_JIT_C_COMPILER_NAME : 

            {
                va_start (ap, field) ;
                char *C_compiler = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_compiler (C_compiler)) ;
            }

        case GxB_JIT_C_COMPILER_FLAGS : 

            {
                va_start (ap, field) ;
                char *C_flags = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_flags (C_flags)) ;
            }

        case GxB_JIT_C_LINKER_FLAGS : 

            {
                va_start (ap, field) ;
                char *C_link = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_link_flags (C_link)) ;
            }

        case GxB_JIT_C_LIBRARIES : 

            {
                va_start (ap, field) ;
                char *C_libraries = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_libraries (C_libraries)) ;
            }

        case GxB_JIT_C_CMAKE_LIBS : 

            {
                va_start (ap, field) ;
                char *C_libraries = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_cmake_libs (C_libraries)) ;
            }

        case GxB_JIT_C_PREFACE : 

            {
                va_start (ap, field) ;
                char *C_preface = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_C_preface (C_preface)) ;
            }

        case GxB_JIT_USE_CMAKE : 

            {
                va_start (ap, field) ;
                int value = va_arg (ap, int) ;
                va_end (ap) ;
                GB_jitifyer_set_use_cmake ((bool) value) ;
            }
            break ;

        case GxB_JIT_C_CONTROL : 

            {
                va_start (ap, field) ;
                int value = va_arg (ap, int) ;
                va_end (ap) ;
                GB_jitifyer_set_control (value) ;
            }
            break ;

        case GxB_JIT_ERROR_LOG : 

            {
                va_start (ap, field) ;
                char *error_log = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_error_log (error_log)) ;
            }

        case GxB_JIT_CACHE_PATH : 

            {
                va_start (ap, field) ;
                char *cache_path = va_arg (ap, char *) ;
                va_end (ap) ;
                return (GB_jitifyer_set_cache_path (cache_path)) ;
            }

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

