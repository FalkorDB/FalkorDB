//------------------------------------------------------------------------------
// GB_mex_test29: test GrB_get and GrB_set (global)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

int myprintf (const char *restrict format, ...) ;

int myprintf (const char *restrict format, ...)
{
    printf ("[[myprintf:") ;
    va_list ap ;
    va_start (ap, format) ;
    vprintf (format, ap) ;
    va_end (ap) ;
    printf ("]]") ;
    return (1) ;
}

int myflush (void) ;

int myflush (void)
{
    printf ("myflush\n") ;
    fflush (stdout) ;
    return (0) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048], defn2 [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // global set/get
    //--------------------------------------------------------------------------

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_LIBRARY_VER_MAJOR)) ;
    CHECK (i == GxB_IMPLEMENTATION_MAJOR) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_LIBRARY_VER_MINOR)) ;
    CHECK (i == GxB_IMPLEMENTATION_MINOR) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_LIBRARY_VER_PATCH)) ;
    CHECK (i == GxB_IMPLEMENTATION_SUB) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_API_VER_MAJOR)) ;
    CHECK (i == GxB_SPEC_MAJOR) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_API_VER_MINOR)) ;
    CHECK (i == GxB_SPEC_MINOR) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_API_VER_PATCH)) ;
    CHECK (i == GxB_SPEC_SUB) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_BLOCKING_MODE)) ;
    CHECK (i == GrB_NONBLOCKING) ;

    i = -1 ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_MODE)) ;
    CHECK (i == GrB_NONBLOCKING) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    CHECK (i == GrB_COLMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_ROWMAJOR,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_ROWMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_COLMAJOR,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_COLMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_BOTH,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_ROWMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_COLMAJOR,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_COLMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_UNKNOWN,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_ROWMAJOR) ;

        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, GrB_COLMAJOR,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        CHECK (i == GrB_COLMAJOR) ;
    
        expected = GrB_INVALID_VALUE ;
        ERR (GrB_Global_set_INT32_ (GrB_GLOBAL, 999,
            GrB_STORAGE_ORIENTATION_HINT)) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_FORMAT)) ;
    ERR (GrB_Global_set_INT32_ (GrB_GLOBAL, 999, GxB_FORMAT)) ;
    CHECK (i == GxB_BY_COL) ;

    int32_t nth ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &nth, GxB_GLOBAL_NTHREADS)) ;
    printf ("nthreads: %d\n", nth) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, 2, GxB_GLOBAL_NTHREADS)) ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_GLOBAL_NTHREADS)) ;
    CHECK (i == 2) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, nth, GxB_GLOBAL_NTHREADS)) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_BURBLE)) ;
    printf ("burble:   %d\n", i) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, 1, GxB_BURBLE)) ;
    OK (GrB_Matrix_new (&A, GrB_FP32, 3, 3)) ;
    OK (GrB_assign (A, NULL, NULL, 3, GrB_ALL, 3, GrB_ALL, 3, NULL)) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, 0, GxB_BURBLE)) ;
    OK (GrB_assign (A, NULL, NULL, 4, GrB_ALL, 3, GrB_ALL, 3, NULL)) ;
    OK (GxB_print (A, 2)) ;

    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_LIBRARY_OPENMP)) ;
    printf ("GraphBLAS compiled with OpenMP: %d\n", i) ;

    int32_t onebase ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &onebase, GxB_PRINT_1BASED)) ;
    printf ("1based:   %d\n", i) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, 1, GxB_PRINT_1BASED)) ;
    OK (GxB_print (A, 2)) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, onebase, GxB_PRINT_1BASED)) ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_PRINT_1BASED)) ;
    CHECK (i == onebase) ;

    int32_t control ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    printf ("jit ctrl: %d\n", control) ;
    for (int c = 0 ; c <= GxB_JIT_ON ; c++)
    {
        int32_t b ;
        OK (GrB_Global_set_INT32_ (GrB_GLOBAL, c, GxB_JIT_C_CONTROL)) ;
        OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &b, GxB_JIT_C_CONTROL)) ;
        CHECK (c == b) ;
    }
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, control, GxB_JIT_C_CONTROL)) ;

    int32_t use_cmake ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &use_cmake, GxB_JIT_USE_CMAKE)) ;
    printf ("jit cmake %d\n", use_cmake) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, 1, GxB_JIT_USE_CMAKE)) ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i,GxB_JIT_USE_CMAKE)) ;
    CHECK (i == 1) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, use_cmake, GxB_JIT_USE_CMAKE)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Global_set_INT32_ (GrB_GLOBAL, 1, GrB_BLOCKING_MODE)) ;
    expected = GrB_EMPTY_OBJECT ;
    ERR (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_int32,
        GrB_STORAGE_ORIENTATION_HINT)) ;
    expected = GrB_INVALID_VALUE ;
    OK (GrB_Scalar_setElement_INT32 (s_int32, 1)) ;
    ERR (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_int32, GrB_BLOCKING_MODE)) ;

    OK (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_int32, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Scalar_setElement_INT32 (s_int32, 2)) ;
    OK (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_int32, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Scalar_extractElement (&i, s_int32)) ;
    CHECK (i == 1) ;
    OK (GrB_Global_set_INT32_ (GrB_GLOBAL, control, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32_ (GrB_GLOBAL, &i, GxB_JIT_C_CONTROL)) ;
    CHECK (i == control) ;

    OK (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_fp64, GxB_HYPER_SWITCH)) ;
    OK (GrB_Scalar_extractElement (&dvalue, s_fp64)) ;
    printf ("hyper switch: %g\n", dvalue) ;
    OK (GrB_Scalar_setElement (s_fp64, 0.75)) ;
    OK (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_fp64, GxB_HYPER_SWITCH)) ;
    OK (GrB_Scalar_clear (s_fp64)) ;
    OK (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_fp64, GxB_HYPER_SWITCH)) ;
    OK (GrB_Scalar_extractElement (&dvalue, s_fp64)) ;
    CHECK (dvalue == 0.75) ;

    OK (GrB_Scalar_setElement_FP64 (s_fp64, 0.75)) ;
    OK (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_fp64, GxB_HYPER_SWITCH)) ;
    OK (GrB_Scalar_clear (s_fp64)) ;

    OK (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_fp64, GxB_CHUNK)) ;
    OK (GrB_Scalar_extractElement (&dvalue, s_fp64)) ;
    printf ("chunk:        %g\n", dvalue) ;
    OK (GrB_Scalar_setElement (s_fp64, 8901)) ;
    OK (GrB_Global_set_Scalar_ (GrB_GLOBAL, s_fp64, GxB_CHUNK)) ;
    OK (GrB_Scalar_clear (s_fp64)) ;
    OK (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_fp64, GxB_CHUNK)) ;
    OK (GrB_Scalar_extractElement (&dvalue, s_fp64)) ;
    CHECK (dvalue == 8901) ;


    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Global_get_Scalar_ (GrB_GLOBAL, s_fp64, GrB_EL_TYPE_CODE)) ;
    ERR (GrB_Global_get_INT32_  (GrB_GLOBAL, &i, GrB_EL_TYPE_CODE)) ;
    ERR (GrB_Global_get_SIZE_   (GrB_GLOBAL, &size, GrB_EL_TYPE_CODE)) ;
    ERR (GrB_Global_get_String_ (GrB_GLOBAL, name, GrB_EL_TYPE_CODE)) ;
    ERR (GrB_Global_get_VOID_   (GrB_GLOBAL, nothing, GrB_EL_TYPE_CODE)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GrB_NAME)) ;
    printf ("library name: [%s]\n", name) ;
    CHECK (MATCH (name, GxB_IMPLEMENTATION_NAME)) ;

    name [0] = 0 ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_LIBRARY_NAME)) ;
    printf ("library name: [%s]\n", name) ;
    CHECK (MATCH (name, GxB_IMPLEMENTATION_NAME)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_LIBRARY_DATE)) ;
    printf ("library date: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_LIBRARY_ABOUT)) ;
    printf ("library about: [%s]\n", defn) ;
    CHECK (MATCH (defn, GxB_IMPLEMENTATION_ABOUT)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_LIBRARY_LICENSE)) ;
    printf ("library license: [%s]\n", defn) ;
    CHECK (MATCH (defn, GxB_IMPLEMENTATION_LICENSE)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_LIBRARY_COMPILE_DATE)) ;
    printf ("library compile date: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_LIBRARY_COMPILE_TIME)) ;
    printf ("library compile time: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_LIBRARY_URL)) ;
    printf ("library url: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_API_DATE)) ;
    printf ("api date: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_API_ABOUT)) ;
    printf ("api about: [%s]\n", defn) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, name, GxB_API_URL)) ;
    printf ("api url: [%s]\n", name) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_COMPILER_NAME)) ;
    printf ("compiler: [%s]\n", defn) ;


    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_COMPILER_NAME)) ;
    printf ("JIT C compiler: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "cc", GxB_JIT_C_COMPILER_NAME)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_COMPILER_NAME)) ;
    CHECK (MATCH (defn2, "cc")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_COMPILER_NAME)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_COMPILER_NAME)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_COMPILER_FLAGS)) ;
    printf ("JIT C compiler flags: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "-O", GxB_JIT_C_COMPILER_FLAGS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_COMPILER_FLAGS)) ;
    CHECK (MATCH (defn2, "-O")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_COMPILER_FLAGS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_COMPILER_FLAGS)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_LINKER_FLAGS)) ;
    printf ("JIT C link flags: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "-stuff", GxB_JIT_C_LINKER_FLAGS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_LINKER_FLAGS)) ;
    CHECK (MATCH (defn2, "-stuff")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_LINKER_FLAGS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_LINKER_FLAGS)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_LIBRARIES)) ;
    printf ("JIT C libraries: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "-lm", GxB_JIT_C_LIBRARIES)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_LIBRARIES)) ;
    CHECK (MATCH (defn2, "-lm")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_LIBRARIES)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_LIBRARIES)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_CMAKE_LIBS)) ;
    printf ("JIT C cmake libs: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "m;dl", GxB_JIT_C_CMAKE_LIBS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_CMAKE_LIBS)) ;
    CHECK (MATCH (defn2, "m;dl")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_CMAKE_LIBS)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_CMAKE_LIBS)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_C_PREFACE)) ;
    printf ("JIT C preface: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "// stuff", GxB_JIT_C_PREFACE)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_PREFACE)) ;
    CHECK (MATCH (defn2, "// stuff")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_C_PREFACE)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_C_PREFACE)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_CUDA_PREFACE)) ;
    printf ("JIT CUDA preface: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "// cu", GxB_JIT_CUDA_PREFACE)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_CUDA_PREFACE)) ;
    CHECK (MATCH (defn2, "// cu")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_CUDA_PREFACE)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_CUDA_PREFACE)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_ERROR_LOG)) ;
    printf ("JIT error log: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "errlog.txt", GxB_JIT_ERROR_LOG)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_ERROR_LOG)) ;
    CHECK (MATCH (defn2, "errlog.txt")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_ERROR_LOG)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_ERROR_LOG)) ;
    CHECK (MATCH (defn2, defn)) ;

    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn, GxB_JIT_CACHE_PATH)) ;
    printf ("JIT cache: [%s]\n", defn) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, "/tmp/stuff", GxB_JIT_CACHE_PATH)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_CACHE_PATH)) ;
    CHECK (MATCH (defn2, "/tmp/stuff")) ;
    OK (GrB_Global_set_String_ (GrB_GLOBAL, defn, GxB_JIT_CACHE_PATH)) ;
    OK (GrB_Global_get_String_ (GrB_GLOBAL, defn2, GxB_JIT_CACHE_PATH)) ;
    CHECK (MATCH (defn2, defn)) ;
    int ignore = system ("ls /tmp/stuff ; rm -rf /tmp/stuff") ;

    ERR (GrB_Global_set_String_ (GrB_GLOBAL, defn, GrB_NAME)) ;

    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_JIT_CACHE_PATH)) ;
    CHECK (size == strlen (defn) + 1) ;

    double sw [GxB_NBITMAP_SWITCH] ;
    double s2 [GxB_NBITMAP_SWITCH] ;
    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_BITMAP_SWITCH)) ;
    CHECK (size == sizeof (double) * GxB_NBITMAP_SWITCH) ;
    OK (GrB_Global_get_VOID_ (GrB_GLOBAL, (void *) sw, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Global_get_VOID_ (GrB_GLOBAL, (void *) s2, GxB_BITMAP_SWITCH)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        printf ("bitmap switch [%d] = %g\n", k, sw [k]) ;
        sw [k] = ((double) k) / 8. ; 
    }

    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) sw, GxB_BITMAP_SWITCH,
        size)) ;
    memset (sw, 0, size) ;
    OK (GrB_Global_get_VOID_ (GrB_GLOBAL, (void *) sw, GxB_BITMAP_SWITCH)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (sw [k] == ((double) k) / 8.) ; 
    }

    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) NULL, GxB_BITMAP_SWITCH,
        0)) ;
    OK (GrB_Global_get_VOID_ (GrB_GLOBAL, (void *) sw, GxB_BITMAP_SWITCH)) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        CHECK (sw [k] == s2 [k]) ;
    }

    ERR (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) s2, GxB_BITMAP_SWITCH,
        1)) ;

    ERR (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) NULL, 0, 0)) ;

    ERR (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) NULL, GxB_PRINTF, 0)) ;
    ERR (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) NULL, GxB_FLUSH, 0)) ;
    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) myprintf, GxB_PRINTF,
        sizeof (GB_printf_function_t))) ;
    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) myflush, GxB_FLUSH,
        sizeof (GB_flush_function_t))) ;
    OK (GxB_print (s_int32, 3)) ;
    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) mexPrintf, GxB_PRINTF,
        sizeof (GB_printf_function_t))) ;
    OK (GrB_Global_set_VOID_ (GrB_GLOBAL, (void *) NULL, GxB_FLUSH,
        sizeof (GB_flush_function_t))) ;
    OK (GxB_print (s_int32, 3)) ;

    int32_t cv [3] ;
    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_COMPILER_VERSION)) ;
    CHECK (size == sizeof (int32_t) * 3) ;
    OK (GrB_Global_get_VOID_ (GrB_GLOBAL, (void *) cv, GxB_COMPILER_VERSION)) ;

    for (int k = 0 ; k < 3 ; k++)
    {
        printf ("compiler version [%d] = %d\n", k, cv [k]) ;
    }

    void *f = NULL ;
    OK (GrB_Global_get_VOID_(GrB_GLOBAL, (void *) &f, GxB_MALLOC_FUNCTION)) ;
    CHECK (f == mxMalloc) ;
    OK (GrB_Global_get_VOID_(GrB_GLOBAL, (void *) &f, GxB_REALLOC_FUNCTION)) ;
    CHECK (f == mxRealloc) ;
    OK (GrB_Global_get_VOID (GrB_GLOBAL, (void *) &f, GxB_CALLOC_FUNCTION)) ;
    CHECK (f == mxCalloc) ;
    OK (GrB_Global_get_VOID (GrB_GLOBAL, (void *) &f, GxB_FREE_FUNCTION)) ;
    CHECK (f == mxFree) ;

    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_MALLOC_FUNCTION)) ;
    CHECK (size == sizeof (void *)) ;
    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_REALLOC_FUNCTION)) ;
    CHECK (size == sizeof (void *)) ;
    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_CALLOC_FUNCTION)) ;
    CHECK (size == sizeof (void *)) ;
    OK (GrB_Global_get_SIZE_ (GrB_GLOBAL, &size, GxB_FREE_FUNCTION)) ;
    CHECK (size == sizeof (void *)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&A) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test29:  all tests passed\n\n") ;
}

