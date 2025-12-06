//------------------------------------------------------------------------------
// GB_mex_test11: JIT testing and set/get
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/jitifyer/GB_file.h"
#include "../Source/jitifyer/GB_jitifyer.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void gb_myplus_11 (float *z, const float *x, const float *y) ;
void gb_myplus_11 (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }
#define MYPLUS_DEFN \
"void gb_myplus_11 (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }"

void gb_myinc_11 (float *z, const float *x) ;
void gb_myinc_11 (float *z, const float *x) { (*z) = (*x)+1 ; }
#define MYINC_DEFN \
"void gb_myinc_11 (float *z, const float *x) { (*z) = (*x)+1 ; }"

void gb_myidx_11 (int64_t *z, const void *x, uint64_t i, uint64_t j, const void *y);
void gb_myidx_11 (int64_t *z, const void *x, uint64_t i, uint64_t j, const void *y)
{
    (*z) = i + j ;
}
#define MYIDX_DEFN \
"void gb_myidx_11 (int64_t *z, const void *x, uint64_t i, uint64_t j," \
" const void *y)\n" \
"{ \n" \
"    (*z) = (int64_t) (i + j) ; \n" \
"}"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Scalar scalar = NULL ;
    const char *s = NULL, *t = NULL, *c = NULL, *cache = NULL ;
    GrB_Info info, expected ;
    bool onebased = false, iso = false ;
    int use_cmake_int = 0, onebased_int = 1, control = 99 ;
    size_t mysize = 99 ;
    bool use_cmake = false ;

    #define MAXLEN 4096
    char save_string [MAXLEN+1] ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    bool malloc_debug = GB_mx_get_global (true) ;
    printf ("GB_mex_test11: malloc_debug: %d\n", malloc_debug) ;

    //--------------------------------------------------------------------------
    // remove temp files and folders
    //--------------------------------------------------------------------------

    remove ("/tmp/grberr2.txt") ;
    remove ("/tmp/grb_error_log.txt") ;
    int ignore = system ("rm -rf /tmp/grb_cache") ;

    //--------------------------------------------------------------------------
    // determine if GraphBLAS was compiled with GRAPHBLAS_USE_JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    bool jit_enabled = (control == GxB_JIT_ON) ;

    //--------------------------------------------------------------------------
    // get/set tests
    //--------------------------------------------------------------------------

if (jit_enabled)
{
    printf ("JIT enabled:\n") ;

    OK (GrB_Matrix_new (&A, GrB_FP32, 3, 4)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 3, GrB_ALL, 4, NULL)) ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (iso) ;
    int iso2 = false ;
    OK (GrB_Matrix_get_INT32 (A, &iso2, GxB_ISO)) ;
    CHECK (iso == iso2) ;

    OK (GrB_Matrix_setElement (A, 3, 0, 0)) ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (!iso) ;
    OK (GrB_Matrix_get_INT32 (A, &iso2, GxB_ISO)) ;
    CHECK (iso == iso2) ;

    OK (GrB_Vector_new (&v, GrB_FP32, 3)) ;
    OK (GrB_assign (v, NULL, NULL, 1, GrB_ALL, 3, NULL)) ;
    OK (GxB_Vector_iso (&iso, v)) ;
    CHECK (iso) ;
    OK (GrB_Vector_get_INT32 (v, &iso2, GxB_ISO)) ;
    CHECK (iso == iso2) ;

    OK (GrB_Vector_setElement (v, 3, 0)) ;
    OK (GxB_Vector_iso (&iso, v)) ;
    CHECK (!iso) ;
    OK (GrB_Vector_get_INT32 (v, &iso2, GxB_ISO)) ;
    CHECK (iso == iso2) ;

    OK (GrB_free (&v)) ;

    OK (GxB_set (GxB_PRINT_1BASED, true)) ;
    OK (GxB_print (A, 3)) ;
    OK (GxB_get (GxB_PRINT_1BASED, &onebased)) ;
    CHECK (onebased == true) ;

    OK (GxB_Global_Option_set_INT32 (GxB_PRINT_1BASED, false)) ;
    OK (GxB_print (A, 3)) ;
    OK (GxB_Global_Option_get_INT32 (GxB_PRINT_1BASED, &onebased_int)) ;
    CHECK (onebased_int == 0) ;
    OK (GrB_free (&A)) ;

    //--------------------------------------------------------------------------
    // try cmake
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, true)) ;
    OK (GxB_set (GxB_JIT_USE_CMAKE, true)) ;
    OK (GxB_get (GxB_JIT_USE_CMAKE, &use_cmake)) ;
    CHECK (use_cmake == true) ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_USE_CMAKE, &use_cmake_int)) ;
    CHECK (use_cmake_int == 1) ;

    GrB_Type MyType = NULL ;
    info = GxB_Type_new (&MyType, 0, "gb_mytype", "typedef double gb_mytype ;") ;
    if (info != GrB_SUCCESS || MyType->size != sizeof (double))
    {
        // cmake didn't work
        OK (GrB_free (&MyType)) ;
        OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
        OK (GxB_Global_Option_set_INT32 (GxB_JIT_USE_CMAKE, false)) ;
        OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef double gb_mytype ;")) ;
    }
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;

    //--------------------------------------------------------------------------
    // test compiler name
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &c)) ;
    printf ("default compiler [%s]\n", c) ;
    strncpy (save_string, c, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_C_COMPILER_NAME, "cc")) ;
    OK (GxB_get (GxB_JIT_C_COMPILER_NAME, &s)) ;
    CHECK (MATCH (s, "cc")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &t)) ;
    CHECK (MATCH (t, "cc")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_NAME, "gcc")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &t)) ;
    CHECK (MATCH (t, "gcc")) ;

    // reset the compiler back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_NAME, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test compiler flags
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &s)) ;
    printf ("default flags [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_C_COMPILER_FLAGS, "-g")) ;
    OK (GxB_get (GxB_JIT_C_COMPILER_FLAGS, &s)) ;
    CHECK (MATCH (s, "-g")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &t)) ;
    CHECK (MATCH (t, "-g")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_FLAGS, "-O0")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &t)) ;
    CHECK (MATCH (t, "-O0")) ;

    // reset the flags back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_FLAGS, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test libraries for cmake
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_C_CMAKE_LIBS, &s)) ;
    printf ("default C cmake libs [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    printf ("set cmake libs:\n") ;
    OK (GxB_set (GxB_JIT_C_CMAKE_LIBS, "m")) ;
    printf ("get cmake libs:\n") ;
    OK (GxB_get (GxB_JIT_C_CMAKE_LIBS, &s)) ;
    CHECK (MATCH (s, "m")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_CMAKE_LIBS, &t)) ;
    CHECK (MATCH (t, "m")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_CMAKE_LIBS, "m;dl")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_CMAKE_LIBS, &t)) ;
    CHECK (MATCH (t, "m;dl")) ;

    // reset the cmake libraries back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_CMAKE_LIBS, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_CMAKE_LIBS, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test new type
    //--------------------------------------------------------------------------

    OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef int32_t gb_mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;
    OK (GrB_free (&MyType)) ;

    //--------------------------------------------------------------------------
    // switch back from cmake to use the C compiler
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set_INT32 (GxB_JIT_USE_CMAKE, false)) ;
    OK (GxB_get (GxB_JIT_USE_CMAKE, &use_cmake)) ;
    CHECK (use_cmake == false) ;

    //--------------------------------------------------------------------------
    // test C linker flags
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_C_LINKER_FLAGS, &s)) ;
    printf ("default linker flags [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_C_LINKER_FLAGS, "-shared")) ;
    OK (GxB_get (GxB_JIT_C_LINKER_FLAGS, &s)) ;
    CHECK (MATCH (s, "-shared")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LINKER_FLAGS, &t)) ;
    CHECK (MATCH (t, "-shared")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LINKER_FLAGS, " -shared  ")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LINKER_FLAGS, &t)) ;
    CHECK (MATCH (t, " -shared  ")) ;

    // reset the C linker flags back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LINKER_FLAGS, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LINKER_FLAGS, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test C libraries
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_C_LIBRARIES, &s)) ;
    printf ("default C libraries [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_C_LIBRARIES, "-lm")) ;
    OK (GxB_get (GxB_JIT_C_LIBRARIES, &s)) ;
    CHECK (MATCH (s, "-lm")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LIBRARIES, &t)) ;
    CHECK (MATCH (t, "-lm")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LIBRARIES, "-lm -ldl")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LIBRARIES, &t)) ;
    CHECK (MATCH (t, "-lm -ldl")) ;

    // reset the C libraries back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LIBRARIES, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LIBRARIES, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test C preface
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_C_PREFACE, &s)) ;
    printf ("default C preface [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_C_PREFACE, "// stuff here")) ;
    OK (GxB_get (GxB_JIT_C_PREFACE, &s)) ;
    CHECK (MATCH (s, "// stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_PREFACE, &t)) ;
    CHECK (MATCH (t, "// stuff here")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_PREFACE, "// more stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_PREFACE, &t)) ;
    CHECK (MATCH (t, "// more stuff here")) ;

    // reset the C preface back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_PREFACE, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_PREFACE, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test CUDA preface
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_CUDA_PREFACE, &s)) ;
    printf ("default CUDA preface [%s]\n", s) ;
    strncpy (save_string, s, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_CUDA_PREFACE, "// cuda stuff here")) ;
    OK (GxB_get (GxB_JIT_CUDA_PREFACE, &s)) ;
    CHECK (MATCH (s, "// cuda stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_CUDA_PREFACE, &t)) ;
    CHECK (MATCH (t, "// cuda stuff here")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_CUDA_PREFACE,
        "// more cuda stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_CUDA_PREFACE, &t)) ;
    CHECK (MATCH (t, "// more cuda stuff here")) ;

    // reset the CUDA preface back to the default
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_CUDA_PREFACE, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_CUDA_PREFACE, &t)) ;
    CHECK (MATCH (t, save_string)) ;

    //--------------------------------------------------------------------------
    // test JIT error handling
    //--------------------------------------------------------------------------

    OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef double gb_mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;

    OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef int32_t gb_mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;
    OK (GrB_free (&MyType)) ;

    printf ("\n--------------------------- intentional compile errors:\n") ;

    expected = GxB_JIT_ERROR ;
    ERR (GxB_Type_new (&MyType, 0, "gb_mytype2_crud", "garbage")) ;
    CHECK (MyType == NULL) ;
    printf ("\n-------------------------------------------------------\n\n") ;

    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    CHECK (control == GxB_JIT_LOAD) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    CHECK (control == GxB_JIT_ON) ;

    OK (GxB_get (GxB_JIT_ERROR_LOG, &s)) ;
    printf ("default error log: [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_ERROR_LOG, "/tmp/grb_error_log.txt")) ;
    OK (GxB_get (GxB_JIT_ERROR_LOG, &t)) ;
    printf ("new error log: [%s]\n", t) ;
    CHECK (MATCH (t, "/tmp/grb_error_log.txt")) ;

    ERR (GxB_Type_new (&MyType, 0, "gb_mytype2_crud", "garbage")) ;
    CHECK (MyType == NULL) ;

    printf ("\n------------------------ compile error log (intentional):\n") ;
    ignore = system ("cat /tmp/grb_error_log.txt") ;
    printf ("\n-------------------------------------------------------\n\n") ;

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_ERROR_LOG, &s)) ;
    CHECK (MATCH (s, "/tmp/grb_error_log.txt")) ;

    OK (GxB_Global_Option_set_CHAR (GxB_JIT_ERROR_LOG, "/tmp/grberr2.txt")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_ERROR_LOG, &s)) ;
    CHECK (MATCH (s, "/tmp/grberr2.txt")) ;

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    ERR (GxB_Type_new (&MyType, 0, "gb_mytype2_crud", "more garbage")) ;
    CHECK (MyType == NULL) ;

    printf ("\n------------------------ compile error log (intentional):\n") ;
    ignore = system ("cat /tmp/grberr2.txt") ;
    printf ("\n-------------------------------------------------------\n\n") ;

    //--------------------------------------------------------------------------
    // test JIT cache path
    //--------------------------------------------------------------------------

    OK (GxB_get (GxB_JIT_CACHE_PATH, &cache)) ;
    printf ("default cache path: [%s]\n", cache) ;
    strncpy (save_string, cache, MAXLEN) ;
    save_string [MAXLEN] = '\0' ;

    OK (GxB_set (GxB_JIT_CACHE_PATH, "/tmp/grb_cache")) ;
    OK (GxB_get (GxB_JIT_CACHE_PATH, &s)) ;
    printf ("new cache path: [%s]\n", s) ;
    CHECK (MATCH (s, "/tmp/grb_cache")) ;

    OK (GxB_set (GxB_JIT_CACHE_PATH, "/tmp/::tmp/grb_@cache \\gunk??")) ;
    OK (GxB_get (GxB_JIT_CACHE_PATH, &s)) ;
    printf ("\nfixed cache path: [%s]\n", s) ;
    CHECK (MATCH (s, "/tmp/__tmp/grb__cache /gunk__")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Global_Option_set_CHAR (999, "gunk")) ;

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef double gb_mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;

    OK (GxB_Type_new (&MyType, 0, "gb_mytype", "typedef int32_t gb_mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;

    expected = GrB_INVALID_OBJECT ;
    MyType->name_len++ ;
    ERR (GxB_print (MyType, 3)) ;
    MyType->name_len-- ;
    OK (GxB_print (MyType, 3)) ;
    OK (GrB_free (&MyType)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, NULL, NULL)) ;
    ERR (GxB_Type_new (&MyType, 0, NULL, "typedef int32_t gb_mytype ;")) ;
    ERR (GxB_Type_new (&MyType, 0, "gb_mytype", NULL)) ;

    printf ("\nhere %d control is now: %d\n", __LINE__,
        GB_jitifyer_get_control ( )) ;

    // invalid cache path
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, "/root/noperm")) ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    CHECK (control == GxB_JIT_RUN) ;
    printf ("\nhere %d control is now: %d\n", __LINE__,
        GB_jitifyer_get_control ( )) ;

    // restore cache path
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    printf ("\nhere %d control is now: %d\n", __LINE__,
        GB_jitifyer_get_control ( )) ;
    printf ("\nset back to default cache path: [%s]\n", save_string) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, save_string)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_CACHE_PATH, &s)) ;
    printf ("cache [%s]\n" , save_string) ;
    printf ("s     [%s]\n" , s) ;
    CHECK (MATCH (s, save_string)) ;
}

    //--------------------------------------------------------------------------
    // GrB_Semiring_new memory tests
    //--------------------------------------------------------------------------

    GrB_Semiring sr = NULL ;
    GrB_BinaryOp op = NULL ;
    GrB_Monoid monoid = NULL ;
    METHOD (GrB_Semiring_new (&sr, GrB_PLUS_MONOID_FP32, GrB_TIMES_FP32)) ;
    OK (GxB_print (sr, 3)) ;
    GrB_free (&sr) ;

    OK (GxB_BinaryOp_new (&op, (GxB_binary_function) gb_myplus_11,
        GrB_FP32, GrB_FP32, GrB_FP32, "gb_myplus_11", MYPLUS_DEFN)) ;
    float zero = 0 ;
    OK (GrB_Monoid_new (&monoid, op, zero)) ;
    OK (GxB_print (op, 3)) ;
    OK (GxB_print (monoid, 3)) ;

    METHOD (GrB_Semiring_new (&sr, monoid, op)) ;
    OK (GxB_print (sr, 3)) ;

    expected = GrB_INVALID_OBJECT ;
    op->name_len-- ;
    ERR (GxB_print (op, 3)) ;
    op->name_len++ ;
    OK (GxB_print (op, 3)) ;

    sr->name_len-- ;
    ERR (GxB_print (sr, 3)) ;
    sr->name_len++ ;
    OK (GxB_print (sr, 3)) ;

    GrB_free (&sr) ;
    GrB_free (&monoid) ;
    GrB_free (&op) ;

    //--------------------------------------------------------------------------
    // GxB_select (deprecated but still functional)
    //--------------------------------------------------------------------------

    GrB_Type type = GrB_INT64 ;
    OK (GxB_SelectOp_ttype (&type, GxB_TRIL)) ;
    CHECK (type == NULL) ;
    type = GrB_INT64 ;
    OK (GxB_SelectOp_xtype (&type, GxB_TRIL)) ;
    CHECK (type == NULL) ;
    expected = GrB_NULL_POINTER ;
    ERR (GxB_SelectOp_fprint (NULL, "null", 3, stdout)) ;

    OK (GrB_Matrix_new (&A, GxB_FC32, 3, 4)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 3, GrB_ALL, 4, NULL)) ;
    OK (GrB_Matrix_setElement (A, 2, 0, 0)) ;
    OK (GrB_Matrix_setElement (A, 3, 0, 1)) ;
    OK (GrB_Scalar_new (&scalar, GxB_FC32)) ;
    OK (GrB_Scalar_setElement (scalar, (float) 2)) ;
    OK (GxB_Matrix_select (A, NULL, NULL, GxB_NE_THUNK, A, scalar, NULL)) ;
    OK (GxB_print (A, 3)) ;
    OK (GrB_Scalar_setElement (scalar, (float) 1)) ;
    OK (GxB_Matrix_select (A, NULL, NULL, GxB_EQ_THUNK, A, scalar, NULL)) ;
    OK (GxB_print (A, 3)) ;
    GrB_free (&A) ;
    GrB_free (&scalar) ;

    //--------------------------------------------------------------------------
    // GxB_print, GrB_IndexUnaryOp, and GrB_UnaryOp
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GxB_FC32, 30, 30)) ;
    OK (GrB_Matrix_setElement (A, 2, 4, 5)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    expected = GrB_INVALID_OBJECT ;
    A->nvals++ ;
    ERR (GxB_print (A, 3)) ;
    scalar = (GrB_Scalar) A ;
    ERR (GxB_print (scalar, 3)) ;
    A->nvals-- ;
    OK (GxB_print (A, 3)) ;

    GrB_IndexUnaryOp MyIdxOp = NULL ;
    #undef GrB_IndexUnaryOp_new
    #undef GrM_IndexUnaryOp_new
    OK (GrM_IndexUnaryOp_new (&MyIdxOp, (GxB_index_unary_function) gb_myidx_11,
        GrB_INT64, GrB_INT64, GrB_INT64)) ;
    OK (GxB_print (MyIdxOp, 3)) ;
    OK (GrB_apply (A, NULL, NULL, MyIdxOp, A, 0, NULL)) ;
    OK (GxB_print (A, 3)) ;

    MyIdxOp->name_len++ ;
    ERR (GxB_print (MyIdxOp, 3)) ;
    MyIdxOp->name_len-- ;
    OK (GxB_print (MyIdxOp, 3)) ;
    GrB_free (&MyIdxOp) ;

    GrB_UnaryOp MyUnOp = NULL ;
    OK (GrB_UnaryOp_new (&MyUnOp, (GxB_unary_function) gb_myinc_11, GrB_FP32, GrB_FP32)) ;
    OK (GxB_print (MyUnOp, 3)) ;
    OK (GrB_apply (A, NULL, NULL, MyUnOp, A, NULL)) ;
    OK (GxB_print (A, 3)) ;

    MyUnOp->name_len++ ;
    ERR (GxB_print (MyUnOp, 3)) ;
    MyUnOp->name_len-- ;
    OK (GxB_print (MyUnOp, 3)) ;
    GrB_free (&MyUnOp) ;

    GrB_free (&MyUnOp) ;
    GrB_free (&A) ;

    //--------------------------------------------------------------------------
    // GB_file tests
    //--------------------------------------------------------------------------

    if (jit_enabled)
    {
        printf ("JIT enabled:\n") ;
        bool ok = GB_file_mkdir (NULL) ;
        CHECK (!ok) ;
    }

    //--------------------------------------------------------------------------
    // GxB_Context
    //--------------------------------------------------------------------------

    GxB_Context context1 = NULL, context2 = NULL ;
    OK (GxB_Context_new (&context1)) ;
    OK (GxB_Context_new (&context2)) ;
    OK (GxB_Context_engage (context1)) ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Context_disengage (context2)) ;
    OK (GxB_Context_set (context1, GxB_CHUNK, (double) -1)) ;
    double chunk = 0 ;
    OK (GxB_Context_get (context1, GxB_CHUNK, &chunk)) ;
    CHECK (chunk == GB_CHUNK_DEFAULT) ;

    int nth ;
    OK (GxB_Context_set_INT32 (context1, GxB_CONTEXT_NTHREADS, 33)) ;
    OK (GxB_Context_get_INT32 (context1, GxB_CONTEXT_NTHREADS, &nth)) ;
    CHECK (nth == 33) ;

    OK (GxB_Context_set_FP64 (context1, GxB_CONTEXT_CHUNK, 1234)) ;
    OK (GxB_Context_get_FP64 (context1, GxB_CONTEXT_CHUNK, &chunk)) ;
    CHECK (chunk == 1234) ;

    OK (GxB_Context_disengage (NULL)) ;
    GrB_free (&context1) ;
    GrB_free (&context2) ;

    //--------------------------------------------------------------------------
    // memory usage
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP32, 5, 5)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 5, GrB_ALL, 5, NULL)) ;
    OK (GrB_Matrix_removeElement (A, 0, 0)) ;
    OK (GxB_print (A, 3)) ;
    size_t mem = 0 ;
    OK (GxB_Matrix_memoryUsage (&mem, A)) ;
    printf ("memory: %lu\n", mem) ;
    int64_t nallocs, nallocs2 ;
    size_t mem_deep, mem_deep2, mem_shallow, mem_shallow2 ;
    GB_memoryUsage (&nallocs, &mem_deep, &mem_shallow, A, false) ;
    printf ("nallocs: %ld deep: %lu shallow %lu\n", nallocs,
        mem_deep, mem_shallow) ;
    A->b_shallow = true ;
    OK (GxB_print (A, 3)) ;
    GB_memoryUsage (&nallocs2, &mem_deep2, &mem_shallow2, A, false) ;
    printf ("nallocs: %ld deep: %lu shallow %lu\n", nallocs2,
        mem_deep2, mem_shallow2) ;
    CHECK (nallocs == nallocs2 + 1) ;
    CHECK (mem_deep == mem_deep2 + mem_shallow2) ;
    CHECK (mem_shallow2 == 25) ;
    A->b_shallow = false ;
    GrB_free (&A) ;

    //--------------------------------------------------------------------------
    // GB_ijproperties
    //--------------------------------------------------------------------------

    GB_WERK ("about11") ;
    uint64_t I [1] = {0} ;  // OK
    int Ikind = GB_LIST ;
    int64_t Icolon [3] = {0,0,0}, imin_result, imax_result ;
    bool I_is_unsorted, I_has_dupl, I_is_contig ;
    OK (GB_ijproperties (
        // input:
        I, false, 0, 0, 5,
        // input/output:
        &Ikind, Icolon,
        // output:
        &I_is_unsorted, &I_has_dupl, &I_is_contig,
        &imin_result, &imax_result, Werk)) ;
    printf ("ijproperties: imin %ld imax %ld\n", imin_result, imax_result) ;
    CHECK (imin_result == 5) ;
    CHECK (imax_result == -1) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    // remove temp files and folders
    remove ("/tmp/grberr2.txt") ;
    remove ("/tmp/grb_error_log.txt") ;
    ignore = system ("rm -rf /tmp/grb_cache") ;

    OK (GxB_set (GxB_BURBLE, false)) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test11: (compiler errors above expected) "
        "all tests passed\n\n") ;
}

