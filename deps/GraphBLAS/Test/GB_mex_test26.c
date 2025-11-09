//------------------------------------------------------------------------------
// GB_mex_test26: test GrB_get and GrB_set (type, scalar, vector, matrix)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef struct { int32_t stuff ; } mytype ;
#define MYTYPE_DEFN \
"typedef struct { int32_t stuff ; } mytype ;"

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
    GrB_Vector v = NULL ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL,
        s_uint64 = NULL ;
    GrB_Type type = NULL ;
    uint64_t u64 ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;
    OK (GrB_Scalar_new (&s_uint64, GrB_UINT64)) ;

    //--------------------------------------------------------------------------
    // GrB_Type get/set
    //--------------------------------------------------------------------------

    // type name size
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GrB_NAME)) ;
    CHECK (size == strlen ("GrB_BOOL") + 1) ;

    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GxB_JIT_C_NAME)) ;
    CHECK (size == strlen ("bool") + 1) ;

    // type name
    OK (GrB_Type_get_String_(GrB_BOOL, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_BOOL")) ;

    OK (GrB_Type_get_String_(GrB_INT8, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT8")) ;

    OK (GrB_Type_get_String_(GrB_INT16, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT16")) ;

    OK (GrB_Type_get_String_(GrB_INT32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT32")) ;

    OK (GrB_Type_get_String_(GrB_INT64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT64")) ;

    OK (GrB_Type_get_String_(GrB_UINT8, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT8")) ;

    OK (GrB_Type_get_String_(GrB_UINT16, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT16")) ;

    OK (GrB_Type_get_String_(GrB_UINT32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT32")) ;

    OK (GrB_Type_get_String_(GrB_UINT64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT64")) ;

    OK (GrB_Type_get_String_(GrB_FP32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Type_get_String_(GrB_FP64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_FP64")) ;

    OK (GrB_Type_get_String_(GxB_FC32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GxB_FC32")) ;

    OK (GrB_Type_get_String_(GxB_FC64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GxB_FC64")) ;

    // type JIT name
    OK (GrB_Type_get_String_(GrB_BOOL, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "bool")) ;

    OK (GrB_Type_get_String_(GrB_INT8, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "int8_t")) ;

    OK (GrB_Type_get_String_(GrB_INT16, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "int16_t")) ;

    OK (GrB_Type_get_String_(GrB_INT32, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "int32_t")) ;

    OK (GrB_Type_get_String_(GrB_INT64, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "int64_t")) ;

    OK (GrB_Type_get_String_(GrB_UINT8, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "uint8_t")) ;

    OK (GrB_Type_get_String_(GrB_UINT16, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "uint16_t")) ;

    OK (GrB_Type_get_String_(GrB_UINT32, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "uint32_t")) ;

    OK (GrB_Type_get_String_(GrB_UINT64, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "uint64_t")) ;

    OK (GrB_Type_get_String_(GrB_FP32, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "float")) ;

    OK (GrB_Type_get_String_(GrB_FP64, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "double")) ;

    OK (GrB_Type_get_String_(GxB_FC32, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "GxB_FC32_t")) ;

    OK (GrB_Type_get_String_(GxB_FC64, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "GxB_FC64_t")) ;

    // type code
    OK (GrB_Type_get_INT32_(GrB_BOOL, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_INT8, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_INT8_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_INT16, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_INT16_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_INT32, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_INT32_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_INT64, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_INT64_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_UINT8, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_UINT8_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_UINT16, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_UINT16_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_UINT32, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_UINT32_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_UINT64, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_UINT64_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_FP32, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_Type_get_INT32_(GrB_FP64, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_Type_get_INT32_(GxB_FC32, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GxB_FC32_CODE) ;

    OK (GrB_Type_get_INT32_(GxB_FC64, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GxB_FC64_CODE) ;

    // type size (using a GrB_Scalar): recommended type of GrB_UINT64
    OK (GrB_Type_get_Scalar_(GrB_BOOL, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (bool)) ;

    OK (GrB_Type_get_Scalar_(GrB_INT8, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (int8_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_INT16, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (int16_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_INT32, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (int32_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_INT64, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (int64_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_UINT8, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (uint8_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_UINT16, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (uint16_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_UINT32, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (uint32_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_UINT64, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (uint64_t)) ;

    OK (GrB_Type_get_Scalar_(GrB_FP32, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (float)) ;

    OK (GrB_Type_get_Scalar_(GrB_FP64, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (double)) ;

    OK (GrB_Type_get_Scalar_(GxB_FC32, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (float complex)) ;

    OK (GrB_Type_get_Scalar_(GxB_FC64, s_uint64, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_UINT64_(&u64, s_uint64)) ;
    CHECK (u64 == sizeof (double complex)) ;


    // type size (using a size_t)
    size = 0 ;
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (bool)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT8, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int8_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT16, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int16_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int32_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int64_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT8, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint8_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT16, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint16_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint32_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint64_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_FP32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (float)) ;

    OK (GrB_Type_get_SIZE_(GrB_FP64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (double)) ;

    OK (GrB_Type_get_SIZE_(GxB_FC32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (float complex)) ;

    OK (GrB_Type_get_SIZE_(GxB_FC64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (double complex)) ;



    // built-in type definition
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GxB_JIT_C_DEFINITION)) ;
    CHECK (size == 1) ;
    OK (GrB_Type_get_String_(GrB_BOOL, defn, GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH (defn, "")) ;

    // user-defined type
    OK (GrB_Type_new (&type, sizeof (mytype))) ;
    OK (GxB_print (type, 3)) ;
    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_set_String_(type, "", GxB_JIT_C_NAME)) ;
    OK (GrB_Type_set_String_(type, "mytype", GxB_JIT_C_NAME)) ;
    CHECK (type->hash == UINT64_MAX) ;
    OK (GrB_Type_set_String_(type, MYTYPE_DEFN, GxB_JIT_C_DEFINITION)) ;
    OK (GxB_print (type, 3)) ;
    CHECK (type->hash != UINT64_MAX) ;
    printf ("    hash: %016lx\n", type->hash) ;

    printf ("    test user name:\n") ;
    OK (GrB_Type_get_SIZE_(type, &size, GrB_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_Type_set_String_ (type, "user name of a type", GrB_NAME)) ;
    OK (GrB_Type_get_SIZE_(type, &size, GrB_NAME)) ;
    CHECK (size == strlen ("user name of a type") + 1) ;
    OK (GrB_Type_get_String_ (type, name, GrB_NAME)) ;
    CHECK (MATCH (name, "user name of a type")) ;
    OK (GxB_print (type, 3)) ;

    expected = GrB_ALREADY_SET ;
    ERR (GrB_Type_set_String_ (type, "another user name of a type", GrB_NAME)) ;
    printf ("    test GrB_ALREADY_SET: ok\n") ;

    OK (GrB_Type_get_String_(type, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "mytype")) ;
    OK (GrB_Type_get_SIZE_(type, &size, GxB_JIT_C_NAME)) ;
    CHECK (size == strlen ("mytype") + 1) ;

    OK (GrB_Type_get_SIZE_(type, &size, GxB_JIT_C_DEFINITION)) ;
    CHECK (size == strlen (MYTYPE_DEFN) + 1) ;
    OK (GrB_Type_get_String_(type, defn, GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH (defn, MYTYPE_DEFN)) ;

    OK (GrB_Type_get_Scalar_(type, s_int32, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == sizeof (mytype)) ;

    OK (GrB_Type_get_INT32_(type, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_UDT_CODE) ;

    OK (GrB_Type_get_String_(type, name, GrB_EL_TYPE_STRING)) ;
    CHECK (MATCH (name, "user name of a type")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_get_INT32_(type, &code, GrB_EL_TYPE_STRING)) ;

    i = -1 ;
    OK (GrB_Type_get_Scalar_(type, s_int32, GrB_EL_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_UDT_CODE) ;

    OK (GrB_Type_get_Scalar_(type, s_int32, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == sizeof (mytype)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_get_Scalar_(type, s_int32, GrB_OUTP)) ;
    ERR (GrB_Type_get_String_(type, name, GrB_OUTP)) ;
    ERR (GrB_Type_get_SIZE_(type, &size, GrB_OUTP)) ;

    ERR (GrB_Type_get_SIZE_(GrB_FP32, &i, GrB_SIZE)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_get_VOID_(type, nothing, 0)) ;
    ERR (GrB_Type_set_Scalar_(type, s_int32, 0)) ;
    ERR (GrB_Type_set_INT32_(type, 3, 0)) ;
    ERR (GrB_Type_set_VOID_(type, nothing, 0, 256)) ;

    //--------------------------------------------------------------------------
    // GrB_Scalar get/set
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&s, GrB_FP32)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Scalar_get_VOID_(s, nothing, 0)) ;

    OK (GrB_Scalar_get_SIZE_(s, &size, GrB_EL_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;
    OK (GrB_Scalar_get_String_(s, name, GrB_EL_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Scalar_get_String_(s, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Scalar_get_INT32_(s, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    i = -1 ;
    OK (GrB_Scalar_get_Scalar_(s, s_int32, GrB_EL_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_FP32_CODE) ;

    GxB_print (s, 3) ;

    OK (GrB_Scalar_get_INT32_(s, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Scalar_get_INT32_(s, &i, GxB_FORMAT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GxB_BY_COL) ;

    OK (GrB_Scalar_get_INT32_(s, &i, GxB_SPARSITY_CONTROL)) ;
    printf ("sparsity control: %d\n", i) ;
    CHECK (i == GxB_AUTO_SPARSITY) ;

    GxB_print (s_int32, 3) ;
    OK (GrB_Scalar_get_INT32_(s_int32, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Scalar_get_INT32_(s_int32, &i, 0)) ;
    ERR (GrB_Scalar_get_SIZE_(s, &size, 0)) ;

    ERR (GrB_Scalar_set_Scalar_(s, s_int32, 0)) ;
    OK (GrB_Scalar_set_Scalar_(s, s_int32, GrB_STORAGE_ORIENTATION_HINT)) ;

    ERR (GrB_Scalar_set_INT32_(s, 0, 0)) ;
    OK (GrB_Scalar_set_INT32_(s, 0, GrB_STORAGE_ORIENTATION_HINT)) ;

    OK (GrB_Scalar_set_String_(s, "scalar name", GrB_NAME)) ;
    OK (GrB_Scalar_get_String_(s, name, GrB_NAME)) ;
    OK (GrB_Scalar_get_SIZE_(s, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "scalar name")) ;
    CHECK (size == strlen (name) + 1) ;

    OK (GrB_Scalar_set_String_(s, "another scalar name", GrB_NAME)) ;
    OK (GrB_Scalar_get_String_(s, name, GrB_NAME)) ;
    OK (GrB_Scalar_get_SIZE_(s, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "another scalar name")) ;
    CHECK (size == strlen (name) + 1) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Scalar_set_VOID_(s, nothing, 0, 0)) ;

    //--------------------------------------------------------------------------
    // GrB_Vector get/set
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&v, GrB_FP32, 10)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Vector_get_VOID_(v, nothing, 0)) ;

    OK (GrB_Vector_get_SIZE_(v, &size, GrB_EL_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;
    OK (GrB_Vector_get_String_(v, name, GrB_EL_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Vector_get_String_(v, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Vector_get_INT32_(v, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    i = -1 ;
    OK (GrB_Vector_get_Scalar_(v, s_int32, GrB_EL_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_FP32_CODE) ;

    GxB_print (v, 3) ;

    OK (GrB_Vector_get_INT32_(v, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    printf ("vector storage: %d\n", i) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Vector_get_INT32_(v, &i, GxB_FORMAT)) ;
    printf ("vector storage: %d\n", i) ;
    CHECK (i == GxB_BY_COL) ;

    OK (GrB_Vector_set_INT32_(v, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Vector_get_INT32_(v, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Vector_get_INT32_(v, &i, GxB_SPARSITY_CONTROL)) ;
    printf ("sparsity control: %d\n", i) ;
    CHECK (i == GxB_AUTO_SPARSITY) ;

    OK (GrB_assign (v, NULL, NULL, 1, GrB_ALL, 10, NULL)) ;
    GxB_print (v, 3) ;

    OK (GrB_Vector_get_INT32_(v, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Vector_get_INT32_(v, &i, 0)) ;
    ERR (GrB_Vector_get_SIZE_(v, &size, 0)) ;

    fvalue = -1 ;
    OK (GrB_Vector_get_Scalar_(v, s_fp32, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP32_(&fvalue, s_fp32)) ;
    printf ("bitmap switch: %g\n", fvalue) ;
    CHECK (fabs (fvalue - 0.04) < 1e-6) ;

    OK (GrB_Scalar_setElement_FP32_(s_fp32, 0.5)) ;
    OK (GrB_Vector_set_Scalar_(v, s_fp32, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Vector_get_Scalar_(v, s_fp64, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP64_(&dvalue, s_fp64)) ;
    printf ("bitmap switch: %g\n", dvalue) ;
    CHECK (fabs (dvalue - 0.5) < 1e-6) ;

    OK (GrB_Scalar_setElement_INT32_(s_int32, GxB_BITMAP)) ;
    OK (GrB_Vector_set_Scalar_(v, s_int32, GxB_SPARSITY_CONTROL)) ;
    GxB_print (v, 3) ;

    OK (GrB_Vector_get_INT32_(v, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_BITMAP) ;

    OK (GrB_Vector_set_INT32_(v, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Vector_get_INT32_(v, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_SPARSE) ;

    ERR (GrB_Vector_set_Scalar_(v, s_int32, GxB_HYPER_SWITCH)) ;
    ERR (GrB_Vector_get_Scalar_(v, s_int32, GxB_HYPER_SWITCH)) ;

    OK (GrB_Vector_set_String_(v, "vector name", GrB_NAME)) ;
    OK (GrB_Vector_get_String_(v, name, GrB_NAME)) ;
    OK (GrB_Vector_get_SIZE_(v, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "vector name")) ;
    CHECK (size == strlen (name) + 1) ;

    OK (GrB_Vector_set_String_(v, "another vector name", GrB_NAME)) ;
    OK (GrB_Vector_get_String_(v, name, GrB_NAME)) ;
    OK (GrB_Vector_get_SIZE_(v, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "another vector name")) ;
    CHECK (size == strlen (name) + 1) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Vector_set_VOID_(v, nothing, 0, 1)) ;

    expected = GrB_EMPTY_OBJECT ;
    OK (GrB_Scalar_clear (s_int32)) ;
    ERR (GrB_Vector_set_Scalar_(v, s_int32, GxB_FORMAT)) ;

    //--------------------------------------------------------------------------
    // GrB_Matrix get/set
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP32, 5, 5)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Matrix_get_VOID_(A, nothing, 0)) ;

    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_EL_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_EL_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Matrix_get_INT32_(A, &code, GrB_EL_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    i = -1 ;
    OK (GrB_Matrix_get_Scalar_(A, s_int32, GrB_EL_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_FP32_CODE) ;

    GxB_print (A, 3) ;

    OK (GrB_Matrix_get_INT32_(A, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    printf ("matrix storage: %d\n", i) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Matrix_get_INT32_(A, &i, GxB_FORMAT)) ;
    printf ("matrix storage: %d\n", i) ;
    CHECK (i == GxB_BY_COL) ;

    OK (GrB_Matrix_get_INT32_(A, &i, GxB_SPARSITY_CONTROL)) ;
    printf ("sparsity control: %d\n", i) ;
    CHECK (i == GxB_AUTO_SPARSITY) ;

    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 5, GrB_ALL, 5, NULL)) ;
    GxB_print (A, 3) ;

    OK (GrB_Matrix_get_INT32_(A, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Matrix_get_INT32_(A, &i, 0)) ;
    ERR (GrB_Matrix_get_SIZE_(A, &size, 0)) ;

    fvalue = -1 ;
    OK (GrB_Matrix_get_Scalar_(A, s_fp32, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP32_(&fvalue, s_fp32)) ;
    printf ("bitmap switch: %g\n", fvalue) ;
    CHECK (fabs (fvalue - 0.08) < 1e-6) ;

    OK (GrB_Scalar_setElement_FP32_(s_fp32, 0.5)) ;
    OK (GrB_Matrix_set_Scalar_(A, s_fp32, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Matrix_get_Scalar_(A, s_fp64, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP64_(&dvalue, s_fp64)) ;
    printf ("bitmap switch: %g\n", dvalue) ;
    CHECK (fabs (dvalue - 0.5) < 1e-6) ;

    OK (GrB_Scalar_setElement_INT32_(s_int32, GxB_BITMAP)) ;
    OK (GrB_Matrix_set_Scalar_(A, s_int32, GxB_SPARSITY_CONTROL)) ;
    GxB_print (A, 3) ;

    OK (GrB_Matrix_get_INT32_(A, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_BITMAP) ;

    OK (GrB_Scalar_setElement_FP32_(s_fp32, 0.25)) ;
    OK (GrB_Matrix_set_Scalar_(A, s_fp32, GxB_HYPER_SWITCH)) ;
    OK (GrB_Matrix_get_Scalar_(A, s_fp64, GxB_HYPER_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP64_(&dvalue, s_fp64)) ;
    printf ("hyper switch: %g\n", dvalue) ;
    CHECK (fabs (dvalue - 0.25) < 1e-6) ;

    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_NAME)) ;
    CHECK (size == 1) ;

    OK (GrB_Matrix_set_String_(A, "matrix name", GrB_NAME)) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "matrix name")) ;
    CHECK (size == strlen (name) + 1) ;

    OK (GrB_Matrix_set_String_(A, "another matrix name", GrB_NAME)) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_NAME)) ;
    CHECK (MATCH (name, "another matrix name")) ;
    CHECK (size == strlen (name) + 1) ;

    OK (GrB_Matrix_get_String_(A, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "float")) ;
    OK (GrB_Matrix_get_SIZE_(A, &size, GxB_JIT_C_NAME)) ;
    CHECK (size == strlen ("float") + 1) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Matrix_set_String_ (A, "garbage", 999)) ;
    ERR (GrB_Matrix_set_VOID_(A, nothing, 0, 1)) ;
    ERR (GrB_Matrix_get_SIZE_(A, &size, 999)) ;

    OK (GrB_Matrix_set_INT32_(A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    CHECK (i == GrB_ROWMAJOR) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GxB_FORMAT)) ;
    CHECK (i == GxB_BY_ROW) ;
    GxB_print (A, 3) ;

    OK (GrB_Matrix_set_INT32_(A, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    CHECK (i == GrB_COLMAJOR) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GxB_FORMAT)) ;
    CHECK (i == GxB_BY_COL) ;
    GxB_print (A, 3) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Matrix_set_INT32_(A, 99, GxB_FORMAT)) ;
    ERR (GrB_Matrix_set_INT32_(A, 99, 999)) ;
    ERR (GrB_Matrix_get_String_(A, defn, 999)) ;
    ERR (GrB_Matrix_get_Scalar(A, s_int32, 999)) ;

    expected = GrB_EMPTY_OBJECT ;
    OK (GrB_Scalar_clear (s_int32)) ;
    ERR (GrB_Matrix_set_Scalar_(A, s_int32, GxB_FORMAT)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&v) ;
    GrB_free (&s) ;
    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&s_uint64) ;
    GrB_free (&type) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test26:  all tests passed.\n\n") ;
}

