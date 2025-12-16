//------------------------------------------------------------------------------
// GB_mex_test35: test GrB_get for a serialized blob
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
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GrB_Type type = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    uint64_t blob_size = 0 ;
    char name [256] ;
    char defn [2048] ;
    int32_t code, i, iso ;
    float fvalue ;
    double dvalue ;
    void *blob = NULL ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // create a non-iso test matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP32, 5, 5)) ;
    OK (GrB_Matrix_setElement (A, 0, 0, 1)) ;
    OK (GrB_Matrix_setElement (A, 1, 0, 2)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_set (A, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_set (A, 64, GxB_OFFSET_INTEGER_HINT)) ;
    OK (GrB_set (A, 64, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_set (A, 64, GxB_COLINDEX_INTEGER_HINT)) ;

    //--------------------------------------------------------------------------
    // test global hints
    //--------------------------------------------------------------------------

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_set (GrB_GLOBAL, 0, GxB_OFFSET_INTEGER_HINT)) ;
    ERR (GrB_set (GrB_GLOBAL, 0, GxB_ROWINDEX_INTEGER_HINT)) ;
    ERR (GrB_set (GrB_GLOBAL, 0, GxB_COLINDEX_INTEGER_HINT)) ;

    OK (GrB_set (GrB_GLOBAL, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_OFFSET_INTEGER_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_OFFSET_INTEGER_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_set (GrB_GLOBAL, 32, GxB_COLINDEX_INTEGER_HINT)) ;

    //--------------------------------------------------------------------------
    // test GxB_Serialized_get
    //--------------------------------------------------------------------------

    for (int p_control = 0 ; p_control <= 64 ; p_control += 32)
    for (int r_control = 0 ; r_control <= 64 ; r_control += 32)
    for (int c_control = 0 ; c_control <= 64 ; c_control += 32)
    {
        printf ("\n-----------------controls: (%d %d %d)\n",
            p_control, r_control, c_control) ;
        OK (GrB_set (A, p_control, GxB_OFFSET_INTEGER_HINT)) ;
        OK (GrB_set (A, r_control, GxB_ROWINDEX_INTEGER_HINT)) ;
        OK (GrB_set (A, c_control, GxB_COLINDEX_INTEGER_HINT)) ;
        OK (GxB_print (A, 5)) ;

        expected = GrB_INVALID_VALUE ;
        ERR (GrB_set (A, 1, GxB_OFFSET_INTEGER_HINT)) ;
        ERR (GrB_set (A, 1, GxB_ROWINDEX_INTEGER_HINT)) ;
        ERR (GrB_set (A, 1, GxB_COLINDEX_INTEGER_HINT)) ;

        // create the blob
        OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_OFFSET_INTEGER_HINT, blob_size)) ;
        CHECK (i == p_control) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_OFFSET_INTEGER_HINT)) ;
        CHECK (i == p_control) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_ROWINDEX_INTEGER_HINT, blob_size)) ;
        CHECK (i == r_control) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_ROWINDEX_INTEGER_HINT)) ;
        CHECK (i == r_control) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_COLINDEX_INTEGER_HINT, blob_size)) ;
        CHECK (i == c_control) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_COLINDEX_INTEGER_HINT)) ;
        CHECK (i == c_control) ;

        iso = true ;
        OK (GxB_Serialized_get_INT32_(blob, &iso, GxB_ISO, blob_size)) ;
        CHECK (!iso) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_OFFSET_INTEGER_BITS, blob_size)) ;
        if (p_control != 0) CHECK (i == p_control) ;
        CHECK (i == 32 || i == 64) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_OFFSET_INTEGER_BITS)) ;
        if (p_control != 0) CHECK (i == p_control) ;
        CHECK (i == 32 || i == 64) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_ROWINDEX_INTEGER_BITS, blob_size)) ;
        if (r_control != 0) CHECK (i == r_control) ;
        CHECK (i == 32 || i == 64) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_ROWINDEX_INTEGER_BITS)) ;
        if (r_control != 0) CHECK (i == r_control) ;
        CHECK (i == 32 || i == 64) ;

        i = 911 ;
        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_COLINDEX_INTEGER_BITS, blob_size)) ;
        if (c_control != 0) CHECK (i == c_control) ;
        CHECK (i == 32 || i == 64) ;

        i = 911 ;
        OK (GrB_Matrix_get_INT32_(A, &i, GxB_COLINDEX_INTEGER_BITS)) ;
        if (c_control != 0) CHECK (i == c_control) ;
        CHECK (i == 32 || i == 64) ;

        ERR (GxB_Serialized_get_VOID_(blob, nothing, 0, blob_size)) ;

        OK (GxB_Serialized_get_SIZE_(blob, &size, GrB_EL_TYPE_STRING, blob_size)) ;
        CHECK (size == strlen ("GrB_FP32") + 1) ;
        OK (GxB_Serialized_get_String_(blob, name, GrB_EL_TYPE_STRING, blob_size)) ;
        CHECK (MATCH (name, "GrB_FP32")) ;

        OK (GxB_Serialized_get_SIZE_(blob, &size, GxB_JIT_C_NAME, blob_size)) ;
        CHECK (size == strlen ("float") + 1) ;
        OK (GxB_Serialized_get_String_(blob, name, GxB_JIT_C_NAME, blob_size)) ;
        CHECK (MATCH (name, "float")) ;

        OK (GxB_Serialized_get_String_(blob, name, GrB_NAME, blob_size)) ;
        CHECK (MATCH (name, "")) ;

        OK (GxB_Serialized_get_String_(blob, name, GrB_EL_TYPE_STRING, blob_size)) ;
        CHECK (MATCH (name, "GrB_FP32")) ;

        OK (GxB_Serialized_get_SIZE_(blob, &size, GrB_EL_TYPE_STRING, blob_size)) ;
        CHECK (size == strlen ("GrB_FP32") + 1) ;

        OK (GxB_Serialized_get_INT32_(blob, &code, GrB_EL_TYPE_CODE, blob_size)) ;
        CHECK (code == GrB_FP32_CODE) ;

        i = -1 ;
        OK (GxB_Serialized_get_Scalar_(blob, s_int32, GrB_EL_TYPE_CODE, blob_size)) ;
        OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
        CHECK (i == GrB_FP32_CODE) ;

        OK (GxB_Serialized_get_INT32_(blob, &i, GrB_STORAGE_ORIENTATION_HINT,
            blob_size)) ;
        printf ("blob storage: %d\n", i) ;
        CHECK (i == GrB_COLMAJOR) ;

        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_FORMAT, blob_size)) ;
        printf ("blob storage: %d\n", i) ;
        CHECK (i == GxB_BY_COL) ;

        OK (GxB_Serialized_get_INT32_(blob, &i, GxB_SPARSITY_CONTROL, blob_size)) ;
        printf ("blob sparsity control: %d\n", i) ;
        CHECK (i == GxB_HYPERSPARSE) ;

        // free the blob
        mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    }

    //--------------------------------------------------------------------------
    // change the matrix to dense
    //--------------------------------------------------------------------------

    OK (GrB_set (A, GxB_AUTO_SPARSITY, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 5, GrB_ALL, 5, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    OK (GrB_Matrix_get_String_ (A, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "float")) ;

    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Matrix_set_String_(A, "A matrix", GrB_NAME)) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    CHECK (MATCH (name, "A matrix")) ;

    // create the blob
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_INT32_(blob, &i, GxB_SPARSITY_STATUS, blob_size)) ;
    printf ("blob sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    OK (GxB_Serialized_get_String_ (blob, name, GrB_NAME, blob_size)) ;
    printf ("name: [%s]\n", name) ;
    CHECK (MATCH (name, "A matrix")) ;
    OK (GxB_Serialized_get_String_ (blob, &size, GrB_NAME, blob_size)) ;
    CHECK (size == strlen ("A matrix") + 1) ;

    OK (GxB_Serialized_get_String_ (blob, name, GrB_EL_TYPE_STRING, blob_size)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;
    OK (GxB_Serialized_get_String_(blob, &size, GrB_EL_TYPE_STRING, blob_size)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;

    OK (GxB_Serialized_get_String_ (blob, name, GxB_JIT_C_NAME, blob_size)) ;
    CHECK (MATCH (name, "float")) ;
    OK (GxB_Serialized_get_String_ (blob, &size, GxB_JIT_C_NAME, blob_size)) ;
    CHECK (size == strlen ("float") + 1) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Serialized_get_INT32_(blob, &i, 0, blob_size)) ;
    ERR (GxB_Serialized_get_SIZE_(blob, &size, 0, blob_size)) ;
    ERR (GxB_Serialized_get_INT32_(blob, &i, GxB_JIT_C_NAME, blob_size)) ;

    fvalue = -1 ;
    OK (GxB_Serialized_get_Scalar_(blob, s_fp32, GxB_BITMAP_SWITCH,
        blob_size)) ;
    OK (GrB_Scalar_extractElement_FP32_(&fvalue, s_fp32)) ;
    printf ("blob bitmap switch: %g\n", fvalue) ;
    CHECK (fabs (fvalue - 0.08) < 1e-6) ;

    // free the blob
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;

    //--------------------------------------------------------------------------
    // change the matrix to bitmap
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_set_INT32_(A, GxB_BITMAP, GxB_SPARSITY_CONTROL)) ;

    //--------------------------------------------------------------------------
    // create the blob
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_String_(A, name, GxB_JIT_C_NAME)) ;

    OK (GxB_Serialized_get_INT32_(blob, &i, GxB_SPARSITY_STATUS, blob_size)) ;
    printf ("blob sparsity status: %d\n", i) ;
    CHECK (i == GxB_BITMAP) ;

    OK (GrB_Scalar_setElement_FP32_(s_fp32, 0.25)) ;
    OK (GrB_Matrix_set_Scalar_(A, s_fp32, GxB_HYPER_SWITCH)) ;

    // free the blob and recreate it
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_Scalar_(blob, s_fp64, GxB_HYPER_SWITCH, blob_size)) ;
    OK (GrB_Scalar_extractElement_FP64_(&dvalue, s_fp64)) ;
    printf ("blob hyper switch: %g\n", dvalue) ;
    CHECK (fabs (dvalue - 0.25) < 1e-6) ;

    OK (GrB_Matrix_set_INT32_(A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    CHECK (i == GrB_ROWMAJOR) ;
    OK (GrB_Matrix_get_INT32_(A, &i, GxB_FORMAT)) ;
    CHECK (i == GxB_BY_ROW) ;
    // GxB_print (A, 3) ;

    // free the blob and recreate it
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;
    GxB_print (A, 3) ;

    OK (GxB_Serialized_get_INT32_(blob, &i, GrB_STORAGE_ORIENTATION_HINT,
        blob_size)) ;
    CHECK (i == GrB_ROWMAJOR) ;
    OK (GxB_Serialized_get_INT32_(blob, &i, GxB_FORMAT, blob_size)) ;
    CHECK (i == GxB_BY_ROW) ;
    // GxB_print (A, 3) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Matrix_get_String_(A, defn, 999)) ;
    ERR (GrB_Matrix_get_Scalar(A, s_int32, 999)) ;

    OK (GrB_Matrix_get_SIZE_(A, &size, GrB_NAME)) ;
    CHECK (size == strlen ("A matrix") + 1) ;

    expected = GrB_INVALID_OBJECT ;
    uint8_t *b = (uint8_t *) blob ;
    ERR (GxB_Serialized_get_INT32_(blob, &i, GxB_FORMAT, 20)) ;
    b [0]++ ;
    ERR (GxB_Serialized_get_INT32_(blob, &i, GxB_FORMAT, blob_size)) ;
    b [0]-- ;
    OK (GxB_Serialized_get_INT32_(blob, &i, GxB_FORMAT, blob_size)) ;
    CHECK (i == GxB_BY_ROW) ;

    OK (GxB_Serialized_get_Scalar_(blob, s_int32, GrB_STORAGE_ORIENTATION_HINT,
        blob_size)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_ROWMAJOR) ;

    OK (GxB_Serialized_get_Scalar_(blob, s_int32, GxB_FORMAT, blob_size)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GxB_BY_ROW) ;

    OK (GxB_Serialized_get_Scalar_(blob, s_int32, GxB_SPARSITY_CONTROL,
        blob_size)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GxB_BITMAP) ;

    OK (GxB_Serialized_get_Scalar_(blob, s_int32, GxB_SPARSITY_STATUS,
        blob_size)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GxB_BITMAP) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Serialized_get_Scalar_(blob, s_int32, GrB_NAME, blob_size)) ;
    ERR (GxB_Serialized_get_Scalar_(blob, name, 9999, blob_size)) ;

    OK (GrB_Type_new (&type, sizeof (mytype))) ;
    OK (GrB_Type_set_String_ (type, "mytype", GxB_JIT_C_NAME)) ;
    OK (GrB_Type_set_String_ (type, MYTYPE_DEFN, GxB_JIT_C_DEFINITION)) ;
    GrB_free (&A) ;

    int32_t one = 1 ;
    OK (GrB_Matrix_new (&A, type, 5, 5)) ;
    OK (GrB_Matrix_setElement (A, (void *) &one, 0, 0)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_print (A, 3)) ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    // free the blob and recreate it
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_String_(blob, name, GrB_NAME, blob_size)) ;
    CHECK (MATCH (name, "")) ;

    OK (GxB_Serialized_get_String_(blob, name, GxB_JIT_C_NAME, blob_size)) ;
    CHECK (MATCH (name, "mytype")) ;

    OK (GxB_Serialized_get_String_(blob, name, GrB_EL_TYPE_STRING, blob_size)) ;
    CHECK (MATCH (name, "")) ;

    GrB_free (&A) ;
    GrB_free (&type) ;
    OK (GrB_Type_new (&type, sizeof (mytype))) ;
    OK (GrB_Matrix_new (&A, type, 50, 50)) ;
    OK (GrB_Matrix_setElement (A, (void *) &one, 0, 0)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_set_INT32_(A, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;

    OK (GrB_Matrix_set_String_(A, "A hyper", GrB_NAME)) ;
    OK (GrB_Matrix_get_String_(A, name, GrB_NAME)) ;
    printf ("name [%s]\n", name) ;
    CHECK (MATCH (name, "A hyper")) ;
    GxB_print (A, 3) ;

    // free the blob and recreate it
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_String_(blob, name, GxB_JIT_C_NAME, blob_size)) ;
    CHECK (MATCH (name, "")) ;
    OK (GxB_Serialized_get_String_(blob, name, GrB_NAME, blob_size)) ;
    printf ("name [%s]\n", name) ;
    CHECK (MATCH (name, "A hyper")) ;

    OK (GrB_Type_set_String_ (type, "mytype", GxB_JIT_C_NAME)) ;
    OK (GrB_Type_set_String_ (type, "my type", GrB_NAME)) ;

    // free the blob and recreate it
    mxFree (blob) ; blob = NULL ; blob_size = 0 ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    OK (GxB_Serialized_get_String_(blob, name, GxB_JIT_C_NAME, blob_size)) ;
    CHECK (MATCH (name, "mytype")) ;
    OK (GxB_Serialized_get_String_(blob, name, GrB_EL_TYPE_STRING, blob_size)) ;
    CHECK (MATCH (name, "my type")) ;
    OK (GxB_Serialized_get_String_(blob, name, GrB_NAME, blob_size)) ;
    printf ("name [%s]\n", name) ;
    CHECK (MATCH (name, "A hyper")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Serialized_get_String_(blob, name, GrB_EL_TYPE_CODE, blob_size)) ;
    ERR (GxB_Serialized_get_SIZE_(blob, &size, GrB_EL_TYPE_CODE, blob_size)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&s) ;
    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&type) ;
    if (blob != NULL) mxFree (blob) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test35:  all tests passed.\n\n") ;
}

