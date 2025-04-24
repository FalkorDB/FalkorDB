//------------------------------------------------------------------------------
// LAGraph/src/test/test_IsEqual.c:  test LAGraph_*_IsEqual
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#include "LAGraph_test.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

int status ;
GrB_Info info ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B = NULL ;
GrB_Vector u = NULL, v = NULL ;
GrB_Type atype = NULL ;
#define LEN 512
char filename [LEN+1] ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;

//------------------------------------------------------------------------------
// test matrices
//------------------------------------------------------------------------------

typedef struct
{
    bool isequal ;
    bool isequal0 ;
    const char *matrix1 ;
    const char *matrix2 ;
}
matrix_info ;

const matrix_info files [ ] =
{
    //   iseq        matrix1             matrix2
    {    0, 0, "A.mtx"           , "cover.mtx" },
    {    0, 1, "A.mtx"           , "A2.mtx" },
    {    0, 0, "cover.mtx"       , "cover_structure.mtx" },
    {    0, 0, "cover.mtx"       , "cover_structure.mtx" },
    {    1, 1, "LFAT5.mtx"       , "LFAT5.mtx" },
    {    0, 0, "sample2.mtx"     , "sample.mtx" },
    {    1, 1, "sample.mtx"      , "sample.mtx" },
    {    1, 1, "matrix_int32.mtx", "matrix_int32.mtx" },
    {    1, 1, "matrix_int32.mtx", "matrix_int32.mtx" },
    {    0, 0, "matrix_int32.mtx", "matrix_int64.mtx" },
    {    0, 0, "matrix_int32.mtx", "matrix_int64.mtx" },
    {    1, 1, "west0067.mtx"    , "west0067_jumbled.mtx" },
    {    1, 1, "west0067.mtx"    , "west0067_noheader.mtx"},
    {    0, 0, "LFAT5.mtx"       , "west0067.mtx" },
    {    0, 0, "empty.mtx"       , "full.mtx" },
    {    1, 1, "full.mtx"        , "full_noheader.mtx" },
    {    0, 0, ""                , "" }
} ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_IsEqual: test LAGraph_Matrix_IsEqual
//------------------------------------------------------------------------------

void test_IsEqual (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;
    printf ("\nTesting IsEqual:\n") ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth pair of files
        //----------------------------------------------------------------------

        const char *aname = files [k].matrix1 ;
        const char *bname = files [k].matrix2 ;
        const bool isequal = files [k].isequal ;
        const bool isequal0 = files [k].isequal0 ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("test %2d: %s %s\n", k, aname, bname) ;

        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        GrB_Index ancols ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        snprintf (filename, LEN, LG_DATA_DIR "%s", bname) ;
        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", bname) ;
        GrB_Index bncols ;
        OK (GrB_Matrix_ncols (&bncols, B)) ;

        //----------------------------------------------------------------------
        // compare the two matrices
        //----------------------------------------------------------------------

        bool same = false ;

        OK (LAGraph_Matrix_IsEqual (&same, A, B, msg)) ;
        TEST_CHECK (same == isequal) ;

        OK (LAGraph_Matrix_IsEqual (&same, A, A, msg)) ;
        TEST_CHECK (same == true) ;

        //----------------------------------------------------------------------
        // compare the two matrices with a given op
        //----------------------------------------------------------------------

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        OK (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

        GrB_BinaryOp op = NULL ;
        if      (atype == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
        else if (atype == GrB_INT8  ) op = GrB_EQ_INT8   ;
        else if (atype == GrB_INT16 ) op = GrB_EQ_INT16  ;
        else if (atype == GrB_INT32 ) op = GrB_EQ_INT32  ;
        else if (atype == GrB_INT64 ) op = GrB_EQ_INT64  ;
        else if (atype == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
        else if (atype == GrB_UINT16) op = GrB_EQ_UINT16 ;
        else if (atype == GrB_UINT32) op = GrB_EQ_UINT32 ;
        else if (atype == GrB_UINT64) op = GrB_EQ_UINT64 ;
        else if (atype == GrB_FP32  ) op = GrB_EQ_FP32   ;
        else if (atype == GrB_FP64  ) op = GrB_EQ_FP64   ;

        OK (LAGraph_Matrix_IsEqualOp (&same, A, B, op, msg)) ;
        TEST_CHECK (same == isequal) ;

        OK (LAGraph_Matrix_IsEqualOp (&same, A, A, op, msg)) ;
        TEST_CHECK (same == true) ;

        //----------------------------------------------------------------------
        // compare two vectors
        //----------------------------------------------------------------------

        OK (GrB_Vector_new (&u, atype, ancols)) ;
        OK (GrB_Vector_new (&v, atype, bncols)) ;
        OK (GrB_Col_extract (u, NULL, NULL, A, GrB_ALL, ancols, 0,
            GrB_DESC_T0)) ;
        OK (GrB_Col_extract (v, NULL, NULL, B, GrB_ALL, bncols, 0,
            GrB_DESC_T0)) ;

        OK (LAGraph_Vector_IsEqual (&same, u, v, msg)) ;
        TEST_CHECK (same == isequal0) ;

        OK (LAGraph_Vector_IsEqual (&same, u, u, msg)) ;
        TEST_CHECK (same == true) ;

        OK (LAGraph_Vector_IsEqual (&same, u, u, msg)) ;
        TEST_CHECK (same == true) ;

        OK (GrB_free (&u)) ;
        OK (GrB_free (&v)) ;
        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_IsEqual_brutal:
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_IsEqual_brutal (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    OK (LG_brutal_setup (msg)) ;
    printf ("\nTesting IsEqual:\n") ;
    GxB_set (GxB_BURBLE, false) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth pair of files
        //----------------------------------------------------------------------

        const char *aname = files [k].matrix1 ;
        const char *bname = files [k].matrix2 ;
        const bool isequal = files [k].isequal ;
        const bool isequal0 = files [k].isequal0 ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("test %2d: %s %s\n", k, aname, bname) ;

        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        GrB_Index ancols ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        snprintf (filename, LEN, LG_DATA_DIR "%s", bname) ;
        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", bname) ;
        GrB_Index bncols ;
        OK (GrB_Matrix_ncols (&bncols, B)) ;

        //----------------------------------------------------------------------
        // compare the two matrices
        //----------------------------------------------------------------------

        bool same = false ;

        LG_BRUTAL (LAGraph_Matrix_IsEqual (&same, A, B, msg)) ;
        TEST_CHECK (same == isequal) ;

        LG_BRUTAL (LAGraph_Matrix_IsEqual (&same, A, A, msg)) ;
        TEST_CHECK (same == true) ;

        //----------------------------------------------------------------------
        // compare the two matrices with a given op
        //----------------------------------------------------------------------

        LG_BRUTAL (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        LG_BRUTAL (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

        GrB_BinaryOp op = NULL ;
        if      (atype == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
        else if (atype == GrB_INT8  ) op = GrB_EQ_INT8   ;
        else if (atype == GrB_INT16 ) op = GrB_EQ_INT16  ;
        else if (atype == GrB_INT32 ) op = GrB_EQ_INT32  ;
        else if (atype == GrB_INT64 ) op = GrB_EQ_INT64  ;
        else if (atype == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
        else if (atype == GrB_UINT16) op = GrB_EQ_UINT16 ;
        else if (atype == GrB_UINT32) op = GrB_EQ_UINT32 ;
        else if (atype == GrB_UINT64) op = GrB_EQ_UINT64 ;
        else if (atype == GrB_FP32  ) op = GrB_EQ_FP32   ;
        else if (atype == GrB_FP64  ) op = GrB_EQ_FP64   ;

        LG_BRUTAL (LAGraph_Matrix_IsEqualOp (&same, A, B, op, msg)) ;
        TEST_CHECK (same == isequal) ;

        LG_BRUTAL (LAGraph_Matrix_IsEqualOp (&same, A, A, op, msg)) ;
        TEST_CHECK (same == true) ;

        //----------------------------------------------------------------------
        // compare two vectors
        //----------------------------------------------------------------------

        LG_BRUTAL (GrB_Vector_new (&u, atype, ancols)) ;
        LG_BRUTAL (GrB_Vector_new (&v, atype, bncols)) ;
        LG_BRUTAL (GrB_Col_extract (u, NULL, NULL, A, GrB_ALL, ancols, 0,
            GrB_DESC_T0)) ;
        LG_BRUTAL (GrB_Col_extract (v, NULL, NULL, B, GrB_ALL, bncols, 0,
            GrB_DESC_T0)) ;

        LG_BRUTAL (LAGraph_Vector_IsEqual (&same, u, v, msg)) ;
        TEST_CHECK (same == isequal0) ;

        LG_BRUTAL (LAGraph_Vector_IsEqual (&same, u, u, msg)) ;
        TEST_CHECK (same == true) ;

        LG_BRUTAL (LAGraph_Vector_IsEqual (&same, u, u, msg)) ;
        TEST_CHECK (same == true) ;

        LG_BRUTAL (GrB_free (&u)) ;
        OK (GrB_free (&v)) ;
        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// test_IsEqual_failures: test error handling of LAGraph_Matrix_IsEqual*
//------------------------------------------------------------------------------

typedef int myint ;

void test_IsEqual_failures (void)
{
    setup ( ) ;
    printf ("\nTest IsEqual: error handling and special cases\n") ;

    bool same = false ;
    // not a failure, but a special case:
    OK (LAGraph_Matrix_IsEqual (&same, NULL, NULL, msg)) ;
    TEST_CHECK (same == true) ;

    OK (LAGraph_Vector_IsEqual (&same, NULL, NULL, msg)) ;
    TEST_CHECK (same == true) ;

    int result = LAGraph_Matrix_IsEqual (NULL, NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Matrix_IsEqual (NULL, NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    OK (GrB_Matrix_new (&A, GrB_BOOL, 2, 2)) ;
    OK (GrB_Matrix_new (&B, GrB_BOOL, 2, 2)) ;

    OK (GrB_Vector_new (&u, GrB_BOOL, 2)) ;
    OK (GrB_Vector_new (&v, GrB_BOOL, 2)) ;

    result = LAGraph_Matrix_IsEqual (NULL, A, B, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Matrix_IsEqualOp (&same, A, B, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Vector_IsEqual (NULL, u, v, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Vector_IsEqualOp (&same, u, v, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Matrix_IsEqual (NULL, A, B, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    result = LAGraph_Matrix_IsEqual (NULL, A, B, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    OK (LAGraph_Matrix_IsEqual (&same, A, B, msg)) ;
    TEST_CHECK (same == true) ;

    OK (GrB_free (&u)) ;
    OK (GrB_free (&v)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_Vector_IsEqual: test LAGraph_Vector_isEqual
//------------------------------------------------------------------------------

void test_Vector_IsEqual (void)
{
    setup ( ) ;

    bool same = false ;
    OK (LAGraph_Vector_IsEqualOp (&same, NULL, NULL, GrB_EQ_BOOL, msg)) ;
    TEST_CHECK (same == true) ;

    OK (GrB_Vector_new (&u, GrB_BOOL, 3)) ;
    OK (GrB_Vector_new (&v, GrB_BOOL, 2)) ;

    OK (LAGraph_Vector_IsEqualOp (&same, u, v, GrB_EQ_BOOL, msg)) ;
    TEST_CHECK (same == false) ;

    OK (GrB_free (&u)) ;
    OK (GrB_Vector_new (&u, GrB_BOOL, 2)) ;

    OK (LAGraph_Vector_IsEqualOp (&same, u, v, GrB_EQ_BOOL, msg)) ;
    TEST_CHECK (same == true) ;

    OK (GrB_Vector_setElement (u, true, 0)) ;
    OK (GrB_Vector_setElement (v, true, 1)) ;
    OK (LAGraph_Vector_IsEqualOp (&same, u, v, GrB_EQ_BOOL, msg)) ;
    TEST_CHECK (same == false) ;

    OK (LAGraph_Vector_IsEqual (&same, u, v, msg)) ;
    TEST_CHECK (same == false) ;

    OK (GrB_free (&u)) ;
    OK (GrB_free (&v)) ;

    OK (GrB_Vector_new (&u, GrB_BOOL, 3)) ;
    OK (GrB_Vector_new (&v, GrB_FP32, 3)) ;
    OK (LAGraph_Vector_IsEqual (&same, u, v, msg)) ;
    TEST_CHECK (same == false) ;

    OK (GrB_free (&u)) ;
    OK (GrB_free (&v)) ;

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//------------------------------------------------------------------------------

TEST_LIST =
{
    { "IsEqual", test_IsEqual },
    { "Vector_IsEqual", test_Vector_IsEqual },
    { "IsEqual_failures", test_IsEqual_failures },
    #if LAGRAPH_SUITESPARSE
    { "IsEqual_brutal", test_IsEqual_brutal },
    #endif
    { NULL, NULL }
} ;
