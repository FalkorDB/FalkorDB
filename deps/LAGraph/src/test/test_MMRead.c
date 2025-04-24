//------------------------------------------------------------------------------
// LAGraph/src/test/test_MMRead.c:  test LAGraph_MMRead and LAGraph_MMWrite
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
#include "LG_internal.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

int status ;
GrB_Info info ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B = NULL ;
const char *name, *date ;
int ver [3] ;
GrB_Index nrows, ncols, nvals ;
#define LEN 512
char filename [LEN+1] ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;
char btype_name [LAGRAPH_MAX_NAME_LEN] ;

//------------------------------------------------------------------------------
// test matrices
//------------------------------------------------------------------------------

typedef struct
{
    GrB_Index nrows ;
    GrB_Index ncols ;
    GrB_Index nvals ;
    const char *type ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    // nrows ncols nvals type         name
    {    7,    7,    30, "bool",    "A.mtx" },
    {    7,    7,    12, "int32_t", "cover.mtx" },
    {    7,    7,    12, "bool",    "cover_structure.mtx" },
    { 1138, 1138,  7450, "bool",    "jagmesh7.mtx" },
    {    8,    8,    18, "bool",    "ldbc-cdlp-directed-example.mtx" },
    {    8,    8,    24, "bool",    "ldbc-cdlp-undirected-example.mtx" },
    {   10,   10,    17, "bool",    "ldbc-directed-example-bool.mtx" },
    {   10,   10,    17, "double",  "ldbc-directed-example.mtx" },
    {   10,   10,    17, "bool",    "ldbc-directed-example-unweighted.mtx" },
    {    9,    9,    24, "bool",    "ldbc-undirected-example-bool.mtx" },
    {    9,    9,    24, "double",  "ldbc-undirected-example.mtx" },
    {    9,    9,    24, "bool",    "ldbc-undirected-example-unweighted.mtx"},
    {   10,   10,    30, "int64_t", "ldbc-wcc-example.mtx" },
    {   14,   14,    46, "double",  "LFAT5.mtx" },
    {    6,    6,     8, "int64_t", "msf1.mtx" },
    {    8,    8,    12, "int64_t", "msf2.mtx" },
    {    5,    5,     7, "int64_t", "msf3.mtx" },
    {    8,    8,    28, "bool",    "sample2.mtx" },
    {    8,    8,    12, "bool",    "sample.mtx" },
    {   64,    1,    64, "int64_t", "sources_7.mtx" },
    { 1000, 1000,  3996, "double",  "olm1000.mtx" },
    { 2003, 2003, 83883, "double",  "bcsstk13.mtx" },
    { 2500, 2500, 12349, "double",  "cryg2500.mtx" },
    {    6,    6,    10, "int64_t", "tree-example.mtx" },
    {   67,   67,   294, "double",  "west0067.mtx" },
    {   27,   51,   102, "double",  "lp_afiro.mtx" },
    {   27,   51,   102, "bool",    "lp_afiro_structure.mtx" },
    {   34,   34,   156, "bool",    "karate.mtx" },
    {    7,    7,    12, "bool",    "matrix_bool.mtx" },
    {    7,    7,    12, "int8_t",  "matrix_int8.mtx" },
    {    7,    7,    12, "int16_t", "matrix_int16.mtx" },
    {    7,    7,    12, "int32_t", "matrix_int32.mtx" },
    {    7,    7,    12, "int64_t", "matrix_int64.mtx" },
    {    7,    7,    12, "uint8_t", "matrix_uint8.mtx" },
    {    7,    7,    12, "uint16_t","matrix_uint16.mtx" },
    {    7,    7,    12, "uint32_t","matrix_uint32.mtx" },
    {    7,    7,    12, "uint64_t","matrix_uint64.mtx" },
    {    7,    7,    12, "float",   "matrix_fp32.mtx" },
    {    7,    7,    12, "bool",    "matrix_fp32_structure.mtx" },
    {    7,    7,    12, "double",  "matrix_fp64.mtx" },
    {   67,   67,   294, "double",  "west0067_jumbled.mtx" },
    {    6,    6,    20, "float",    "skew_fp32.mtx" },
    {    6,    6,    20, "double",  "skew_fp64.mtx" },
    {    6,    6,    20, "int8_t",  "skew_int8.mtx" },
    {    6,    6,    20, "int16_t", "skew_int16.mtx" },
    {    6,    6,    20, "int32_t", "skew_int32.mtx" },
    {    6,    6,    20, "int64_t", "skew_int64.mtx" },
    {    7,    7,    12, "int32_t", "structure.mtx" },
    {    3,    3,     9, "double",  "full.mtx" },
    {    4,    4,    16, "double",  "full_symmetric.mtx" },
    {    3,    4,     0, "int32_t", "empty.mtx" },
    { 0, 0, 0, "", "" },
} ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    printf ("\nsetup: %s\n", __FILE__) ;
    printf ("data is in [%s]\n", LG_DATA_DIR) ;
    OK (LAGraph_Init (msg)) ;
    #if LAGRAPH_SUITESPARSE
    OK (GxB_get (GxB_LIBRARY_NAME, &name)) ;
    OK (GxB_get (GxB_LIBRARY_DATE, &date)) ;
    OK (GxB_get (GxB_LIBRARY_VERSION, ver)) ;
    #endif
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    #if LAGRAPH_SUITESPARSE
    printf ("\n%s %d.%d.%d (%s)\n", name, ver [0], ver [1], ver [2], date) ;
    #endif
    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    TEST_CHECK (A == NULL) ;
    TEST_CHECK (B == NULL) ;
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_MMRead:  read a set of matrices, check their stats, and write them out
//------------------------------------------------------------------------------

void test_MMRead (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file
        //----------------------------------------------------------------------

        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;

        //----------------------------------------------------------------------
        // check its stats
        //----------------------------------------------------------------------

        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        OK (GrB_Matrix_nvals (&nvals, A)) ;
        TEST_CHECK (nrows == files [k].nrows) ;
        TEST_CHECK (ncols == files [k].ncols) ;
        TEST_CHECK (nvals == files [k].nvals) ;

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        printf ("types: [%s] [%s]\n", atype_name, files [k].type) ;
        TEST_CHECK (MATCHNAME (atype_name, files [k].type)) ;
        TEST_MSG ("Stats are wrong for %s\n", aname) ;

        //----------------------------------------------------------------------
        // pretty-print the matrix
        //----------------------------------------------------------------------

        for (int pr = 0 ; pr <= 2 ; pr++)
        {
            printf ("\nPretty-print %s: pr=%d:\n", aname, pr) ;
            LAGraph_PrintLevel prl = pr ;
            OK (LAGraph_Matrix_Print (A, prl, stdout, msg)) ;
        }

        //----------------------------------------------------------------------
        // write it to a temporary file
        //----------------------------------------------------------------------

        f = tmpfile ( ) ;
        OK (LAGraph_MMWrite (A, f, NULL, msg)) ;
        TEST_MSG ("Failed to write %s to a temp file\n", aname) ;

        //----------------------------------------------------------------------
        // load it back in again
        //----------------------------------------------------------------------

        rewind (f) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;
        TEST_MSG ("Failed to load %s from a temp file\n", aname) ;
        OK (fclose (f)) ;       // close and delete the temporary file

        //----------------------------------------------------------------------
        // ensure A and B are the same
        //----------------------------------------------------------------------

        OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
        TEST_CHECK (MATCHNAME (atype_name, btype_name)) ;
        bool ok ;
        OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
        TEST_CHECK (ok) ;
        TEST_MSG ("Failed test for equality, file: %s\n", aname) ;

        //----------------------------------------------------------------------
        // free workspace
        //----------------------------------------------------------------------

        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_karate: read in karate graph from a file and compare it known graph
//-----------------------------------------------------------------------------

void test_karate (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    //--------------------------------------------------------------------------
    // load in the data/karate.mtx file as the matrix A
    //--------------------------------------------------------------------------

    FILE *f = fopen (LG_DATA_DIR "karate.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
    TEST_CHECK (MATCHNAME (atype_name, "bool")) ;
    OK (fclose (f)) ;
    OK (LAGraph_Matrix_Print (A, LAGraph_SHORT, stdout, msg)) ;
    TEST_MSG ("Loading of A matrix failed: karate matrix") ;

    //--------------------------------------------------------------------------
    // load in the matrix defined by graph_zachary_karate.h as the matrix B
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&B, GrB_BOOL, ZACHARY_NUM_NODES, ZACHARY_NUM_NODES)) ;
    OK (GrB_Matrix_build (B, ZACHARY_I, ZACHARY_J, ZACHARY_V,
        ZACHARY_NUM_EDGES, GrB_LOR)) ;
    OK (LAGraph_Matrix_Print (B, LAGraph_SHORT, stdout, msg)) ;
    TEST_MSG ("Loading of B matrix failed: karate matrix") ;

    //--------------------------------------------------------------------------
    // ensure A and B are the same
    //--------------------------------------------------------------------------

    bool ok ;
    OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
    TEST_CHECK (ok) ;
    TEST_MSG ("Test for A and B equal failed: karate matrix") ;

    //--------------------------------------------------------------------------
    // free workspace and finish the test
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_failures: test for failure modes of LAGraph_MMRead and MMWrite
//-----------------------------------------------------------------------------

typedef struct
{
    int error ;
    const char *name ;
}
mangled_matrix_info ;

const mangled_matrix_info mangled_files [ ] =
{
//  error             filename              how the matrix is mangled
    LAGRAPH_IO_ERROR, "mangled1.mtx",       // bad header
    LAGRAPH_IO_ERROR, "mangled2.mtx",       // bad header
    LAGRAPH_IO_ERROR, "mangled3.mtx",       // bad type
    GrB_NOT_IMPLEMENTED, "complex.mtx",     // valid complex, but not supported
    LAGRAPH_IO_ERROR, "mangled4.mtx",       // bad format
    LAGRAPH_IO_ERROR, "mangled5.mtx",       // invalid format options
    LAGRAPH_IO_ERROR, "mangled6.mtx",       // invalid format options
    LAGRAPH_IO_ERROR, "mangled7.mtx",       // invalid GraphBLAS type
    LAGRAPH_IO_ERROR, "mangled8.mtx",       // invalid first line
    LAGRAPH_IO_ERROR, "mangled9.mtx",       // symmetric and rectangular
    LAGRAPH_IO_ERROR, "mangled10.mtx",      // truncated
    LAGRAPH_IO_ERROR, "mangled11.mtx",      // entries mangled
    LAGRAPH_IO_ERROR, "mangled12.mtx",      // entries mangled
    GrB_INDEX_OUT_OF_BOUNDS, "mangled13.mtx",// indices out of range
    GrB_INVALID_VALUE, "mangled14.mtx",     // duplicate entries
    LAGRAPH_IO_ERROR, "mangled_bool.mtx",   // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_int8.mtx",   // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_int16.mtx",  // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_int32.mtx",  // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_uint8.mtx",  // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_uint16.mtx", // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_uint32.mtx", // entry value out of range
    LAGRAPH_IO_ERROR, "mangled_skew.mtx",   // unsigned skew invalid
    GrB_NOT_IMPLEMENTED, "mangled15.mtx",   // complex not supported
    GrB_NOT_IMPLEMENTED, "mangled16.mtx",   // complex not supported
    LAGRAPH_IO_ERROR, "mangled_format.mtx", // "array pattern" invalid
    0, "",
} ;

void test_MMRead_failures (void)
{
    setup ( ) ;
    printf ("\nTesting error handling of LAGraph_MMRead when giving it "
        "mangled matrices:\n") ;

    // input arguments are NULL
    TEST_CHECK (LAGraph_MMRead (NULL, NULL, msg) == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;
    TEST_CHECK (LAGraph_MMRead (&A, NULL, msg) == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;

    // matrix files are mangled in some way, or unsupported
    for (int k = 0 ; ; k++)
    {
        const char *aname = mangled_files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        int error = mangled_files [k].error ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        printf ("file: [%s]\n", filename) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        int status = LAGraph_MMRead (&A, f, msg) ;
        printf ("error expected: %d %d [%s]\n", error, status, msg) ;
        TEST_CHECK (status == error) ;
        OK (fclose (f)) ;
        TEST_CHECK (A == NULL) ;
    }

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_jumbled: test reading a jumbled matrix
//-----------------------------------------------------------------------------

void test_jumbled (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    //--------------------------------------------------------------------------
    // load in the data/west0067.mtx file as the matrix A
    //--------------------------------------------------------------------------

    FILE *f = fopen (LG_DATA_DIR "west0067.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
    TEST_CHECK (MATCHNAME (atype_name, "double")) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of west0067.mtx failed") ;

    //--------------------------------------------------------------------------
    // load in the data/west0067_jumbled.mtx file as the matrix B
    //--------------------------------------------------------------------------

    f = fopen (LG_DATA_DIR "west0067_jumbled.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&B, f, msg)) ;
    OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
    TEST_CHECK (MATCHNAME (btype_name, "double")) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of west0067_jumbled.mtx failed") ;

    //--------------------------------------------------------------------------
    // ensure A and B are the same
    //--------------------------------------------------------------------------

    bool ok ;
    OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
    TEST_CHECK (ok) ;
    TEST_MSG ("Test for A and B equal failed: west0067_jumbled.mtx matrix") ;

    //--------------------------------------------------------------------------
    // free workspace and finish the test
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_MMWrite: test LAGraph_MMWrite
//-----------------------------------------------------------------------------

const char* files_for_MMWrite [ ] =
{
    "west0067.mtx",
    "full.mtx",
    "cover.mtx",
    ""
} ;

void test_MMWrite (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file
        //----------------------------------------------------------------------

        const char *aname = files_for_MMWrite [k] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;

        //----------------------------------------------------------------------
        // create a file for comments
        //----------------------------------------------------------------------

        FILE *fcomments = fopen (LG_DATA_DIR "comments.txt", "wb") ;
        TEST_CHECK (fcomments != NULL) ;
        fprintf (fcomments, " comments for %s\n", aname) ;
        fprintf (fcomments, " this file was created by test_MMRead.c\n") ;
        fclose (fcomments) ;
        TEST_MSG ("Failed to create comments.txt") ;

        //----------------------------------------------------------------------
        // write the matrix to the data/comments_*.mtx file
        //----------------------------------------------------------------------

        snprintf (filename, LEN, LG_DATA_DIR "comments_%s", aname) ;
        fcomments = fopen (LG_DATA_DIR "comments.txt", "r") ;
        FILE *foutput = fopen (filename, "wb") ;
        TEST_CHECK (foutput != NULL) ;
        TEST_CHECK (fcomments != NULL) ;
        OK (LAGraph_MMWrite (A, foutput, fcomments, msg)) ;
        fclose (fcomments) ;
        fclose (foutput) ;
        TEST_MSG ("Failed to create %s", filename) ;

        //----------------------------------------------------------------------
        // load in the data/comments_.mtx file as the matrix B
        //----------------------------------------------------------------------

        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;

        OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
        TEST_CHECK (MATCHNAME (atype_name, btype_name)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of %s failed", filename) ;

        //----------------------------------------------------------------------
        // ensure A and B are the same
        //----------------------------------------------------------------------

        bool ok ;
        OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
        TEST_CHECK (ok) ;
        TEST_MSG ("Test for A and B equal failed: %s", filename) ;

        //----------------------------------------------------------------------
        // write a nan
        //----------------------------------------------------------------------

        if (k == 0)
        {
            OK (GrB_Matrix_setElement (A, NAN, 0, 0)) ;
            double a ;
            OK (GrB_Matrix_extractElement (&a, A, 0, 0)) ;
            TEST_CHECK (isnan (a)) ;
            foutput = fopen (filename, "wb") ;
            fcomments = fopen (LG_DATA_DIR "comments.txt", "r") ;
            TEST_CHECK (foutput != NULL) ;
            OK (LAGraph_MMWrite (A, foutput, fcomments, msg)) ;
            fclose (fcomments) ;
            fclose (foutput) ;
            OK (GrB_free (&A)) ;
            f = fopen (filename, "r") ;
            TEST_CHECK (f != NULL) ;
            OK (LAGraph_MMRead (&A, f, msg)) ;
            fclose (f) ;
            a = 0 ;
            OK (GrB_Matrix_extractElement (&a, A, 0, 0)) ;
            TEST_CHECK (isnan (a)) ;
        }

        //----------------------------------------------------------------------
        // free workspace
        //----------------------------------------------------------------------

        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_MMWrite_failures: test error handling of LAGraph_MMWrite
//-----------------------------------------------------------------------------

typedef int mytype ;

void test_MMWrite_failures (void)
{
    setup ( ) ;
    GrB_Type atype = NULL ;
    printf ("\nTesting error handling of LAGraph_MMWrite\n") ;

    // input arguments are NULL
    TEST_CHECK (LAGraph_MMWrite (NULL, NULL, NULL, msg) == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;

    // attempt to print a matrix with a user-defined type, which should fail
    FILE *f = tmpfile ( ) ;
    TEST_CHECK (f != NULL) ;
    OK (GrB_Type_new (&atype, sizeof (mytype))) ;
    OK (GrB_Matrix_new (&A, atype, 4, 4)) ;
    int status = LAGraph_Matrix_Print (A, LAGraph_COMPLETE, stdout, msg) ;
    printf ("msg: [%s]\n", msg) ;
    TEST_CHECK (status == GrB_NOT_IMPLEMENTED) ;
    status = LAGraph_MMWrite (A, f, NULL, msg) ;
    printf ("msg: %d [%s]\n", status, msg) ;
    TEST_CHECK (status == GrB_NOT_IMPLEMENTED) ;
    OK (GrB_free (&atype)) ;
    OK (GrB_free (&A)) ;
    OK (fclose (f)) ;       // close and delete the temporary file

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_MMReadWrite_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_MMReadWrite_brutal (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file
        //----------------------------------------------------------------------

        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        printf ("\n") ;

        //----------------------------------------------------------------------
        // write it to a temporary file
        //----------------------------------------------------------------------

        for (int nbrutal = 0 ; ; nbrutal++)
        {
            /* allow for only nbrutal mallocs before 'failing' */
            printf (".") ;
            LG_brutal = nbrutal ;
            /* try the method with brutal malloc */
            f = tmpfile ( ) ;   // create a new temp file for each trial
            int brutal_result = LAGraph_MMWrite (A, f, NULL, msg) ;
            if (brutal_result >= 0)
            {
                /* the method finally succeeded */
                // leave the file open for the next phase
                printf (" MMWrite ok: %d mallocs\n", nbrutal) ;
                break ;
            }
            OK (fclose (f)) ;   // close and delete the file and try again
            if (nbrutal > 10000) { printf ("Infinite!\n") ; abort ( ) ; }
        }
        LG_brutal = -1 ;  /* turn off brutal mallocs */

        //----------------------------------------------------------------------
        // load it back in again
        //----------------------------------------------------------------------

        for (int nbrutal = 0 ; ; nbrutal++)
        {
            /* allow for only nbrutal mallocs before 'failing' */
            printf (".") ;
            LG_brutal = nbrutal ;
            /* try the method with brutal malloc */
            rewind (f) ;        // rewind the temp file for each trial
            int brutal_result = LAGraph_MMRead (&B, f, msg) ;
            if (brutal_result >= 0)
            {
                /* the method finally succeeded */
                printf (" MMRead ok: %d mallocs\n", nbrutal) ;
                OK (fclose (f)) ;   // finally close and delete the temp file
                break ;
            }
            if (nbrutal > 10000) { printf ("Infinite!\n") ; abort ( ) ; }
        }
        LG_brutal = -1 ;  /* turn off brutal mallocs */

        //----------------------------------------------------------------------
        // ensure A and B are the same
        //----------------------------------------------------------------------

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
        TEST_CHECK (MATCHNAME (atype_name, btype_name)) ;

        bool ok ;
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
        OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
        LG_BRUTAL (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
        TEST_CHECK (ok) ;
        TEST_MSG ("Failed test for equality, file: %s\n", aname) ;

        //----------------------------------------------------------------------
        // free workspace
        //----------------------------------------------------------------------

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
// test_array_pattern
//------------------------------------------------------------------------------

void test_array_pattern ( )
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    OK (LG_brutal_setup (msg)) ;

    //--------------------------------------------------------------------------
    // construct a dense 3-by-3 matrix of all 1's (iso-valued)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT64, 3, 3)) ;
    OK (GrB_Matrix_assign_INT64 (A, NULL, NULL, 1, GrB_ALL, 3, GrB_ALL, 3,
        NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    printf ("\nA matrix:\n") ;
    OK (LAGraph_Matrix_Print (A, LAGraph_COMPLETE, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // write it to a temporary file
    //--------------------------------------------------------------------------

    FILE *f = tmpfile ( ) ; // fopen ("/tmp/mine.mtx", "wb") ;
    OK (LAGraph_MMWrite (A, f, NULL, msg)) ;
    TEST_MSG ("Failed to write matrix to a temp file\n") ;
//  OK (fclose (f)) ;

    //--------------------------------------------------------------------------
    // load it back in again
    //--------------------------------------------------------------------------

    rewind (f) ;
//  f = fopen ("/tmp/mine.mtx", "r") ;
    OK (LAGraph_MMRead (&B, f, msg)) ;
    TEST_MSG ("Failed to load matrix from a temp file\n") ;
    OK (fclose (f)) ;       // close and delete the temporary file

    printf ("\nB matrix:\n") ;
    OK (LAGraph_Matrix_Print (B, LAGraph_COMPLETE, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // ensure A and B are the same
    //--------------------------------------------------------------------------

    OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
    TEST_CHECK (MATCHNAME ("int64_t", btype_name)) ;
    bool ok ;
    OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
    TEST_CHECK (ok) ;
    TEST_MSG ("Failed test for equality, dense 3-by-3\n") ;

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;

    OK (LG_brutal_teardown (msg)) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "MMRead", test_MMRead },
    { "karate", test_karate },
    { "MMRead_failures", test_MMRead_failures },
    { "jumbled", test_jumbled },
    { "MMWrite", test_MMWrite },
    { "MMWrite_failures", test_MMWrite_failures },
    #if LAGRAPH_SUITESPARSE
    { "MMReadWrite_brutal", test_MMReadWrite_brutal },
    #endif
    { "array_pattern", test_array_pattern },
    { NULL, NULL }
} ;
