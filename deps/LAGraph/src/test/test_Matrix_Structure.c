//------------------------------------------------------------------------------
// LAGraph/src/test/test_Matrix_Structure.c:  test LAGraph_Matrix_Structure
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

char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B = NULL, C = NULL ;
GrB_Vector w = NULL, u = NULL, z = NULL ;
#define LEN 512
char filename [LEN+1] ;
char btype_name [LAGRAPH_MAX_NAME_LEN] ;

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
// test_Matrix_Structure:  test LAGraph_Matrix_Structure
//------------------------------------------------------------------------------

const char *files [ ] =
{
    "cover",
    "lp_afiro",
    "matrix_fp32",
    ""
} ;

void test_Matrix_Structure (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {
        // load the valued matrix as A
        const char *aname = files [k] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s.mtx", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;

        // load the structure as B
        snprintf (filename, LEN, LG_DATA_DIR "%s_structure.mtx", aname) ;
        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;
        OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
        TEST_CHECK (MATCHNAME (btype_name, "bool")) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of structure matrix failed") ;

        // C = structure (A)
        OK (LAGraph_Matrix_Structure (&C, A, msg)) ;

        // ensure B and C are the same
        bool ok ;
        OK (LAGraph_Matrix_IsEqual (&ok, C, B, msg)) ;
        TEST_CHECK (ok) ;
        TEST_MSG ("Test for C and B equal failed") ;

        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
        OK (GrB_free (&C)) ;
    }
    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_Matrix_Structure_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Matrix_Structure_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {
        // load the valued matrix as A
        const char *aname = files [k] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s.mtx", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;

        // load the structure as B
        snprintf (filename, LEN, LG_DATA_DIR "%s_structure.mtx", aname) ;
        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&B, f, msg)) ;
        OK (LAGraph_Matrix_TypeName (btype_name, B, msg)) ;
        TEST_CHECK (MATCHNAME (btype_name, "bool")) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of structure matrix failed") ;

        // C = structure (A)
        LG_BRUTAL (LAGraph_Matrix_Structure (&C, A, msg)) ;

        // ensure B and C are the same
        bool ok ;
        OK (LAGraph_Matrix_IsEqual (&ok, C, B, msg)) ;
        TEST_CHECK (ok) ;
        TEST_MSG ("Test for C and B equal failed") ;

        OK (GrB_free (&A)) ;
        OK (GrB_free (&B)) ;
        OK (GrB_free (&C)) ;
    }
    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// test_Matrix_Structure_failures: test LAGraph_Matrix_Structure error handling
//------------------------------------------------------------------------------

void test_Matrix_Structure_failures (void)
{
    setup ( ) ;

    C = NULL ;
    int result = LAGraph_Matrix_Structure (NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("\nmsg: [%s]\n", msg) ;
    result = LAGraph_Matrix_Structure (&C, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: [%s]\n", msg) ;
    TEST_CHECK (C == NULL) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "Matrix_Structure", test_Matrix_Structure },
    { "Matrix_Structure_failures", test_Matrix_Structure_failures },
    #if LAGRAPH_SUITESPARSE
    { "Matrix_Structure_brutal", test_Matrix_Structure_brutal },
    #endif
    { NULL, NULL }
} ;
