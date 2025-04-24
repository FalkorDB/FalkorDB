//-----------------------------------------------------------------------------
// LAGraph/src/test/test_fopen.c:  test fopen
//-----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//-----------------------------------------------------------------------------

#include "LAGraph_test.h"

//-----------------------------------------------------------------------------
// test fopen, as an example of how to use LG_SOURCE_DIR
//-----------------------------------------------------------------------------

void test_fopen(void)
{
    char buffer [1000] ;
    printf ("\nLAGraph source directory: [%s]\n", LG_SOURCE_DIR) ;
    FILE *f = fopen (LG_SOURCE_DIR "/data/A.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    char *r = fgets (buffer, 512, f) ;
    TEST_CHECK (r != NULL) ;
    printf ("[%s]\n", buffer) ;
    fclose (f) ;
}

void test_fopen_failure (void)
{
    FILE *f = fopen ("garbage", "r") ;
    TEST_CHECK (f == NULL) ;
}

//-----------------------------------------------------------------------------
// run the test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "fopen", test_fopen },
    { "fopen_failure", test_fopen_failure },
    // no brutal test needed
    { NULL, NULL }
} ;
