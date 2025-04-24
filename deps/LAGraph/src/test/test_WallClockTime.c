//------------------------------------------------------------------------------
// LAGraph/src/test/test_WallClockTime.c:  test LAGraph_WallClockTime
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

double t ;
char msg [LAGRAPH_MSG_LEN] ;

//------------------------------------------------------------------------------
// test_WallClockTime:  test LAGraph_WallClockTime
//------------------------------------------------------------------------------

void test_WallClockTime (void)
{

    OK (LAGraph_Init (msg)) ;

    // start the timer
    double t = LAGraph_WallClockTime ( ) ;

    // do some useless work
    double x = msg [0] ;
    for (int64_t k = 0 ; k < 10000 ; k++)
    {
        for (int64_t i = 0 ; i < 10000 ; i++)
        {
            x = x + 1 ;
            if (x > 100) x = x/2 ;
        }
    }

    // stop the timer
    t = LAGraph_WallClockTime ( ) - t ;

    // print the result so the compiler doesn't remove the loops above
    printf ("\nresult: %g, time: %g sec\n", x, t) ;

    OK (LAGraph_Finalize (msg)) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "WallClockTime", test_WallClockTime },
    // no brutal test needed
    { NULL, NULL }
} ;
