//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/simple_demo.c: tests simple_rand
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
    A simple test that illustrates the use of simple_rand.
    This test does not require C11, nor GraphBLAS.  It only tests the
    simple_* Demo functions.  The output of this test should look like the
    following.  The random numbers should be the same.  On a Mac OSX system:

        first 10 random numbers:
            0.513871
            0.175726
            0.308634
            0.534532
            0.947630
            0.171728
            0.702231
            0.226417
            0.494766
            0.124699

    The random numbers are identical on linux, as desired.
*/

#include "simple_rand.h"
#include "simple_rand.c"
#include <stdio.h>
#include <stdlib.h>

#define LEN 10000000

int main (void)
{
    double *x ;
    int i ;

    fprintf (stderr, "simple_demo:\n") ;

    // calloc the space for more accurate timing
    x = (double *) calloc (LEN, sizeof (double)) ;
    if (x == NULL)
    {
        fprintf (stderr, "simple_demo: out of memory\n") ;
        exit (1) ;
    }

    // generate random numbers
    for (i = 0 ; i < LEN ; i++)
    {
        x [i] = simple_rand_x ( ) ;
    }

    // these should be the same on any system and any compiler
    printf ("first 10 random numbers:\n") ;
    for (i = 0 ; i < 10 ; i++)
    {
        printf ("%12.6f\n", x [i]) ;
    }

    // generate random uint64_t numbers
    for (i = 0 ; i < LEN ; i++)
    {
        simple_rand_i ( ) ;
    }

    free (x) ;
}

