//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/wathen_demo.c: test wathen
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Construct a matrix using the wathen method.
//
//  wathen_demo nx ny method nthreads
//
//  nx and ny default to 10.
//
//  methods:
//      0: build (the default, the fastest method)
//      1: assign scalars
//      2: assign finite elements, create F with a loop
//      3: assign finite elements, create F all at once
//
//  nthreads: defaults # of threads from OpenMP

#include "graphblas_demos.h"
#include "wathen.c"

// macro used by OK(...) to free workspace if an error occurs
#undef  FREE_ALL
#define FREE_ALL            \
    GrB_Matrix_free (&A) ;  \

int main (int argc, char **argv)
{
    GrB_Matrix A = NULL ;
    GrB_Info info ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
//  OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_BURBLE)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int64_t nx = 10, ny = 10 ;
    int method = 0 ;
    int nthreads ;
    if (argc > 1) nx = strtol (argv [1], NULL, 0) ;
    if (argc > 2) ny = strtol (argv [2], NULL, 0) ;
    if (argc > 3) method = strtol (argv [3], NULL, 0) ;
    if (argc > 4)
    {
        nthreads = strtol (argv [4], NULL, 0) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads, GxB_NTHREADS)) ;
    }
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;
    fprintf (stderr, "Wathen: nx %d ny %d method: %d nthreads: %d ",
        (int) nx, (int) ny, method, nthreads) ;

    //--------------------------------------------------------------------------
    // create a Wathen matrix
    //--------------------------------------------------------------------------

    uint64_t state = 1 ;
    double t = WALLCLOCK ;
    OK (wathen (&A, nx, ny, false, method, NULL, &state)) ;
    t = WALLCLOCK - t ;
    fprintf (stderr, "time: %g\n", t) ;
    GrB_Index n ;
    OK (GrB_Matrix_nrows (&n, A)) ;
    if (n < 1000)
    {
        OK (GxB_Matrix_fprint (A, "A", GxB_SUMMARY, stdout)) ;
    }
    FREE_ALL ;
    OK (GrB_finalize ( )) ;
}

