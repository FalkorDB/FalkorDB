//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/import_demo.c: test import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Read a graph from a file and test import/export. Usage:
//
//  import_demo < infile

#include "graphblas_demos.h"
#include "simple_rand.h"
#include "simple_rand.c"
#include "usercomplex.h"
#include "usercomplex.c"
#include "wathen.c"
#include "get_matrix.c"
#include "random_matrix.c"
#include "import_test.c"
#include "read_matrix.c"

// macro used by OK(...) to free workspace if an error occurs
#undef  FREE_ALL
#define FREE_ALL            \
    GrB_Matrix_free (&A) ;  \

int main (int argc, char **argv)
{
    GrB_Matrix A = NULL ;
    GrB_Info info ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "import_demo: nthreads: %d\n", nthreads) ;

    //--------------------------------------------------------------------------
    // get a matrix
    //--------------------------------------------------------------------------

    // usage:  ./main  < file
    //         ./main 0 dump < file
    //         ./main 1 dump < file
    //
    // default is 0-based, for the matrices in the Matrix/ folder

    bool one_based = false ;
    bool dump = false ;
    if (argc > 1) one_based = strtol (argv [1], NULL, 0) ;
    if (argc > 2) dump      = strtol (argv [2], NULL, 0) ;

    OK (read_matrix (&A, stdin, false, false, one_based, false, false)) ;

    for (int hyper = 0 ; hyper <= 1 ; hyper++)
    {
        for (int csc = 0 ; csc <= 1 ; csc++)
        {
            double h = hyper ? GxB_ALWAYS_HYPER : GxB_NEVER_HYPER ;
            GxB_Format_Value f = csc ? GxB_BY_COL : GxB_BY_ROW ;

            printf ("\n######### input A: hyper %d csc %d\n", hyper, csc) ;

            for (int format = 0 ; format <= 3 ; format++)
            {
                OK (GxB_Matrix_Option_set (A, GxB_HYPER_SWITCH, h)) ;
                OK (GxB_Matrix_Option_set (A, GxB_FORMAT, f)) ;
                OK (import_test (&A, format, dump)) ;
            }
        }
    }

    FREE_ALL ;

    OK (GrB_finalize ( )) ;
    printf ("import_demo: all tests passed\n") ;
}

