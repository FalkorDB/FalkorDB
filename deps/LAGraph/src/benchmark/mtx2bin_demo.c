//------------------------------------------------------------------------------
// LAGraph/src/benchmark/mtx2bin_demo.c: convert Matrix Market file to SS:GrB binary file
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

// usage:
// mtx2bin infile.mtx outfile.grb

#include "LAGraph_demo.h"

#define LG_FREE_ALL                 \
{                                   \
    GrB_free (&A) ;                 \
}

int main (int argc, char **argv)
{
    GrB_Info info ;
    GrB_Matrix A = NULL ;
    char msg [LAGRAPH_MSG_LEN] ;

    if (argc < 3)
    {
        printf ("Usage: mxt2bin infile.mtx outfile.grb\n") ;
        exit (1) ;
    }

    printf ("infile:  %s\n", argv [1]) ;
    printf ("outfile: %s\n", argv [2]) ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    //--------------------------------------------------------------------------
    // read matrix from input file
    //--------------------------------------------------------------------------

    double t_read = LAGraph_WallClockTime ( ) ;

    // read in the file in Matrix Market format from the input file
    FILE *f = fopen (argv [1], "r") ;
    if (f == NULL)
    {
        printf ("Matrix file not found: [%s]\n", argv [1]) ;
        exit (1) ;
    }
    LAGRAPH_TRY (LAGraph_MMRead (&A, f, msg)) ;
    fclose (f) ;

    GRB_TRY (GrB_wait (A, GrB_MATERIALIZE)) ;

    t_read = LAGraph_WallClockTime ( ) - t_read ;
    printf ("read time: %g sec\n", t_read) ;

    //--------------------------------------------------------------------------
    // write to output file
    //--------------------------------------------------------------------------

    double t_binwrite = LAGraph_WallClockTime ( ) ;
    f = fopen (argv [2], "w") ;
    if (f == NULL)
    {
        printf ("Unable to open binary output file: [%s]\n", argv [2]) ;
        exit (1) ;
    }
    if (binwrite (&A, f, argv [1]) != 0)
    {
        printf ("Unable to create binary file\n") ;
        exit (1) ;
    }
    t_binwrite = LAGraph_WallClockTime ( ) - t_binwrite ;
    printf ("binary write time: %g sec\n", t_binwrite) ;

    LG_FREE_ALL ;
    return (GrB_SUCCESS) ;
}
