//------------------------------------------------------------------------------
// LAGraph/experimental/benchmark/helloworld_demo.c: a simple demo
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A Davis, Texas A&M University

//------------------------------------------------------------------------------

// This main program is a simple driver for testing and benchmarking the
// LAGraph_HelloWorld "algorithm", in experimental/algorithm.  To use it,
// compile LAGraph while in the build folder with these commands:
//
//      cd LAGraph/build
//      cmake ..
//      make -j8
//
// Then run this demo with an input matrix.  For example:
//
//      ./experimental/benchmark/helloworld_demo ../data/west0067.mtx
//      ./experimental/benchmark/helloworld_demo < ../data/west0067.mtx
//      ./experimental/benchmark/helloworld_demo ../data/karate.mtx
//
// If you create your own algorithm and want to mimic this main program, call
// it write in experimental/benchmark/whatever_demo.c (with "_demo.c" as the
// end of the filename), and the cmake will find it and compile it.

// This main program makes use of supporting utilities in
// src/benchmark/LAGraph_demo.h and src/utility/LG_internal.h.
// See helloworld2_demo.c for a main program that just uses the
// user-callable methods in LAGraph.h and LAGraphX.h.

#include "../../src/benchmark/LAGraph_demo.h"
#include "LAGraphX.h"
#include "LG_internal.h"

// LG_FREE_ALL is required by LG_TRY
#undef  LG_FREE_ALL
#define LG_FREE_ALL                             \
{                                               \
    GrB_free (&Y) ;                             \
    LAGraph_Delete (&G, msg) ;                  \
}

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // startup LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;        // for error messages from LAGraph
    LAGraph_Graph G = NULL ;
    GrB_Matrix Y = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;               // set true for diagnostic outputs
    demo_init (burble) ;

    //--------------------------------------------------------------------------
    // read in the graph: this method is defined in LAGraph_demo.h
    //--------------------------------------------------------------------------

    // readproblem can read in a file in Matrix Market format, or in a binary
    // format created by binwrite (see LAGraph_demo.h, or the main program,
    // mtx2bin_demo).

    double t = LAGraph_WallClockTime ( ) ;
    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    LG_TRY (readproblem (
        &G,         // the graph that is read from stdin or a file
        NULL,       // source nodes (none, if NULL)
        false,      // make the graph undirected, if true
        false,      // remove self-edges, if true
        false,      // return G->A as structural, if true,
        NULL,       // prefered GrB_Type of G->A; null if no preference
        false,      // ensure all entries are positive, if true
        argc, argv)) ;  // input to this main program
    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time to read the graph:      %g sec\n", t) ;

    printf ("\n==========================The input graph matrix G:\n") ;
    LG_TRY (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // try the LAGraph_HelloWorld "algorithm"
    //--------------------------------------------------------------------------

    t = LAGraph_WallClockTime ( ) ;
    LG_TRY (LAGraph_HelloWorld (&Y, G, msg)) ;
    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time for LAGraph_HelloWorld: %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // check the results (make sure Y is a copy of G->A)
    //--------------------------------------------------------------------------

    bool isequal ;
    t = LAGraph_WallClockTime ( ) ;
    LG_TRY (LAGraph_Matrix_IsEqual (&isequal, Y, G->A, msg)) ;
    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time to check results:       %g sec\n", t) ;
    if (isequal)
    {
        printf ("Test passed.\n") ;
    }
    else
    {
        printf ("Test failure!\n") ;
    }

    //--------------------------------------------------------------------------
    // print the results (Y is just a copy of G->A)
    //--------------------------------------------------------------------------

    printf ("\n===============================The result matrix Y:\n") ;
    LG_TRY (LAGraph_Matrix_Print (Y, LAGraph_SHORT, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // free everyting and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    LG_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
