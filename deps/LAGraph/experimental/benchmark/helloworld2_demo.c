//------------------------------------------------------------------------------
// LAGraph/experimental/benchmark/helloworld2_demo.c: a simple demo
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
//      ./experimental/benchmark/helloworld2_demo < ../data/west0067.mtx
//      ./experimental/benchmark/helloworld2_demo < ../data/karate.mtx
//
// If you create your own algorithm and want to mimic this main program, call
// it write in experimental/benchmark/whatever_demo.c (with "_demo.c" as the
// end of the filename), and the cmake will find it and compile it.

// This main program only uses the user-callable methods in LAGraph.h and
// LAGraphX.h.  See helloworld_demo.c for another example that relies on
// internal methods defined in the src/benchmark and src/utility.

#include "LAGraphX.h"

// LAGRAPH_CATCH is required by LAGRAPH_TRY.  If an error occurs, this macro
// catches it and takes corrective action, then terminates this program.
#define LAGRAPH_CATCH(info)                     \
{                                               \
    GrB_free (&Y) ;                             \
    GrB_free (&A) ;                             \
    LAGraph_Delete (&G, msg) ;                  \
    return (info) ;                             \
}

// GRB_CATCH is required by GRB_TRY (although GRB_TRY isn't used here)
#define GRB_CATCH(info) LAGRAPH_CATCH(info)

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // startup LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;        // for error messages from LAGraph
    LAGraph_Graph G = NULL ;
    GrB_Matrix Y = NULL, A = NULL ;

    // start GraphBLAS and LAGraph
    LAGRAPH_TRY (LAGraph_Init (msg)) ;

    //--------------------------------------------------------------------------
    // read in the graph via a Matrix Market file from stdin
    //--------------------------------------------------------------------------

    double t = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LAGraph_MMRead (&A, stdin, msg)) ;
    LAGRAPH_TRY (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time to read the graph:      %g sec\n", t) ;

    printf ("\n==========================The input graph matrix G:\n") ;
    LAGRAPH_TRY (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // try the LAGraph_HelloWorld "algorithm"
    //--------------------------------------------------------------------------

    t = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LAGraph_HelloWorld (&Y, G, msg)) ;
    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time for LAGraph_HelloWorld: %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // check the results (make sure Y is a copy of G->A)
    //--------------------------------------------------------------------------

    bool isequal ;
    t = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LAGraph_Matrix_IsEqual (&isequal, Y, G->A, msg)) ;
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
    LAGRAPH_TRY (LAGraph_Matrix_Print (Y, LAGraph_SHORT, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // free everything and finish
    //--------------------------------------------------------------------------

    GrB_free (&Y) ;
    LAGraph_Delete (&G, msg) ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
