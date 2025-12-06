//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/graphblas_demos.h: include file for all demo programs
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GRAPHBLAS_DEMOS_H
#define GRAPHBLAS_DEMOS_H

//------------------------------------------------------------------------------
// manage compiler warnings
//------------------------------------------------------------------------------

#if defined __INTEL_COMPILER
#pragma warning (disable: 58 167 144 177 181 186 188 589 593 869 981 1418 1419 1572 1599 2259 2282 2557 2547 3280 )
#elif defined __GNUC__

// disable warnings for gcc 5.x and higher:
#if (__GNUC__ > 4)
// disable warnings
// #pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wformat-truncation="
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
// enable these warnings as errors
#pragma GCC diagnostic error "-Wmisleading-indentation"
#endif

#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if !defined ( __cplusplus )
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#else
#pragma GCC diagnostic ignored "-Wwrite-strings"
#endif
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wtype-limits"

// enable these warnings as errors
#pragma GCC diagnostic error "-Wswitch-default"
#endif

#ifndef GB_MICROSOFT
#if ( defined (_MSC_VER) && !defined(__INTEL_COMPILER) )
#define GB_MICROSOFT 1
#else
#define GB_MICROSOFT 0
#endif
#endif

// do not use any historical methods in GraphBLAS.h
#define NHISTORICAL

// do not use any _Generic macros
#define GxB_STDC_VERSION 199001L

#include "GraphBLAS.h"
#include "simple_rand.h"
#undef I
#ifdef _OPENMP
    #include "omp.h"
    #define WALLCLOCK omp_get_wtime ()
#else
    #define WALLCLOCK 0
#endif

#undef MIN
#undef MAX
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

GrB_Info read_matrix        // read a double-precision matrix
(
    GrB_Matrix *A,          // handle of matrix to create
    FILE *f,                // file to read the tuples from
    bool make_symmetric,    // if true, return A as symmetric
    bool no_self_edges,     // if true, then remove self edges from A
    bool one_based,         // if true, input matrix is 1-based
    bool boolean,           // if true, input is GrB_BOOL, otherwise GrB_FP64
    bool printstuff         // if true, print status to stdout
) ;

GrB_Info random_matrix      // create a random double-precision matrix
(
    GrB_Matrix *A_output,   // handle of matrix to create
    bool make_symmetric,    // if true, return A as symmetric
    bool no_self_edges,     // if true, then do not create self edges
    int64_t nrows,          // number of rows
    int64_t ncols,          // number of columns
    int64_t ntuples,        // number of entries (x2 if made symmetric)
    int method,             // method to use: 0:setElement, 1:build
    bool A_complex,         // if true, create a Complex matrix
    uint64_t *state         // random number state (revised on output)
) ;

GrB_Info get_matrix         // get a matrix from stdin, or create random one
(
    GrB_Matrix *A_output,   // matrix to create
    int argc,               // command-line arguments
    char **argv,
    bool no_self_edges,     // if true, ensure the matrix has no self-edges
    bool boolean,           // if true, file is read as GrB_BOOL, else GrB_FP64
    bool spones,            // if true, return all entries equal to 1
    uint64_t *state         // random state, revised on output
) ;

GrB_Info wathen             // construct a random Wathen matrix
(
    GrB_Matrix *A_output,   // output matrix
    int64_t nx,             // grid dimension nx
    int64_t ny,             // grid dimension ny
    bool scale,             // if true, scale the rows
    int method,             // 0 to 3
    double *rho_given,      // nx-by-ny dense matrix, if NULL use random rho
    uint64_t *state         // random state, revised on output
) ;

GrB_Info isequal_type       // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op         // should be GrB_EQ_<type>, for the type of A and B
) ;

GrB_Info isequal            // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp userop     // for A and B with user-defined types.  ignored
                            // if A and B are of built-in types
) ;

//------------------------------------------------------------------------------
// CHECK: expr must be true; if not, return an error condition
//------------------------------------------------------------------------------

// the #include'ing file must define the FREE_ALL macro

#define CHECK(expr,info)                                                \
{                                                                       \
    if (! (expr))                                                       \
    {                                                                   \
        /* free the result and all workspace, and return NULL */        \
        FREE_ALL ;                                                      \
        printf ("Failure: %d, line %d file %s\n", info,                 \
            __LINE__, __FILE__) ;                                       \
        return (info) ;                                                 \
    }                                                                   \
}

//------------------------------------------------------------------------------
// OK  call a GraphBLAS method and check the result
//------------------------------------------------------------------------------

// OK(method) is a macro that calls a GraphBLAS method and checks the status;
// if a failure occurs, it handles the error via the CHECK macro above, and
// returns the error status to the caller.

#define OK(method)                                                      \
{                                                                       \
    info = method ;                                                     \
    if (!(info == GrB_SUCCESS || info == GrB_NO_VALUE))                 \
    {                                                                   \
        printf ("GraphBLAS error: %d\n", info) ;                        \
        fprintf (stderr, "GraphBLAS error: %d\n", info) ;               \
        CHECK (false, info) ;                                           \
    }                                                                   \
}

#endif
