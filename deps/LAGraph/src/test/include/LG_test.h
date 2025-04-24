//------------------------------------------------------------------------------
// LG_test.h: include file for LAGraph test library
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#ifndef LG_TEST_H
#define LG_TEST_H

#include <GraphBLAS.h>
#include <LAGraph.h>
#undef I

#if ( _MSC_VER && !__INTEL_COMPILER && LG_TEST_DLL )
    #ifdef LG_TEST_LIBRARY
        // compiling LAGraph itself, exporting symbols to user apps
        #define LG_TEST_PUBLIC __declspec ( dllexport )
    #else
        // compiling the user application, importing symbols from LAGraph
        #define LG_TEST_PUBLIC __declspec ( dllimport )
    #endif
#else
    // for other compilers
    #define LG_TEST_PUBLIC
#endif

int LG_check_bfs
(
    // input
    GrB_Vector Level,       // optional; may be NULL
    GrB_Vector Parent,      // optional; may be NULL
    LAGraph_Graph G,
    GrB_Index src,
    char *msg
) ;

int LG_check_tri        // -1 if out of memory, 0 if successful
(
    // output
    uint64_t *ntri,     // # of triangles in A
    // input
    LAGraph_Graph G,    // the structure of G->A must be symmetric
    char *msg
) ;

int LG_check_cc
(
    // input
    GrB_Vector Component,   // Component(i)=k if node is in the kth Component
    LAGraph_Graph G,
    char *msg
) ;

int LG_check_vector
(
    int64_t *x,         // x (0:n-1) = X (0:n-1), of type int64_t
    GrB_Vector X,       // vector of size n
    int64_t n,
    int64_t missing     // value to assign to x(i) if X(i) is not present
) ;

int LG_check_sssp
(
    // input
    GrB_Vector Path_Length,     // Path_Length(i) is the length of the
                                // shortest path from src to node i.
    LAGraph_Graph G,            // all edge weights must be > 0
    GrB_Index src,
    char *msg
) ;

int LG_check_export
(
    // input
    LAGraph_Graph G,        // export G->A in CSR format
    // output
    GrB_Index **Ap_handle,  // size Ap_len on output
    GrB_Index **Aj_handle,  // size Aj_len on output
    void **Ax_handle,       // size Ax_len * typesize on output
    GrB_Index *Ap_len,
    GrB_Index *Aj_len,
    GrB_Index *Ax_len,
    size_t *typesize,       // size of the type of A
    char *msg
) ;

//------------------------------------------------------------------------------
// LG_brutal_*:  brutal memory tests
//------------------------------------------------------------------------------

// Brutal memory tests use a global variable (LG_brutal) to tell them how many
// times malloc may be successfully called.  Once this counter reaches zero,
// LG_brutal_malloc, LG_brutal_calloc, and LG_brutal_realloc all return NULL.

// SuiteSparse:GraphBLAS is required for brutal memory testing.  The
// LG_brutal_malloc/calloc/realloc/free methods are passed to SuiteSparse
// GraphBLAS via GxB_init.  This way, out-of-memory conditions returned by
// GraphBLAS can be checked and handled by LAGraph.

// Use LG_brutal_setup to start LAGraph for brutal memory tests, and
// LG_brutal_teardown to finish.  To test a method with brutal memory tests,
// use LG_BRUTAL (LAGraph_method (...)).  The LAGraph_method(...) will be
// called with LG_brutal set to 0 (no mallocs allowed), 1 (one malloc), 2, ...
// until the method succeeds by returning a result >= 0.  If a method never
// returns a non-negative result, LG_BRUTAL will get stuck in an infinite loop.

// If LG_brutal starts as negative, then the brutal memory tests are not
// performed, and LG_brutal_malloc/calloc/etc never pretend to fail.

// A count (LG_nmalloc) is kept of the number of allocated blocks that have not
// yet been freed.  If this count is not zero after finalizing GraphBLAS
// and LAGraph, an error is reported.  No report is provided as to what blocks
// of memory are still allocated; use valgrind if that occurs.

// For methods that access files, or have other side effects, the LG_BRUTAL
// macro will not work.  It will reach a failure state from which it cannot
// recover.  For example:
//
//  (1) LG_BRUTAL (LAGraph_MMRead (...)) would fail, because it would read the
//      file and then fail because LG_brutal_malloc returns a NULL.  To handle
//      this, each iteration of the brutal loop must rewind the file for the
//      next brutal trial.  See src/test/test_MMRead for details.
//
//  (2) If any GrB_Matrix or GrB_Vector component of G has pending work (G->A
//      for example), and GraphBLAS attempts to finish it, the method will fail
//      if LG_brutal_malloc returns NULL.  In this case, GraphBLAS will return
//      the GrB_Matrix or GrB_Vector as an invalid object, and then LG_BRUTAL
//      will never succeed.  If this occurs, simply use GrB_wait to finalize
//      all components of G first, before trying a brutal test on a method that
//      uses G.  See src/test/test_Graph_Print.c for an example.
//
//  (3) LG_BRUTAL will fail for methods that unpack/pack their input matrix
//      G->A (such as several of the LG_check_* methods, and the current draft
//      of LG_CC_FastSV6).  Those methods will return an empty G->A with no
//      entries, if they fail in the middle.  For these methods, the brutal
//      tests must keep a copy of G->A outside of G, and reconstruct G->A for
//      each trial.
//
//  (4) Some GrB* methods leave their output matrix in an invalid state, with
//      all entries cleared, if an out-of-memory failure occurs (GrB_assign
//      in particular).  See src/test/test_vector for an example.

LG_TEST_PUBLIC int LG_brutal_setup (char *msg) ;
LG_TEST_PUBLIC int LG_brutal_teardown (char *msg) ;

LG_TEST_PUBLIC extern int64_t LG_brutal ;
LG_TEST_PUBLIC extern int64_t LG_nmalloc ;

LG_TEST_PUBLIC
void *LG_brutal_malloc      // return pointer to allocated block of memory
(
    size_t size             // # of bytes to allocate
) ;

LG_TEST_PUBLIC
void *LG_brutal_calloc      // return pointer to allocated block of memory
(
    size_t nitems,          // # of items to allocate
    size_t itemsize         // # of bytes per item
) ;

LG_TEST_PUBLIC
void LG_brutal_free
(
    void *p                 // block to free
) ;

LG_TEST_PUBLIC
void *LG_brutal_realloc     // return pointer to reallocated memory
(
    void *p,                // block to realloc
    size_t size             // new size of the block
) ;

// brutal memory testing of a GraphBLAS or LAGraph method, no burble
#define LG_BRUTAL(Method)                                       \
{                                                               \
    for (int nbrutal = 0 ; ; nbrutal++)                         \
    {                                                           \
        /* allow for only nbrutal mallocs before 'failing' */   \
        LG_brutal = nbrutal ;                                   \
        /* try the method with brutal malloc */                 \
        int brutal_result = Method ;                            \
        if (brutal_result >= 0)                                 \
        {                                                       \
            /* the method finally succeeded */                  \
            break ;                                             \
        }                                                       \
        if (nbrutal > 10000)                                    \
        {                                                       \
            printf ("Line %d Infinite! result: %d\n",           \
                __LINE__, brutal_result) ;                      \
            abort ( ) ;                                         \
        }                                                       \
    }                                                           \
    LG_brutal = -1 ;  /* turn off brutal mallocs */             \
}

// brutal memory testing of a GraphBLAS or LAGraph method, and print results
#define LG_BRUTAL_BURBLE(Method)                                \
{                                                               \
    printf ("brutal test at line %4d: LG_nmalloc: %g\n",        \
        __LINE__, (double) LG_nmalloc) ;                        \
    printf ("method: " LG_XSTR(Method) "\n") ;                  \
    for (int nbrutal = 0 ; ; nbrutal++)                         \
    {                                                           \
        /* allow for only nbrutal mallocs before 'failing' */   \
        LG_brutal = nbrutal ;                                   \
        /* try the method with brutal malloc */                 \
        int brutal_result = Method ;                            \
        if (brutal_result >= 0)                                 \
        {                                                       \
            /* the method finally succeeded */                  \
            printf ("brutal test at line %4d: LG_nmalloc: %g,"  \
                " succeeded with %d mallocs\n", __LINE__,       \
                (double) LG_nmalloc, nbrutal) ;                 \
            break ;                                             \
        }                                                       \
        if (nbrutal > 10000)                                    \
        {                                                       \
            printf ("Line %d Infinite! result: %d %s\n",        \
                __LINE__, brutal_result, msg) ;                 \
            abort ( ) ;                                         \
        }                                                       \
    }                                                           \
    LG_brutal = -1 ;  /* turn off brutal mallocs */             \
}

#endif
