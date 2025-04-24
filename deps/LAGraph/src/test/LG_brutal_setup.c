//------------------------------------------------------------------------------
// LG_brutal_setup.c: setup an LAGraph test with brutal memory testing
// -----------------------------------------------------------------------------

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

#include "LG_internal.h"
#include "LG_test.h"

int LG_brutal_setup (char *msg)
{
    LG_brutal = -1 ;        // disable brutal testing for now
    LG_nmalloc = 0 ;        // assuming nothing is malloc'd
    int result = LAGr_Init (GrB_NONBLOCKING,
        LG_brutal_malloc, LG_brutal_calloc,
        LG_brutal_realloc, LG_brutal_free, msg) ;
    if (result != 0) return (result) ;
    #if LAGRAPH_SUITESPARSE
    // disable the SuiteSparse:GraphBLAS memory pool
    int64_t free_pool_limit [64] ;
    memset (free_pool_limit, 0, 64 * sizeof (int64_t)) ;
    result = GxB_set (GxB_MEMORY_POOL, free_pool_limit) ;
    #endif
    return (result) ;
}
