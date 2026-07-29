//------------------------------------------------------------------------------
// GB_mx_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Initialize GraphBLAS to use malloc/free for arena 0.  The GB_ARENA_TEST (2)
// is initialized to use mxMalloc/mxCalloc/mxRealloc/mxFree and then set as
// the default arena.

#include "GB_mex.h"
#include "GB_mex_errors.h"

GrB_Info GB_mx_init ( void )
{
    GrB_Info info ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
    OK (GxB_arena_init (GB_ARENA_TEST, mxMalloc, mxCalloc, mxRealloc, mxFree)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GB_ARENA_TEST, GxB_ARENA_DATA)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GB_ARENA_TEST, GxB_ARENA_HEADER)) ;
    return (GrB_SUCCESS) ;
}

