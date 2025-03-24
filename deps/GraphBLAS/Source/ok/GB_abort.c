//------------------------------------------------------------------------------
// GB_abort.c: hard assertions for all of GraphBLAS, including JIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function is always active; it is not controlled by #ifdef GB_DEBUG.

#include "GB.h"

void GB_abort
(
    const char *file,
    int line
)
{
    GBDUMP ("\nGraphBLAS assertion failed: [ %s ]: line %d\n", file, line) ;
    GB_Global_abort ( ) ;
}

