//------------------------------------------------------------------------------
// GB_mx_at_exit: terminate GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is called by MATLAB when the mexFunction that called GrB_init
// (or GxB_init) is cleared.

#include "GB_mex.h"

void GB_mx_at_exit ( void )
{
    // Finalize GraphBLAS, clearing all JIT kernels and freeing the hash table.
    // MATLAB can only use GraphBLAS if GrB_init / GxB_init is called again.

    // The call to GB_Global_GrB_init_called_set allows GrB_init or GxB_init to
    // be called again.  This is an extension to the spec that is possible with
    // SuiteSparse:GraphBLAS but not available via a documented function.
    // Instead, an internal method is used.  If this flag is set, the next call
    // to any @GrB mexFunction will first call gb_usage, which calls GxB_init
    // to re-initialize GraphBLAS.  That method will re-load the hash table
    // with all PreJIT kernels.

    // These 2 lines are placed together so a "grep GrB_finalize" reports
    // both of them.

    GrB_finalize ( ) ; GB_Global_GrB_init_called_set (false) ;
}

