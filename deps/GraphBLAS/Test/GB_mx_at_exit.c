//------------------------------------------------------------------------------
// GB_mx_at_exit: terminate GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is called by MATLAB when the mexFunction that called GrB_init
// (or GxB_init) is cleared.

#include "GB_mex.h"

void GB_mx_at_exit ( void )
{
    // Finalize GraphBLAS, clearing all JIT kernels and freeing the hash table.
    // MATLAB can only use GraphBLAS if GrB_init / GxB_init is called again.
    GB_Global_persistent_set (mexMakeMemoryPersistent) ;
    GrB_finalize ( ) ;
}

