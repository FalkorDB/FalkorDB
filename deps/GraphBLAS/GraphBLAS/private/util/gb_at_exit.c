//------------------------------------------------------------------------------
// gb_at_exit: terminate GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is called by MATLAB when the mexFunction that called GrB_init
// (or GxB_init) is cleared.

void gb_at_exit ( void )
{
    // Finalize GraphBLAS, clearing all JIT kernels and freeing the hash table.
    // MATLAB can only use GraphBLAS if GrB_init / GxB_init is called again.

    // SuiteSparse:GraphBLAS allows GrB_init or GxB_init to be called again, as
    // an extension to the spec.  G[rx]B_init can be called if GxB_initialized
    // returns false, or if GxB_finalized returns true.

    // If GraphBLAS has not ever been initialized or if it has been finalized,
    // the next call to any GrB mexFunction will first call gbmx_usage, which
    // calls GrB_init to re-initialize GraphBLAS.  That method will re-load the
    // hash table with all PreJIT kernels.

    GrB_finalize ( ) ;
}

