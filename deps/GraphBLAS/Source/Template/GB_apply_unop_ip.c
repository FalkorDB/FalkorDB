//------------------------------------------------------------------------------
// GB_apply_unop_ip: C = op (A), depending only on i
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A can be jumbled.  If A is jumbled, so is C.

{

    //--------------------------------------------------------------------------
    // Cx = op (A)
    //--------------------------------------------------------------------------

    int64_t p ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    { 
        if (!GBB_A (Ab, p)) continue ;
        // Cx [p] = op (A (i,j))
        GB_APPLY_OP (p) ;
    }
}

#undef GB_APPLY_OP

