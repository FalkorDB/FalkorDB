//------------------------------------------------------------------------------
// GB_apply_unop_ip_template: C = op (A), depending only on i
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A can be jumbled.  If A is jumbled, so is C.

{

    //--------------------------------------------------------------------------
    // Cx = op (A)
    //--------------------------------------------------------------------------

    int64_t p ;
    if (GB_A_IS_BITMAP)
    {
        // A is bitmap
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        { 
            if (!Ab [p]) continue ;
            // Cx [p] = op (A (i,j))
            GB_APPLY_OP (p, p) ;
        }
    }
    else
    {
        // A is sparse, hypersparse, or full
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        { 
            // Cx [p] = op (A (i,j))
            GB_APPLY_OP (p, p) ;
        }
    }
}

#undef GB_APPLY_OP

