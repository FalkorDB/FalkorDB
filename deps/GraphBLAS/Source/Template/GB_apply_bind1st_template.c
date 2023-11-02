//------------------------------------------------------------------------------
// GB_apply_bind1st_template: Cx = op (x,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_output ;
    GB_X_TYPE   x = (*((GB_X_TYPE *) x_input)) ;
    GB_B_TYPE *Bx = (GB_B_TYPE *) Bx_input ;
    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < bnz ; p++)
    { 
        if (!GBB_B (Bb, p)) continue ;
        GB_DECLAREB (bij) ;
        GB_GETB (bij, Bx, p, false) ;
        GB_EWISEOP (Cx, p, x, bij, 0, 0) ;
    }
}

