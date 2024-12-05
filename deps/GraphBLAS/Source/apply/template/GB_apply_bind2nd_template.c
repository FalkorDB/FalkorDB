//------------------------------------------------------------------------------
// GB_apply_bind2nd_template: Cx = op (A,y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_output ;
    GB_A_TYPE *Ax = (GB_A_TYPE *) Ax_input ;
    GB_Y_TYPE   y = (*((GB_Y_TYPE *) y_input)) ;
    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < anz ; p++)
    { 
        if (!GBB_A (Ab, p)) continue ;
        GB_DECLAREA (aij) ;
        GB_GETA (aij, Ax, p, false) ;
        GB_EWISEOP (Cx, p, aij, y, 0, 0) ;
    }
}

