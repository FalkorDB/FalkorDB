//------------------------------------------------------------------------------
// GB_apply_unop_template.c: C = op(A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_out ;
    GB_A_TYPE *Ax = (GB_A_TYPE *) Ax_in ;
    int64_t p ;
    if (Ab == NULL)
    { 
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        {
            // Cx [p] = unop (Ax [p])
            GB_APPLY_OP (p, p) ;
        }
    }
    else
    { 
        // bitmap case, no transpose; A->b already memcpy'd into C->b
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        {
            if (!Ab [p]) continue ;
            // Cx [p] = unop (Ax [p])
            GB_APPLY_OP (p, p) ;
        }
    }
}

