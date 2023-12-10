//------------------------------------------------------------------------------
// GB_add_full_32:  C=A+B, C and A are full, B is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method32: C and A are full; B is sparse or hypersparse
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        #if GB_IS_EWISEUNION
        { 
            // C (i,j) = A(i,j) + beta
            GB_LOAD_A (aij, Ax, p, A_iso) ;
            GB_EWISEOP (Cx, p, aij, beta_scalar, p % vlen, p / vlen) ;
        }
        #else
        { 
            // C (i,j) = A (i,j)
            GB_COPY_A_to_C (Cx, p, Ax, p, A_iso) ;
        }
        #endif
    }

    const int64_t *kfirst_Bslice = B_ek_slicing ;
    const int64_t *klast_Bslice  = B_ek_slicing + B_ntasks ;
    const int64_t *pstart_Bslice = B_ek_slicing + B_ntasks*2 ;

    #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < B_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of B(:,k) for this task
            int64_t j = GBH_B (Bh, k) ;
            GB_GET_PA (pB_start, pB_end, taskid, k, kfirst, klast,
                pstart_Bslice, GBP_B (Bp, k, vlen), GBP_B (Bp, k+1, vlen)) ;
            int64_t pC_start = j * vlen ;
            // traverse over B(:,j), the kth vector of B
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                int64_t i = Bi [pB] ;
                int64_t p = pC_start + i ;
                GB_LOAD_A (aij, Ax, p , A_iso) ;
                GB_LOAD_B (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, p, aij, bij, i, j) ;
            }
        }
    }
}
