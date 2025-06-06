//------------------------------------------------------------------------------
// GB_add_full_34:  C=A+B; C and B are full, A is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method34: C and B are full; A is hypersparse or sparse
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        #if GB_IS_EWISEUNION
        { 
            // C (i,j) = alpha + B(i,j)
            GB_LOAD_B (bij, Bx, p, B_iso) ;
            GB_EWISEOP (Cx, p, alpha_scalar, bij, p % vlen, p / vlen) ;
        }
        #else
        { 
            // C (i,j) = B (i,j)
            GB_COPY_B_to_C (Cx, p, Bx, p, B_iso) ;
        }
        #endif
    }

    const int64_t *kfirst_Aslice = A_ek_slicing ;
    const int64_t *klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *pstart_Aslice = A_ek_slicing + A_ntasks*2 ;

    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < A_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Aslice [taskid] ;
        int64_t klast  = klast_Aslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of A(:,k) for this task
            int64_t j = GBh_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, taskid, k, kfirst, klast,
                pstart_Aslice, GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;
            int64_t pC_start = j * vlen ;
            // traverse over A(:,j), the kth vector of A
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                int64_t i = GB_IGET (Ai, pA) ;
                int64_t p = pC_start + i ;
                GB_LOAD_A (aij, Ax, pA, A_iso) ;
                GB_LOAD_B (bij, Bx, p , B_iso) ;
                GB_EWISEOP (Cx, p, aij, bij, i, j) ;
            }
        }
    }
}

