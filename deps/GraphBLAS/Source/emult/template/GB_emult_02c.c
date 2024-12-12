//------------------------------------------------------------------------------
// GB_emult_02c: C<#M>=A.*B when A is sparse/hyper; M and B are bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as A.
// M is bitmap/full.  A is sparse/hyper, and B is bitmap/full.

{

    //--------------------------------------------------------------------------
    // Method2(c): C<#M>=A.*B, A is sparse/hyper, M and B are bitmap/full
    //--------------------------------------------------------------------------

    const int8_t  *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (Mask_struct) ? NULL : ((GB_M_TYPE *) M->x) ;
    const size_t msize = M->type->size ;

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBH_A (Ah, k) ;
            int64_t pB_start = j * vlen ;
            GB_GET_PA_AND_PC (pA, pA_end, pC, tid, k, kfirst, klast,
                pstart_Aslice, Cp_kfirst,
                GBP_A (Ap, k, vlen), GBP_A (Ap, k+1, vlen),
                GBP_C (Cp, k, vlen)) ;
            for ( ; pA < pA_end ; pA++)
            { 
                int64_t i = Ai [pA] ;
                int64_t pB = pB_start + i ;
                if (!GBB_B (Bb, pB)) continue ;
                bool mij = GBB_M (Mb, pB) && GB_MCAST (Mx, pB, msize) ;
                mij = mij ^ Mask_comp ;
                if (!mij) continue ;
                // C (i,j) = A (i,j) .* B (i,j)
                Ci [pC] = i ;
                #ifndef GB_ISO_EMULT
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, pA, A_iso) ;     
                GB_DECLAREB (bij) ;
                GB_GETB (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, pC, aij, bij, i, j) ;
                #endif
                pC++ ;
            }
        }
    }
}

