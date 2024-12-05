//------------------------------------------------------------------------------
// GB_emult_02b: C = A.*B when A is sparse/hyper and B is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as A.  No mask is present.
// A is sparse/hyper, and B is full.

{

    //--------------------------------------------------------------------------
    // Method2(b): C=A.*B where A is sparse/hyper and B is full
    //--------------------------------------------------------------------------

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
            GB_GET_PA (pA, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                GBP_A (Ap, k, vlen), GBP_A (Ap, k+1, vlen)) ;
            for ( ; pA < pA_end ; pA++)
            { 
                // C (i,j) = A (i,j) .* B (i,j)
                int64_t i = Ai [pA] ;
                int64_t pB = pB_start + i ;
                // Ci [pA] = i ; already defined
                #ifndef GB_ISO_EMULT
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, pA, A_iso) ;
                GB_DECLAREB (bij) ;
                GB_GETB (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, pA, aij, bij, i, j) ;
                #endif
            }
        }
    }
}

