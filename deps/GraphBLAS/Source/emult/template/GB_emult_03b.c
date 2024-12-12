//------------------------------------------------------------------------------
// GB_emult_03b: C = A.*B when A is full and B is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as B.  No mask is present.
// A is full, and B is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // Method3(b): C=A.*B where A is full and B is sparse/hyper
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < B_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Bslice [tid] ;
        int64_t klast  = klast_Bslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBH_B (Bh, k) ;
            int64_t pA_start = j * vlen ;
            GB_GET_PA (pB, pB_end, tid, k, kfirst, klast, pstart_Bslice,
                GBP_B (Bp, k, vlen), GBP_B (Bp, k+1, vlen)) ;
            for ( ; pB < pB_end ; pB++)
            { 
                // C (i,j) = A (i,j) .* B (i,j)
                int64_t i = Bi [pB] ;
                int64_t pA = pA_start + i ;
                // Ci [pB] = i ; already defined
                #ifndef GB_ISO_EMULT
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, pA, A_iso) ;
                GB_DECLAREB (bij) ;
                GB_GETB (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, pB, aij, bij, i, j) ;
                #endif
            }
        }
    }
}

