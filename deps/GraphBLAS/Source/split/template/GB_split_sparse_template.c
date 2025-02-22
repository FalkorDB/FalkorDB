//------------------------------------------------------------------------------
// GB_split_sparse_template: split a single tile from a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  C has the same sparsity format as A.

{

    //--------------------------------------------------------------------------
    // get A and C, and the slicing of C
    //--------------------------------------------------------------------------

    #ifndef GB_ISO_SPLIT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    GB_Cp_DECLARE (Cp,      ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    const GB_Ap_TYPE *restrict Wp = (const GB_Ap_TYPE *) Wp_workspace ;
    const int64_t *restrict kfirst_Cslice = C_ek_slicing ;
    const int64_t *restrict klast_Cslice  = C_ek_slicing + C_ntasks ;
    const int64_t *restrict pstart_Cslice = C_ek_slicing + C_ntasks * 2 ;
    #endif

    //--------------------------------------------------------------------------
    // copy the tile from A to C
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < C_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Cslice [tid] ;
        int64_t klast  = klast_Cslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // int64_t jA = GBh_A (Ah, k+akstart) ; not needed 
            int64_t p0 = GB_IGET (Cp, k) ;
            GB_GET_PA (pC_start, pC_end, tid, k,
                kfirst, klast, pstart_Cslice, p0, GB_IGET (Cp, k+1)) ;
            int64_t pA_offset = GB_IGET (Wp, k + akstart) ;
            // copy the vector from A to C
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            { 
                // get the index of A(iA,jA)
                int64_t pA = pA_offset + pC - p0 ;
                int64_t iA = GB_IGET (Ai, pA) ;
                // shift the index and copy into C(i,j)
                int64_t iC = iA - aistart ;
                GB_ISET (Ci, pC, iC) ;      // Ci [pC] = iC ;
                GB_COPY (pC, pA) ;
            }
        }
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_SPLIT

