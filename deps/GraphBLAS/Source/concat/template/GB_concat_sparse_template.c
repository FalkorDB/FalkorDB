//------------------------------------------------------------------------------
// GB_concat_sparse_template: concatenate a tile into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The tile A is hypersparse, sparse, or full, not bitmap.  If C is iso, then
// so is A, and the values are not copied here.

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) || GB_IS_FULL (A)) ;

    #ifndef GB_ISO_CONCAT
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    int64_t avlen = A->vlen ;
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;
    #if GB_Cp_IS_32
    const uint32_t *restrict W = W_parameter ;
    #else
    const uint64_t *restrict W = W_parameter ;
    #endif
    #endif

    //--------------------------------------------------------------------------
    // copy the tile A into C
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBh_A (Ah, k) ;
            const int64_t pC_start = GB_IGET (W, j) ;

            //------------------------------------------------------------------
            // find the part of the kth vector A(:,j) for this task
            //------------------------------------------------------------------

            const int64_t p0 = GBp_A (Ap, k, avlen) ;
            GB_GET_PA (pA_start, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                p0, GBp_A (Ap, k+1, avlen)) ;

            //------------------------------------------------------------------
            // append A(:,j) onto C(:,j)
            //------------------------------------------------------------------

            GB_PRAGMA_SIMD
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                int64_t i = GBi_A (Ai, pA, avlen) ;     // i = Ai [pA]
                int64_t pC = pC_start + pA - p0 ;
                int64_t ci = cistart + i ;
                GB_ISET (Ci, pC, ci) ;                  // Ci [pC] = ci ; 
                // Cx [pC] = Ax [pA] ;
                GB_COPY (pC, pA, A_iso) ;
            }
        }
    }
}

#undef GB_C_TYPE
#undef GB_A_TYPE
#undef GB_ISO_CONCAT

