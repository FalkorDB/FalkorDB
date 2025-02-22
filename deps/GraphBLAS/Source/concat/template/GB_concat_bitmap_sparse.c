//------------------------------------------------------------------------------
// GB_concat_bitmap_sparse: concatenate a sparse tile into a bitmap matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_IS_BITMAP (C)) ;

    #ifdef GB_JIT_KERNEL
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    #undef  GB_FREE_ALL
    #define GB_FREE_ALL GB_WERK_POP (A_ek_slicing, int64_t) ;
    #endif

    int A_nthreads, A_ntasks ;
    GB_A_NHELD (A_nnz_held) ;
    GB_SLICE_MATRIX_WORK (A, 1, A_nnz_held + A->nvec, A_nnz_held) ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBh_A (Ah, k) ;
            int64_t jC = cvstart + j ;
            int64_t pC_start = cistart + jC * cvlen ;
            GB_GET_PA (pA_start, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;
            GB_PRAGMA_SIMD
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                int64_t i = GB_IGET (Ai, pA) ;
                int64_t pC = pC_start + i ;
                // Cx [pC] = Ax [pA] ;
                GB_COPY (pC, pA, A_iso) ;
                Cb [pC] = 1 ;
            }
        }
    }

    #ifdef GB_JIT_KERNEL
    GB_FREE_ALL ;
    #endif
}

