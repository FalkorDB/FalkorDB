//------------------------------------------------------------------------------
// GB_emult_04_template: C<M>= A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as M.
// A and B are both bitmap/full.

{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    const int8_t *restrict Ab = A->b ;
    const int8_t *restrict Bb = B->b ;

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;
    #endif

    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    #ifdef GB_JIT_KERNEL
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    ASSERT (!Mask_comp) ;
    #endif

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;

    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) ((Mask_struct) ? NULL : M->x) ;
    const int64_t vlen = M->vlen ;
    const size_t  msize = M->type->size ;

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;

    const int64_t *restrict kfirst_Mslice = M_ek_slicing ;
    const int64_t *restrict klast_Mslice  = M_ek_slicing + M_ntasks ;
    const int64_t *restrict pstart_Mslice = M_ek_slicing + M_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // Method4: C<M>=A.*B where M is sparse/hyper, A and B are bitmap/full
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < M_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Mslice [tid] ;
        int64_t klast  = klast_Mslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBh_M (Mh, k) ;
            int64_t pstart = j * vlen ;
            GB_GET_PA_AND_PC (pM, pM_end, pC, tid, k, kfirst, klast,
                pstart_Mslice, Cp_kfirst,
                GB_IGET (Mp, k), GB_IGET (Mp, k+1), GB_IGET (Cp, k)) ;
            for ( ; pM < pM_end ; pM++)
            {
                int64_t i = GB_IGET (Mi, pM) ;
                if (GB_MCAST (Mx, pM, msize)
                    && GBb_A (Ab, pstart + i)
                    && GBb_B (Bb, pstart + i))
                { 
                    int64_t p = pstart + i ;
                    // C (i,j) = A (i,j) .* B (i,j)
                    GB_ISET (Ci, pC, i) ;   // Ci [pC] = i
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, p, A_iso) ;
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, p, B_iso) ;
                    GB_EWISEOP (Cx, pC, aij, bij, i, j) ;
                    #endif
                    pC++ ;
                }
            }
        }
    }
}

