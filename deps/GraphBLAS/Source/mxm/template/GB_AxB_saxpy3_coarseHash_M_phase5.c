//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_M_phase5: C<M>=A*B, coarse Hash, phase 5
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase5: coarse hash task, C<M>=A*B
    //--------------------------------------------------------------------------

    // M is sparse and scattered into Hf

    // Initially, Hf [...] < mark for all of Hf.
    // Let h = Hi [hash] and f = Hf [hash].

    // f < mark            : M(i,j)=0, C(i,j) is ignored.
    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        int64_t pC = Cp [kk] ;
        int64_t cjnz = Cp [kk+1] - pC ;
        if (cjnz == 0) continue ;   // nothing to do
        GB_GET_M_j ;                // get M(:,j)
        GB_GET_M_j_RANGE (64) ;     // get 1st & last in M(:,j)
        mark += 2 ;
        int64_t mark1 = mark+1 ;
        GB_HASH_M_j ;               // hash M(:,j)
        GB_GET_B_j ;                // get B(:,j)
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        { 
            GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            #define GB_IKJ                                              \
            {                                                           \
                for (GB_HASH (i))       /* find i in hash */            \
                {                                                       \
                    int64_t f = Hf [hash] ;                             \
                    if (f < mark) break ; /* M(i,j)=0, ignore*/         \
                    if (Hi [hash] == i)                                 \
                    {                                                   \
                        GB_MULT_A_ik_B_kj ;     /* t = aik*bkj */       \
                        if (f == mark)          /* if true, i is new */ \
                        {                                               \
                            /* C(i,j) is new */                         \
                            Hf [hash] = mark1 ;     /* mark seen */     \
                            GB_HX_WRITE (hash, t) ; /* Hx[hash] = t */  \
                            Ci [pC++] = i ;                             \
                        }                                               \
                        else                                            \
                        {                                               \
                            /* C(i,j) has been seen; update */          \
                            GB_HX_UPDATE (hash, t) ;                    \
                        }                                               \
                        break ;                                         \
                    }                                                   \
                }                                                       \
            }
            GB_SCAN_M_j_OR_A_k (A_ok_for_binary_search) ;
            #undef GB_IKJ
        }
        GB_SORT_AND_GATHER_HASHED_C_j (mark1) ;
    }
}

