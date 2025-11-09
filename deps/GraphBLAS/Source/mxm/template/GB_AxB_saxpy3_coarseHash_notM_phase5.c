//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_notM_phase5: C<!M>=A*B, coarse Hash, phase 5
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase5: coarse hash task, C<!M>=A*B
    //--------------------------------------------------------------------------

    // M is sparse and scattered into Hf

    // Initially, Hf [...] < mark for all of Hf.
    // Let h = Hi [hash] and f = Hf [hash].

    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        int64_t pC_start = GB_Cp_IGET (kk) ;
        int64_t pC_end = GB_Cp_IGET (kk+1) ;
        int64_t pC = pC_start ;
        int64_t cjnz = pC_end - pC ;
        ASSERT (pC_start >= 0) ;
        ASSERT (pC_start <= pC_end) ;
        ASSERT (pC_end <= cnz) ;

        if (cjnz == 0) continue ;   // nothing to do
        GB_GET_M_j ;                // get M(:,j)
        mark += 2 ;
        uint64_t mark1 = mark+1 ;
        GB_HASH_M_j ;               // hash M(:,j)
        GB_GET_B_j ;                // get B(:,j)
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of A(i,k)
                for (GB_HASH (i))       // find i in hash
                {
                    uint64_t f = Hf [hash] ;
                    if (f < mark)   // if true, i is new
                    { 
                        // C(i,j) is new
                        Hf [hash] = mark1 ;             // mark C(i,j) seen
                        Hi [hash] = i ;
                        GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                        GB_HX_WRITE (hash, t) ;         // Hx [hash] = t
                        GB_ISET (Ci, pC, i) ;           // Ci [pC] = i
                        pC++ ;
                        break ;
                    }
                    if (Hi [hash] == i)
                    {
                        if (f == mark1)
                        { 
                            // C(i,j) has been seen; update it.
                            GB_MULT_A_ik_B_kj ;         //t = A(i,k)*B(k,j)
                            GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
                        }
                        break ;
                    }
                }
            }
        }
        GB_SORT_AND_GATHER_HASHED_C_j (mark1) ;
    }
}

