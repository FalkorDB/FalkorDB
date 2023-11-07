//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseGus_noM_phase1: symbolic coarse Gustavson, no mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Initially, Hf [...] < mark for all Hf.
// Hf [i] is set to mark when C(i,j) is found.

{
    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        GB_GET_B_j ;                    // get B(:,j)

        //----------------------------------------------------------------------
        // special cases when B(:,j) has zero or one entries
        //----------------------------------------------------------------------

        #if ( GB_B_IS_SPARSE || GB_B_IS_HYPER )
        if (bjnz == 0)
        { 
            Cp [kk] = 0 ;               // C(:,j) is empty
            continue ;
        }
        #if ( GB_A_IS_SPARSE )
        if (bjnz == 1)
        { 
            GB_GET_B_kj_INDEX ;         // get index k of entry B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            Cp [kk] = aknz ;            // nnz (C (:,j)) = nnz (A (:,k))
            continue ;
        }
        #endif
        #endif

        //----------------------------------------------------------------------
        // count nnz in C(:,j), terminating early if C(:,j) becomes dense
        //----------------------------------------------------------------------

        const int64_t f = (++mark) ;
        int64_t cjnz = 0 ;
        for ( ; pB < pB_end && cjnz < cvlen ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get index k of entry B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of entry A(i,k)
                if (Hf [i] != f)        // if true, i is new
                { 
                    Hf [i] = f ;        // mark C(i,j) as seen
                    cjnz++ ;            // C(i,j) is a new entry
                }
            }
        }
        Cp [kk] = cjnz ;                // save count of entries in C(:,j)
    }
}

