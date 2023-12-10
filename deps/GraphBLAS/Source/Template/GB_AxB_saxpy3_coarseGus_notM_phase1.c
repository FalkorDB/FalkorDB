//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseGus_notM_phase1: symbolic coarse Gustavson, with !M
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    // Initially, Hf [...] < mark for all of Hf.

    // Hf [i] < mark    : M(i,j)=0, C(i,j) is not yet seen.
    // Hf [i] == mark   : M(i,j)=1, so C(i,j) is ignored.
    // Hf [i] == mark+1 : M(i,j)=0, and C(i,j) has been seen.

    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        GB_GET_B_j ;            // get B(:,j)
        Cp [kk] = 0 ;

        //----------------------------------------------------------------------
        // special case when B(:,j) is empty
        //----------------------------------------------------------------------

        #if ( GB_B_IS_SPARSE || GB_B_IS_HYPER )
        if (bjnz == 0) continue ;
        #endif

        //----------------------------------------------------------------------
        // get M(:,j) and scatter it into the Hf workspace
        //----------------------------------------------------------------------

        GB_GET_M_j ;                                // get M(:,j)
        mark += 2 ;
        const int64_t f0 = mark ;
        const int64_t f1 = mark+1 ;
        GB_SCATTER_M_j (pM_start, pM_end, f0) ;     // scatter M(:,j)

        //----------------------------------------------------------------------
        // count nnz in C(:,j)
        //----------------------------------------------------------------------

        int64_t cjnz = 0 ;
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of A(i,k)
                if (Hf [i] < f0)        // if true, M(i,j) is 0
                { 
                    Hf [i] = f1 ;       // flag C(i,j) as seen
                    cjnz++ ;            // C(i,j) is a new entry
                }
            }
        }
        Cp [kk] = cjnz ;                // count the entries in C(:,j)
    }
}

