//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_notM_phase1: symbolic coarse hash method, with !M
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    // Initially, Hf [...] < mark for all of Hf.
    // Let h = Hi [hash] and f = Hf [hash].

    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

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

        GB_GET_M_j ;            // get M(:,j)
        mark += 2 ;
        int64_t mark1 = mark+1 ;
        GB_HASH_M_j ;           // hash M(:,j)

        //----------------------------------------------------------------------
        // count nnz in C(:,j)
        //----------------------------------------------------------------------

        int64_t cjnz = 0 ;
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of A(i,k)
                for (GB_HASH (i))       // find i in hash
                {
                    if (Hf [hash] < mark)   // if true, i is new
                    { 
                        Hf [hash] = mark1 ; // mark C(i,j) seen
                        Hi [hash] = i ;
                        cjnz++ ;        // C(i,j) is a new entry
                        break ;
                    }
                    if (Hi [hash] == i) break ;
                }
            }
        }
        Cp [kk] = cjnz ;                // count the entries in C(:,j)
    }
}

