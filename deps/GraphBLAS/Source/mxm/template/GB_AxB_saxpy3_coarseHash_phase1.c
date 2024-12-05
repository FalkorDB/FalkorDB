//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_phase1: symbolic coarse Hash, optional dense mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase1: coarse hash task, C=A*B, or C<#M>=A*B if M is dense
    //--------------------------------------------------------------------------

    // Initially, Hf [...] < mark for all of Hf.
    // Let f = Hf [hash] and h = Hi [hash]

    // f < mark          : unoccupied.
    // h == i, f == mark : occupied with C(i,j)

    // The mask M can be optionally checked, if it is bitmap, or as-if-full and
    // checked in place.  This method is not used if M is present and sparse.

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
        // get M(:,j), or handle the case when B(:,j) has one entry
        //----------------------------------------------------------------------

        #ifdef GB_CHECK_MASK_ij

            // The mask M is bitmap or as-if-full.  Get pointers Mjb and Mjx
            // into the M(:,j) vector.
            GB_GET_M_j ;
            const M_TYPE *restrict Mjx = Mask_struct ? NULL :
                ((M_TYPE *) Mx) + (M_SIZE * pM_start) ;
            const int8_t *restrict Mjb = M_is_bitmap ? (Mb+pM_start) : NULL ;

        #else

            // M is not present
            #if ( GB_A_IS_SPARSE || GB_A_IS_HYPER )
            if (bjnz == 1)
            { 
                GB_GET_B_kj_INDEX ;     // get index k of B(k,j)
                GB_GET_A_k ;            // get A(:,k)
                Cp [kk] = aknz ;
                continue ;
            }
            #endif

        #endif

        mark++ ;

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
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j) is protected by
                // the mask
                GB_CHECK_MASK_ij ;
                #endif
                int64_t hash ;
                bool marked = false ;
                bool done = false ;
                for (hash = GB_HASHF (i, hash_bits) ; ;
                    GB_REHASH (hash, i, hash_bits))
                { 
                    // if the hash entry is marked then it is occuppied with
                    // some row index in the current C(:,j).
                    marked = (Hf [hash] == mark) ;
                    // if found, then the hash entry holds the row index i.
                    bool found = marked && (Hi [hash] == i) ;
                    // if the hash entry is unmarked, then it is empty, and i
                    // is not in the hash table.  In this case, C(i,j) is a new
                    // entry.  The search terminates if either i is found, or
                    // if an empty (unmarked) slot is found.
                    if (found || !marked) break ;
                }
                if (!marked)
                { 
                    // empty slot found, insert C(i,j)
                    Hf [hash] = mark ;
                    Hi [hash] = i ;
                    cjnz++ ;            // C(i,j) is a new entry
                }
            }
        }
        Cp [kk] = cjnz ;                // count the entries in C(:,j)
    }
}

