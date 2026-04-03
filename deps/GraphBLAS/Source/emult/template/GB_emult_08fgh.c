//------------------------------------------------------------------------------
// GB_emult_08fgh: C<#M>=A.*B when C is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is bitmap/full.  A and B are both sparse/hyper

{

    //--------------------------------------------------------------------------
    // Method08(f): M is bitmap or full, for either C<M>=A.*B or C<!M>=A.*B
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C       <M>=        A       .*      B
    //      ------------------------------------------
    //      sparse  bitmap      sparse          sparse  (method: 8)
    //      sparse  full        sparse          sparse  (method: 8)

    //      ------------------------------------------
    //      C       <!M>=       A       .*      B
    //      ------------------------------------------
    //      sparse  bitmap      sparse          sparse  (method: 8)
    //      sparse  full        sparse          sparse  (method: 8)

    // both A and B are sparse/hyper
    ASSERT (A_is_sparse || A_is_hyper) ;
    ASSERT (B_is_sparse || B_is_hyper) ;

    int64_t pM_start = j * vlen ;

    if (ajnz > 32 * bjnz)
    {

        //----------------------------------------------------------------------
        // Method8(f): A(:,j) much denser than B(:,j), M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = GB_IGET (Bi, pB) ;
            // get M(i,j)
            int64_t pM = pM_start + i ;
            bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
            if (Mask_comp) mij = !mij ;
            if (mij)
            {
                // find i in A(:,j)
                int64_t pright = pA_end - 1 ;
                bool found ;
                found = GB_binary_search (i, Ai, GB_Ai_IS_32, &pA, &pright) ;
                if (found)
                { 
                    // C (i,j) = A (i,j) .* B (i,j)
                    #if ( GB_EMULT_08_PHASE == 1 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, pA, A_iso) ;
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, aij, bij, i, j) ;
                    #endif
                    pC++ ;
                    #endif
                }
            }
        }

        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif

    }
    else if (bjnz > 32 * ajnz)
    {

        //----------------------------------------------------------------------
        // Method8(g): B(:,j) much denser than A(:,j), M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pA < pA_end ; pA++)
        {
            int64_t i = GB_IGET (Ai, pA) ;
            // get M(i,j)
            int64_t pM = pM_start + i ;
            bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
            if (Mask_comp) mij = !mij ;
            if (mij)
            {
                // find i in B(:,j)
                int64_t pright = pB_end - 1 ;
                bool found ;
                found = GB_binary_search (i, Bi, GB_Bi_IS_32, &pB, &pright) ;
                if (found)
                { 
                    // C (i,j) = A (i,j) .* B (i,j)
                    #if ( GB_EMULT_08_PHASE == 1 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, pA, A_iso) ;
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, aij, bij, i, j) ;
                    #endif
                    pC++ ;
                    #endif
                }
            }
        }

        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // Method8(h): A(:,j) and B(:,j) about same, M bitmap/full
        //----------------------------------------------------------------------

        // linear-time scan of A(:,j) and B(:,j)

        while (pA < pA_end && pB < pB_end)
        {
            int64_t iA = GB_IGET (Ai, pA) ;
            int64_t iB = GB_IGET (Bi, pB) ;
            if (iA < iB)
            { 
                // A(i,j) exists but not B(i,j)
                pA++ ;
            }
            else if (iB < iA)
            { 
                // B(i,j) exists but not A(i,j)
                pB++ ;
            }
            else
            {
                // both A(i,j) and B(i,j) exist
                int64_t i = iA ;
                // get M(i,j)
                int64_t pM = pM_start + i ;
                bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                if (Mask_comp) mij = !mij ;
                if (mij)
                { 
                    // C (i,j) = A (i,j) .* B (i,j)
                    #if ( GB_EMULT_08_PHASE == 1 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, pA, A_iso) ;
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, aij, bij, iB, j) ;
                    #endif
                    pC++ ;
                    #endif
                }
                pA++ ;
                pB++ ;
            }
        }

        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif
    }
}

