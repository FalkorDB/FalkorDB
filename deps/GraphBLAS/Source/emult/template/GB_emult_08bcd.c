//------------------------------------------------------------------------------
// GB_emult_08bcd: C=A.*B; C, A, and B are all sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------
    // Method8(b,c,d): C = A.*B, no mask
    //--------------------------------------------------------------

    //      ------------------------------------------
    //      C       =           A       .*      B
    //      ------------------------------------------
    //      sparse  .           sparse          sparse  (method: 8)
    //      sparse  sparse      sparse          sparse  (8, M later)

    // both A and B are sparse/hyper
    ASSERT (A_is_sparse || A_is_hyper) ;
    ASSERT (B_is_sparse || B_is_hyper) ;

    if (ajnz > 32 * bjnz)
    {

        //----------------------------------------------------------
        // Method8(b): A(:,j) is much denser than B(:,j)
        //----------------------------------------------------------

        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = GB_IGET (Bi, pB) ;
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
                GB_ISET (Ci, pC, i) ;   // Ci [pC] = i ;
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
        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif

    }
    else if (bjnz > 32 * ajnz)
    {

        //----------------------------------------------------------
        // Method8(c): B(:,j) is much denser than A(:,j)
        //----------------------------------------------------------

        for ( ; pA < pA_end ; pA++)
        {
            int64_t i = GB_IGET (Ai, pA) ;
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
                GB_ISET (Ci, pC, i) ;   // Ci [pC] = i ;
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
        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif

    }
    else
    {

        //----------------------------------------------------------
        // Method8(d): A(:,j) and B(:,j) about the sparsity
        //----------------------------------------------------------

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
                // C (i,j) = A (i,j) .* B (i,j)
                #if ( GB_EMULT_08_PHASE == 1 )
                cjnz++ ;
                #else
                ASSERT (pC < pC_end) ;
                GB_ISET (Ci, pC, iB) ;  // Ci [pC] = iB ;
                #ifndef GB_ISO_EMULT
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, pA, A_iso) ;
                GB_DECLAREB (bij) ;
                GB_GETB (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, pC, aij, bij, iB, j) ;
                #endif
                pC++ ;
                #endif
                pA++ ;
                pB++ ;
            }
        }

        #if ( GB_EMULT_08_PHASE == 2 )
        ASSERT (pC == pC_end) ;
        #endif
    }
}

