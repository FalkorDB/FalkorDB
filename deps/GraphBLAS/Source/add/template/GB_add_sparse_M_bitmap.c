//------------------------------------------------------------------------------
// GB_add_sparse_M_bitmap: C(:,j)<M>=A(:,j)+B(:,j), C sparse/hyper, M bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is bitmap.  It may be complemented or not, and valued or structural.
// A and B are both sparse or hyper.

{ 

    //--------------------------------------------------------------
    // M is bitmap or full, for either C<M>=A+B or C<!M>=A+B
    //--------------------------------------------------------------

    //      ------------------------------------------
    //      C      <M> =        A       +       B
    //      ------------------------------------------
    //      sparse  bitmap      sparse          sparse
    //      sparse  full        sparse          sparse

    //      ------------------------------------------
    //      C      <!M> =       A       +       B
    //      ------------------------------------------
    //      sparse  bitmap      sparse          sparse
    //      sparse  full        sparse          sparse

    // This method is very efficient for any mask, and should always be used if
    // M is bitmap or full, even if the mask must also be applied later in
    // GB_mask or GB_accum_mask.  Exploiting the mask here adds no extra search
    // time, and it reduces the size of C on output.

    // GB_GET_MIJ: get M(i,j) where M is bitmap or full
    #undef  GB_GET_MIJ
    #define GB_GET_MIJ(i)                                     \
        int64_t pM = pM_start + i ;                           \
        bool mij = GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ; \
        if (Mask_comp) mij = !mij ;

    // A and B are sparse or hypersparse, not bitmap or full, but individual
    // vectors of A and B might have all entries present (adense and/or
    // bdense).
    ASSERT (A_is_sparse || A_is_hyper) ;
    ASSERT (B_is_sparse || B_is_hyper) ;

    int64_t pM_start = j * vlen ;

    if (adense && bdense)
    {

        //----------------------------------------------------------------------
        // Method15: A(:,j) and B(:,j) dense, M bitmap/full
        //----------------------------------------------------------------------

        ASSERT (ajnz == bjnz) ;
        ASSERT (iA_first == iB_first) ;
        ASSERT (iA_last  == iB_last ) ;
        for (int64_t p = 0 ; p < ajnz ; p++)
        {
            int64_t i = p + iA_first ;
            ASSERT (Ai [pA + p] == i) ;
            ASSERT (Bi [pB + p] == i) ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                GB_LOAD_A (aij, Ax, pA + p, A_iso) ;
                GB_LOAD_B (bij, Bx, pB + p, B_iso) ;
                GB_EWISEOP (Cx, pC, aij, bij, i, j) ;
                #endif
                pC++ ;
                #endif
            }
        }

    }
    else if (ajnz == 0)
    {

        //----------------------------------------------------------------------
        // Method16: A(:,j) is empty, M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = Bi [pB] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                // C (i,j) = B (i,j), or alpha + B(i,j)
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, alpha_scalar, bij, i, j) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

    }
    else if (bjnz == 0)
    {

        //----------------------------------------------------------------------
        // Method17: B(:,j) is empty, M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pA < pA_end ; pA++)
        {
            int64_t i = Ai [pA] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = A(i,j) + beta
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_EWISEOP (Cx, pC, aij, beta_scalar, i, j) ;
                }
                #else
                { 
                    // C (i,j) = A (i,j)
                    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

    }
    else if (iA_last < iB_first)
    {

        //----------------------------------------------------------------------
        // Method18:last A(:,j) before 1st B(:,j), M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pA < pA_end ; pA++)
        {
            int64_t i = Ai [pA] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = A(i,j) + beta
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_EWISEOP (Cx, pC, aij, beta_scalar, i, j) ;
                }
                #else
                { 
                    // C (i,j) = A (i,j)
                    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = Bi [pB] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, alpha_scalar, bij, i, j) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

    }
    else if (iB_last < iA_first)
    {

        //----------------------------------------------------------------------
        // Method19:last B(:,j) before 1st A(:,j), M bitmap/full
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)
        {
            int64_t i = Bi [pB] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                // C (i,j) = B (i,j), or alpha + B(i,j)
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, alpha_scalar, bij, i, j) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

        for ( ; pA < pA_end ; pA++)
        {
            int64_t i = Ai [pA] ;
            GB_GET_MIJ (i) ;
            if (mij)
            { 
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = A(i,j) + beta
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_EWISEOP (Cx, pC, aij, beta_scalar, i, j) ;
                }
                #else
                { 
                    // C (i,j) = A (i,j)
                    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // Method20: merge A(:,j) and B(:,j), M bitmap/full
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t iA = Ai [pA] ;
            int64_t iB = Bi [pB] ;
            if (iA < iB)
            {
                GB_GET_MIJ (iA) ;
                if (mij)
                { 
                    #if ( GB_ADD_PHASE == 1 )
                    cjnz++ ;
                    #else
                    Ci [pC] = iA ;
                    #ifndef GB_ISO_ADD
                    #if GB_IS_EWISEUNION
                    { 
                        // C (iA,j) = A(iA,j) + beta
                        GB_LOAD_A (aij, Ax, pA, A_iso) ;
                        GB_EWISEOP (Cx, pC, aij, beta_scalar, iA, j);
                    }
                    #else
                    { 
                        // C (iA,j) = A (iA,j)
                        GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;
                    }
                    #endif
                    #endif
                    pC++ ;
                    #endif
                }
                pA++ ;
            }
            else if (iA > iB)
            {
                GB_GET_MIJ (iB) ;
                if (mij)
                { 
                    // C (iB,j) = B (iB,j), or alpha + B(iB,j)
                    #if ( GB_ADD_PHASE == 1 )
                    cjnz++ ;
                    #else
                    Ci [pC] = iB ;
                    #ifndef GB_ISO_ADD
                    #if GB_IS_EWISEUNION
                    { 
                        // C (iB,j) = alpha + B(iB,j)
                        GB_LOAD_B (bij, Bx, pB, B_iso) ;
                        GB_EWISEOP (Cx, pC, alpha_scalar, bij, iB, j) ;
                    }
                    #else
                    { 
                        // C (iB,j) = B (iB,j)
                        GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;
                    }
                    #endif
                    #endif
                    pC++ ;
                    #endif
                }
                pB++ ;
            }
            else
            {
                GB_GET_MIJ (iB) ;
                if (mij)
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    #if ( GB_ADD_PHASE == 1 )
                    cjnz++ ;
                    #else
                    Ci [pC] = iB ;
                    #ifndef GB_ISO_ADD
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, aij, bij, iB, j) ;
                    #endif
                    pC++ ;
                    #endif
                }
                pA++ ;
                pB++ ;
            }
        }

        //----------------------------------------------------------------------
        // A (:,j) or B (:,j) have entries left; not both
        //----------------------------------------------------------------------

        for ( ; pA < pA_end ; pA++)
        {
            int64_t iA = Ai [pA] ;
            GB_GET_MIJ (iA) ;
            if (mij)
            { 
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = iA ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (iA,j) = A(iA,j) + beta
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_EWISEOP (Cx, pC, aij, beta_scalar, iA, j) ;
                }
                #else
                { 
                    // C (iA,j) = A (iA,j)
                    GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

        for ( ; pB < pB_end ; pB++)
        {
            int64_t iB = Bi [pB] ;
            GB_GET_MIJ (iB) ;
            if (mij)
            { 
                // C (iB,j) = B (iB,j), or alpha + B(iB,j)
                #if ( GB_ADD_PHASE == 1 )
                cjnz++ ;
                #else
                Ci [pC] = iB ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (iB,j) = alpha + B(iB,j)
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_EWISEOP (Cx, pC, alpha_scalar, bij, iB, j) ;
                }
                #else
                { 
                    // C (iB,j) = B (iB,j)
                    GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }
    }
}

