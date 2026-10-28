//------------------------------------------------------------------------------
// GB_add_sparse_noM: C(:,j)=A(:,j)+B(:,j), C sparse/hyper, no mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // M is not present, or !M is sparse but not applied here
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C       =           A       +       B
    //      ------------------------------------------
    //      sparse  .           sparse          sparse

    //      ------------------------------------------
    //      C     <!M> =        A       +       B
    //      ------------------------------------------
    //      sparse  sparse      sparse          sparse  (mask later)

    // If all four matrices are sparse or hypersparse, and Mask_comp is true,
    // the mask M is passed in to this method as NULL.  C=A+B is computed with
    // no mask, and !M is applied later.

    // A and B are both sparse or hypersparse, not bitmap or full, but
    // individual vectors of A and B might have all entries present (adense
    // and/or bdense).
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;

    #if ( GB_ADD_PHASE == 1 )

    if (A_and_B_are_disjoint)
    { 

        // only used by GB_wait, which computes A+T where T is the matrix of
        // pending tuples for A.  The pattern of pending tuples is always
        // disjoint with the pattern of A.

        cjnz = ajnz + bjnz ;

    }
    else

    #endif

    if (adense && bdense)
    {

        //----------------------------------------------------------------------
        // Method01: A(:,j) and B(:,j) dense: thus C(:,j) dense
        //----------------------------------------------------------------------

        ASSERT (ajnz == bjnz) ;
        ASSERT (iA_first == iB_first) ;
        ASSERT (iA_last  == iB_last ) ;
        #if ( GB_ADD_PHASE == 1 )
        cjnz = ajnz ;
        #else
        ASSERT (cjnz == ajnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            // C (i,j) = A (i,j) + B (i,j)
            int64_t i = p + iA_first ;
            GB_ISET (Ci, pC + p, i) ;       // Ci [pC + p] = i ;
            ASSERT (GB_IGET (Ai, pA + p) == i) ;
            ASSERT (GB_IGET (Bi, pB + p) == i) ;
            GB_ADD_AIJ_PLUS_BIJ (Cx, pC+p,
                Ax, pA+p, A_iso, Bx, pB+p, B_iso, i, j) ;
        }
        #endif

    }
    else if (adense)
    {

        //----------------------------------------------------------------------
        // Method02: A(:,j) dense, B(:,j) sparse: C(:,j) dense
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = ajnz ;
        #else
        ASSERT (cjnz == ajnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            int64_t i = p + iA_first ;
            GB_ISET (Ci, pC + p, i) ;       // Ci [pC + p] = i ;
            ASSERT (GB_IGET (Ai, pA + p) == i) ;
            // C (i,j) = A (i,j), or A(i,j) + beta for eWiseUnion
            GB_ADD_AIJ_PLUS_BETA (Cx, pC+p,
                Ax, pA+p, A_iso, beta_scalar, i, j) ;
        }
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            // C (i,j) = A (i,j) + B (i,j)
            int64_t i = GB_IGET (Bi, pB + p) ;
            int64_t ii = i - iA_first ;
            ASSERT (GB_IGET (Ai, pA + ii) == i) ;
            GB_ADD_AIJ_PLUS_BIJ (Cx, pC+ii,
                Ax, pA+ii, A_iso, Bx, pB+p, B_iso, i, j) ;
        }
        #endif
        #endif

    }
    else if (bdense)
    {

        //----------------------------------------------------------------------
        // Method03: A(:,j) sparse, B(:,j) dense: C(:,j) dense
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = bjnz ;
        #else
        ASSERT (cjnz == bjnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            int64_t i = p + iB_first ;
            GB_ISET (Ci, pC + p, i) ;       // Ci [pC + p] = i ;
            ASSERT (GB_IGET (Bi, pB + p) == i) ;
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC+p,
                alpha_scalar, Bx, pB+p, B_iso, i, j) ;
        }
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            // C (i,j) = A (i,j) + B (i,j)
            int64_t i = GB_IGET (Ai, pA + p) ;
            int64_t ii = i - iB_first ;
            ASSERT (GB_IGET (Bi, pB + ii) == i) ;
            GB_ADD_AIJ_PLUS_BIJ (Cx, pC+ii,
                Ax, pA+p, A_iso, Bx, pB+ii, B_iso, i, j) ;
        }
        #endif
        #endif

    }
    else if (ajnz == 0)
    {

        //----------------------------------------------------------------------
        // Method04: A(:,j) is empty
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = bjnz ;
        #else
        ASSERT (cjnz == bjnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            int64_t i = GB_IGET (Bi, pB+p) ;    // i = Bi [pB+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC+p,
                alpha_scalar, Bx, pB+p, B_iso, i, j) ;
        }
        #endif

    }
    else if (bjnz == 0)
    {

        //----------------------------------------------------------------------
        // Method05: B(:,j) is empty
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = ajnz ;
        #else
        ASSERT (cjnz == ajnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            int64_t i = GB_IGET (Ai, pA+p) ;    // i = Ai [pA+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            // C (i,j) = A (i,j), or A(i,j) + beta for eWiseUnion
            GB_ADD_AIJ_PLUS_BETA (Cx, pC+p,
                Ax, pA+p, A_iso, beta_scalar, i, j) ;
        }
        #endif

    }
    else if (iA_last < iB_first)
    {

        //----------------------------------------------------------------------
        // Method06: last A(:,j) comes before 1st B(:,j)
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = ajnz + bjnz ;
        #else
        ASSERT (cjnz == ajnz + bjnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            int64_t i = GB_IGET (Ai, pA+p) ;    // i = Ai [pA+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            // C (i,j) = A (i,j), or A(i,j) + beta for eWiseUnion
            GB_ADD_AIJ_PLUS_BETA (Cx, pC+p,
                Ax, pA+p, A_iso, beta_scalar, i, j) ;
        }
        pC += ajnz ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            int64_t i = GB_IGET (Bi, pB+p) ;    // i = Bi [pB+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC+p,
                alpha_scalar, Bx, pB+p, B_iso, i, j) ;
        }
        #endif

    }
    else if (iB_last < iA_first)
    {

        //----------------------------------------------------------------------
        // Method07: last B(:,j) comes before 1st A(:,j)
        //----------------------------------------------------------------------

        #if ( GB_ADD_PHASE == 1 )
        cjnz = ajnz + bjnz ;
        #else
        ASSERT (cjnz == ajnz + bjnz) ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            int64_t i = GB_IGET (Bi, pB+p) ;    // i = Bi [pB+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC+p,
                alpha_scalar, Bx, pB+p, B_iso, i, j) ;
        }
        pC += bjnz ;
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            int64_t i = GB_IGET (Ai, pA+p) ;    // i = Ai [pA+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            // C (i,j) = A (i,j), or A(i,j) + beta for eWiseUnion
            GB_ADD_AIJ_PLUS_BETA (Cx, pC+p,
                Ax, pA+p, A_iso, beta_scalar, i, j) ;
        }
        #endif

    }

    #if ( GB_ADD_PHASE == 1 )
    else if (ajnz > 32 * bjnz)
    {

        //----------------------------------------------------------------------
        // Method08: A(:,j) is much denser than B(:,j)
        //----------------------------------------------------------------------

        // cjnz = ajnz + bjnz - nnz in the intersection

        cjnz = ajnz + bjnz ;
        for ( ; pB < pB_end ; pB++)
        { 
            int64_t i = GB_IGET (Bi, pB) ;
            // find i in A(:,j)
            int64_t pright = pA_end - 1 ;
            bool found ;
            found = GB_binary_search (i, Ai, GB_Ai_IS_32, &pA, &pright) ;
            if (found) cjnz-- ;
        }

    }
    else if (bjnz > 32 * ajnz)
    {

        //----------------------------------------------------------------------
        // Method09: B(:,j) is much denser than A(:,j)
        //----------------------------------------------------------------------

        // cjnz = ajnz + bjnz - nnz in the intersection

        cjnz = ajnz + bjnz ;
        for ( ; pA < pA_end ; pA++)
        { 
            int64_t i = GB_IGET (Ai, pA) ;
            // find i in B(:,j)
            int64_t pright = pB_end - 1 ;
            bool found ;
            found = GB_binary_search (i, Bi, GB_Bi_IS_32, &pB, &pright) ;
            if (found) cjnz-- ;
        }

    }
    #endif

    else
    {

        //----------------------------------------------------------------------
        // Method10: A(:,j) and B(:,j) about the same sparsity
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t iA = GB_IGET (Ai, pA) ;
            int64_t iB = GB_IGET (Bi, pB) ;
            if (iA < iB)
            { 
                #if ( GB_ADD_PHASE == 2 )
                GB_ISET (Ci, pC, iA) ;      // Ci [pC] = iA ;
                // C (iA,j) = A(iA,j), or A(iA,j) + beta for ewiseUnion
                GB_ADD_AIJ_PLUS_BETA (Cx, pC, Ax, pA, A_iso, beta_scalar,
                    iA, j) ;
                #endif
                pA++ ;
            }
            else if (iA > iB)
            { 
                #if ( GB_ADD_PHASE == 2 )
                GB_ISET (Ci, pC, iB) ;      // Ci [pC] = iB ;
                // C (iB,j) = B (iB,j), or = alpha + B(iB,j) for eWiseUnion:
                GB_ADD_ALPHA_PLUS_BIJ (Cx, pC, alpha_scalar, Bx, pB, B_iso,
                    iB, j) ;
                #endif
                pB++ ;
            }
            else
            { 
                // C (i,j) = A (i,j) + B (i,j)
                #if ( GB_ADD_PHASE == 2 )
                GB_ISET (Ci, pC, iB) ;      // Ci [pC] = iB ;
                GB_ADD_AIJ_PLUS_BIJ (Cx, pC, Ax, pA, A_iso, Bx, pB, B_iso,
                    iB, j) ;
                #endif
                pA++ ;
                pB++ ;
            }
            #if ( GB_ADD_PHASE == 2 )
            pC++ ;
            #else
            cjnz++ ;
            #endif
        }

        //----------------------------------------------------------------------
        // A (:,j) or B (:,j) have entries left; not both
        //----------------------------------------------------------------------

        ajnz = (pA_end - pA) ;
        bjnz = (pB_end - pB) ;
        ASSERT (ajnz == 0 || bjnz == 0) ;
        #if ( GB_ADD_PHASE == 1 )
        cjnz += ajnz + bjnz ;
        #else
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            int64_t i = GB_IGET (Ai, pA+p) ;    // i = Ai [pA+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            // C (i,j) = A (i,j), or A(i,j) + beta for eWiseUnion
            GB_ADD_AIJ_PLUS_BETA (Cx, pC+p,
                Ax, pA+p, A_iso, beta_scalar, i, j) ;
        }
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            int64_t i = GB_IGET (Bi, pB+p) ;    // i = Bi [pB+p]
            GB_ISET (Ci, pC+p, i) ;             // Ci [pC+p] = i
            // C (i,j) = B (i,j), or = alpha + B(i,j) for eWiseUnion:
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC+p,
                alpha_scalar, Bx, pB+p, B_iso, i, j) ;
        }
        ASSERT (pC + ajnz + bjnz == pC_end) ;
        #endif
    }
}

