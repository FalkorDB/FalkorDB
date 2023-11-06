//------------------------------------------------------------------------------
// GB_add_sparse_noM: C(:,j)=A(:,j)+B(:,j), C sparse/hyper, no mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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

    ASSERT (A_is_sparse || A_is_hyper) ;
    ASSERT (B_is_sparse || B_is_hyper) ;

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
            Ci [pC + p] = i ;
            ASSERT (Ai [pA + p] == i) ;
            ASSERT (Bi [pB + p] == i) ;
            #ifndef GB_ISO_ADD
            GB_LOAD_A (aij, Ax, pA + p, A_iso) ;
            GB_LOAD_B (bij, Bx, pB + p, B_iso) ;
            GB_EWISEOP (Cx, pC + p, aij, bij, i, j) ;
            #endif
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
            Ci [pC + p] = i ;
            ASSERT (Ai [pA + p] == i) ;
            #ifndef GB_ISO_ADD
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = A(i,j) + beta
                GB_LOAD_A (aij, Ax, pA+p, A_iso) ;
                GB_EWISEOP (Cx, pC+p, aij, beta_scalar, i, j) ;
            }
            #else
            { 
                // C (i,j) = A (i,j)
                GB_COPY_A_to_C (Cx, pC+p, Ax, pA+p, A_iso) ;
            }
            #endif
            #endif
        }
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            // C (i,j) = A (i,j) + B (i,j)
            int64_t i = Bi [pB + p] ;
            int64_t ii = i - iA_first ;
            ASSERT (Ai [pA + ii] == i) ;
            #ifndef GB_ISO_ADD
            GB_LOAD_A (aij, Ax, pA + ii, A_iso) ;
            GB_LOAD_B (bij, Bx, pB + p, B_iso) ;
            GB_EWISEOP (Cx, pC + ii, aij, bij, i, j) ;
            #endif
        }
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
            Ci [pC + p] = i ;
            ASSERT (Bi [pB + p] == i) ;
            #ifndef GB_ISO_ADD
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = alpha + B(i,j)
                GB_LOAD_B (bij, Bx, pB+p, B_iso) ;
                // GB_COMPILER_MSC_2019 workaround: the following line of code
                // triggers a bug in the MSC 19.2x compiler in Visual Studio
                // 2019, only for the FIRST_FC32 and SECOND_FC32 operators.  As
                // a workaround, this template is not used for those operators
                // when compiling GraphBLAS with this compiler.
                GB_EWISEOP (Cx, pC+p, alpha_scalar, bij, i, j) ;
            }
            #else
            { 
                // C (i,j) = B (i,j)
                GB_COPY_B_to_C (Cx, pC+p, Bx, pB+p, B_iso) ;
            }
            #endif
            #endif
        }
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            // C (i,j) = A (i,j) + B (i,j)
            int64_t i = Ai [pA + p] ;
            int64_t ii = i - iB_first ;
            ASSERT (Bi [pB + ii] == i) ;
            #ifndef GB_ISO_ADD
            GB_LOAD_A (aij, Ax, pA + p, A_iso) ;
            GB_LOAD_B (bij, Bx, pB + ii, B_iso) ;
            GB_EWISEOP (Cx, pC + ii, aij, bij, i, j) ;
            #endif
        }
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
        memcpy (Ci + pC, Bi + pB, bjnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = alpha + B(i,j)
                GB_LOAD_B (bij, Bx, pB+p, B_iso) ;
                GB_EWISEOP (Cx, pC+p, alpha_scalar, bij, Bi [pB+p], j) ;
            }
            #else
            { 
                // C (i,j) = B (i,j)
                GB_COPY_B_to_C (Cx, pC+p, Bx, pB+p, B_iso) ;
            }
            #endif
        }
        #endif
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
        memcpy (Ci + pC, Ai + pA, ajnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = A(i,j) + beta
                GB_LOAD_A (aij, Ax, pA+p, A_iso) ;
                GB_EWISEOP (Cx, pC+p, aij, beta_scalar, Ai [pA+p], j) ;
            }
            #else
            { 
                // C (i,j) = A (i,j)
                GB_COPY_A_to_C (Cx, pC+p, Ax, pA+p, A_iso) ;
            }
            #endif
        }
        #endif
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
        memcpy (Ci + pC, Ai + pA, ajnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = A(i,j) + beta
                GB_LOAD_A (aij, Ax, pA+p, A_iso) ;
                GB_EWISEOP (Cx, pC+p, aij, beta_scalar, Ai [pA+p], j) ;
            }
            #else
            { 
                // C (i,j) = A (i,j)
                GB_COPY_A_to_C (Cx, pC+p, Ax, pA+p, A_iso) ;
            }
            #endif
        }
        #endif
        pC += ajnz ;
        memcpy (Ci + pC, Bi + pB, bjnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = alpha + B(i,j)
                GB_LOAD_B (bij, Bx, pB+p, B_iso) ;
                GB_EWISEOP (Cx, pC+p, alpha_scalar, bij, Bi [pB+p], j) ;
            }
            #else
            { 
                // C (i,j) = B (i,j)
                GB_COPY_B_to_C (Cx, pC+p, Bx, pB+p, B_iso) ;
            }
            #endif
        }
        #endif
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
        memcpy (Ci + pC, Bi + pB, bjnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = alpha + B(i,j)
                GB_LOAD_B (bij, Bx, pB+p, B_iso) ;
                GB_EWISEOP (Cx, pC+p, alpha_scalar, bij, Bi [pB+p], j) ;
            }
            #else
            { 
                // C (i,j) = B (i,j)
                GB_COPY_B_to_C (Cx, pC+p, Bx, pB+p, B_iso) ;
            }
            #endif
        }
        #endif
        pC += bjnz ;
        memcpy (Ci + pC, Ai + pA, ajnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = A(i,j) + beta
                GB_LOAD_A (aij, Ax, pA+p, A_iso) ;
                GB_EWISEOP (Cx, pC+p, aij, beta_scalar, Ai [pA+p], j) ;
            }
            #else
            { 
                // C (i,j) = A (i,j)
                GB_COPY_A_to_C (Cx, pC+p, Ax, pA+p, A_iso) ;
            }
            #endif
        }
        #endif
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
            int64_t i = Bi [pB] ;
            // find i in A(:,j)
            int64_t pright = pA_end - 1 ;
            bool found ;
            GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
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
            int64_t i = Ai [pA] ;
            // find i in B(:,j)
            int64_t pright = pB_end - 1 ;
            bool found ;
            GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
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
            int64_t iA = Ai [pA] ;
            int64_t iB = Bi [pB] ;
            if (iA < iB)
            { 
                #if ( GB_ADD_PHASE == 2 )
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
                #endif
                pA++ ;
            }
            else if (iA > iB)
            { 
                #if ( GB_ADD_PHASE == 2 )
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
                #endif
                pB++ ;
            }
            else
            { 
                // C (i,j) = A (i,j) + B (i,j)
                #if ( GB_ADD_PHASE == 2 )
                Ci [pC] = iB ;
                #ifndef GB_ISO_ADD
                GB_LOAD_A (aij, Ax, pA, A_iso) ;
                GB_LOAD_B (bij, Bx, pB, B_iso) ;
                GB_EWISEOP (Cx, pC, aij, bij, iB, j) ;
                #endif
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
        memcpy (Ci + pC, Ai + pA, ajnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        for (int64_t p = 0 ; p < ajnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = A(i,j) + beta
                GB_LOAD_A (aij, Ax, pA+p, A_iso) ;
                GB_EWISEOP (Cx, pC+p, aij, beta_scalar, Ai [pA+p], j) ;
            }
            #else
            { 
                // C (i,j) = A (i,j)
                GB_COPY_A_to_C (Cx, pC+p, Ax, pA+p, A_iso) ;
            }
            #endif
        }
        #endif
        memcpy (Ci + pC, Bi + pB, bjnz * sizeof (int64_t)) ;
        #ifndef GB_ISO_ADD
        for (int64_t p = 0 ; p < bjnz ; p++)
        { 
            #if GB_IS_EWISEUNION
            { 
                // C (i,j) = alpha + B(i,j)
                GB_LOAD_B (bij, Bx, pB+p, B_iso) ;
                GB_EWISEOP (Cx, pC+p, alpha_scalar, bij, Bi [pB+p], j) ;
            }
            #else
            { 
                // C (i,j) = B (i,j)
                GB_COPY_B_to_C (Cx, pC+p, Bx, pB+p, B_iso) ;
            }
            #endif
        }
        #endif
        ASSERT (pC + ajnz + bjnz == pC_end) ;
        #endif
    }
}

