//------------------------------------------------------------------------------
// GB_emult_08e: C<M>=A.*B when C and M are sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and M are both sparse/hyper.  M is not complemented.
// A and B can have any format, except at least one is sparse/hyper.

{

    //--------------------------------------------------------------
    // Method8(e): C and M are sparse or hypersparse
    //--------------------------------------------------------------

    //      ------------------------------------------
    //      C       <M>=        A       .*      B
    //      ------------------------------------------
    //      sparse  sparse      sparse          sparse  (method: 8)
    //      sparse  sparse      sparse          bitmap  (9 or 2)
    //      sparse  sparse      sparse          full    (9 or 2)
    //      sparse  sparse      bitmap          sparse  (10 or 3)
    //      sparse  sparse      full            sparse  (10 or 3)

    // Methods 9 and 10 are not yet implemented; using Method 8
    // (GB_emult_08_phase[012]) instead.

    // ether A or B are sparse/hyper
    ASSERT (A_is_sparse || A_is_hyper || B_is_sparse || B_is_hyper);

    //------------------------------------------------------------------
    // get M(:,j)
    //------------------------------------------------------------------

    int64_t pM = -1 ;
    int64_t pM_end = -1 ;

    if (fine_task)
    { 
        // A fine task operates on Mi,Mx [pM...pM_end-1], which is
        // a subset of the vector M(:,j)
        pM     = TaskList [taskid].pM ;
        pM_end = TaskList [taskid].pM_end ;
    }
    else
    {
        int64_t kM = -1 ;
        if (Ch == Mh)
        { 
            // Ch is the same as Mh (a shallow copy), or both NULL
            kM = k ;
        }
        else
        { 
            kM = (C_to_M == NULL) ? j : C_to_M [k] ;
        }
        if (kM >= 0)
        { 
            pM     = GBP_M (Mp, kM, vlen) ;
            pM_end = GBP_M (Mp, kM+1, vlen) ;
        }
    }

    //--------------------------------------------------------------------------
    // iterate across M(:,j)
    //--------------------------------------------------------------------------

    for ( ; pM < pM_end ; pM++)
    {

        //----------------------------------------------------------------------
        // get M(i,j) for A(i,j) .* B (i,j)
        //----------------------------------------------------------------------

        int64_t i = GBI_M (Mi, pM, vlen) ;
        bool mij = GB_MCAST (Mx, pM, msize) ;
        if (!mij) continue ;

        //----------------------------------------------------------------------
        // get A(i,j)
        //----------------------------------------------------------------------

        bool afound ;
        if (adense)
        { 
            // A(:,j) is dense, bitmap, or full; use quick lookup
            pA = pA_start + i - iA_first ;
            afound = GBB_A (Ab, pA) ;
        }
        else
        { 
            // A(:,j) is sparse; use binary search for A(i,j)
            int64_t apright = pA_end - 1 ;
            GB_BINARY_SEARCH (i, Ai, pA, apright, afound) ;
        }
        if (!afound) continue ;
        ASSERT (GBI_A (Ai, pA, vlen) == i) ;

        //----------------------------------------------------------------------
        // get B(i,j)
        //----------------------------------------------------------------------

        bool bfound ;
        if (bdense)
        { 
            // B(:,j) is dense; use direct lookup for B(i,j)
            pB = pB_start + i - iB_first ;
            bfound = GBB_B (Bb, pB) ;
        }
        else
        { 
            // B(:,j) is sparse; use binary search for B(i,j)
            int64_t bpright = pB_end - 1 ;
            GB_BINARY_SEARCH (i, Bi, pB, bpright, bfound) ;
        }
        if (!bfound) continue ;
        ASSERT (GBI_B (Bi, pB, vlen) == i) ;

        //----------------------------------------------------------------------
        // C(i,j) = A(i,j) .* B(i,j)
        //----------------------------------------------------------------------

        // C (i,j) = A (i,j) .* B (i,j)
        #if ( GB_EMULT_08_PHASE == 1 )
        cjnz++ ;
        #else
        Ci [pC] = i ;
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

    #if ( GB_EMULT_08_PHASE == 2 )
    ASSERT (pC == pC_end) ;
    #endif
}

