//------------------------------------------------------------------------------
// GB_subassign_08n_template: C(I,J)<M> += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 08n: C(I,J)<M> += A ; no S

// M:           present
// Mask_struct: true or false
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// C not bitmap; C can be full since no zombies are inserted in that case.
// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.
// M, A: not bitmap; Method 08s is used instead if M or A are bitmap.

//------------------------------------------------------------------------------
// GB_PHASE1_ACTION
//------------------------------------------------------------------------------

// action to take for phase 1 when A(i,j) exists and M(i,j)=1
#define GB_PHASE1_ACTION                                                    \
{                                                                           \
    if (cjdense)                                                            \
    {                                                                       \
        /* direct lookup of C(iC,jC) */                                     \
        GB_iC_DENSE_LOOKUP ;                                                \
        /* ----[C A 1] or [X A 1]------------------------------- */         \
        /* [C A 1]: action: ( =C+A ): apply accum                */         \
        /* [X A 1]: action: ( undelete ): zombie lives           */         \
        GB_withaccum_C_A_1_matrix ;                                         \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* binary search for C(iC,jC) in C(:,jC) */                         \
        GB_iC_BINARY_SEARCH (may_see_zombies_phase1) ;                      \
        if (cij_found)                                                      \
        {                                                                   \
            /* ----[C A 1] or [X A 1]--------------------------- */         \
            /* [C A 1]: action: ( =C+A ): apply accum            */         \
            /* [X A 1]: action: ( undelete ): zombie lives       */         \
            GB_withaccum_C_A_1_matrix ;                                     \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* ----[. A 1]-------------------------------------- */         \
            /* [. A 1]: action: ( insert )                       */         \
            task_pending++ ;                                                \
        }                                                                   \
    }                                                                       \
}

//------------------------------------------------------------------------------
// GB_PHASE2_ACTION
//------------------------------------------------------------------------------

// action to take for phase 2 when A(i,j) exists and M(i,j)=1
#define GB_PHASE2_ACTION                                                    \
{                                                                           \
    ASSERT (!cjdense) ;                                                     \
    {                                                                       \
        /* binary search for C(iC,jC) in C(:,jC) */                         \
        GB_iC_BINARY_SEARCH (may_see_zombies_phase2) ;                      \
        if (!cij_found)                                                     \
        {                                                                   \
            /* ----[. A 1]-------------------------------------- */         \
            /* [. A 1]: action: ( insert )                       */         \
            GB_PENDING_INSERT_aij ;                                         \
        }                                                                   \
    }                                                                       \
}

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    const bool may_see_zombies_phase1 = (C->nzombies > 0) ;
    GB_GET_C_HYPER_HASH ;
    GB_GET_MASK ;
    GB_GET_ACCUM_MATRIX ;
    bool A_is_full = GB_IS_FULL (A) ;

    //--------------------------------------------------------------------------
    // Method 08n: C(I,J)<M> += A ; no S
    //--------------------------------------------------------------------------

    // Time: Close to optimal. Omega (sum_j (min (nnz (A(:,j)), nnz (M(:,j)))),
    // since only the intersection of A.*M needs to be considered.  If either
    // M(:,j) or A(:,j) are very sparse compared to the other, then the shorter
    // is traversed with a linear-time scan and a binary search is used for the
    // other.  If the number of nonzeros is comparable, a linear-time scan is
    // used for both.  Once two entries M(i,j)=1 and A(i,j) are found with the
    // same index i, the entry A(i,j) is accumulated or inserted into C.

    // The algorithm is very much like the eWise multiplication of A.*M, so the
    // parallel scheduling relies on GB_emult_08_phase0 and GB_ewise_slice.

    //--------------------------------------------------------------------------
    // Parallel: slice the eWiseMult of Z=A.*M (Method 08n only)
    //--------------------------------------------------------------------------

    // Method 08n only.  If C is sparse, it is sliced for a fine task, so that
    // it can do a binary search via GB_iC_BINARY_SEARCH.  But if C(:,jC) is
    // dense, C(:,jC) is not sliced, so the fine task must do a direct lookup
    // via GB_iC_DENSE_LOOKUP.  Otherwise a race condition will occur.
    // The Z matrix is not constructed, except for its hyperlist (Zh_shallow)
    // and mapping to A and M.

    // No matrix (C, M, or A) can be bitmap.  C, M, A can be sparse/hyper/full,
    // in any combination.

    int64_t Znvec ;
    GB_MDECL (Zh_shallow, const, u) ;
    bool Zj_is_32 ;
    GB_OK (GB_subassign_08n_slice (
        &TaskList, &TaskList_size, &ntasks, &nthreads,
        &Znvec, &Zh_shallow, &Z_to_A, &Z_to_A_size, &Z_to_M, &Z_to_M_size,
        &Zj_is_32, C,
        I, GB_I_IS_32, nI, GB_I_KIND, Icolon,
        J, GB_J_IS_32, nJ, GB_J_KIND, Jcolon,
        A, M, Werk)) ;
    GB_IPTR (Zh_shallow, Zj_is_32) ;
    GB_ALLOCATE_NPENDING_WERK ;

    //--------------------------------------------------------------------------
    // phase 1: undelete zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR_PHASE1 ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t j = GBh (Zh_shallow, k) ;

            int64_t pA = -1, pA_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on a slice of A(:,k)
                pA     = TaskList [taskid].pA ;
                pA_end = TaskList [taskid].pA_end ;
            }
            else
            { 
                // vectors are never sliced for a coarse task
                int64_t kA = (Zh_shallow == Ah) ? k :
                    ((Z_to_A == NULL) ? j : Z_to_A [k]) ;
                if (kA >= 0)
                { 
                    pA     = GBp_A (Ap, kA, Avlen) ;
                    pA_end = GBp_A (Ap, kA+1, Avlen) ;
                }
            }

            int64_t pM = -1, pM_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on a slice of M(:,k)
                pM     = TaskList [taskid].pB ;
                pM_end = TaskList [taskid].pB_end ;
            }
            else
            { 
                // vectors are never sliced for a coarse task
                int64_t kM = (Zh_shallow == Mh) ? k :
                    ((Z_to_M == NULL) ? j : Z_to_M [k]) ;
                if (kM >= 0)
                { 
                    pM     = GBp_M (Mp, kM, Mvlen) ;
                    pM_end = GBp_M (Mp, kM+1, Mvlen) ;
                }
            }

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t mjnz = pM_end - pM ;
            if (ajnz == 0 || mjnz == 0) continue ;
            int64_t iA_first = GBi_A (Ai, pA, Avlen) ;
            int64_t iA_last  = GBi_A (Ai, pA_end-1, Avlen) ;
            int64_t iM_first = GBi_M (Mi, pM, Mvlen) ;
            int64_t iM_last  = GBi_M (Mi, pM_end-1, Mvlen) ;
            if (iA_last < iM_first || iM_last < iA_first) continue ;
            int64_t pM_start = pM ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;


            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += A(:,j) ; no S
            //------------------------------------------------------------------

            if (GB_A_IS_FULL)
            {

                //--------------------------------------------------------------
                // A is a full matrix
                //--------------------------------------------------------------

                int64_t pA_start = j * Avlen ;
                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;
                        // get iA in A(:,j)
                        pA = pA_start + iA ;
                        GB_PHASE1_ACTION ;
                    }
                }

            }
            else if (ajnz > 32 * mjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than M(:,j)
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;
                        // find iA in A(:,j)
                        int64_t pright = pA_end - 1 ;
                        bool found = GB_binary_search (iA, Ai, GB_Ai_IS_32,
                            &pA, &pright) ;
                        if (found) GB_PHASE1_ACTION ;
                    }
                }

            }
            else if (mjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // M(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // FUTURE::: exploit dense mask
                bool mjdense = false ;

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (mij) GB_PHASE1_ACTION ;
                }

            }
            else
            {

                //----------------------------------------------------------
                // A(:,j) and M(:,j) have about the same # of entries
                //----------------------------------------------------------

                // linear-time scan of A(:,j) and M(:,j)

                while (pA < pA_end && pM < pM_end)
                {
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    int64_t iM = GBi_M (Mi, pM, Mvlen) ;
                    if (iA < iM)
                    { 
                        // A(i,j) exists but not M(i,j)
                        pA++ ;  // go to the next entry in A(:,j)
                    }
                    else if (iM < iA)
                    { 
                        // M(i,j) exists but not A(i,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                    else
                    { 
                        // both A(i,j) and M(i,j) exist
                        if (GB_MCAST (Mx, pM, msize)) GB_PHASE1_ACTION ;
                        pA++ ;  // go to the next entry in A(:,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                }
            }
        }

        GB_PHASE1_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    // All zombies might have just been brought back to life, so recheck the
    // may_see_zombies condition.

    GB_PENDING_CUMSUM ;
    const bool may_see_zombies_phase2 = (C->nzombies > 0) ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(&&:pending_sorted)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR_PHASE2 ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t j = GBh (Zh_shallow, k) ;

            int64_t pA = -1, pA_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on a slice of A(:,k)
                pA     = TaskList [taskid].pA ;
                pA_end = TaskList [taskid].pA_end ;
            }
            else
            { 
                // vectors are never sliced for a coarse task
                int64_t kA = (Zh_shallow == Ah) ? k :
                    ((Z_to_A == NULL) ? j : Z_to_A [k]) ;
                if (kA >= 0)
                { 
                    pA     = GBp_A (Ap, kA, Avlen) ;
                    pA_end = GBp_A (Ap, kA+1, Avlen) ;
                }
            }

            int64_t pM = -1, pM_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on a slice of M(:,k)
                pM     = TaskList [taskid].pB ;
                pM_end = TaskList [taskid].pB_end ;
            }
            else
            { 
                // vectors are never sliced for a coarse task
                int64_t kM = (Zh_shallow == Mh) ? k :
                    ((Z_to_M == NULL) ? j : Z_to_M [k]) ;
                if (kM >= 0)
                { 
                    pM     = GBp_M (Mp, kM, Mvlen) ;
                    pM_end = GBp_M (Mp, kM+1, Mvlen) ;
                }
            }

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t mjnz = pM_end - pM ;
            if (ajnz == 0 || mjnz == 0) continue ;
            int64_t iA_first = GBi_A (Ai, pA, Avlen) ;
            int64_t iA_last  = GBi_A (Ai, pA_end-1, Avlen) ;
            int64_t iM_first = GBi_M (Mi, pM, Mvlen) ;
            int64_t iM_last  = GBi_M (Mi, pM_end-1, Mvlen) ;
            if (iA_last < iM_first || iM_last < iA_first) continue ;
            int64_t pM_start = pM ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;
            if (cjdense) continue ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += A(:,j) ; no S
            //------------------------------------------------------------------

            if (GB_A_IS_FULL)
            {

                //--------------------------------------------------------------
                // A is a full matrix
                //--------------------------------------------------------------

                int64_t pA_start = j * Avlen ;
                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;
                        // get iA in A(:,j)
                        pA = pA_start + iA ;
                        GB_PHASE2_ACTION ;
                    }
                }

            }
            else if (ajnz > 32 * mjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than M(:,j)
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;
                        // find iA in A(:,j)
                        int64_t pright = pA_end - 1 ;
                        bool found = GB_binary_search (iA, Ai, GB_Ai_IS_32,
                            &pA, &pright) ;
                        if (found) GB_PHASE2_ACTION ;
                    }
                }

            }
            else if (mjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // M(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // FUTURE::: exploit dense mask
                bool mjdense = false ;

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (mij) GB_PHASE2_ACTION ;
                }

            }
            else
            {

                //----------------------------------------------------------
                // A(:,j) and M(:,j) have about the same # of entries
                //----------------------------------------------------------

                // linear-time scan of A(:,j) and M(:,j)

                while (pA < pA_end && pM < pM_end)
                {
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    int64_t iM = GBi_M (Mi, pM, Mvlen) ;
                    if (iA < iM)
                    { 
                        // A(i,j) exists but not M(i,j)
                        pA++ ;  // go to the next entry in A(:,j)
                    }
                    else if (iM < iA)
                    { 
                        // M(i,j) exists but not A(i,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                    else
                    { 
                        // both A(i,j) and M(i,j) exist
                        if (GB_MCAST (Mx, pM, msize)) GB_PHASE2_ACTION ;
                        pA++ ;  // go to the next entry in A(:,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                }
            }
        }

        GB_PHASE2_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

