//------------------------------------------------------------------------------
// GB_subassign_27_template: C<C,struct> += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 27: C<C,struct> += A ; no S

// M:           present, and aliased exactly with C
// Mask_struct: true
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// C not bitmap; C can be full since no zombies are inserted in that case.
// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    const bool may_see_zombies = (C->nzombies > 0) ;
    GB_GET_C_HYPER_HASH ;
    GB_GET_ACCUM_MATRIX ;
    bool A_is_full = GB_IS_FULL (A) ;

    //--------------------------------------------------------------------------
    // Method 27: C<C,struct> += A ; no S
    //--------------------------------------------------------------------------

    // Time: Close to optimal. Omega (sum_j (min (nnz (A(:,j)), nnz (C(:,j))))),
    // since only the intersection of A.*C needs to be considered.  If either
    // C(:,j) or A(:,j) are very sparse compared to the other, then the shorter
    // is traversed with a linear-time scan and a binary search is used for the
    // other.  If the number of nonzeros is comparable, a linear-time scan is
    // used for both.  Once two entries C(i,j) and A(i,j) are found with the
    // same index i, the entry A(i,j) is accumulated into C.

    // The algorithm is very much like the eWise multiplication of A.*C, so the
    // parallel scheduling relies on GB_emult_08_phase0 and GB_ewise_slice.

    //--------------------------------------------------------------------------
    // Parallel: slice the eWiseMult of Z=A.*C (Method 27 only)
    //--------------------------------------------------------------------------

    // Method 27 only.  If C is sparse, it is sliced for a fine task, so that
    // it can do a binary search.  But if C(:,jC) is dense, C(:,jC) is not
    // sliced, so the fine task must do a direct lookup.  The Z matrix is not
    // constructed, except for its hyperlist (Zh_shallow) and mapping to A and
    // C.

    // No matrix (C or A) can be bitmap.  C and A can be sparse/hyper/full,
    // in any combination.

    int64_t Znvec ;
    GB_MDECL (Zh_shallow, const, u) ;
    bool Zj_is_32 ;
    GB_OK (GB_subassign_08n_slice (
        &TaskList, &TaskList_size, &ntasks, &nthreads,
        &Znvec, &Zh_shallow, &Z_to_A, &Z_to_A_size, &Z_to_M, &Z_to_M_size,
        &Zj_is_32, C,
        /* I, I_is_32, nI, Ikind, Icolon: */ NULL, false, 0, GB_ALL, NULL,
        /* J, J_is_32, nJ, Jkind, Jcolon: */ NULL, false, 0, GB_ALL, NULL,
        A, /* C is aliased to M: */ C, Werk)) ;
    GB_IPTR (Zh_shallow, Zj_is_32) ;

    //--------------------------------------------------------------------------
    // phase 1: update entries
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
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
            // get A(:,j) and M(:,j) (the latter is the same as C(:,j))
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

            int64_t pC = -1, pC_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on a slice of C(:,k)
                pC     = TaskList [taskid].pB ;
                pC_end = TaskList [taskid].pB_end ;
            }
            else
            { 
                // vectors are never sliced for a coarse task
                int64_t kC = (Zh_shallow == Ch) ? k :
                    ((Z_to_M == NULL) ? j : Z_to_M [k]) ;
                if (kC >= 0)
                { 
                    pC     = GBp_C (Cp, kC, Cvlen) ;
                    pC_end = GBp_C (Cp, kC+1, Cvlen) ;
                }
            }

            int64_t pC_start = pC ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and C(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t cjnz = pC_end - pC ;
            if (ajnz == 0 || cjnz == 0) continue ;
            int64_t iA_first = GBi_A (Ai, pA, Avlen) ;
            int64_t iA_last  = GBi_A (Ai, pA_end-1, Avlen) ;
            int64_t iC_first = GBi_C (Ci, pC, Cvlen) ;
            int64_t iC_last  = GBi_C (Ci, pC_end-1, Cvlen) ;
            if (iA_last < iC_first || iC_last < iA_first) continue ;

            //------------------------------------------------------------------
            // C(:,j)<C(:,j),struct> += A(:,j) ; no S
            //------------------------------------------------------------------

            if (GB_A_IS_FULL)
            {

                //--------------------------------------------------------------
                // A is a full matrix
                //--------------------------------------------------------------

                int64_t pA_start = j * Avlen ;
                for ( ; pC < pC_end ; pC++)
                { 
                    // get iC in C(:,j)
                    int64_t iC = GBi_C (Ci, pC, Cvlen) ;
                    bool is_zombie = GB_IS_ZOMBIE (iC) ;
                    if (!is_zombie)
                    { 
                        // ----[C A 1] with accum
                        // action: ( =C+A ): apply the accumulator
                        int64_t pA = pA_start + iC ;
                        GB_ACCUMULATE_aij (Cx, pC, Ax, pA, GB_A_ISO, ywork,
                            GB_C_ISO) ;
                    }
                }

            }
            else if (cjdense)
            {

                //--------------------------------------------------------------
                // C(:,j) is dense
                //--------------------------------------------------------------

                for ( ; pA < pA_end ; pA++)
                { 
                    // get iA in A(:,j)
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    // lookup iC in C(:,j)
                    int64_t pC = pC_start + iA ;
                    int64_t iC = GBi_C (Ci, pC, Cvlen) ;
                    bool is_zombie = GB_IS_ZOMBIE (iC) ;
                    if (!is_zombie)
                    { 
                        // ----[C A 1] with accum
                        // action: ( =C+A ): apply the accumulator
                        GB_ACCUMULATE_aij (Cx, pC, Ax, pA, GB_A_ISO, ywork,
                            GB_C_ISO) ;
                    }
                }

            }
            else if (ajnz > 32 * cjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than C(:,j)
                //--------------------------------------------------------------

                for ( ; pC < pC_end ; pC++)
                { 
                    int64_t iC = GBi_C (Ci, pC, Cvlen) ;
                    bool is_zombie = GB_IS_ZOMBIE (iC) ;
                    if (!is_zombie)
                    {
                        // find iC in A(:,j)
                        int64_t pright = pA_end - 1 ;
                        bool found = GB_binary_search (iC, Ai, GB_Ai_IS_32,
                            &pA, &pright) ;
                        if (found)
                        { 
                            // ----[C A 1] with accum
                            // action: ( =C+A ): apply the accumulator
                            GB_ACCUMULATE_aij (Cx, pC, Ax, pA, GB_A_ISO, ywork,
                                GB_C_ISO) ;
                        }
                    }
                }

            }
            else if (cjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // C(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    // find iA in C(:,j)
                    bool is_zombie ;
                    int64_t pright = pC_end - 1 ;
                    bool found = GB_binary_search_zombie (iA, Ci, GB_Ci_IS_32,
                        &pC, &pright, may_see_zombies, &is_zombie) ;
                    if (found && !is_zombie)
                    { 
                        // ----[C A 1] with accum
                        // action: ( =C+A ): apply the accumulator
                        GB_ACCUMULATE_aij (Cx, pC, Ax, pA, GB_A_ISO, ywork,
                            GB_C_ISO) ;
                    }
                }

            }
            else
            {

                //----------------------------------------------------------
                // A(:,j) and C(:,j) have about the same # of entries
                //----------------------------------------------------------

                // linear-time scan of A(:,j) and C(:,j)

                while (pA < pA_end && pC < pC_end)
                {
                    int64_t iA = GBi_A (Ai, pA, Avlen) ;
                    int64_t iC = GBi_C (Ci, pC, Cvlen) ;
                    bool is_zombie = GB_IS_ZOMBIE (iC) ;
                    if (is_zombie) iC = GB_DEZOMBIE (iC) ; 
                    if (iA < iC)
                    { 
                        // A(i,j) exists but not C(i,j)
                        pA++ ;  // go to the next entry in A(:,j)
                    }
                    else if (iC < iA)
                    { 
                        // C(i,j) exists but not A(i,j)
                        pC++ ;  // go to the next entry in C(:,j)
                    }
                    else
                    { 
                        if (!is_zombie)
                        { 
                            // both A(i,j) and C(i,j) exist
                            // ----[C A 1] with accum
                            // action: ( =C+A ): apply the accumulator
                            GB_ACCUMULATE_aij (Cx, pC, Ax, pA, GB_A_ISO, ywork,
                                GB_C_ISO) ;
                        }
                        pA++ ;  // go to the next entry in A(:,j)
                        pC++ ;  // go to the next entry in C(:,j)
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C output from subassign_27", GB0_Z) ;
    return (GrB_SUCCESS) ;
}

