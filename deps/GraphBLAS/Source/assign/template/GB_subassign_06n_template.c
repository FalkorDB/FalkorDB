//------------------------------------------------------------------------------
// GB_subassign_06n_template: C(I,J)<M> = A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 06n: C(I,J)<M> = A ; no S

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           none (see also GB_subassign_06s)

// FULL: if A and C are dense, then C remains dense.

// If A is sparse and C dense, C will likely become sparse, except if M(i,j)=0
// wherever A(i,j) is not present.  So if M==A is aliased and A is sparse, then
// C remains dense.  Need C(I,J)<A,struct>=A kernel.  Then in that case, if C
// is dense it remains dense, even if A is sparse.   If that change is made,
// this kernel can start with converting C to sparse if A is sparse.

// C is not bitmap: GB_bitmap_assign is used if C is bitmap.
// M and A are not bitmap: 06s is used instead, if M or A are bitmap.

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    int64_t zorig = C->nzombies ;
    const int64_t Cnvec = C->nvec ;
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    GB_GET_C_HYPER_HASH ;
    GB_GET_MASK ;
    GB_GET_A ;
    const int64_t *restrict Ah = A->h ;
    const int64_t Anvec = A->nvec ;
    const bool A_is_hyper = (Ah != NULL) ;

    GB_OK (GB_hyper_hash_build (A, Werk)) ;
    const int64_t *restrict A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *restrict A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *restrict A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    //--------------------------------------------------------------------------
    // Method 06n: C(I,J)<M> = A ; no S
    //--------------------------------------------------------------------------

    // Time: O(nnz(M)*(log(a)+log(c)), where a and c are the # of entries in a
    // vector of A and C, respectively.  The entries in the intersection of M
    // (where the entries are true) and the matrix addition C(I,J)+A must be
    // examined.  This method scans M, and searches for entries in A and C(I,J)
    // using two binary searches.  If M is very dense, this method can be
    // slower than Method 06s.  This method is selected if nnz (A) >= nnz (M).

    // Compare with Methods 05 and 07, which use a similar algorithmic outline
    // and parallelization strategy.

    //--------------------------------------------------------------------------
    // Parallel: slice M into coarse/fine tasks (Method 05, 06n, 07)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_ONE_SLICE (M) ;    // M cannot be jumbled

    //--------------------------------------------------------------------------
    // phase 1: create zombies, update entries, and count pending tuples
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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = GBH_M (Mh, k) ;
            GB_GET_VECTOR_M (pM, pM_end, pA, pA_end, Mp, k, Mvlen) ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;
            GB_LOOKUP_VECTOR_A (j, pA, pA_end) ;
            int64_t ajnz = pA_end - pA ;
            bool ajdense = (ajnz == Avlen) ;
            int64_t pA_start = pA ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            int64_t cjnz = pC_end - pC_start ;
            if (cjnz == 0 && ajnz == 0) continue ;
            bool cjdense = (cjnz == Cvlen) ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = A(:,j) ; no S
            //------------------------------------------------------------------

            if (cjdense && ajdense)
            {

                //--------------------------------------------------------------
                // C(:,jC) and A(:,j) are both dense
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBI_M (Mi, pM, Mvlen) ;
                        GB_iC_DENSE_LOOKUP ;

                        // find iA in A(:,j)
                        // A(:,j) is dense; no need for binary search
                        pA = pA_start + iA ;
                        ASSERT (GBI_A (Ai, pA, Avlen) == iA) ;
                        // ----[C A 1] or [X A 1]-----------------------
                        // [C A 1]: action: ( =A ): copy A to C, no acc
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_noaccum_C_A_1_matrix ;
                    }
                }

            }
            else if (cjdense)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense, A(:,j) is sparse
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (GB_MCAST (Mx, pM, msize))
                    {
                        int64_t iA = GBI_M (Mi, pM, Mvlen) ;
                        GB_iC_DENSE_LOOKUP ;

                        // find iA in A(:,j)
                        bool aij_found ;
                        int64_t apright = pA_end - 1 ;
                        GB_BINARY_SEARCH (iA, Ai, pA, apright, aij_found) ;

                        if (!aij_found)
                        { 
                            // C (iC,jC) is present but A (i,j) is not
                            // ----[C . 1] or [X . 1]---------------------------
                            // [C . 1]: action: ( delete ): becomes zombie
                            // [X . 1]: action: ( X ): still zombie
                            GB_DELETE_ENTRY ;
                        }
                        else
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A to C, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_noaccum_C_A_1_matrix ;
                        }
                    }
                }

            }
            else if (ajdense)
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse, A(:,j) is dense
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (GB_MCAST (Mx, pM, msize))
                    {
                        int64_t iA = GBI_M (Mi, pM, Mvlen) ;

                        // find C(iC,jC) in C(:,jC)
                        GB_iC_BINARY_SEARCH ;

                        // lookup iA in A(:,j)
                        pA = pA_start + iA ;
                        ASSERT (GBI_A (Ai, pA, Avlen) == iA) ;

                        if (cij_found)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A into C, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_noaccum_C_A_1_matrix ;
                        }
                        else
                        { 
                            // C (iC,jC) is not present, A (i,j) is present
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C(:,jC) and A(:,j) are both sparse
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (GB_MCAST (Mx, pM, msize))
                    {
                        int64_t iA = GBI_M (Mi, pM, Mvlen) ;

                        // find C(iC,jC) in C(:,jC)
                        GB_iC_BINARY_SEARCH ;

                        // find iA in A(:,j)
                        bool aij_found ;
                        int64_t apright = pA_end - 1 ;
                        GB_BINARY_SEARCH (iA, Ai, pA, apright, aij_found) ;

                        if (cij_found && aij_found)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A into C, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_noaccum_C_A_1_matrix ;
                        }
                        else if (!cij_found && aij_found)
                        { 
                            // C (iC,jC) is not present, A (i,j) is present
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
                        else if (cij_found && !aij_found)
                        { 
                            // C (iC,jC) is present but A (i,j) is not
                            // ----[C . 1] or [X . 1]---------------------------
                            // [C . 1]: action: ( delete ): becomes zombie
                            // [X . 1]: action: ( X ): still zombie
                            GB_DELETE_ENTRY ;
                        }
                    }
                }
            }
        }

        GB_PHASE1_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    GB_PENDING_CUMSUM ;
    zorig = C->nzombies ;

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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = GBH_M (Mh, k) ;
            GB_GET_VECTOR_M (pM, pM_end, pA, pA_end, Mp, k, Mvlen) ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;
            GB_LOOKUP_VECTOR_A (j, pA, pA_end) ;
            int64_t ajnz = pA_end - pA ;
            if (ajnz == 0) continue ;
            bool ajdense = (ajnz == Avlen) ;
            int64_t pA_start = pA ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            bool cjdense = ((pC_end - pC_start) == Cvlen) ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = A(:,j)
            //------------------------------------------------------------------

            if (!cjdense)
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (GB_MCAST (Mx, pM, msize))
                    {
                        int64_t iA = GBI_M (Mi, pM, Mvlen) ;

                        // find iA in A(:,j)
                        if (ajdense)
                        { 
                            // A(:,j) is dense; no need for binary search
                            pA = pA_start + iA ;
                            ASSERT (GBI_A (Ai, pA, Avlen) == iA) ;
                        }
                        else
                        { 
                            // A(:,j) is sparse; use binary search
                            int64_t apright = pA_end - 1 ;
                            bool aij_found ;
                            GB_BINARY_SEARCH (iA, Ai, pA, apright, aij_found) ;
                            if (!aij_found) continue ;
                        }

                        // find C(iC,jC) in C(:,jC)
                        GB_iC_BINARY_SEARCH ;
                        if (!cij_found)
                        { 
                            // C (iC,jC) is not present, A (i,j) is present
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            GB_PENDING_INSERT_aij ;
                        }
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

