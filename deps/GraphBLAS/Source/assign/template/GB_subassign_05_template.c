//------------------------------------------------------------------------------
// GB_subassign_05_template: C(I,J)<M> = scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 05: C(I,J)<M> = scalar ; no S

// No entries can be deleted (no new zombies), but entries can be inserted.
// Prior zombies may exist in C.

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

// C: not bitmap
// M: any sparsity

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    const bool may_see_zombies_phase1 = (C->nzombies > 0) ;
    GB_GET_C_HYPER_HASH ;
    GB_GET_MASK ;
    GB_GET_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 05: C(I,J)<M> = scalar ; no S
    //--------------------------------------------------------------------------

    // Time: Close to Optimal:  the method must iterate over all entries in M,
    // so the time is Omega(nnz(M)).  For each entry M(i,j)=1, the
    // corresponding entry in C must be found and updated (inserted or
    // modified).  This method does this with a binary search of C(:,jC) or a
    // direct lookup if C(:,jC) is dense.  The time is thus O(nnz(M)*log(n)) in
    // the worst case, usually less than that since C(:,jC) often has O(1)
    // entries.  An additional time of O(|J|*log(Cnvec)) is added if C is
    // hypersparse.  There is no equivalent method that computes
    // C(I,J)<M>=scalar using the matrix S.

    // Method 05 and Method 07 are very similar.  Also compare with Method 06n.

    //--------------------------------------------------------------------------
    // Parallel: slice M into coarse/fine tasks (Method 05, 06n, 07)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_ONE_SLICE (M) ;    // M cannot be jumbled

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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = GBh_M (Mh, k) ;
            GB_GET_VECTOR_M ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            int64_t cjnz = pC_end - pC_start ;
            bool cjdense = (cjnz == Cvlen) ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = scalar ; no S
            //------------------------------------------------------------------

            if (cjdense)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense so the binary search of C is not needed
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    { 
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;
                        GB_iC_DENSE_LOOKUP ;

                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =A ): copy A into C, no accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_noaccum_C_A_1_scalar ;
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // update C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    {
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;

                        // find C(iC,jC) in C(:,jC)
                        GB_iC_BINARY_SEARCH (may_see_zombies_phase1) ;
                        if (cij_found)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A into C, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_noaccum_C_A_1_scalar ;
                        }
                        else
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = GBh_M (Mh, k) ;
            GB_GET_VECTOR_M ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC ;
            bool cjdense = ((pC_end - pC_start) == Cvlen) ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = scalar ; no S
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

                    bool mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    {
                        int64_t iA = GBi_M (Mi, pM, Mvlen) ;

                        // find C(iC,jC) in C(:,jC)
                        GB_iC_BINARY_SEARCH (may_see_zombies_phase2) ;
                        if (!cij_found)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            GB_PENDING_INSERT_scalar ;
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

