//------------------------------------------------------------------------------
// GB_subassign_13_template: C(I,J)<!M> = scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 13: C(I,J)<!M> = scalar ; using S

// M:           present
// Mask_comp:   true
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           constructed

// C: not bitmap, but can be full since no zombies are inserted in that case
// M: not bitmap

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    const int64_t Cnvec = C->nvec ;
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    GB_GET_MASK ;
    GB_GET_MASK_HYPER_HASH ;
    GB_GET_SCALAR ;
    GB_GET_S ;

    //--------------------------------------------------------------------------
    // Method 13: C(I,J)<!M> = scalar ; using S
    //--------------------------------------------------------------------------

    // Time: Close to optimal; must visit all IxJ, so Omega(|I|*|J|) is
    // required.  The sparsity of !M cannot be exploited.

    // Methods 13, 15, 17, and 19 are very similar.

    //--------------------------------------------------------------------------
    // Parallel: all IxJ (Methods 01, 03, 13, 15, 17, 19)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_IXJ_SLICE ;

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

        GB_GET_IXJ_TASK_DESCRIPTOR_PHASE1 (iA_start, iA_end) ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, j, GB_J_KIND, Jcolon) ;

            //------------------------------------------------------------------
            // get S(iA_start:end,j) and M(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_S_FOR_IXJ (j, pS, pS_end, iA_start) ;
            GB_LOOKUP_VECTOR_M_FOR_IXJ (j, pM, pM_end, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M,repl> = scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {

                //--------------------------------------------------------------
                // Get the indices at the top of each list.
                //--------------------------------------------------------------

                int64_t iS = (pS < pS_end) ? GBI_S (Si, pS, Svlen) : INT64_MAX ;
                int64_t iM = (pM < pM_end) ? GBI_M (Mi, pM, Mvlen) : INT64_MAX ;

                //--------------------------------------------------------------
                // find the smallest index of [iS iA iM] (always iA)
                //--------------------------------------------------------------

                int64_t i = iA ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                if (i == iM)
                { 
                    // mij = (bool) M [pM]
                    mij = GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    pM++ ;  // go to the next entry in M(:,j)
                }
                else
                { 
                    // mij not present, implicitly false
                    ASSERT (i < iM) ;
                    mij = false ;
                }

                // complement the mask entry mij since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // assign the entry
                //--------------------------------------------------------------

                if (i == iS)
                {
                    ASSERT (i == iA) ;
                    {
                        // both S (i,j) and A (i,j) present
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_C_S_LOOKUP ;
                            GB_noaccum_C_A_1_scalar ;
                        }
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                }
                else
                {
                    ASSERT (i == iA) ;
                    {
                        // S (i,j) is not present, A (i,j) is present
                        if (mij)
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

    GB_PENDING_CUMSUM ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(&&:pending_sorted)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_IXJ_TASK_DESCRIPTOR_PHASE2 (iA_start, iA_end) ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, j, GB_J_KIND, Jcolon) ;

            //------------------------------------------------------------------
            // get S(iA_start:end,j) and M(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_S_FOR_IXJ (j, pS, pS_end, iA_start) ;
            GB_LOOKUP_VECTOR_M_FOR_IXJ (j, pM, pM_end, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M,repl> = scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {

                //--------------------------------------------------------------
                // Get the indices at the top of each list.
                //--------------------------------------------------------------

                int64_t iS = (pS < pS_end) ? GBI_S (Si, pS, Svlen) : INT64_MAX ;
                int64_t iM = (pM < pM_end) ? GBI_M (Mi, pM, Mvlen) : INT64_MAX ;

                //--------------------------------------------------------------
                // find the smallest index of [iS iA iM] (always iA)
                //--------------------------------------------------------------

                int64_t i = iA ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                if (i == iM)
                { 
                    // mij = (bool) M [pM]
                    mij = GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    pM++ ;  // go to the next entry in M(:,j)
                }
                else
                { 
                    // mij not present, implicitly false
                    ASSERT (i < iM) ;
                    mij = false ;
                }

                // complement the mask entry mij since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // assign the entry
                //--------------------------------------------------------------

                if (i == iS)
                {
                    ASSERT (i == iA) ;
                    { 
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                }
                else
                {
                    ASSERT (i == iA) ;
                    {
                        // S (i,j) is not present, A (i,j) is present
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            int64_t iC = GB_ijlist (I, iA, GB_I_KIND, Icolon) ;
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

