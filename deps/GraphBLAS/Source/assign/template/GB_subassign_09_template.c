//------------------------------------------------------------------------------
// GB_subassign_09_template: C(I,J)<M,repl> = scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 09: C(I,J)<M,repl> = scalar ; using S

// M:           present
// Mask_struct: true or false
// Mask_comp:   false
// C_replace:   true
// accum:       NULL
// A:           scalar
// S:           constructed

// C: not bitmap or full

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_GET_C ;      // C must not be bitmap
    GB_GET_MASK ;
    GB_GET_SCALAR ;
    GB_GET_S ;

    //--------------------------------------------------------------------------
    // Method 09: C(I,J)<M,repl> = scalar ; using S
    //--------------------------------------------------------------------------

    // Time: Optimal.  All entries in M+S must be examined.  All entries in S
    // are modified:  if M(i,j)=1 then S(i,j) is used to write to the
    // corresponding entry in C.  If M(i,j) is not present, or zero, then the
    // entry in C is cleared (because of C_replace).  If S(i,j) is not present,
    // and M(i,j)=1, then the scalar is inserted into C.  The only case that
    // can be skipped is if neither S nor M is present.  As a result, this
    // method need not traverse all of IxJ.  It can limit its traversal to the
    // pattern of M+S.

    // Method 09 and Method 11 are very similar.

    //--------------------------------------------------------------------------
    // Parallel: M+S (Methods 02, 04, 09, 10, 11, 12, 14, 16, 18, 20)
    //--------------------------------------------------------------------------

    if (GB_M_IS_BITMAP)
    { 
        // all of IxJ must be examined
        GB_SUBASSIGN_IXJ_SLICE ;
    }
    else
    { 
        // traverse all M+S
        GB_SUBASSIGN_TWO_SLICE (M, S) ;
    }

    //--------------------------------------------------------------------------
    // phase 1: create zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    if (GB_M_IS_BITMAP)
    {

        //----------------------------------------------------------------------
        // phase1: M is bitmap
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:nzombies)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_IXJ_TASK_DESCRIPTOR_PHASE1 (iM_start, iM_end) ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t j = kfirst ; j <= klast ; j++)
            {

                //--------------------------------------------------------------
                // get S(iM_start:iM_end,j)
                //--------------------------------------------------------------

                GB_LOOKUP_VECTOR_S_FOR_IXJ (j, pS, pS_end, iM_start) ;
                int64_t pM_start = j * Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(iM_start:iM_end,j) and M(ditto,j)
                //--------------------------------------------------------------

                for (int64_t iM = iM_start ; iM < iM_end ; iM++)
                {

                    int64_t pM = pM_start + iM ;
                    bool Sfound = (pS < pS_end) && (GBI_S (Si,pS,Svlen) == iM) ;
                    bool mij = Mb [pM] && GB_MCAST (Mx, pM, msize) ;

                    if (Sfound && !mij)
                    { 
                        // S (i,j) is present but M (i,j) is false
                        // ----[C A 0] or [X A 0]-------------------------------
                        // [X A 0]: action: ( X ): still a zombie
                        // [C A 0]: C_repl: action: ( delete ): becomes zombie
                        GB_C_S_LOOKUP ;
                        GB_DELETE_ENTRY ;
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                    else if (!Sfound && mij)
                    { 
                        // S (i,j) is not present, M (i,j) is true
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        task_pending++ ;
                    }
                    else if (Sfound && mij)
                    { 
                        // S (i,j) present and M (i,j) is true
                        GB_C_S_LOOKUP ;
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =A ): copy A, no accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_noaccum_C_A_1_scalar ;
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                }
            }
            GB_PHASE1_TASK_WRAPUP ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // phase1: M is hypersparse, sparse, or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:nzombies)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_TASK_DESCRIPTOR_PHASE1 ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // get S(:,j) and M(:,j)
                //--------------------------------------------------------------

                int64_t j = GBH (Zh, k) ;
                GB_GET_MAPPED (pM, pM_end, pA, pA_end, Mp, j, k, Z_to_X, Mvlen);
                GB_GET_MAPPED (pS, pS_end, pB, pB_end, Sp, j, k, Z_to_S, Svlen);

                //--------------------------------------------------------------
                // do a 2-way merge of S(:,j) and M(:,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                // int64_t jC = GB_ijlist (J, j, GB_J_KIND, Jcolon) ;

                // while both list S (:,j) and M (:,j) have entries
                while (pS < pS_end && pM < pM_end)
                {
                    int64_t iS = GBI_S (Si, pS, Svlen) ;
                    int64_t iM = GBI_M (Mi, pM, Mvlen) ;

                    if (iS < iM)
                    { 
                        // S (i,j) is present but M (i,j) is not
                        // ----[C A 0] or [X A 0]-------------------------------
                        // [X A 0]: action: ( X ): still a zombie
                        // [C A 0]: C_repl: action: ( delete ): becomes zombie
                        GB_C_S_LOOKUP ;
                        GB_DELETE_ENTRY ;
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                    else if (iM < iS)
                    {
                        // S (i,j) is not present, M (i,j) is present
                        if (GB_MCAST (Mx, pM, msize))
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                    else
                    {
                        // both S (i,j) and M (i,j) present
                        GB_C_S_LOOKUP ;
                        if (GB_MCAST (Mx, pM, msize))
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): copy A, no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_noaccum_C_A_1_scalar ;
                        }
                        else
                        { 
                            // ----[C A 0] or [X A 0]---------------------------
                            // [X A 0]: action: ( X ): still a zombie
                            // [C A 0]: C_repl: action: ( delete ): now zombie
                            GB_DELETE_ENTRY ;
                        }
                        pS++ ;  // go to the next entry in S(:,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                }

                // while list S (:,j) has entries.  List M (:,j) exhausted.
                while (pS < pS_end)
                { 
                    // S (i,j) is present but M (i,j) is not
                    // ----[C A 0] or [X A 0]-----------------------------------
                    // [X A 0]: action: ( X ): still a zombie
                    // [C A 0]: C_repl: action: ( delete ): becomes zombie
                    GB_C_S_LOOKUP ;
                    GB_DELETE_ENTRY ;
                    pS++ ;  // go to the next entry in S(:,j)
                }

                // while list M (:,j) has entries.  List S (:,j) exhausted.
                while (pM < pM_end)
                {
                    // S (i,j) is not present, M (i,j) is present
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        task_pending++ ;
                    }
                    pM++ ;  // go to the next entry in M(:,j)
                }
            }

            GB_PHASE1_TASK_WRAPUP ;
        }
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    GB_PENDING_CUMSUM ;

    if (GB_M_IS_BITMAP)
    {

        //----------------------------------------------------------------------
        // phase2: M is bitmap
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(&&:pending_sorted)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_IXJ_TASK_DESCRIPTOR_PHASE2 (iM_start, iM_end) ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t j = kfirst ; j <= klast ; j++)
            {

                //--------------------------------------------------------------
                // get S(iM_start:iM_end,j)
                //--------------------------------------------------------------

                GB_LOOKUP_VECTOR_S_FOR_IXJ (j, pS, pS_end, iM_start) ;
                int64_t pM_start = j * Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(iM_start:iM_end,j) and M(ditto,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                int64_t jC = GB_ijlist (J, j, GB_J_KIND, Jcolon) ;

                for (int64_t iM = iM_start ; iM < iM_end ; iM++)
                {
                    int64_t pM = pM_start + iM ;
                    bool Sfound = (pS < pS_end) && (GBI_S (Si,pS,Svlen) == iM) ;
                    bool mij = Mb [pM] && GB_MCAST (Mx, pM, msize) ;

                    if (!Sfound && mij)
                    { 
                        // S (i,j) is not present, M (i,j) is true
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        int64_t iC = GB_ijlist (I, iM, GB_I_KIND, Icolon) ;
                        GB_PENDING_INSERT_scalar ;
                    }
                    else if (Sfound)
                    { 
                        // S (i,j) present
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                }
            }
            GB_PHASE2_TASK_WRAPUP ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // phase2: M is hypersparse, sparse, or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(&&:pending_sorted)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_TASK_DESCRIPTOR_PHASE2 ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // get S(:,j) and M(:,j)
                //--------------------------------------------------------------

                int64_t j = GBH (Zh, k) ;
                GB_GET_MAPPED (pM, pM_end, pA, pA_end, Mp, j, k, Z_to_X, Mvlen);
                GB_GET_MAPPED (pS, pS_end, pB, pB_end, Sp, j, k, Z_to_S, Svlen);

                //--------------------------------------------------------------
                // do a 2-way merge of S(:,j) and M(:,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                int64_t jC = GB_ijlist (J, j, GB_J_KIND, Jcolon) ;

                // while both list S (:,j) and M (:,j) have entries
                while (pS < pS_end && pM < pM_end)
                {
                    int64_t iS = GBI_S (Si, pS, Svlen) ;
                    int64_t iM = GBI_M (Mi, pM, Mvlen) ;

                    if (iS < iM)
                    { 
                        // S (i,j) is present but M (i,j) is not
                        pS++ ;  // go to the next entry in S(:,j)
                    }
                    else if (iM < iS)
                    {
                        // S (i,j) is not present, M (i,j) is present
                        if (GB_MCAST (Mx, pM, msize))
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            int64_t iC = GB_ijlist (I, iM, GB_I_KIND, Icolon) ;
                            GB_PENDING_INSERT_scalar ;
                        }
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                    else
                    { 
                        // both S (i,j) and M (i,j) present
                        pS++ ;  // go to the next entry in S(:,j)
                        pM++ ;  // go to the next entry in M(:,j)
                    }
                }

                // while list M (:,j) has entries.  List S (:,j) exhausted.
                while (pM < pM_end)
                {
                    // S (i,j) is not present, M (i,j) is present
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        int64_t iM = GBI_M (Mi, pM, Mvlen) ;
                        int64_t iC = GB_ijlist (I, iM, GB_I_KIND, Icolon) ;
                        GB_PENDING_INSERT_scalar ;
                    }
                    pM++ ;  // go to the next entry in M(:,j)
                }
            }

            GB_PHASE2_TASK_WRAPUP ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

