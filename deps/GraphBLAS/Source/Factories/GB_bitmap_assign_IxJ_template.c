//------------------------------------------------------------------------------
// GB_bitmap_assign_IxJ_template: iterate over all of C(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Iterate over all positions in the IxJ Cartesian product.  This is all
// entries C(i,j) where i is in the list I and j is in the list J.  This
// traversal occurs whether or not C(i,j) is an entry present in C.

// The C matrix is accessed at C(I,J).  The A matrix is size |I|-by-|J|.
// For bitmap assignent, C(I,J)=A is being computed.  For bitmap extraction,
// C=A(I,J) so the roles of A and C are swapped (see GB_bitmap_subref.c).

{

    //--------------------------------------------------------------------------
    // create the tasks to iterate over IxJ
    //--------------------------------------------------------------------------

    int ntasks = 0, nthreads ;
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;
    GB_OK (GB_subassign_IxJ_slice (&TaskList, &TaskList_size, &ntasks,
        &nthreads, /* I, */ nI, /* Ikind, Icolon, J, */ nJ,
        /* Jkind, Jcolon, */ Werk)) ;

    //--------------------------------------------------------------------------
    // iterate over all IxJ
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        int64_t task_cnvals = 0 ;
        bool fine_task = (klast == -1) ;
        int64_t iA_start = 0, iA_end = nI ;
        if (fine_task)
        { 
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
            iA_start = TaskList [taskid].pA ;
            iA_end   = TaskList [taskid].pA_end ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t jA = kfirst ; jA <= klast ; jA++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, jA, Jkind, Jcolon) ;
            int64_t pC0 = jC * vlen ;       // first entry in C(:,jC)
            int64_t pA0 = jA * nI ;         // first entry in A(:,jA)

            //------------------------------------------------------------------
            // operate on C (I(iA_start,iA_end-1),jC)
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            { 
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                int64_t pC = iC + pC0 ;
                int64_t pA = iA + pA0 ;
                // operate on C(iC,jC) at pC (if C is bitmap or full)
                // and A(iA,jA) or M(iA,jA) at pA, if A and/or M are
                // bitmap or full.  M(iA,jA) is accessed only for the
                // subassign method when M is bitmap or full.
                GB_IXJ_WORK (pC, pA) ;
            }
        }
        cnvals += task_cnvals ;
    }

    //--------------------------------------------------------------------------
    // free workpace
    //--------------------------------------------------------------------------

    GB_FREE_WORK (&TaskList, TaskList_size) ;
}

