//------------------------------------------------------------------------------
// GB_bitmap_assign_IxJ_template: iterate over all of C(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Iterate over all positions in the IxJ Cartesian product.  This is all
// entries C(i,j) where i is in the list I and j is in the list J.  This
// traversal occurs whether or not C(i,j) is an entry present in C.

// The C matrix is accessed at C(I,J).  The A matrix is size |I|-by-|J|.
// For bitmap assignent, C(I,J)=A is being computed.  For bitmap extraction,
// C=A(I,J) so the roles of A and C are swapped (see GB_bitmap_subref.c).

// The workspace must already have been declared as follow:
//
//      GB_task_struct *TaskList_IxJ = NULL ; size_t TaskList_IxJ_size = 0 ;
//      int ntasks_IxJ = 0, nthreads_IxJ = 0 ;

// This template is used in the GB_bitmap_assign_* methods, and
// GB_bitmap_subref.  vlen = C->vlen must be assigned.

// The workspace is allocated and tasks are computed, if not already done.
// It is not freed, so it can be used for subsequent uses of this template.
// To free the workspace, the method that uses this template must do:
//
//      GB_FREE_MEMORY (&TaskList_IxJ, TaskList_IxJ_size) ;

{

    //--------------------------------------------------------------------------
    // slice IxJ
    //--------------------------------------------------------------------------

    if (TaskList_IxJ == NULL)
    { 
        GB_OK (GB_subassign_IxJ_slice (&TaskList_IxJ, &TaskList_IxJ_size,
            &ntasks_IxJ, &nthreads_IxJ, nI, nJ, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // iterate over all IxJ
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads_IxJ) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (taskid = 0 ; taskid < ntasks_IxJ ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList_IxJ [taskid].kfirst ;
        int64_t klast  = TaskList_IxJ [taskid].klast ;
        #ifndef GB_NO_CNVALS
        int64_t task_cnvals = 0 ;
        #endif
        bool fine_task = (klast == -1) ;
        int64_t iA_start = 0, iA_end = nI ;
        if (fine_task)
        { 
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
            iA_start = TaskList_IxJ [taskid].pA ;
            iA_end   = TaskList_IxJ [taskid].pA_end ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t jA = kfirst ; jA <= klast ; jA++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_IJLIST (J, jA, GB_J_KIND, Jcolon) ;
            int64_t pC0 = jC * vlen ;       // first entry in C(:,jC)
            int64_t pA0 = jA * nI ;         // first entry in A(:,jA)

            //------------------------------------------------------------------
            // operate on C (I(iA_start,iA_end-1),jC)
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            { 
                int64_t iC = GB_IJLIST (I, iA, GB_I_KIND, Icolon) ;
                int64_t pC = iC + pC0 ;
                int64_t pA = iA + pA0 ;
                // operate on C(iC,jC) at pC (if C is bitmap or full)
                // and A(iA,jA) or M(iA,jA) at pA, if A and/or M are
                // bitmap or full.  M(iA,jA) is accessed only for the
                // subassign method when M is bitmap or full.
                GB_IXJ_WORK (pC, pA) ;
            }
        }
        #ifndef GB_NO_CNVALS
        cnvals += task_cnvals ;
        #endif
    }
}

