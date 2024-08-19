//------------------------------------------------------------------------------
// GB_add_sparse_template:  C=A+B, C<M>=A+B when C is sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse or hypersparse:

        //      ------------------------------------------
        //      C       =           A       +       B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse

        //      ------------------------------------------
        //      C      <M> =        A       +       B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse
        //      sparse  sparse      sparse          bitmap
        //      sparse  sparse      sparse          full
        //      sparse  sparse      bitmap          sparse
        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      bitmap          full
        //      sparse  sparse      full            sparse
        //      sparse  sparse      full            bitmap
        //      sparse  sparse      full            full

        //      sparse  bitmap      sparse          sparse
        //      sparse  full        sparse          sparse

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      sparse  bitmap      sparse          sparse
        //      sparse  full        sparse          sparse

        // If all four matrices are sparse/hypersparse, and C<!M>=A+B is being
        // computed, then M is passed in as NULL to GB_add_phase*.
        // GB_add_sparsity returns apply_mask as false.  The methods below do
        // not handle the case when C is sparse, M is sparse, and !M is used.
        // All other uses of !M when M is sparse result in a bitmap structure
        // for C, and this is handled by GB_add_bitmap_template.

        // For this case: the mask is done later, so C=A+B is computed here:

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse      (mask later)

{

    //--------------------------------------------------------------------------
    // phase1: count entries in each C(:,j)
    // phase2: compute C
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < C_ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast == -1) ;
        int64_t len ;
        if (fine_task)
        { 
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
            len = TaskList [taskid].len ;
        }
        else
        { 
            // a coarse task operates on one or more whole vectors
            len = vlen ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get j, the kth vector of C
            //------------------------------------------------------------------

            int64_t j = GBH_C (Ch, k) ;

            #if ( GB_ADD_PHASE == 1 )
            int64_t cjnz = 0 ;
            #else
            int64_t pC, pC_end ;
            if (fine_task)
            { 
                // A fine task computes a slice of C(:,j)
                pC     = TaskList [taskid  ].pC ;
                pC_end = TaskList [taskid+1].pC ;
                ASSERT (Cp [k] <= pC && pC <= pC_end && pC_end <= Cp [k+1]) ;
            }
            else
            { 
                // The vectors of C are never sliced for a coarse task.
                pC     = Cp [k  ] ;
                pC_end = Cp [k+1] ;
            }
            int64_t cjnz = pC_end - pC ;
            if (cjnz == 0) continue ;
            #endif

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA = -1, pA_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Ai,Ax [pA...pA_end-1], which is
                // a subset of the vector A(:,j)
                pA     = TaskList [taskid].pA ;
                pA_end = TaskList [taskid].pA_end ;
            }
            else
            {
                // A coarse task operates on the entire vector A (:,j)
                int64_t kA = (C_to_A == NULL) ? j : C_to_A [k] ;
                if (kA >= 0)
                { 
                    pA     = GBP_A (Ap, kA, vlen) ;
                    pA_end = GBP_A (Ap, kA+1, vlen) ;
                }
            }

            int64_t ajnz = pA_end - pA ;    // nnz in A(:,j) for this slice
            int64_t pA_start = pA ;
            bool adense = (ajnz == len) ;

            // get the first and last indices in A(:,j) for this vector
            int64_t iA_first = -1, iA_last = -1 ;
            if (ajnz > 0)
            { 
                iA_first = GBI_A (Ai, pA, vlen) ;
                iA_last  = GBI_A (Ai, pA_end-1, vlen) ;
            }

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t pB = -1, pB_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Bi,Bx [pB...pB_end-1], which is
                // a subset of the vector B(:,j)
                pB     = TaskList [taskid].pB ;
                pB_end = TaskList [taskid].pB_end ;
            }
            else
            {
                // A coarse task operates on the entire vector B (:,j)
                int64_t kB = (C_to_B == NULL) ? j : C_to_B [k] ;
                if (kB >= 0)
                { 
                    pB     = GBP_B (Bp, kB, vlen) ;
                    pB_end = GBP_B (Bp, kB+1, vlen) ;
                }
            }

            int64_t bjnz = pB_end - pB ;    // nnz in B(:,j) for this slice
            int64_t pB_start = pB ;
            bool bdense = (bjnz == len) ;

            // get the first and last indices in B(:,j) for this vector
            int64_t iB_first = -1, iB_last = -1 ;
            if (bjnz > 0)
            { 
                iB_first = GBI_B (Bi, pB, vlen) ;
                iB_last  = GBI_B (Bi, pB_end-1, vlen) ;
            }

            //------------------------------------------------------------------
            // C(:,j)<optional mask> = A (:,j) + B (:,j) or subvector
            //------------------------------------------------------------------

            #ifdef GB_JIT_KERNEL
            {
                #if GB_NO_MASK
                {
                    #include "GB_add_sparse_noM.c"
                }
                #elif (GB_M_IS_SPARSE || GB_M_IS_HYPER)
                {
                    #include "GB_add_sparse_M_sparse.c"
                }
                #else
                {
                    #include "GB_add_sparse_M_bitmap.c"
                }
                #endif
            }
            #else
            {
                if (M == NULL)
                { 
                    #include "GB_add_sparse_noM.c"
                }
                else if (M_is_sparse_or_hyper)
                { 
                    #include "GB_add_sparse_M_sparse.c"
                }
                else
                { 
                    #include "GB_add_sparse_M_bitmap.c"
                }
            }
            #endif

            //------------------------------------------------------------------
            // final count of nnz (C (:,j))
            //------------------------------------------------------------------

            #if ( GB_ADD_PHASE == 1 )
            if (fine_task)
            { 
                TaskList [taskid].pC = cjnz ;
            }
            else
            { 
                Cp [k] = cjnz ;
            }
            #endif
        }
    }
}

