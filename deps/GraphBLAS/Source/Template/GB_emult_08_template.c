//------------------------------------------------------------------------------
// GB_emult_08_template: C=A.*B, C<M or !M>=A.*B when C is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C=A.*B, C<M>=A.*B, or C<!M>=A.*B when C is sparse or hypersparse:

// phase1: does not compute C itself, but just counts the # of entries in each
// vector of C.  Fine tasks compute the # of entries in their slice of a
// single vector of C, and the results are cumsum'd.

// phase2: computes C, using the counts computed by phase1.

// No input matrix can be jumbled, and C is constructed as unjumbled.

// The following cases are handled:

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse  (method: 8bcd)

        //      ------------------------------------------
        //      C       <M>=        A       .*      B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse  (method: 8e)
        //      sparse  bitmap      sparse          sparse  (method: 8fgh)
        //      sparse  full        sparse          sparse  (method: 8fgh)
        //      sparse  sparse      sparse          bitmap  (9  (8e) or 2)
        //      sparse  sparse      sparse          full    (9  (8e) or 2)
        //      sparse  sparse      bitmap          sparse  (10 (8e) or 3)
        //      sparse  sparse      full            sparse  (10 (8e) or 3)

        //      ------------------------------------------
        //      C       <!M>=       A       .*      B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse  (8bcd: M later)
        //      sparse  bitmap      sparse          sparse  (method: 8fgh)
        //      sparse  full        sparse          sparse  (method: 8fgh)

// Methods 9 and 10 are not yet implemented, and are currently handled by this
// Method 8 instead.  See GB_emult_sparsity for this decision.
// "M later" means that C<!M>=A.*B is being computed, but the mask is not
// handled here; insteadl T=A.*B is computed and C<!M>=T is done later.

{

    int taskid ;
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

            #if ( GB_EMULT_08_PHASE == 1 )
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
                pC     = Cp [k] ;
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
                int64_t kA = (Ch == Ah) ? k :
                            ((C_to_A == NULL) ? j : C_to_A [k]) ;
                if (kA >= 0)
                { 
                    pA     = GBP_A (Ap, kA, vlen) ;
                    pA_end = GBP_A (Ap, kA+1, vlen) ;
                }
            }

            int64_t ajnz = pA_end - pA ;        // nnz in A(:,j) for this slice
            int64_t pA_start = pA ;
            bool adense = (ajnz == len) ;

            // get the first and last indices in A(:,j) for this vector
            int64_t iA_first = -1 ;
            if (ajnz > 0)
            { 
                iA_first = GBI_A (Ai, pA, vlen) ;
            }
            #if ( GB_EMULT_08_PHASE == 1 ) || defined ( GB_DEBUG )
            int64_t iA_last = -1 ;
            if (ajnz > 0)
            { 
                iA_last  = GBI_A (Ai, pA_end-1, vlen) ;
            }
            #endif

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
                int64_t kB = (Ch == Bh) ? k :
                            ((C_to_B == NULL) ? j : C_to_B [k]) ;
                if (kB >= 0)
                { 
                    pB     = GBP_B (Bp, kB, vlen) ;
                    pB_end = GBP_B (Bp, kB+1, vlen) ;
                }
            }

            int64_t bjnz = pB_end - pB ;        // nnz in B(:,j) for this slice
            int64_t pB_start = pB ;
            bool bdense = (bjnz == len) ;

            // get the first and last indices in B(:,j) for this vector
            int64_t iB_first = -1 ;
            if (bjnz > 0)
            { 
                iB_first = GBI_B (Bi, pB, vlen) ;
            }
            #if ( GB_EMULT_08_PHASE == 1 ) || defined ( GB_DEBUG )
            int64_t iB_last = -1 ;
            if (bjnz > 0)
            { 
                iB_last  = GBI_B (Bi, pB_end-1, vlen) ;
            }
            #endif

            //------------------------------------------------------------------
            // C(:,j)<optional mask> = A (:,j) .* B (:,j) or subvector
            //------------------------------------------------------------------

            #if ( GB_EMULT_08_PHASE == 1 )
            if (ajnz == 0 || bjnz == 0)
            { 
                // Method8(a): A(:,j) and/or B(:,j) are empty
                ;
            }
            else if (iA_last < iB_first || iB_last < iA_first)
            { 
                // Method8(a): intersection of A(:,j) and B(:,j) is empty
                // the last entry of A(:,j) comes before the first entry
                // of B(:,j), or visa versa
                ;
            }
            else
            #endif

            #ifdef GB_JIT_KERNEL
            {
                #if GB_NO_MASK
                {
                    // C=A.*B, all matrices sparse/hyper
                    #include "GB_emult_08bcd.c"
                }
                #elif (GB_M_IS_SPARSE || GB_M_IS_HYPER)
                {
                    // C<M>=A.*B, C and M are sparse/hyper
                    // either A or B are sparse/hyper
                    #include "GB_emult_08e.c"
                }
                #else
                {
                    // C<#M>=A.*B; C, A and B are sparse/hyper; M is bitmap/full
                    #include "GB_emult_08fgh.c"
                }
                #endif
            }
            #else
            {
                if (M == NULL)
                {
                    // C=A.*B, all matrices sparse/hyper
                    #include "GB_emult_08bcd.c"
                }
                else if (M_is_sparse_or_hyper)
                {
                    // C<M>=A.*B, C and M are sparse/hyper
                    // either A or B are sparse/hyper
                    #include "GB_emult_08e.c"
                }
                else
                {
                    // C<#M>=A.*B; C, A and B are sparse/hyper; M is bitmap/full
                    #include "GB_emult_08fgh.c"
                }
            }
            #endif

            //------------------------------------------------------------------
            // final count of nnz (C (:,j))
            //------------------------------------------------------------------

            #if ( GB_EMULT_08_PHASE == 1 )
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

