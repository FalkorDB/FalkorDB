//------------------------------------------------------------------------------
// GB_AxB_dot3_phase1_template: analysis phase for dot3; C<M> = A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Purely symbolic phase1 for the dot3 method of GrB_mxm.  This does not
// access any values, except for the mask.

{
    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast == -1) ;
        if (fine_task)
        { 
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get j, the kth vector of C and M
            //------------------------------------------------------------------

            #if defined ( GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED )
            // M and C are sparse
            const int64_t j = k ;
            #else
            // M and C are either both sparse or both hypersparse
            const int64_t j = GBH_C (Ch, k) ;
            #endif

            GB_GET_VECTOR (pM, pM_end, pM, pM_end, Mp, k, mvlen) ;

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
                // B is hyper: find B(:,j) using the B->Y hyper hash
                int64_t pB_start, pB_end ;
                GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx,
                    B_hash_bits, j, &pB_start, &pB_end) ;
            #elif GB_B_IS_SPARSE
                // B is sparse
                const int64_t pB_start = Bp [j] ;
                const int64_t pB_end = Bp [j+1] ;
            #else
                // B is bitmap or full
                const int64_t pB_start = j * vlen ;
                const int64_t pB_end = (j+1) * vlen ;
            #endif
            const int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // estimate the work to compute each entry of C(:,j)
            //------------------------------------------------------------------

            // A decent estimate of the work to compute the dot product C(i,j)
            // = A(:,i)'*B(:,j) is min (|A(:,i)|, |B(:,j)|) + 1.  This is a
            // lower bound.  The actual work could require a binary search of
            // either A(:,i) or B(:,j), or a merge of the two vectors.  Or it
            // could require no work at all if all entries in A(:,i) appear
            // before all entries in B(:,j), or visa versa.  No work is done if
            // M(i,j)=0.

            if (bjnz == 0)
            {
                // B(:,j) is empty, so C(:,j) is empty as well.  No work is to
                // be done, but it still takes unit work to flag each C(:,j) as
                // a zombie
                for ( ; pM < pM_end ; pM++)
                { 
                    Cwork [pM] = 1 ;
                }
            }
            else
            {
                for ( ; pM < pM_end ; pM++)
                {
                    int64_t work = 1 ;
                    #if !defined ( GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED )
                    // if M is structural, no need to check its values
                    if (GB_MCAST (Mx, pM, msize))
                    #endif
                    { 
                        const int64_t i = Mi [pM] ;
                        #if GB_A_IS_HYPER
                        // A is hyper: find A(:,i) using the A->Y hyper hash
                        int64_t pA, pA_end ;
                        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx,
                            A_hash_bits, i, &pA, &pA_end) ;
                        const int64_t ainz = pA_end - pA ;
                        work += GB_IMIN (ainz, bjnz) ;
                        #elif GB_A_IS_SPARSE
                        // A is sparse
                        const int64_t pA = Ap [i] ;
                        const int64_t pA_end = Ap [i+1] ;
                        const int64_t ainz = pA_end - pA ;
                        work += GB_IMIN (ainz, bjnz) ;
                        #else
                        // A is bitmap or full
                        work += bjnz ;
                        #endif
                    }
                    Cwork [pM] = work ;
                }
            }
        }
    }
}

