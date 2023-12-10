//------------------------------------------------------------------------------
// GB_AxB_dot3_template: C<M>=A'*B via dot products, where C is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and M are both sparse or both hyper, and C->h is a copy of M->h.
// M is present, and not complemented.  It may be valued or structural.

{

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [tid].kfirst ;
        int64_t klast  = TaskList [tid].klast ;
        int64_t pC_first = TaskList [tid].pC ;
        int64_t pC_last  = TaskList [tid].pC_end ;
        int64_t task_nzombies = 0 ;     // # of zombies found by this task

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get C(:,k) and M(:k)
            //------------------------------------------------------------------

            #if defined ( GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED )
            // M and C are both sparse
            const int64_t j = k ;
            #else
            // M and C are either both sparse or both hypersparse
            const int64_t j = GBH_C (Ch, k) ;
            #endif

            int64_t pC_start = Cp [k] ;
            int64_t pC_end   = Cp [k+1] ;
            if (k == kfirst)
            { 
                // First vector for task; may only be partially owned.
                pC_start = pC_first ;
                pC_end   = GB_IMIN (pC_end, pC_last) ;
            }
            else if (k == klast)
            { 
                // Last vector for task; may only be partially owned.
                pC_end   = pC_last ;
            }
            else
            { 
                // task completely owns this vector C(:,k).
            }

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
            #endif

            #if (GB_B_IS_SPARSE || GB_B_IS_HYPER)
                const int64_t bjnz = pB_end - pB_start ;
                if (bjnz == 0)
                {
                    // no work to do if B(:,j) is empty, except for zombies
                    task_nzombies += (pC_end - pC_start) ;
                    for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                    { 
                        // C(i,j) is a zombie
                        int64_t i = Mi [pC] ;
                        Ci [pC] = GB_FLIP (i) ;
                    }
                    continue ;
                }
                #if (GB_A_IS_SPARSE || GB_A_IS_HYPER)
                    // Both A and B are sparse; get first and last in B(:,j)
                    const int64_t ib_first = Bi [pB_start] ;
                    const int64_t ib_last  = Bi [pB_end-1] ;
                #endif
            #endif

            //------------------------------------------------------------------
            // C(:,j)<M(:,j)> = A(:,i)'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            {

                //--------------------------------------------------------------
                // get C(i,j) and M(i,j)
                //--------------------------------------------------------------

                bool cij_exists = false ;
                GB_CIJ_DECLARE (cij) ;
                #if GB_IS_PLUS_PAIR_REAL_SEMIRING
                cij = 0 ;
                #endif

                // get the value of M(i,j)
                int64_t i = Mi [pC] ;
                #if !defined ( GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED )
                // if M is structural, no need to check its values
                if (GB_MCAST (Mx, pC, msize))
                #endif
                { 

                    //----------------------------------------------------------
                    // the mask allows C(i,j) to be computed
                    //----------------------------------------------------------

                    #if GB_A_IS_HYPER
                    // A is hyper: find A(:,i) using the A->Y hyper hash
                    int64_t pA, pA_end ;
                    GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx,
                        A_hash_bits, i, &pA, &pA_end) ;
                    const int64_t ainz = pA_end - pA ;
                    if (ainz > 0)
                    #elif GB_A_IS_SPARSE
                    // A is sparse
                    int64_t pA = Ap [i] ;
                    const int64_t pA_end = Ap [i+1] ;
                    const int64_t ainz = pA_end - pA ;
                    if (ainz > 0)
                    #else
                    // A is bitmap or full
                    const int64_t pA = i * vlen ;
                    #endif
                    { 
                        // C(i,j) = A(:,i)'*B(:,j)
                        #include "GB_AxB_dot_cij.c"
                    }
                }

                if (!GB_CIJ_EXISTS)
                { 
                    // C(i,j) is a zombie
                    task_nzombies++ ;
                    Ci [pC] = GB_FLIP (i) ;
                }
            }
        }
        nzombies += task_nzombies ;
    }
}

#undef GB_A_IS_SPARSE
#undef GB_A_IS_HYPER
#undef GB_A_IS_BITMAP
#undef GB_A_IS_FULL
#undef GB_B_IS_SPARSE
#undef GB_B_IS_HYPER
#undef GB_B_IS_BITMAP
#undef GB_B_IS_FULL

