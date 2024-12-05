//------------------------------------------------------------------------------
// GB_bitmap_assign_M_all_template:  traverse M for GB_ASSIGN
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is sparse or hypersparse, not bitmap or full.  C<M>(I,J) = ... is being
// computed (or !M), and all entries in M are traversed.  For a given entry
// M (iM,jM) in the mask, the entry C(iM,jM) is accessed at location pC.

// C is bitmap/full.  M is sparse/hyper, and can be jumbled.

{
    ASSERT (GB_IS_BITMAP (C) || GB_IS_FULL (C)) ;
    ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;

    const int64_t *restrict kfirst_Mslice = M_ek_slicing ;
    const int64_t *restrict klast_Mslice  = M_ek_slicing + M_ntasks ;
    const int64_t *restrict pstart_Mslice = M_ek_slicing + M_ntasks * 2 ;

    int tid ;
    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < M_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Mslice [tid] ;
        int64_t klast  = klast_Mslice  [tid] ;
        int64_t task_cnvals = 0 ;

        //----------------------------------------------------------------------
        // traverse over M (:,kfirst:klast)
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of M(:,k) for this task
            //------------------------------------------------------------------

            int64_t jM = GBH_M (Mh, k) ;
            GB_GET_PA (pM_start, pM_end, tid, k, kfirst, klast, pstart_Mslice,
                Mp [k], Mp [k+1]) ;

            //------------------------------------------------------------------
            // traverse over M(:,jM), the kth vector of M
            //------------------------------------------------------------------

            // for assign: M is a matrix the same size as C
            int64_t jC = jM ;

            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                bool mij = GB_MCAST (Mx, pM, msize) ;
                if (mij)
                { 
                    int64_t iC = Mi [pM] ;
                    int64_t pC = iC + jC * Cvlen ;
                    GB_MASK_WORK (pC) ;
                }
            }
        }
        #ifndef GB_NO_CNVALS
        cnvals += task_cnvals ;
        #endif
    }
}

