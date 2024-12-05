//------------------------------------------------------------------------------
// GB_bitmap_assign_M_col_template:  traverse M for GB_COL_ASSIGN
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is a (C->vlen)-by-1 hypersparse or sparse matrix, for
// GrB_Row_assign (if C is CSR) or GrB_Col_assign (if C is CSC).

// C is bitmap/full.  M is sparse/hyper, and can be jumbled.

{
    ASSERT (GB_IS_BITMAP (C) || GB_IS_FULL (C)) ;
    ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;

    const int64_t *restrict kfirst_Mslice = M_ek_slicing ;
    const int64_t *restrict klast_Mslice  = M_ek_slicing + M_ntasks ;
    const int64_t *restrict pstart_Mslice = M_ek_slicing + M_ntasks * 2 ;

    int64_t jC = J [0] ;
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

            ASSERT (k == 0) ;
            ASSERT (GBH_M (Mh, k) == 0) ;
            GB_GET_PA (pM_start, pM_end, tid, k, kfirst, klast, pstart_Mslice,
                Mp [k], Mp [k+1]) ;

            //------------------------------------------------------------------
            // traverse over M(:,0), the kth vector of M
            //------------------------------------------------------------------

            // for col_assign: M is a single vector, jC = J [0]
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

