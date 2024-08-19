//------------------------------------------------------------------------------
// GB_bitmap_assign_M_sub_template:  traverse M for GB_SUBASSIGN
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is sparse or hypersparse, not bitmap or full.  C(I,J)<M>= ... is being
// computed (or !M), and all entries in M are traversed.  For a given entry
// M(iM,jM) in the mask, at location pM.  The entry C(iC,jC) is at location pC,
// where iC = I [iM] and jC = J [jM].  The matrix C is bitmap or full.

// C is bitmap/full.  M is sparse/hyper, and can be jumbled.

{
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
            int64_t pM_start, pM_end ;
            GB_get_pA (&pM_start, &pM_end, tid, k, kfirst,
                klast, pstart_Mslice, Mp, Mvlen) ;

            //------------------------------------------------------------------
            // traverse over M(:,jM), the kth vector of M
            //------------------------------------------------------------------

            // for subassign, M has same size as C(I,J) and A.
            int64_t jC = GB_ijlist (J, jM, Jkind, Jcolon) ;
            int64_t pC0 = jC * cvlen ;

            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                bool mij = GB_MCAST (Mx, pM, msize) ;
                if (mij)
                { 
                    int64_t iM = Mi [pM] ;
                    int64_t iC = GB_ijlist (I, iM, Ikind, Icolon) ;
                    int64_t pC = iC + pC0 ;
                    GB_MASK_WORK (pC) ;             // operate on Cx [pC]
                }
            }
        }
        cnvals += task_cnvals ;
    }
}

