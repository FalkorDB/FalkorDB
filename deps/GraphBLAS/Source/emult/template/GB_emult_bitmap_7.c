//------------------------------------------------------------------------------
// GB_emult_bitmap_7: C<#M>=A.*B; C bitmap; M, A, and B are bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  M, A, and B are bitmap or full.

{

    //--------------------------------------------------------------------------
    // Method7: C is bitmap; M is bitmap or full
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C      <M> =        A       .*      B
    //      ------------------------------------------
    //      bitmap  bitmap      bitmap          bitmap  (method: 7)
    //      bitmap  bitmap      bitmap          full    (method: 7)
    //      bitmap  bitmap      full            bitmap  (method: 7)

    //      ------------------------------------------
    //      C      <M> =        A       .*      B
    //      ------------------------------------------
    //      bitmap  full        bitmap          bitmap  (method: 7)
    //      bitmap  full        bitmap          full    (method: 7)
    //      bitmap  full        full            bitmap  (method: 7)

    //      ------------------------------------------
    //      C      <!M> =       A       .*      B
    //      ------------------------------------------
    //      bitmap  bitmap      bitmap          bitmap  (method: 7)
    //      bitmap  bitmap      bitmap          full    (method: 7)
    //      bitmap  bitmap      full            bitmap  (method: 7)

    //      ------------------------------------------
    //      C      <!M> =       A       .*      B
    //      ------------------------------------------
    //      bitmap  full        bitmap          bitmap  (method: 7)
    //      bitmap  full        bitmap          full    (method: 7)
    //      bitmap  full        full            bitmap  (method: 7)

    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;

    const int8_t  *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    size_t msize = M->type->size ;

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < C_nthreads ; tid++)
    {
        int64_t pstart, pend, task_cnvals = 0 ;
        GB_PARTITION (pstart, pend, cnz, tid, C_nthreads) ;
        for (int64_t p = pstart ; p < pend ; p++)
        {
            // get M(i,j)
            bool mij = GBB_M (Mb, p) && GB_MCAST (Mx, p, msize) ;
            if (Mask_comp) mij = !mij ; /* TODO: use ^ */
            if (mij)
            {
                // M(i,j) is true, so C(i,j) can be computed
                if (GBB_A (Ab, p) && GBB_B (Bb, p))
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, p, A_iso) ;
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, p, B_iso) ;
                    GB_EWISEOP (Cx, p, aij, bij, p % vlen, p / vlen) ;
                    #endif
                    Cb [p] = 1 ;
                    task_cnvals++ ;
                }
            }
            else
            { 
                // M(i,j) == 1, so C(i,j) is not computed
                Cb [p] = 0 ;
            }
        }
        cnvals += task_cnvals ;
    }
}

