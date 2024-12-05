//------------------------------------------------------------------------------
// GB_emult_bitmap_5: C = A.*B; C is bitmap, M is not present.
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A and B are bitmap or full.  M not present.

{

    //--------------------------------------------------------------------------
    // Method5: C is bitmap, M is not present
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C       =           A       .*      B
    //      ------------------------------------------
    //      bitmap  .           bitmap          bitmap  (method: 5)
    //      bitmap  .           bitmap          full    (method: 5)
    //      bitmap  .           full            bitmap  (method: 5)

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < C_nthreads ; tid++)
    {
        int64_t pstart, pend, task_cnvals = 0 ;
        GB_PARTITION (pstart, pend, cnz, tid, C_nthreads) ;
        for (int64_t p = pstart ; p < pend ; p++)
        {
            if (GBB_A (Ab, p) && GBB_B (Bb,p))
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
        cnvals += task_cnvals ;
    }
}

