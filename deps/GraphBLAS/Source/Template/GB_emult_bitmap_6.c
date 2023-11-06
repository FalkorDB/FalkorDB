//------------------------------------------------------------------------------
// GB_emult_bitmap_6: C<!M>=A.*B, C bitmap, M sparse, A and B are bitmap/full.
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A and B are bitmap or full.  M is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // Method6: C is bitmap, !M is sparse or hyper
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C       <!M>=       A       .*      B
    //      ------------------------------------------
    //      bitmap  sparse      bitmap          bitmap  (method: 6)
    //      bitmap  sparse      bitmap          full    (method: 6)
    //      bitmap  sparse      full            bitmap  (method: 6)

    // M is sparse and complemented.  If M is sparse and not
    // complemented, then C is constructed as sparse, not bitmap.
    ASSERT (M != NULL) ;
    ASSERT (Mask_comp) ;
    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;

    // C(i,j) = A(i,j) .* B(i,j) can only be computed where M(i,j) is
    // not present in the sparse pattern of M, and where it is present
    // but equal to zero.

    //--------------------------------------------------------------------------
    // scatter M into the C bitmap
    //--------------------------------------------------------------------------

    GB_bitmap_M_scatter_whole (C, M, Mask_struct, GB_BITMAP_M_SCATTER_SET_2,
        M_ek_slicing, M_ntasks, M_nthreads) ;

    // C(i,j) has been marked, in Cb, with the value 2 where M(i,j)=1.
    // These positions will not be computed in C(i,j).  C(i,j) can only
    // be modified where Cb [p] is zero.

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < C_nthreads ; tid++)
    {
        int64_t pstart, pend, task_cnvals = 0 ;
        GB_PARTITION (pstart, pend, cnz, tid, C_nthreads) ;
        for (int64_t p = pstart ; p < pend ; p++)
        {
            if (Cb [p] == 0)
            {
                // M(i,j) is zero, so C(i,j) can be computed
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

