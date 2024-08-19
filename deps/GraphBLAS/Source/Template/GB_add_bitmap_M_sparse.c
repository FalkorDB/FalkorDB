//------------------------------------------------------------------------------
// GB_add_bitmap_M_sparse: C<!M>=A+B, C bitmap, M sparse/hyper and comp.
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// M is sparse/hyper and complemented.
// A and B can have any format, except at least one is bitmap/full.

{ 

    //--------------------------------------------------------------------------
    // C is bitmap, M is sparse or hyper and complemented
    //--------------------------------------------------------------------------

    //      ------------------------------------------
    //      C     <!M> =        A       +       B
    //      ------------------------------------------
    //      bitmap  sparse      sparse          bitmap
    //      bitmap  sparse      sparse          full  
    //      bitmap  sparse      bitmap          sparse
    //      bitmap  sparse      bitmap          bitmap
    //      bitmap  sparse      bitmap          full  
    //      bitmap  sparse      full            sparse
    //      bitmap  sparse      full            bitmap
    //      bitmap  sparse      full            full  

    // M is sparse and complemented.  If M is sparse and not complemented, then
    // C is constructed as sparse, not bitmap.

    ASSERT (Mask_comp) ;

    // C(i,j) = A(i,j) + B(i,j) can only be computed where M(i,j) is not
    // present in the sparse pattern of M, and where it is present but equal to
    // zero.

    //--------------------------------------------------------------------------
    // scatter M into the C bitmap
    //--------------------------------------------------------------------------

    const int64_t *kfirst_Mslice = M_ek_slicing ;
    const int64_t *klast_Mslice  = M_ek_slicing + M_ntasks ;
    const int64_t *pstart_Mslice = M_ek_slicing + M_ntasks*2 ;

    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < M_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Mslice [taskid] ;
        int64_t klast  = klast_Mslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of M(:,k) for this task
            int64_t j = GBH_M (Mh, k) ;
            GB_GET_PA (pM_start, pM_end, taskid, k, kfirst, klast,
                pstart_Mslice, GBP_M (Mp, k, vlen), GBP_M (Mp, k+1, vlen)) ;
            int64_t pC_start = j * vlen ;
            // traverse over M(:,j), the kth vector of M
            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                // mark C(i,j) if M(i,j) is true
                bool mij = GB_MCAST (Mx, pM, msize) ;
                if (mij)
                { 
                    int64_t i = Mi [pM] ;
                    int64_t p = pC_start + i ;
                    Cb [p] = 2 ;
                }
            }
        }
    }

    // C(i,j) has been marked, in Cb, with the value 2 where M(i,j)=1.
    // These positions will not be computed in C(i,j).  C(i,j) can only
    // be modified where Cb [p] is zero.

    //--------------------------------------------------------------------------
    // compute C<!M>=A+B using the mask scattered in C
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL

        #if (GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL)
        {
            // A and B are both bitmap/full
            #include "GB_add_bitmap_M_sparse_24.c"
            #define M_cleared true
        }
        #elif (GB_A_IS_BITMAP || GB_A_IS_FULL)
        {
            // A is bitmap/full, B is sparse/hyper
            #include "GB_add_bitmap_M_sparse_25.c"
            #define M_cleared false
        }
        #else
        {
            // A is sparse/hyper, B is bitmap/full
            #include "GB_add_bitmap_M_sparse_26.c"
            #define M_cleared false
        }
        #endif

    #else

        bool M_cleared = false ;
        if ((A_is_bitmap || A_is_full) && (B_is_bitmap || B_is_full))
        { 
            // A and B are both bitmap/full
            #include "GB_add_bitmap_M_sparse_24.c"
            M_cleared = true ;      // M has also been cleared from C
        }
        else if (A_is_bitmap || A_is_full)
        { 
            // A is bitmap/full, B is sparse/hyper
            #include "GB_add_bitmap_M_sparse_25.c"
        }
        else
        { 
            // A is sparse/hyper, B is bitmap/full
            #include "GB_add_bitmap_M_sparse_26.c"
        }

    #endif

    //-------------------------------------------------------------------------
    // clear M from C
    //-------------------------------------------------------------------------

    if (!M_cleared)
    {
        // This step is required if either A or B are sparse/hyper (if
        // one is sparse/hyper, the other must be bitmap).  It requires
        // an extra pass over the mask M, so this might be slower than
        // postponing the application of the mask, and doing it later.

        #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < M_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Mslice [taskid] ;
            int64_t klast  = klast_Mslice  [taskid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // find the part of M(:,k) for this task
                int64_t j = GBH_M (Mh, k) ;
                GB_GET_PA (pM_start, pM_end, taskid, k, kfirst, klast,
                    pstart_Mslice, GBP_M (Mp, k, vlen), GBP_M (Mp, k+1, vlen)) ;
                int64_t pC_start = j * vlen ;
                // traverse over M(:,j), the kth vector of M
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    // mark C(i,j) if M(i,j) is true
                    bool mij = GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    { 
                        int64_t i = Mi [pM] ;
                        int64_t p = pC_start + i ;
                        Cb [p] = 0 ;
                    }
                }
            }
        }
    }
}

