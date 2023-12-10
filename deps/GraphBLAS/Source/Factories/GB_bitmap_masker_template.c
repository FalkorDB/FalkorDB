//------------------------------------------------------------------------------
// GB_bitmap_masker_template:  phase2 for R = masker (C, M, Z), R is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C<M>=Z or C<!M>=Z, returning the result in R, which is bitmap.
// The input matrix C is not modified.  Effectively, this computes R=C and then
// R<M>=Z or R<!M>=Z.  If the C_replace descriptor is enabled, then C has
// already been cleared, and is an empty (but non-NULL) matrix.

// phase2: computes R in a single pass

// C is sparse or hypersparse.  Z is bitmap or full.  R is bitmap.
// M has any sparsity structure.

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      full            bitmap

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

// FUTURE:: add special cases for C==Z, C==M, and Z==M aliases

{
    
    int64_t p, rnvals = 0 ;

    ASSERT (R_sparsity == GxB_BITMAP) ;
    ASSERT (C_is_sparse || C_is_hyper) ;
    ASSERT (Z_is_bitmap || Z_is_full) ;

    //--------------------------------------------------------------------------
    // scatter C into the R bitmap
    //--------------------------------------------------------------------------

    GB_C_NHELD (C_nnz_held) ;
    GB_SLICE_MATRIX_WORK (C, 8, C_nnz_held + C->nvec, C_nnz_held) ;

    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1) \
        reduction(+:rnvals)
    for (taskid = 0 ; taskid < C_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Cslice [taskid] ;
        int64_t klast  = klast_Cslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of C(:,k) for this task
            int64_t j = GBH_C (Ch, k) ;
            int64_t pC_start, pC_end ;
            GB_get_pA (&pC_start, &pC_end, taskid, k, kfirst,
                klast, pstart_Cslice, Cp, vlen) ;
            int64_t pR_start = j * vlen ;
            // traverse over C(:,j), the kth vector of C
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            { 
                // R(i,j) = C(i,j)
                int64_t i = Ci [pC] ;
                int64_t pR = pR_start + i ;
                Rb [pR] = 1 ;
                rnvals++ ;
                #ifndef GB_ISO_MASKER
                memcpy (Rx + (pR)*rsize, Cx + (C_iso? 0:(pC)*rsize), rsize) ;
                #endif
            }
        }
    }

    R->nvals = rnvals ;

    //--------------------------------------------------------------------------
    // R<M>=Z or R<!M>=Z
    //--------------------------------------------------------------------------

    if (M_is_sparse || M_is_hyper)
    {

        //----------------------------------------------------------------------
        // Method05: M is sparse or hypersparse, Z bitmap/full, R bitmap
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      full            bitmap

        ASSERT (Mask_comp) ;

        //----------------------------------------------------------------------
        // scatter M into the R bitmap
        //----------------------------------------------------------------------

        GB_M_NHELD (M_nnz_held) ;
        GB_SLICE_MATRIX_WORK (M, 8, M_nnz_held + M->nvec, M_nnz_held) ;

        #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < M_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Mslice [taskid] ;
            int64_t klast  = klast_Mslice  [taskid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // find the part of M(:,k) for this task
                int64_t j = GBH_M (Mh, k) ;
                int64_t pM_start, pM_end ;
                GB_get_pA (&pM_start, &pM_end, taskid, k, kfirst,
                    klast, pstart_Mslice, Mp, vlen) ;
                int64_t pR_start = j * vlen ;
                // traverse over M(:,j), the kth vector of M
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    // mark R(i,j) if M(i,j) is true
                    bool mij = GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    { 
                        int64_t i = Mi [pM] ;
                        int64_t p = pR_start + i ;
                        Rb [p] += 2 ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // R<!M>=Z, using M scattered into R
        //----------------------------------------------------------------------

        // Rb is marked as follows:
        //  0:  R(i,j) is not present, and M(i,j) is false
        //  1:  R(i,j) is present, and M(i,j) is false
        //  2:  R(i,j) is not present, and M(i,j) is true
        //  3:  R(i,j) is present, and M(i,j) is true

        // M is complemented, but shown uncomplemented in the table below since
        // that is how it is scattered into R.

        // Rb   R(i,j)  M(i,j)  Z(i,j)      modification to R(i,j)
        // 0    -       0       zij         R(i,j) = Z(i,j), new value, rnvals++
        // 0    -       0       -           do nothing
        // 1    rij     0       zij         R(i,j) = Z(i,j), overwrite
        // 1    rij     0       -           delete R(i,j), rnvals--
        // 2    -       1       zij         do nothing, set Rb to 0
        // 2    -       1       -           do nothing, set Rb to 0
        // 3    rij     1       zij         keep R(i,j), set Rb to 1
        // 3    rij     1       -           keep R(i,j), set Rb to 1

        #pragma omp parallel for num_threads(R_nthreads) schedule(static) \
            reduction(+:rnvals)
        for (p = 0 ; p < rnz ; p++)
        {
            int8_t r = Rb [p] ;
            int8_t z = GBB (Zb, p) ;
            switch (r)
            {
                case 0 :    // R(i,j) not present, M(i,j) false
                    if (z)
                    { 
                        // R(i,j) = Z(i,j), insert new value
                        #ifndef GB_ISO_MASKER
                        memcpy (Rx +(p)*rsize, Zx +(Z_iso? 0:(p)*rsize), rsize);
                        #endif
                        Rb [p] = 1 ;
                        rnvals++ ;
                    }
                    break ;

                case 1 :    // R(i,j) present, M(i,j) false
                    if (z)
                    { 
                        // R(i,j) = Z(i,j), update prior value
                        #ifndef GB_ISO_MASKER
                        memcpy (Rx +(p)*rsize, Zx +(Z_iso? 0:(p)*rsize), rsize);
                        #endif
                    }
                    else
                    { 
                        // delete R(i,j)
                        Rb [p] = 0 ;
                        rnvals-- ;
                    }
                    break ;

                case 2 :    // R(i,j) not present, M(i,j) true
                    Rb [p] = 0 ;
                    break ;

                case 3 :    // R(i,j) present, M(i,j) true
                    Rb [p] = 1 ;
                    break ;

                default: ;
            }
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // Method06: M and Z are bitmap or full, R is bitmap
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        // Rb   R(i,j)  M(i,j)  Z(i,j)      modification to R(i,j)

        // 0    -       0       zij         do nothing
        // 0    -       0       -           do nothing
        // 1    rij     0       zij         do nothing
        // 1    rij     0       -           do nothing

        // 0    -       1       zij         R(i,j) = Z(i,j), rnvals++
        // 0    -       1       -           do nothing
        // 1    rij     1       zij         R(i,j) = Z(i,j), no change to rnvals
        // 1    rij     1       -           delete, rnvals--

        #pragma omp parallel for num_threads(R_nthreads) schedule(static) \
            reduction(+:rnvals)
        for (p = 0 ; p < rnz ; p++)
        {
            bool mij = GBB_M (Mb, p) && GB_MCAST (Mx, p, msize) ;
            if (Mask_comp) mij = !mij ;
            if (mij)
            {
                int8_t z = GBB (Zb, p) ;
                int8_t r = Rb [p] ;
                if (r)
                {
                    if (z)
                    { 
                        // R(i,j) = Z(i,j), update, no change to rnvals
                        #ifndef GB_ISO_MASKER
                        memcpy (Rx +(p)*rsize, Zx +(Z_iso? 0:(p)*rsize), rsize);
                        #endif
                    }
                    else
                    { 
                        // delete R(i,j)
                        Rb [p] = 0 ;
                        rnvals-- ;
                    }
                }
                else if (z)
                { 
                    // R(i,j) = Z(i,j), new entry
                    #ifndef GB_ISO_MASKER
                    memcpy (Rx +(p)*rsize, Zx +(Z_iso? 0:(p)*rsize), rsize) ;
                    #endif
                    Rb [p] = 1 ;
                    rnvals++ ;
                }
            }
        }
    }

    R->nvals = rnvals ;
}

