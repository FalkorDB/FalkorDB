//------------------------------------------------------------------------------
// GB_AxB_dot2_template:  C<#M>=A'*B, or C<#M>=A*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A and B are sparse, bitmap, or full; never hypersparse.  If the input
// matrices A and/or B are hypersparse, they are converted into hyper_shallow
// sparse matrices, and C is converted from bitmap to sparse/hypersparse when
// done.

// If A_NOT_TRANSPOSED is #defined, then C=A*B or C<#M>=A*B is computed.
// In this case A is bitmap or full, and B is sparse.

// C is bitmap or full.

// GB_DOT_ALWAYS_SAVE_CIJ: C(i,j) = cij
#undef GB_DOT_ALWAYS_SAVE_CIJ
#if GB_C_IS_FULL
    #define GB_DOT_ALWAYS_SAVE_CIJ      \
    {                                   \
        /* Cx [pC] = cij */             \
        GB_PUTC (cij, Cx, pC) ;         \
    }
#else
    #define GB_DOT_ALWAYS_SAVE_CIJ      \
    {                                   \
        /* Cx [pC] = cij */             \
        GB_PUTC (cij, Cx, pC) ;         \
        Cb [pC] = 1 ;                   \
        task_cnvals++ ;                 \
    }
#endif

// GB_DOT_SAVE_CIJ: C(i,j) = cij, unless already done by GB_DOT
#undef GB_DOT_SAVE_CIJ
#if GB_IS_ANY_MONOID
    // for the ANY monoid, GB_DOT saves C(i,j) as soon as a value is found
    #define GB_DOT_SAVE_CIJ
#else
    // all other monoids: C(i,j) = cij if it exists
    #define GB_DOT_SAVE_CIJ             \
    {                                   \
        if (GB_CIJ_EXISTS)              \
        {                               \
            GB_DOT_ALWAYS_SAVE_CIJ ;    \
        }                               \
    }
#endif

#ifndef GB_NO_MASK
#error "mask undefined"
#endif

#if ( !GB_A_IS_HYPER && !GB_B_IS_HYPER )
{

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B where C is bitmap
    //--------------------------------------------------------------------------

    int tid ;
    #if GB_C_IS_FULL
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    #else
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    #endif
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;
        #if (!GB_C_IS_FULL)
        int64_t task_cnvals = 0 ;
        #endif

        //----------------------------------------------------------------------
        // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t j = kB_start ; j < kB_end ; j++)
        {

            //------------------------------------------------------------------
            // get C(:,j)
            //------------------------------------------------------------------

            const int64_t pC_start = j * cvlen ;

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_SPARSE
                // B is sparse (never hypersparse)
                const int64_t pB_start = Bp [j] ;
                const int64_t pB_end = Bp [j+1] ;
                const int64_t bjnz = pB_end - pB_start ;
                if (bjnz == 0)
                { 
                    // no work to do if B(:,j) is empty, except to clear Cb
                    memset (&Cb [pC_start + kA_start], 0, kA_end - kA_start) ;
                    continue ;
                }
                #if GB_A_IS_SPARSE
                    // Both A and B are sparse; get first and last in B(:,j)
                    const int64_t ib_first = Bi [pB_start] ;
                    const int64_t ib_last  = Bi [pB_end-1] ;
                #endif
            #else
                // B is bitmap or full
                const int64_t pB_start = j * vlen ;
            #endif

            //------------------------------------------------------------------
            // C(:,j)<#M(:,j)> = A'*B(:,j), or C(:,j) = A'*B(:,j) if no mask
            //------------------------------------------------------------------

            for (int64_t i = kA_start ; i < kA_end ; i++)
            {

                //--------------------------------------------------------------
                // get C(i,j), M(i,j), and clear the C(i,j) bitmap if C bitmap
                //--------------------------------------------------------------

                int64_t pC = pC_start + i ;     // get C(i,j) position

                #if GB_C_IS_FULL

                // C is full; nothing to do.  No mask is present
                ASSERT (GB_NO_MASK) ;

                #elif defined ( GB_ANY_SPECIALIZED )

                // the JIT kernel does not need to exploit this as a special
                // case, since Mask_comp, M_is_bitmap, etc, are all
                // compile-time constants.

                // special case: M is bitmap and structural; Mask_comp true
                Cb [pC] = 0 ;       // C is bitmap
                if (!Mb [pC])

                #elif ( !GB_NO_MASK )

                // C is bitmap
                // M is present: arbitrary sparsity and valued/structural
                bool mij ;
                if (M_is_bitmap)
                { 
                    // M is bitmap
                    mij = Mb [pC] && GB_MCAST (Mx, pC, msize) ;
                }
                else if (M_is_full)
                { 
                    // M is full
                    mij = GB_MCAST (Mx, pC, msize) ;
                }
                else // M is sparse or hyper
                { 
                    // M has been scattered into the C bitmap
                    mij = (Cb [pC] > 1) ;
                }
                Cb [pC] = 0 ;
                if (mij ^ Mask_comp)

                #else

                // C is bitmap; M is not present
                Cb [pC] = 0 ;

                #endif
                { 

                    //----------------------------------------------------------
                    // the mask allows C(i,j) to be computed
                    //----------------------------------------------------------

                    #if GB_A_IS_SPARSE
                        // A is sparse
                        int64_t pA = Ap [i] ;
                        const int64_t pA_end = Ap [i+1] ;
                        const int64_t ainz = pA_end - pA ;
                        #if (!GB_C_IS_FULL)
                        if (ainz > 0)       // skip this test if C is full
                        #endif
                    #else
                        // A is bitmap or full
                        #ifdef GB_A_NOT_TRANSPOSED
                        // A(i,:) starts at position i
                        const int64_t pA = i ;
                        #else
                        // A(:,i) starts at position i * vlen
                        const int64_t pA = i * vlen ;
                        #endif
                    #endif
                    { 
                        // C(i,j) = A(:,i)'*B(:,j) or A(i,:)*B(:,j)
                        bool cij_exists = false ;
                        GB_CIJ_DECLARE (cij) ;
                        #if GB_IS_PLUS_PAIR_REAL_SEMIRING
                        cij = 0 ;
                        #endif
                        #include "GB_AxB_dot_cij.c"
                    }
                }
            }
        }
        #if (!GB_C_IS_FULL)
        cnvals += task_cnvals ;
        #endif
    }
}
#endif

#undef GB_A_IS_SPARSE
#undef GB_A_IS_HYPER
#undef GB_A_IS_BITMAP
#undef GB_A_IS_FULL
#undef GB_B_IS_SPARSE
#undef GB_B_IS_HYPER
#undef GB_B_IS_BITMAP
#undef GB_B_IS_FULL
#undef GB_DOT_ALWAYS_SAVE_CIJ
#undef GB_DOT_SAVE_CIJ

