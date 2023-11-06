//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineGus_M_phase2: C<M>=A*B, fine Gustavson, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine Gustavson task, C(:,j)<M(:,j)>=A*B(:,j)
    //--------------------------------------------------------------------------

    // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
    // 0 -> 1 : has already been done in phase0 if M(i,j)=1.

    // 0 -> 0 : to ignore, if M(i,j)=0
    // 1 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    GB_GET_M_j ;                // get M(:,j)
    GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
    { 
        GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
        GB_GET_A_k ;                // get A(:,k)
        if (aknz == 0) continue ;
        GB_GET_B_kj ;               // bkj = B(k,j)

        #if GB_IS_ANY_MONOID && GB_Z_HAS_ATOMIC_WRITE

            //------------------------------------------------------------------
            // C(i,j) += A(i,k)*B(k,j) ; ANY monoids with atomic write
            //------------------------------------------------------------------

            // the double complex ANY monoid cannot use this method

            #define GB_IKJ                                              \
                int8_t f ;                                              \
                GB_ATOMIC_READ                                          \
                f = Hf [i] ;                /* grab the entry */        \
                if (f == 0 || f == 2) continue ;                        \
                GB_ATOMIC_WRITE                                         \
                Hf [i] = 2 ;                /* unlock the entry */      \
                GB_MULT_A_ik_B_kj ;         /* t = A(i,k) * B(k,j) */   \
                GB_Z_ATOMIC_WRITE_HX (i, t) ; /* Hx [i] = t */

        #else

            //------------------------------------------------------------------
            // C(i,j) += A(i,k)*B(k,j) ; all other monoids
            //------------------------------------------------------------------

            #define GB_IKJ                                              \
            {                                                           \
                GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */       \
                int8_t f ;                                              \
                GB_ATOMIC_READ                                          \
                f = Hf [i] ;            /* grab the entry */            \
                if (GB_Z_HAS_ATOMIC_UPDATE && (f == 2))                 \
                {                                                       \
                    /* C(i,j) already seen; update it */                \
                    GB_Z_ATOMIC_UPDATE_HX (i, t) ;    /* Hx [i] += t */ \
                    continue ;       /* C(i,j) has been updated */      \
                }                                                       \
                if (f == 0) continue ; /* M(i,j)=0; ignore C(i,j) */    \
                do  /* lock the entry */                                \
                {                                                       \
                    /* do this atomically: */                           \
                    /* { f = Hf [i] ; Hf [i] = 3 ; } */                 \
                    GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;             \
                } while (f == 3) ; /* lock owner gets f=1 or 2 */       \
                if (f == 1)                                             \
                {                                                       \
                    /* C(i,j) is a new entry */                         \
                    GB_Z_ATOMIC_WRITE_HX (i, t) ;     /* Hx [i] = t */  \
                }                                                       \
                else /* f == 2 */                                       \
                {                                                       \
                    /* C(i,j) already appears in C(:,j) */              \
                    GB_Z_ATOMIC_UPDATE_HX (i, t) ;    /* Hx [i] += t */ \
                }                                                       \
                GB_ATOMIC_WRITE                                         \
                Hf [i] = 2 ;                /* unlock the entry */      \
            }

        #endif

        GB_SCAN_M_j_OR_A_k (A_ok_for_binary_search) ;
        #undef GB_IKJ
    }
}

