//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineHash_M_phase2: C<M>=A*B, fine Hash, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    //--------------------------------------------------------------------------
    // phase2: fine hash task, C(:,j)<M(:,j)>=A*B(:,j)
    //--------------------------------------------------------------------------

    // M is sparse and scattered into Hf

    // Given Hf [hash] split into (h,f)

    // h == 0  , f == 0 : unlocked, unoccupied. C(i,j) ignored
    // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
    //                    C(i,j) has not been seen.
    //                    Hx is not initialized.
    // h == i+1, f == 2 : unlocked, occupied by C(i,j), M(i,j)=1
    //                    Hx is initialized.
    // h == ..., f == 3 : locked.

    // 0 -> 0 : to ignore, if M(i,j)=0
    // 1 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
    { 
        GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
        GB_GET_A_k ;                // get A(:,k)
        if (aknz == 0) continue ;
        GB_GET_B_kj ;               // bkj = B(k,j)
        #define GB_IKJ                                                        \
        {                                                                     \
            GB_MULT_A_ik_B_kj ;      /* t = A(i,k) * B(k,j) */                \
            int64_t i1 = i + 1 ;     /* i1 = one-based index */               \
            int64_t i_unlocked = (i1 << 2) + 2 ;  /* (i+1,2) */               \
            for (GB_HASH (i))        /* find i in hash table */               \
            {                                                                 \
                int64_t hf ;                                                  \
                GB_ATOMIC_READ                                                \
                hf = Hf [hash] ;        /* grab the entry */                  \
                if (GB_Z_HAS_ATOMIC_UPDATE && (hf == i_unlocked))             \
                {                                                             \
                    /* Hx [hash] += t */                                      \
                    GB_Z_ATOMIC_UPDATE_HX (hash, t) ;                         \
                    break ;     /* C(i,j) has been updated */                 \
                }                                                             \
                if (hf == 0) break ; /* M(i,j)=0; ignore Cij */               \
                if ((hf >> 2) == i1) /* if true, i found */                   \
                {                                                             \
                    do /* lock the entry */                                   \
                    {                                                         \
                        /* do this atomically: */                             \
                        /* { hf = Hf [hash] ; Hf [hash] |= 3 ; }*/            \
                        GB_ATOMIC_CAPTURE_INT64_OR (hf, Hf [hash], 3) ;       \
                    } while ((hf & 3) == 3) ; /* own: f=1,2 */                \
                    if ((hf & 3) == 1) /* f == 1 */                           \
                    {                                                         \
                        /* C(i,j) is a new entry in C(:,j) */                 \
                        GB_Z_ATOMIC_WRITE_HX (hash, t) ; /* Hx [hash] = t */  \
                    }                                                         \
                    else /* f == 2 */                                         \
                    {                                                         \
                        /* C(i,j) already appears in C(:,j) */                \
                        GB_Z_ATOMIC_UPDATE_HX (hash, t) ; /* Hx [hash] += t */\
                    }                                                         \
                    GB_ATOMIC_WRITE                                           \
                    Hf [hash] = i_unlocked ; /* unlock entry */               \
                    break ;                                                   \
                }                                                             \
            }                                                                 \
        }
        GB_SCAN_M_j_OR_A_k (A_ok_for_binary_search) ;
        #undef GB_IKJ
    }
}

