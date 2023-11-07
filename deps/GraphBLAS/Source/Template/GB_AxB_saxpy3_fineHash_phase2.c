//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineHash_phase2: C=A*B (or with M in-place), fine Hash, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine hash task, C(:,j)=A*B(:,j)
    //--------------------------------------------------------------------------

    // Given Hf [hash] split into (h,f)

    // h == 0  , f == 0 : unlocked and unoccupied.
    // h == i+1, f == 2 : unlocked, occupied by C(i,j).  Hx is initialized.
    // h == ..., f == 3 : locked.

    // 0 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    // The mask M can be optionally checked, if it is dense and checked in
    // place.  This method is not used if M is present and sparse.

    if (team_size == 1)
    {

        //----------------------------------------------------------------------
        // single-threaded version
        //----------------------------------------------------------------------

        // the hash state 3 is not used, since only a single thread is
        // doing all the work for this vector C(:,j).

        // 0: if i seen for first time
        // 2: if i seen already

        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of A(i,j)
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i_unlocked = ((i+1) << 2) + 2 ;    // (i+1,2)
                // find the entry i in the hash table
                bool hf_unlocked = false ;  // true if i found
                bool hf_empty = false ;     // true if empty slot found
                int64_t hash ;
                for (hash = GB_HASHF (i, hash_bits) ; ;
                    GB_REHASH (hash,i,hash_bits))
                { 
                    int64_t hf = Hf [hash] ;    // grab the entry
                    hf_unlocked = (hf == i_unlocked) ;
                    hf_empty = (hf == 0) ;
                    if (hf_unlocked || hf_empty) break ;
                }
                if (hf_unlocked)    // if true, update C(i,j)
                { 
                    // hash entry occuppied by C(i,j): update it
                    GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
                }
                else // hf_empty:   if true, load the hash entry with C(i,j)
                { 
                    // hash entry unoccuppied: fill it with C(i,j)
                    ASSERT (hf_empty) ;
                    GB_HX_WRITE (hash, t) ;     // Hx [hash] = t
                    Hf [hash] = i_unlocked ;    // unlock entry
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // multi-threaded version
        //----------------------------------------------------------------------

        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                GB_GET_A_ik_INDEX ;     // get index i of A(i,j)
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i1 = i + 1 ;        // i1 = one-based index
                int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                for (GB_HASH (i))           // find i in hash table
                {
                    int64_t hf ;
                    GB_ATOMIC_READ
                    hf = Hf [hash] ;        // grab the entry
                    #if GB_Z_HAS_ATOMIC_UPDATE
                    if (hf == i_unlocked)  // if true, update C(i,j)
                    { 
                        GB_Z_ATOMIC_UPDATE_HX (hash, t) ;   // Hx [hash] += t
                        break ;         // C(i,j) has been updated
                    }
                    #endif
                    int64_t h = (hf >> 2) ;
                    if (h == 0 || h == i1)
                    {
                        // h=0: unoccupied, h=i1: occupied by i
                        do  // lock the entry
                        { 
                            // do this atomically:
                            // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                            GB_ATOMIC_CAPTURE_INT64_OR (hf, Hf [hash], 3) ;
                        } while ((hf & 3) == 3) ; // owner: f=0 or 2

                        if (hf == 0) // f == 0
                        { 
                            // C(i,j) is a new entry in C(:,j)
                            // Hx [hash] = t
                            GB_Z_ATOMIC_WRITE_HX (hash, t) ;
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }
                        if (hf == i_unlocked) // f == 2
                        { 
                            // C(i,j) already appears in C(:,j)
                            GB_Z_ATOMIC_UPDATE_HX (hash, t) ;  // Hx [hash] += t
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }

                        // hash table occupied, but not with i
                        GB_ATOMIC_WRITE
                        Hf [hash] = hf ;  // unlock with prior value
                    }
                }
            }
        }
    }
}

