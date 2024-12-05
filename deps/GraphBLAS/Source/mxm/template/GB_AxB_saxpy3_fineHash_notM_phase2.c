//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineHash_notM_phase2: C<!M>=A*B, fine Hash, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine hash task, C(:,j)<!M(:,j)>=A*B(:,j)
    //--------------------------------------------------------------------------

    // M(:,j) is sparse/hyper and scattered into Hf

    // Given Hf [hash] split into (h,f)

    // h == 0  , f == 0 : unlocked and unoccupied.
    // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
    //                    C(i,j) is ignored.
    // h == i+1, f == 2 : unlocked, occupied by C(i,j).
    //                    Hx is initialized.

    // h == (anything), f == 3: locked.

    // 1 -> 1 : to ignore, if M(i,j)=1
    // 0 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
    {
        GB_GET_B_kj_INDEX ;         // get index k of B(k,j)
        GB_GET_A_k ;                // get A(:,k)
        if (aknz == 0) continue ;
        GB_GET_B_kj ;               // bkj = B(k,j)

        // scan A(:,k)
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
        {
            GB_GET_A_ik_INDEX ;     // get index i of A(i,k)
            GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
            int64_t i1 = i + 1 ;        // i1 = one-based index
            int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
            int64_t i_masked   = (i1 << 2) + 1 ;    // (i+1,1)
            for (GB_HASH (i))           // find i in hash table
            {
                int64_t hf ;
                GB_ATOMIC_READ
                hf = Hf [hash] ;        // grab the entry
                #if GB_Z_HAS_ATOMIC_UPDATE
                {
                    if (hf == i_unlocked)  // if true, update C(i,j)
                    { 
                        GB_Z_ATOMIC_UPDATE_HX (hash, t) ;   // Hx [hash] += t
                        break ;         // C(i,j) has been updated
                    }
                }
                #endif
                if (hf == i_masked) break ; // M(i,j)=1; ignore
                int64_t h = (hf >> 2) ;
                if (h == 0 || h == i1)
                {
                    // h=0: unoccupied, h=i1: occupied by i
                    do // lock the entry
                    { 
                        // do this atomically:
                        // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                        GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3) ;
                    } while ((hf & 3) == 3) ; // owner: f=0,1,2
                    if (hf == 0)            // f == 0
                    { 
                        // C(i,j) is a new entry in C(:,j)
                        // Hx [hash] = t
                        GB_Z_ATOMIC_WRITE_HX (hash, t) ;
                        GB_ATOMIC_WRITE
                        Hf [hash] = i_unlocked ; // unlock entry
                        break ;
                    }
                    if (hf == i_unlocked)   // f == 2
                    { 
                        // C(i,j) already appears in C(:,j)
                        GB_Z_ATOMIC_UPDATE_HX (hash, t) ;   // Hx [hash] += t
                        GB_ATOMIC_WRITE
                        Hf [hash] = i_unlocked ; // unlock entry
                        break ;
                    }
                    // hash table occupied, but not with i,
                    // or with i but M(i,j)=1 so C(i,j) ignored
                    GB_ATOMIC_WRITE
                    Hf [hash] = hf ;  // unlock with prior value
                }
            }
        }
    }
}

