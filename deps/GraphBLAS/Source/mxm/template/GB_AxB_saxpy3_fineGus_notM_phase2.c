//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineGus_notM_phase2: C<!M>=A*B, fine Gustavson, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine Gustavson task, C(:,j)<!M(:,j)>=A*B(:,j)
    //--------------------------------------------------------------------------

    // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
    // 0 -> 1 : has already been done in phase0 if M(i,j)=1

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
            GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
            int8_t f ;

            #if GB_IS_ANY_MONOID && GB_Z_HAS_ATOMIC_WRITE

                //--------------------------------------------------------------
                // ANY monoid with atomic write
                //--------------------------------------------------------------

                // the double complex ANY monoid cannot use this method

                // lock state (3) not needed
                // 0: not seen: update with new value, f becomes 2
                // 1: masked, do nothing, f stays 1
                // 2: already updated, do nothing, f stays 2
                // 3: state not used, f can be 2
                GB_ATOMIC_READ
                f = Hf [i] ;
                if (!f)
                { 
                    GB_ATOMIC_WRITE
                    Hf [i] = 2 ;
                    GB_Z_ATOMIC_WRITE_HX (i, t) ;   // Hx [i] = t
                }


            #else

                //--------------------------------------------------------------
                // all other monoids
                //--------------------------------------------------------------

                GB_ATOMIC_READ
                f = Hf [i] ;            // grab the entry

                #if GB_Z_HAS_ATOMIC_UPDATE
                {
                    // the monoid can be done with a single atomic update
                    if (f == 2)             // if true, update C(i,j)
                    { 
                        GB_Z_ATOMIC_UPDATE_HX (i, t) ;  // Hx [i] += t
                        continue ;          // C(i,j) has been updated
                    }
                }
                #endif

                if (f == 1) continue ; // M(i,j)=1; ignore C(i,j)
                do  // lock the entry
                { 
                    // do this atomically:
                    // { f = Hf [i] ; Hf [i] = 3 ; }
                    GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                } while (f == 3) ; // lock owner of gets f=0 or 2
                if (f == 0)
                { 
                    // C(i,j) is a new entry
                    GB_Z_ATOMIC_WRITE_HX (i, t) ;   // Hx [i] = t
                }
                else // f == 2
                { 
                    // C(i,j) already seen
                    GB_Z_ATOMIC_UPDATE_HX (i, t) ;  // Hx [i] += t
                }
                GB_ATOMIC_WRITE
                Hf [i] = 2 ;                // unlock the entry

            #endif
        }
    }
}

