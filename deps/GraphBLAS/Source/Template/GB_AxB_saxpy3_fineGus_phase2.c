//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineGus_phase2: C=A*B, saxpy3 method, fine Gustavson, phase2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine Gustavson task, C(:,j)=A*B(:,j)
    //--------------------------------------------------------------------------

    // Hf [i] is initially 0.
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
                // C(i,j) += t ; ANY monoids with atomic write
                //--------------------------------------------------------------
            
                // the double complex ANY monoid cannot use this method

                GB_ATOMIC_READ
                f = Hf [i] ;            // grab the entry
                if (f == 2) continue ;  // check if already updated
                GB_Z_ATOMIC_WRITE_HX (i, t) ;   // Hx [i] = t

            #else

                //--------------------------------------------------------------
                // C(i,j) += t ; with all other monoids
                //--------------------------------------------------------------

                #if GB_Z_HAS_ATOMIC_UPDATE

                    // if C(i,j) is already present (f==2), and the monoid can
                    // be done atomically, then do the atomic update.  No need
                    // to modify Hf [i].
                    GB_ATOMIC_READ
                    f = Hf [i] ;        // grab the entry
                    if (f == 2)         // if true, update C(i,j)
                    { 
                        GB_Z_ATOMIC_UPDATE_HX (i, t) ;  // Hx [i] += t
                        continue ;      // C(i,j) has been updated
                    }

                #endif

                do  // lock the entry
                { 
                    // do this atomically:
                    // { f = Hf [i] ; Hf [i] = 3 ; }
                    GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                } while (f == 3) ; // lock owner gets f=0 or 2
                if (f == 0)
                { 
                    // C(i,j) is a new entry
                    GB_Z_ATOMIC_WRITE_HX (i, t) ;   // Hx [i] = t
                }
                else // f == 2
                { 
                    // C(i,j) already appears in C(:,j)
                    GB_Z_ATOMIC_UPDATE_HX (i, t) ;  // Hx [i] += t
                }

            #endif

            GB_ATOMIC_WRITE
            Hf [i] = 2 ;            // flag/unlock the entry
        }
    }
}

