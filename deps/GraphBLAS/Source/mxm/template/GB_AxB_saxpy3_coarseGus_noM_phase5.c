//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseGus_noM_phase5: numeric coarse Gustavson, no mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {

        //----------------------------------------------------------------------
        // get C(:,j) and B(:,j)
        //----------------------------------------------------------------------

        int64_t pC = Cp [kk] ;
        int64_t cjnz = Cp [kk+1] - pC ;
        if (cjnz == 0) continue ;           // no work to do if C(:,j) empty
        GB_GET_B_j ;

        //----------------------------------------------------------------------
        // special case when C (:,j) is dense
        //----------------------------------------------------------------------

        #ifndef GB_GENERIC
        if (cjnz == cvlen)          // C(:,j) is dense
        { 
            // This is not used for the generic saxpy3.
            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
            continue ;
        }
        #endif

        //----------------------------------------------------------------------
        // C(:,j) = A*B(:,j)
        //----------------------------------------------------------------------

        mark++ ;
        if (bjnz == 1 && (A_is_sparse || A_is_hyper))
        { 

            //------------------------------------------------------------------
            // C(:,j) = A(:,k)*B(k,j) where B(:,j) has a single entry
            //------------------------------------------------------------------

            GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;

        }
        else if (16 * cjnz > cvlen)
        {

            //------------------------------------------------------------------
            // C(:,j) is not very sparse
            //------------------------------------------------------------------

            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
            {
                GB_GET_B_kj_INDEX ;             // get index k of entry B(k,j)
                GB_GET_A_k ;                    // get A(:,k)
                if (aknz == 0) continue ;       // skip if A(:,k) is empty
                GB_GET_B_kj ;                   // bkj = B(k,j)
                // scan A(:,k)
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                {
                    GB_GET_A_ik_INDEX ;         // get index i of entry A(i,k)
                    GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                    if (Hf [i] != mark)
                    { 
                        // C(i,j) = A(i,k) * B(k,j)
                        Hf [i] = mark ;
                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                    }
                    else
                    { 
                        // C(i,j) += A(i,k) * B(k,j)
                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                    }
                }
            }
            GB_GATHER_ALL_C_j (mark) ;          // gather into C(:,j) 

        }
        else
        {

            //------------------------------------------------------------------
            // C(:,j) is very sparse
            //------------------------------------------------------------------

            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
            {
                GB_GET_B_kj_INDEX ;             // get index k of entry B(k,j)
                GB_GET_A_k ;                    // get A(:,k)
                if (aknz == 0) continue ;       // skip if A(:,k) is empty
                GB_GET_B_kj ;                   // bkj = B(k,j)
                // scan A(:,k)
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                {
                    GB_GET_A_ik_INDEX ;         // get index i of entry A(i,k)
                    GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                    if (Hf [i] != mark)
                    { 
                        // C(i,j) = A(i,k) * B(k,j)
                        Hf [i] = mark ;
                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                        Ci [pC++] = i ;
                    }
                    else
                    { 
                        // C(i,j) += A(i,k) * B(k,j)
                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                    }
                }
            }
            GB_SORT_AND_GATHER_C_j ;            // gather into C(:,j)
        }
    }
}

