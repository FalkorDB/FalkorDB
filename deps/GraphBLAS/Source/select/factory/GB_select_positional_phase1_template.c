//------------------------------------------------------------------------------
// GB_select_positional_phase1_template: count entries for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse, hypersparse, or full (just for DIAG case)

{

    //==========================================================================
    // positional op (tril, triu, diag, offdiag, row*, but not col*)
    //==========================================================================

    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)
        || (opcode == GB_DIAG_idxunop_code)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;

    //--------------------------------------------------------------------------
    // tril, triu, diag, offdiag, row*: binary search in each vector
    //--------------------------------------------------------------------------

    int64_t k ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(guided)
    for (k = 0 ; k < anvec ; k++)
    {

        //----------------------------------------------------------------------
        // get A(:,k)
        //----------------------------------------------------------------------

        int64_t pA_start = GBp_A (Ap, k, avlen) ;
        int64_t pA_end   = GBp_A (Ap, k+1, avlen) ;
        int64_t p = pA_start ;
        int64_t cjnz = 0 ;
        int64_t ajnz = pA_end - pA_start ;
        bool found = false ;

        if (ajnz > 0)
        {

            //------------------------------------------------------------------
            // search for the entry A(i,k)
            //------------------------------------------------------------------

            int64_t ifirst = GBi_A (Ai, pA_start, avlen) ;
            int64_t ilast  = GBi_A (Ai, pA_end-1, avlen) ;

            #if defined ( GB_ROWINDEX_SELECTOR )
            int64_t i = -ithunk ;
            #elif defined ( GB_ROWLE_SELECTOR ) || defined ( GB_ROWGT_SELECTOR )
            int64_t i = ithunk ;
            #else
            // TRIL, TRIU, DIAG, OFFDIAG
            int64_t j = GBh_A (Ah, k) ;
            int64_t i = j-ithunk ;
            #endif

            if (i < ifirst)
            { 
                // all entries in A(:,k) come after i
                ;
            }
            else if (i > ilast)
            { 
                // all entries in A(:,k) come before i
                p = pA_end ;
            }
            else if (ajnz == avlen)
            { 
                // A(:,k) is dense (either A is full, or has a dense vector)
                found = true ;
                p += i ;
                ASSERT (GBi_A (Ai, p, avlen) == i) ;
            }
            else
            { 
                // binary search in A(:,k) for A (i,k); sparse/hyper case only
                int64_t pright = pA_end - 1 ;
                ASSERT (Ai != NULL) ;
                found = GB_split_binary_search (i, Ai, Ai_is_32, &p, &pright) ;
            }

            #if defined ( GB_TRIL_SELECTOR )

                // keep p to pA_end-1
                cjnz = pA_end - p ;

            #elif defined ( GB_ROWGT_SELECTOR  )

                // if found, keep p+1 to pA_end-1
                // else keep p to pA_end-1
                if (found)
                { 
                    p++ ;
                    // now in both cases, keep p to pA_end-1
                }
                // keep p to pA_end-1
                cjnz = pA_end - p ;

            #elif defined ( GB_TRIU_SELECTOR   ) \
               || defined ( GB_ROWLE_SELECTOR  )

                // if found, keep pA_start to p
                // else keep pA_start to p-1
                if (found)
                { 
                    p++ ;
                    // now in both cases, keep pA_start to p-1
                }
                // keep pA_start to p-1
                cjnz = p - pA_start ;

            #elif defined ( GB_DIAG_SELECTOR )

                // if found, keep p
                // else keep nothing
                cjnz = found ;
                if (!found) p = -1 ;
                // if (cjnz >= 0) keep p, else keep nothing

            #elif defined ( GB_OFFDIAG_SELECTOR  ) || \
                  defined ( GB_ROWINDEX_SELECTOR )

                // if found, keep pA_start to p-1 and p+1 to pA_end-1
                // else keep pA_start to pA_end
                cjnz = ajnz - found ;
                if (!found)
                { 
                    p = pA_end ;
                    // now just keep pA_start to p-1; p+1 to pA_end is 
                    // now empty
                }
                // in both cases, keep pA_start to p-1 and
                // p+1 to pA_end-1.  If the entry is not found, then
                // p == pA_end, and all entries are kept.

            #endif
        }

        //----------------------------------------------------------------------
        // log the result for the kth vector
        //----------------------------------------------------------------------

        GB_ISET (Zp, k, p) ;        // Zp [k] = p ;
        GB_ISET (Cp, k, cjnz) ;     // Cp [k] = cjnz ;
    }

    //--------------------------------------------------------------------------
    // compute Wfirst and Wlast for each task
    //--------------------------------------------------------------------------

    // Wfirst [0..A_ntasks-1] and Wlast [0..A_ntasks-1] are required for
    // constructing Cp_kfirst [0..A_ntasks-1] in GB_select_sparse.

    for (int tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        Wfirst [tid] = 0 ;
        Wlast  [tid] = 0 ;

        if (kfirst <= klast)
        {
            int64_t pA_start = pstart_Aslice [tid] ;
            int64_t pA_end   = GBp_A (Ap, kfirst+1, avlen) ;
            pA_end = GB_IMIN (pA_end, pstart_Aslice [tid+1]) ;
            int64_t pz = GB_IGET (Zp, kfirst) ;
            if (pA_start < pA_end)
            { 
                #if defined ( GB_TRIL_SELECTOR  ) || \
                    defined ( GB_ROWGT_SELECTOR )

                    // keep Zp [kfirst] to pA_end-1
                    int64_t p = GB_IMAX (pz, pA_start) ;
                    Wfirst [tid] = GB_IMAX (0, pA_end - p) ;

                #elif defined ( GB_TRIU_SELECTOR  ) || \
                      defined ( GB_ROWLE_SELECTOR )

                    // keep pA_start to Zp [kfirst]-1
                    int64_t p = GB_IMIN (pz, pA_end) ;
                    Wfirst [tid] = GB_IMAX (0, p - pA_start) ;

                #elif defined ( GB_DIAG_SELECTOR )

                    // task that owns the diagonal entry does this work
                    int64_t p = pz ;
                    Wfirst [tid] = (pA_start <= p && p < pA_end) ? 1 : 0 ;

                #elif defined ( GB_OFFDIAG_SELECTOR  ) || \
                      defined ( GB_ROWINDEX_SELECTOR )

                    // keep pA_start to Zp [kfirst]-1
                    int64_t p = GB_IMIN (pz, pA_end) ;
                    Wfirst [tid] = GB_IMAX (0, p - pA_start) ;

                    // keep Zp [kfirst]+1 to pA_end-1
                    p = GB_IMAX (pz+1, pA_start) ;
                    Wfirst [tid] += GB_IMAX (0, pA_end - p) ;

                #endif
            }
        }

        if (kfirst < klast)
        {
            int64_t pA_start = GBp_A (Ap, klast, avlen) ;
            int64_t pA_end   = pstart_Aslice [tid+1] ;
            int64_t pz = GB_IGET (Zp, klast) ;
            if (pA_start < pA_end)
            { 
                #if defined ( GB_TRIL_SELECTOR  ) || \
                    defined ( GB_ROWGT_SELECTOR )

                    // keep Zp [klast] to pA_end-1
                    int64_t p = GB_IMAX (pz, pA_start) ;
                    Wlast [tid] = GB_IMAX (0, pA_end - p) ;

                #elif defined ( GB_TRIU_SELECTOR  ) || \
                      defined ( GB_ROWLE_SELECTOR )

                    // keep pA_start to Zp [klast]-1
                    int64_t p = GB_IMIN (pz, pA_end) ;
                    Wlast [tid] = GB_IMAX (0, p - pA_start) ;

                #elif defined ( GB_DIAG_SELECTOR )

                    // task that owns the diagonal entry does this work
                    int64_t p = pz ;
                    Wlast [tid] = (pA_start <= p && p < pA_end) ? 1 : 0 ;

                #elif defined ( GB_OFFDIAG_SELECTOR  ) || \
                      defined ( GB_ROWINDEX_SELECTOR )

                    // keep pA_start to Zp [klast]-1
                    int64_t p = GB_IMIN (pz, pA_end) ;
                    Wlast [tid] = GB_IMAX (0, p - pA_start) ;

                    // keep Zp [klast]+1 to pA_end-1
                    p = GB_IMAX (pz+1, pA_start) ;
                    Wlast [tid] += GB_IMAX (0, pA_end - p) ;

                #endif
            }
        }
    }
}

#undef GB_TRIL_SELECTOR
#undef GB_TRIU_SELECTOR
#undef GB_DIAG_SELECTOR
#undef GB_OFFDIAG_SELECTOR
#undef GB_ROWINDEX_SELECTOR
#undef GB_ROWLE_SELECTOR
#undef GB_ROWGT_SELECTOR

