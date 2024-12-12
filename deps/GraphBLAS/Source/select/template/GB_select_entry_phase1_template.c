//------------------------------------------------------------------------------
// GB_select_entry_phase1_template: count entries for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    //--------------------------------------------------------------------------
    // get A and its slicing
    //--------------------------------------------------------------------------

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    int64_t avlen = A->vlen ;
    int64_t anvec = A->nvec ;

    //==========================================================================
    // entry selector
    //==========================================================================

    // The count of live entries kth vector A(:,k) is reduced to the kth scalar
    // Cp(k).  Each thread computes the reductions on roughly the same number
    // of entries, which means that a vector A(:,k) may be reduced by more than
    // one thread.  The first vector A(:,kfirst) reduced by thread tid may be
    // partial, where the prior thread tid-1 (and other prior threads) may also
    // do some of the reductions for this same vector A(:,kfirst).  The thread
    // tid reduces all vectors A(:,k) for k in the range kfirst+1 to klast-1.
    // The last vector A(:,klast) reduced by thread tid may also be partial.
    // Thread tid+1, and following threads, may also do some of the reduces for
    // A(:,klast).

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    size_t  asize = A->type->size ;
    int64_t avdim = A->vdim ;

    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // reduce each slice
    //--------------------------------------------------------------------------

    // each thread reduces its own part in parallel
    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then thread tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        Wfirst [tid] = 0 ;
        Wlast  [tid] = 0 ;

        //----------------------------------------------------------------------
        // reduce vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be reduced by this thread
            //------------------------------------------------------------------

            int64_t j = GBH_A (Ah, k) ;
            GB_GET_PA (pA, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                Ap [k], Ap [k+1]) ;

            //------------------------------------------------------------------
            // count entries in Ax [pA ... pA_end-1]
            //------------------------------------------------------------------

            int64_t cjnz = 0 ;
            for ( ; pA < pA_end ; pA++)
            { 
                int64_t i = Ai [pA] ;
                GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
                if (keep) cjnz++ ;
            }
            if (k == kfirst)
            { 
                Wfirst [tid] = cjnz ;
            }
            else if (k == klast)
            { 
                Wlast [tid] = cjnz ;
            }
            else
            { 
                Cp [k] = cjnz ; 
            }
        }
    }

    //--------------------------------------------------------------------------
    // reduce the first and last vector of each slice using a single thread
    //--------------------------------------------------------------------------

    GB_ek_slice_merge1 (Cp, Wfirst, Wlast, A_ek_slicing, A_ntasks) ;
}

