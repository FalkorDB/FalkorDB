//------------------------------------------------------------------------------
// GB_subref_template: C = A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_subref_templat extracts a submatrix, C = A(I,J).  The method is done in
// two phases.  Phase 1 just counts the entries in C, and phase 2 constructs
// the pattern and values of C.  There are 3 kinds of subref:
//
//      symbolic:  C(i,j) is the position of A(I(i),J(j)) in the matrix A
//      iso:     C = A(I,J), extracting the pattern only, not the values
//      numeric: C = A(I,J), extracting the pattern and values

#if defined ( GB_SYMBOLIC )

    // symbolic method must tolerate zombies
    #define GB_Ai(p) GBI_UNFLIP (Ai, p, avlen)

#else

    // iso and non-iso numeric methods will not see any zombies
    #define GB_Ai(p) GBI_A (Ai, p, avlen)

#endif

// to iterate across all entries in a bucket:
#define GB_for_each_index_in_bucket(inew,i)     \
    for (int64_t inew = Mark [i] - 1 ; inew >= 0 ; inew = Inext [inew])

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A and I
    //--------------------------------------------------------------------------

    const int64_t *restrict Ai = A->i ;
    const int64_t avlen = A->vlen ;

    // these values are ignored if Ikind == GB_LIST
    int64_t ibegin = Icolon [GxB_BEGIN] ;
    int64_t iinc   = Icolon [GxB_INC  ] ;
    int64_t inc    = (iinc < 0) ? (-iinc) : iinc ;
    #ifdef GB_DEBUG
    int64_t iend   = Icolon [GxB_END  ] ;
    #endif

    //--------------------------------------------------------------------------
    // phase1: count entries in each C(:,kC); phase2: compute C
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast < 0) ;
        if (fine_task)
        {
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
        }

        // a coarse task accesses all of I for all its vectors
        int64_t pI     = 0 ;
        int64_t pI_end = nI ;
        int64_t ilen   = nI ;

        ASSERT (0 <= kfirst && kfirst <= klast && klast < Cnvec) ;

        //----------------------------------------------------------------------
        // compute all vectors C(:,kfirst:klast) for this task
        //----------------------------------------------------------------------

        for (int64_t kC = kfirst ; kC <= klast ; kC++)
        {

            //------------------------------------------------------------------
            // get C(:,kC)
            //------------------------------------------------------------------

            #if defined ( GB_ANALYSIS_PHASE )
            // phase1 simply counts the # of entries in C(*,kC).
            int64_t clen = 0 ;
            #else
            // This task computes all or part of C(:,kC), which are the entries
            // in Ci,Cx [pC:pC_end-1].
            int64_t pC, pC_end ;
            if (fine_task)
            { 
                // A fine task computes a slice of C(:,kC)
                pC     = TaskList [taskid  ].pC ;
                pC_end = TaskList [taskid+1].pC ;
                ASSERT (Cp [kC] <= pC && pC <= pC_end && pC_end <= Cp [kC+1]) ;
            }
            else
            { 
                // The vectors of C are never sliced for a coarse task, so this
                // task computes all of C(:,kC).
                pC     = Cp [kC] ;
                pC_end = Cp [kC+1] ;
            }
            int64_t clen = pC_end - pC ;
            if (clen == 0) continue ;
            #endif

            //------------------------------------------------------------------
            // get A(:,kA)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;

            if (fine_task)
            { 
                // a fine task computes a slice of a single vector C(:,kC).
                // The task accesses Ai,Ax [pA:pA_end-1], which holds either
                // the entire vector A(imin:imax,kA) for method 6, the entire
                // dense A(:,kA) for methods 1 and 2, or a slice of the
                // A(imin:max,kA) vector for all other methods.
                pA     = TaskList [taskid].pA ;
                pA_end = TaskList [taskid].pA_end ;
            }
            else
            { 
                // a coarse task computes the entire vector C(:,kC).  The task
                // accesses all of A(imin:imax,kA), for most methods, or all of
                // A(:,kA) for methods 1 and 2.  The vector A(*,kA) appears in
                // Ai,Ax [pA:pA_end-1].
                pA     = Ap_start [kC] ;
                pA_end = Ap_end   [kC] ;
            }

            int64_t alen = pA_end - pA ;
            if (alen == 0) continue ;

            //------------------------------------------------------------------
            // get I
            //------------------------------------------------------------------

            if (fine_task)
            { 
                // A fine task accesses I [pI:pI_end-1].  For methods 2 and 6,
                // pI:pI_end is a subset of the entire 0:nI-1 list.  For all
                // other methods, pI = 0 and pI_end = nI, and the task can
                // access all of I.
                pI     = TaskList [taskid].pB ;
                pI_end = TaskList [taskid].pB_end ;
                ilen   = pI_end - pI ;
            }

            //------------------------------------------------------------------
            // determine the method to use
            //------------------------------------------------------------------

            int method ;
            if (fine_task)
            { 
                // The method that the fine task uses for its slice of A(*,kA)
                // and C(*,kC) has already been determined by GB_subref_slice.
                method = (int) (-TaskList [taskid].klast) ;
            }
            else
            { 
                // determine the method based on A(*,kA) and I
                method = GB_subref_method (NULL, NULL, alen, avlen, Ikind, nI,
                    (Mark != NULL), need_qsort, iinc, nduplicates) ;
            }

            //------------------------------------------------------------------
            // extract C (:,kC) = A (I,kA): consider all cases
            //------------------------------------------------------------------

            switch (method)
            {

                //--------------------------------------------------------------
                case 1 : // C(:,kC) = A(:,kA) where A(:,kA) is dense
                //--------------------------------------------------------------

                    // A (:,kA) has not been sliced
                    ASSERT (Ikind == GB_ALL) ;
                    ASSERT (pA     == Ap_start [kC]) ;
                    ASSERT (pA_end == Ap_end   [kC]) ;
                    // copy the entire vector and construct indices
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = ilen ;
                    #else
                    for (int64_t k = 0 ; k < ilen ; k++)
                    { 
                        int64_t inew = k + pI ;
                        ASSERT (inew == GB_ijlist (I, inew, Ikind, Icolon)) ;
                        ASSERT (inew == GB_Ai (pA + inew)) ;
                        Ci [pC + k] = inew ;
                    }
                    GB_COPY_RANGE (pC, pA + pI, ilen) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 2 : // C(:,kC) = A(I,kA) where A(I,kA) is dense
                //--------------------------------------------------------------

                    // This method handles any kind of list I, but A(:,kA)
                    // must be dense.  A(:,kA) has not been sliced.
                    ASSERT (pA     == Ap_start [kC]) ;
                    ASSERT (pA_end == Ap_end   [kC]) ;
                    // scan I and get the entry in A(:,kA) via direct lookup
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = ilen ;
                    #else
                    for (int64_t k = 0 ; k < ilen ; k++)
                    { 
                        // C(inew,kC) =  A(i,kA), and it always exists.
                        int64_t inew = k + pI ;
                        int64_t i = GB_ijlist (I, inew, Ikind, Icolon) ;
                        ASSERT (i == GB_Ai (pA + i)) ;
                        Ci [pC + k] = inew ;
                        GB_COPY_ENTRY (pC + k, pA + i) ;
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 3 : // the list I has a single index, ibegin
                //--------------------------------------------------------------

                    // binary search in GB_subref_phase0 has already found it.
                    // This can be any Ikind with nI=1: GB_ALL with A->vlen=1,
                    // GB_RANGE with ibegin==iend, GB_STRIDE such as 0:-1:0
                    // (with length 1), or a GB_LIST with ni=1.

                    // Time: 50x faster

                    ASSERT (!fine_task) ;
                    ASSERT (alen == 1) ;
                    ASSERT (nI == 1) ;
                    ASSERT (GB_Ai (pA) == GB_ijlist (I, 0, Ikind, Icolon)) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = 1 ;
                    #else
                    Ci [pC] = 0 ;
                    GB_COPY_ENTRY (pC, pA) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 4 : // Ikind is ":", thus C(:,kC) = A (:,kA)
                //--------------------------------------------------------------

                    // Time: 1x faster but low speedup on the Mac.  Why?
                    // Probably memory bound since it is just memcpy's.

                    ASSERT (Ikind == GB_ALL && ibegin == 0) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    #if defined ( GB_SYMBOLIC )
                    if (nzombies == 0)
                    { 
                        memcpy (Ci + pC, Ai + pA, alen * sizeof (int64_t)) ;
                    }
                    else
                    {
                        // with zombies
                        for (int64_t k = 0 ; k < alen ; k++)
                        { 
                            // symbolic C(:,kC) = A(:,kA) where A has zombies
                            int64_t i = GB_Ai (pA + k) ;
                            ASSERT (i == GB_ijlist (I, i, Ikind, Icolon)) ;
                            Ci [pC + k] = i ;
                        }
                    }
                    #else
                    memcpy (Ci + pC, Ai + pA, alen * sizeof (int64_t)) ;
                    #endif
                    GB_COPY_RANGE (pC, pA, alen) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 5 : // Ikind is GB_RANGE = ibegin:iend
                //--------------------------------------------------------------

                    // Time: much faster.  Good speedup too.

                    ASSERT (Ikind == GB_RANGE) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    for (int64_t k = 0 ; k < alen ; k++)
                    { 
                        int64_t i = GB_Ai (pA + k) ;
                        int64_t inew = i - ibegin ;
                        ASSERT (i == GB_ijlist (I, inew, Ikind, Icolon)) ;
                        Ci [pC + k] = inew ;
                    }
                    GB_COPY_RANGE (pC, pA, alen) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 6 : // I is short vs nnz (A (:,kA)), use binary search
                //--------------------------------------------------------------

                    // Time: very slow unless I is very short and A(:,kA) is
                    // very long.

                    // This case can handle any kind of I, and A(:,kA) of any
                    // properties.  For a fine task, A(:,kA) has not been
                    // sliced; I has been sliced instead.

                    // If the I bucket inverse has not been created, this
                    // method is the only option.  Alternatively, if nI =
                    // length (I) is << nnz (A (:,kA)), then scanning I and
                    // doing a binary search of A (:,kA) is faster than doing a
                    // linear-time search of A(:,kA) and a lookup into the I
                    // bucket inverse.

                    // The vector of C is constructed in sorted order, so no
                    // sort is needed.

                    // A(:,kA) has not been sliced.
                    ASSERT (pA     == Ap_start [kC]) ;
                    ASSERT (pA_end == Ap_end   [kC]) ;

                    // scan I, in order, and search for the entry in A(:,kA)
                    for (int64_t k = 0 ; k < ilen ; k++)
                    {
                        // C(inew,kC) = A (i,kA), if it exists.
                        // i = I [inew] ; or from a colon expression
                        int64_t inew = k + pI ;
                        int64_t i = GB_ijlist (I, inew, Ikind, Icolon) ;
                        bool found ;
                        int64_t pleft = pA ;
                        int64_t pright = pA_end - 1 ;
                        #if defined ( GB_SYMBOLIC )
                        bool is_zombie ;
                        GB_BINARY_SEARCH_ZOMBIE (i, Ai, pleft, pright, found,
                            nzombies, is_zombie) ;
                        #else
                        GB_BINARY_SEARCH (i, Ai, pleft, pright, found) ;
                        #endif
                        if (found)
                        { 
                            ASSERT (i == GB_Ai (pleft)) ;
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            ASSERT (pC < pC_end) ;
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pleft) ;
                            pC++ ;
                            #endif
                        }
                    }
                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 7 : // I is ibegin:iinc:iend with iinc > 1
                //--------------------------------------------------------------

                    // Time: 1 thread: C=A(1:2:n,:) is 3x slower
                    // but has good speedup.  About as fast with
                    // enough threads.

                    ASSERT (Ikind == GB_STRIDE && iinc > 1) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present; see if it is in ibegin:iinc:iend
                        int64_t i = GB_Ai (pA + k) ;
                        ASSERT (ibegin <= i && i <= iend) ;
                        i = i - ibegin ;
                        if (i % iinc == 0)
                        { 
                            // i is in the sequence ibegin:iinc:iend
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            int64_t inew = i / iinc ;
                            ASSERT (pC < pC_end) ;
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pA + k) ;
                            pC++ ;
                            #endif
                        }
                    }
                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    #endif
                    break ;

                //----------------------------------------------------------
                case 8 : // I = ibegin:(-iinc):iend, with iinc < -1
                //----------------------------------------------------------

                    // Time: 2x slower for iinc = -2 or -8.
                    // Good speedup though.  Faster for
                    // large values (iinc = -128).
                
                    ASSERT (Ikind == GB_STRIDE && iinc < -1) ;
                    for (int64_t k = alen - 1 ; k >= 0 ; k--)
                    {
                        // A(i,kA) present; see if it is in ibegin:iinc:iend
                        int64_t i = GB_Ai (pA + k) ;
                        ASSERT (iend <= i && i <= ibegin) ;
                        i = ibegin - i ;
                        if (i % inc == 0)
                        { 
                            // i is in the sequence ibegin:iinc:iend
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            int64_t inew = i / inc ;
                            ASSERT (pC < pC_end) ;
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pA + k) ;
                            pC++ ;
                            #endif
                        }
                    }
                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    #endif
                    break ;

                //----------------------------------------------------------
                case 9 : // I = ibegin:(-1):iend
                //----------------------------------------------------------

                    // Time: much faster.  Good speedup.

                    ASSERT (Ikind == GB_STRIDE && iinc == -1) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    for (int64_t k = alen - 1 ; k >= 0 ; k--)
                    { 
                        // A(i,kA) is present
                        int64_t i = GB_Ai (pA + k) ;
                        int64_t inew = (ibegin - i) ;
                        ASSERT (i == GB_ijlist (I, inew, Ikind, Icolon)) ;
                        Ci [pC] = inew ;
                        GB_COPY_ENTRY (pC, pA + k) ;
                        pC++ ;
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 10 : // I unsorted, and C needs qsort, duplicates OK
                //--------------------------------------------------------------

                    // Time: with one thread: 2x slower, probably
                    // because of the qsort.  Good speedup however.  This used
                    // if qsort is needed but ndupl == 0.  Try a method that
                    // needs qsort, but no duplicates?

                    // Case 10 works well when I has many entries and A(:,kA)
                    // has few entries. C(:,kC) must be sorted after this pass.

                    ASSERT (Ikind == GB_LIST) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present, look it up in the I inverse buckets
                        int64_t i = GB_Ai (pA + k) ;
                        // traverse bucket i for all indices inew where
                        // i == I [inew] or where i is from a colon expression
                        GB_for_each_index_in_bucket (inew, i)
                        { 
                            ASSERT (inew >= 0 && inew < nI) ;
                            ASSERT (i == GB_ijlist (I, inew, Ikind, Icolon)) ;
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pA + k) ;
                            pC++ ;
                            #endif
                        }
                    }

                    // TODO: skip the sort if C is allowed to be jumbled on
                    // output.  Flag C as jumbled instead.

                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    if (!fine_task)
                    { 
                        // a coarse task owns this entire C(:,kC) vector, so
                        // the sort can be done now.  The sort for vectors
                        // handled by multiple fine tasks must wait until all
                        // task are completed, below in the post sort.
                        pC = Cp [kC] ;

                        #if defined ( GB_ISO_SUBREF )
                        // iso numeric subref C=A(I,J)
                        // just sort the pattern of C(:,kC)
                        GB_qsort_1 (Ci + pC, clen) ;
                        #else
                        // sort the pattern of C(:,kC), and the values
                        GB_qsort_1b (Ci + pC, (GB_void *) (Cx + pC*GB_CSIZE1),
                            GB_CSIZE2, clen) ;
                        #endif
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 11 : // I not contiguous, with duplicates. No qsort needed
                //--------------------------------------------------------------

                    // Case 11 works well when I has many entries and A(:,kA)
                    // has few entries.  It requires that I be sorted on input,
                    // so that no sort is required for C(:,kC).  It is
                    // otherwise identical to Case 10.

                    ASSERT (Ikind == GB_LIST) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present, look it up in the I inverse buckets
                        int64_t i = GB_Ai (pA + k) ;
                        // traverse bucket i for all indices inew where
                        // i == I [inew] or where i is from a colon expression
                        GB_for_each_index_in_bucket (inew, i)
                        { 
                            ASSERT (inew >= 0 && inew < nI) ;
                            ASSERT (i == GB_ijlist (I, inew, Ikind, Icolon)) ;
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pA + k) ;
                            pC++ ;
                            #endif
                        }
                    }

                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 12 : // I not contiguous, no duplicates.  No qsort needed.
                //--------------------------------------------------------------

                    // Identical to Case 11, except GB_for_each_index_in_bucket
                    // just needs to iterate 0 or 1 times.  Works well when I
                    // has many entries and A(:,kA) has few entries.

                    ASSERT (Ikind == GB_LIST && nduplicates == 0) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present, look it up in the I inverse buckets
                        int64_t i = GB_Ai (pA + k) ;
                        // bucket i has at most one index inew such that
                        // i == I [inew]
                        int64_t inew = Mark [i] - 1 ;
                        if (inew >= 0)
                        { 
                            ASSERT (inew >= 0 && inew < nI) ;
                            ASSERT (i == GB_ijlist (I, inew, Ikind, Icolon)) ;
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            Ci [pC] = inew ;
                            GB_COPY_ENTRY (pC, pA + k) ;
                            pC++ ;
                            #endif
                        }
                    }

                    #if defined ( GB_PHASE_2_OF_2 )
                    ASSERT (pC == pC_end) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                default: ;
                //--------------------------------------------------------------
            }

            //------------------------------------------------------------------
            // final count of nnz (C (:,j))
            //------------------------------------------------------------------

            #if defined ( GB_ANALYSIS_PHASE )
            if (fine_task)
            { 
                TaskList [taskid].pC = clen ;
            }
            else
            { 
                Cp [kC] = clen ;
            }
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // phase2: post sort for any vectors handled by fine tasks with method 10
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_2_OF_2 )
    {
        if (post_sort)
        {
            int taskid ;
            #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
            for (taskid = 0 ; taskid < ntasks ; taskid++)
            {
                int64_t kC = TaskList [taskid].kfirst ;
                bool do_post_sort = (TaskList [taskid].len != 0) ;
                if (do_post_sort)
                {
                    // This is the first fine task with method 10 for C(:,kC).
                    // The vector C(:,kC) must be sorted, since method 10 left
                    // it with unsorted indices.
                    int64_t pC = Cp [kC] ;
                    int64_t clen = Cp [kC+1] - pC ;
                    #if defined ( GB_ISO_SUBREF )
                    { 
                        // iso numeric subref C=A(I,J)
                        // just sort the pattern of C(:,kC)
                        GB_qsort_1 (Ci + pC, clen) ;
                    }
                    #else
                    { 
                        // sort the pattern of C(:,kC), and the values
                        GB_qsort_1b (Ci + pC, (GB_void *) (Cx + pC*GB_CSIZE1),
                            GB_CSIZE2, clen) ;
                    }
                    #endif
                }
            }
        }
    }
    #endif

}

#undef GB_Ai
#undef GB_for_each_index_in_bucket
#undef GB_COPY_RANGE
#undef GB_COPY_ENTRY
#undef GB_CSIZE1
#undef GB_CSIZE2
#undef GB_SYMBOLIC
#undef GB_ISO_SUBREF

