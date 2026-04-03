//------------------------------------------------------------------------------
// GB_subref_template: C = A(I,J) where C and A are sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_subref_template extracts a submatrix, C = A(I,J).  The method is done in
// three phases.  Phase 1 and 2 are symbolic, and phase 3 (this phase)
// constructs the pattern and values of C.  There are 3 kinds of subref:
//
//      symbolic:  C(i,j) is the position of A(I(i),J(j)) in the matrix A,
//                  in this case, A can have zombies.
//                  C never has zombies, even if A does.
//      iso:       C = A(I,J), extracting the pattern only, not the values
//      numeric:   C = A(I,J), extracting the pattern and values

// The matrix R holds the "inverse" of I, which is not actually an inverse
// since I can have duplicates.  If i = I [k1] = I [k2] = I [k3], then the
// column indices in R(i,:) are {k1, k2, k3}.  R is held by row, and is either
// sparse or hypersparse.

#define GB_for_each_inew_in_I_inverse_hash(i,pR)                        \
        int64_t pR, pR_end ;                                            \
        if (R_is_hyper)                                                 \
        {                                                               \
            /* R(i,:) is the kth vector in the hypersparse matrix R; */ \
            /* find k so that i = Rh [k] using the R->Y hyper_hash, */  \
            /* and set pR = Rp [k] and pR_end = Rp [k+1]. */            \
            GB_hyper_hash_lookup (Rp_is_32, Rj_is_32,                   \
                Rh, rnvec, Rp, R_Yp, R_Yi, R_Yx, R_hash_bits,           \
                i, &pR, &pR_end) ;                                      \
        }                                                               \
        else                                                            \
        {                                                               \
            /* R(i,:) is the ith vector in the sparse matrix R */       \
            pR = GB_IGET (Rp, i) ;          /* pR = Rp [i] */           \
            pR_end = GB_IGET (Rp, i+1) ;    /* pR_end = Rp [i+1] */     \
        }                                                               \
        /* for each entry in the row R(i,:) */                          \
        for ( ; pR < pR_end ; pR++)
        #if 0
        {
            // get R(i,inew); this is the index i = I [inew]
            int64_t inew = GB_IGET (Ri, pR) ;        // inew = Ri [pR]
        }
        #endif

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A and I
    //--------------------------------------------------------------------------

    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int64_t avlen = A->vlen ;

    // these values are ignored if GB_I_KIND == GB_LIST
    int64_t ibegin = Icolon [GxB_BEGIN] ;
    int64_t iinc   = Icolon [GxB_INC  ] ;
    int64_t inc    = (iinc < 0) ? (-iinc) : iinc ;
    #ifdef GB_DEBUG
    int64_t iend   = Icolon [GxB_END  ] ;
    #endif

    #ifndef GB_JIT_KERNEL
    #define GB_Ai_IS_32 Ai_is_32
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

        //----------------------------------------------------------------------
        // compute all vectors C(:,kfirst:klast) for this task
        //----------------------------------------------------------------------

        for (int64_t kC = kfirst ; kC <= klast ; kC++)
        {

            //------------------------------------------------------------------
            // get C(:,kC)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;

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
                ASSERT (GB_IGET (Cp, kC) <= pC) ;
                ASSERT (pC <= pC_end) ;
                ASSERT (pC_end <= GB_IGET (Cp, kC+1)) ;
            }
            else
            { 
                // The vectors of C are never sliced for a coarse task, so this
                // task computes all of C(:,kC).
                pC     = GB_IGET (Cp, kC) ;
                pC_end = GB_IGET (Cp, kC+1) ;
            }
            int64_t clen = pC_end - pC ;
            if (clen == 0) continue ;
            #endif

            //------------------------------------------------------------------
            // get A(:,kA)
            //------------------------------------------------------------------

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
                pA     = GB_IGET (Ap_start, kC) ;
                pA_end = GB_IGET (Ap_end  , kC) ;
            }

            int64_t alen = pA_end - pA ;
            if (alen == 0) 
            {
                #if defined ( GB_ANALYSIS_PHASE )
                if (fine_task)
                {
                    // this fine task has no entries in A(:,kC) to access
                    TaskList [taskid].pC = 0 ;
                }
                else
                { 
                    // this course task has found that C(:,kC) is entirely empty
                    Cwork [kC] = 0 ;
                }
                #endif
                continue ;
            }

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
                method = GB_subref_method (alen, avlen, GB_I_KIND, nI,
                    GB_NEED_QSORT, iinc) ;
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
                    ASSERT (GB_I_KIND == GB_ALL) ;
                    ASSERT (pA     == GB_IGET (Ap_start, kC)) ;
                    ASSERT (pA_end == GB_IGET (Ap_end  , kC)) ;
                    // copy the entire vector and construct indices
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = ilen ;
                    #else
                    for (int64_t k = 0 ; k < ilen ; k++)
                    { 
                        int64_t inew = k + pI ;
                        ASSERT (inew == GB_IJLIST (I, inew, GB_I_KIND, Icolon));
                        #ifdef GB_DEBUG
                        int64_t iA = GB_IGET (Ai, pA + inew) ;
                        iA = GB_UNZOMBIE (iA) ;
                        ASSERT (inew == iA) ;
                        #endif
                        GB_ISET (Ci, pC + k, inew) ;  // Ci [pC + k] = inew ;
                    }
                    GB_COPY_RANGE (pC, pA + pI, ilen) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 2 : // C(:,kC) = A(I,kA) where A(I,kA) is dense
                //--------------------------------------------------------------

                    // This method handles any kind of list I, but A(:,kA)
                    // must be dense.  A(:,kA) has not been sliced.
                    ASSERT (pA     == GB_IGET (Ap_start, kC)) ;
                    ASSERT (pA_end == GB_IGET (Ap_end  , kC)) ;
                    // scan I and get the entry in A(:,kA) via direct lookup
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = ilen ;
                    #else
                    for (int64_t k = 0 ; k < ilen ; k++)
                    { 
                        // C(inew,kC) =  A(i,kA), and it always exists.
                        int64_t inew = k + pI ;
                        #if defined ( GB_DEBUG ) || !defined ( GB_ISO_SUBREF )
                        int64_t i = GB_IJLIST (I, inew, GB_I_KIND, Icolon) ;
                        #endif
                        #ifdef GB_DEBUG
                        int64_t iA = GB_IGET (Ai, pA + i) ;
                        iA = GB_UNZOMBIE (iA) ;
                        ASSERT (i == iA) ;
                        #endif
                        GB_ISET (Ci, pC + k, inew) ;  // Ci [pC + k] = inew ;
                        GB_COPY_ENTRY (pC + k, pA + i) ;
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 3 : // the list I has a single index, ibegin
                //--------------------------------------------------------------

                    // binary search in GB_subref_phase0 has already found it.
                    // This can be any GB_I_KIND with nI=1: GB_ALL with
                    // A->vlen=1, GB_RANGE with ibegin==iend, GB_STRIDE such as
                    // 0:-1:0 (with length 1), or a GB_LIST with ni=1.

                    // Time: 50x faster

                    #ifdef GB_DEBUG
                    ASSERT (!fine_task) ;
                    ASSERT (alen == 1) ;
                    ASSERT (nI == 1) ;
                    int64_t i0 = GB_IJLIST (I, 0, GB_I_KIND, Icolon) ;
                    int64_t iA = GB_IGET (Ai, pA) ;
                    iA = GB_UNZOMBIE (iA) ;
                    ASSERT (iA == i0) ;
                    #endif

                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = 1 ;
                    #else
                    GB_ISET (Ci, pC, 0) ;       // Ci [pC] = 0 ;
                    GB_COPY_ENTRY (pC, pA) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 4 : // GB_I_KIND is ":", thus C(:,kC) = A (:,kA)
                //--------------------------------------------------------------

                    // Time: 1x faster but low speedup on the Mac.  Why?

                    ASSERT (GB_I_KIND == GB_ALL && ibegin == 0) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    #if defined ( GB_SYMBOLIC )
                    if (may_see_zombies)
                    {
                        // with zombies in A
                        for (int64_t k = 0 ; k < alen ; k++)
                        { 
                            // symbolic C(:,kC) = A(:,kA) where A has zombies;
                            // zombies in A are not tagged as zombies in C.
                            int64_t i = GB_IGET (Ai, pA + k) ;
                            i = GB_UNZOMBIE (i) ;
                            ASSERT (i == GB_IJLIST (I, i, GB_I_KIND, Icolon)) ;
                            GB_ISET (Ci, pC + k, i) ;  // Ci [pC + k] = i ;
                        }
                    }
                    else
                    #endif
                    { 
                        // without zombies in A
                        for (int64_t k = 0 ; k < alen ; k++)
                        {
                            int64_t i = GB_IGET (Ai, pA + k) ;
                            ASSERT (i == GB_IJLIST (I, i, GB_I_KIND, Icolon)) ;
                            GB_ISET (Ci, pC + k, i) ;  // Ci [pC + k] = i ;
                        }
                    }
                    GB_COPY_RANGE (pC, pA, alen) ;
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 5 : // GB_I_KIND is GB_RANGE = ibegin:iend
                //--------------------------------------------------------------

                    // Time: much faster.  Good speedup too.

                    ASSERT (GB_I_KIND == GB_RANGE) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    for (int64_t k = 0 ; k < alen ; k++)
                    { 
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
                        int64_t inew = i - ibegin ;
                        ASSERT (i == GB_IJLIST (I, inew, GB_I_KIND, Icolon)) ;
                        GB_ISET (Ci, pC + k, inew) ;  // Ci [pC + k] = inew ;
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

                    // If nI = length (I) is << nnz (A (:,kA)), then scanning I
                    // and doing a binary search of A (:,kA) is faster than
                    // doing a linear-time scan of A(:,kA) and a lookup into
                    // R for each row index i in A(:,kA).

                    // The vector of C is constructed in sorted order, so no
                    // sort is needed.

                    // A(:,kA) has not been sliced.
                    ASSERT (pA     == GB_IGET (Ap_start, kC)) ;
                    ASSERT (pA_end == GB_IGET (Ap_end  , kC)) ;

                    // scan I, in order, and search for the entry in A(:,kA)
                    for (int64_t k = 0 ; k < ilen ; k++)
                    {
                        // C(inew,kC) = A (i,kA), if it exists.
                        // i = I [inew] ; or from a colon expression
                        int64_t inew = k + pI ;
                        int64_t i = GB_IJLIST (I, inew, GB_I_KIND, Icolon) ;
                        bool found ;
                        int64_t pleft = pA ;
                        int64_t pright = pA_end - 1 ;
                        #if defined ( GB_SYMBOLIC )
                        bool is_zombie ;
                        found = GB_binary_search_zombie (i, Ai, GB_Ai_IS_32,
                            &pleft, &pright, may_see_zombies, &is_zombie) ;
                        #else
                        found = GB_binary_search (i, Ai, GB_Ai_IS_32,
                            &pleft, &pright) ;
                        #endif
                        if (found)
                        { 
                            #ifdef GB_DEBUG
                            int64_t iA = GB_IGET (Ai, pleft) ;
                            iA = GB_UNZOMBIE (iA) ;
                            ASSERT (i == iA) ;
                            #endif
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            ASSERT (pC < pC_end) ;
                            GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
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

                    ASSERT (GB_I_KIND == GB_STRIDE && iinc > 1) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present; see if it is in ibegin:iinc:iend
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
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
                            GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
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

                    ASSERT (GB_I_KIND == GB_STRIDE && iinc < -1) ;
                    for (int64_t k = alen - 1 ; k >= 0 ; k--)
                    {
                        // A(i,kA) present; see if it is in ibegin:iinc:iend
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
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
                            GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
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

                    ASSERT (GB_I_KIND == GB_STRIDE && iinc == -1) ;
                    #if defined ( GB_ANALYSIS_PHASE )
                    clen = alen ;
                    #else
                    for (int64_t k = alen - 1 ; k >= 0 ; k--)
                    { 
                        // A(i,kA) is present
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
                        int64_t inew = (ibegin - i) ;
                        ASSERT (i == GB_IJLIST (I, inew, GB_I_KIND, Icolon)) ;
                        GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
                        GB_COPY_ENTRY (pC, pA + k) ;
                        pC++ ;
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 10 : // I unsorted, and C needs qsort, duplicates OK
                //--------------------------------------------------------------

                    // Time: with one thread: 2x slower, probably because of
                    // the qsort.  Good speedup however.

                    // Case 10 works well when I has many entries and A(:,kA)
                    // has few entries. C(:,kC) must be sorted after this pass.

                    ASSERT (GB_I_KIND == GB_LIST) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present, look it up in R(i,:)
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
                        // traverse R(i,:) for all indices inew where
                        // i == I [inew] or where i is from a colon expression
                        GB_for_each_inew_in_I_inverse_hash (i,pR)
                        { 
                            int64_t inew = GB_IGET (Ri, pR) ; // inew = Ri [pR]
                            ASSERT (inew >= 0 && inew < nI) ;
                            ASSERT (i == GB_IJLIST (I, inew, GB_I_KIND,Icolon));
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
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
                        pC = GB_IGET (Cp, kC) ;
                        // sort C(:,kC)
                        GB_QSORT_1B (Ci, Cx, pC, clen) ;
                    }
                    #endif
                    break ;

                //--------------------------------------------------------------
                case 11 : // I not contiguous, duplicates OK. No qsort needed
                //--------------------------------------------------------------

                    // Case 11 works well when I has many entries and A(:,kA)
                    // has few entries.  It requires that I be sorted on input,
                    // so that no sort is required for C(:,kC).  It is
                    // otherwise identical to Case 10.

                    ASSERT (GB_I_KIND == GB_LIST) ;
                    for (int64_t k = 0 ; k < alen ; k++)
                    {
                        // A(i,kA) present, look it up in R(i,:)
                        int64_t i = GB_IGET (Ai, pA + k) ;
                        #if defined ( GB_SYMBOLIC )
                        i = GB_UNZOMBIE (i) ;
                        #endif
                        // traverse R(i,:) for all indices inew where
                        // i == I [inew] or where i is from a colon expression
                        GB_for_each_inew_in_I_inverse_hash (i,pR)
                        { 
                            int64_t inew = GB_IGET (Ri, pR) ; // inew = Ri [pR]
                            ASSERT (inew >= 0 && inew < nI) ;
                            ASSERT (i == GB_IJLIST (I, inew, GB_I_KIND,Icolon));
                            #if defined ( GB_ANALYSIS_PHASE )
                            clen++ ;
                            #else
                            GB_ISET (Ci, pC, inew) ;  // Ci [pC] = inew ;
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
                Cwork [kC] = clen ;
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
                    int64_t pC = GB_IGET (Cp, kC) ;
                    int64_t clen = GB_IGET (Cp, kC+1) - pC ;
                    // sort C(:,kC)
                    GB_QSORT_1B (Ci, Cx, pC, clen) ;
                }
            }
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // ensure all of Cwork has been computed
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // In debug mode, GB_subref_phase2 sets Cwork [0:Cnvec] = UINT64_MAX,
    // tagging its contents as undefined.  It then clears Cwork [kC] for all
    // fine tasks.  Course tasks set Cwork [kfirst:klast] for all the vectors
    // C (:,kfirst:klast).  Together, this ensures that all Cwork [0:Cnvec-1]
    // has been computed.
    #if defined ( GB_ANALYSIS_PHASE )
    for (int64_t kC = 0 ; kC < Cnvec ; kC++)
    {
        ASSERT (Cwork [kC] != UINT64_MAX) ;
    }
    #endif
    #endif
}

#undef GB_for_each_inew_in_I_inverse_hash
#undef GB_COPY_RANGE
#undef GB_COPY_ENTRY
#undef GB_SYMBOLIC
#undef GB_ISO_SUBREF
#undef GB_QSORT_1B

