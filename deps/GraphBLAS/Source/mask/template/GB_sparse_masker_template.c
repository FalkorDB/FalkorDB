//------------------------------------------------------------------------------
// GB_sparse_masker_template:  R = masker (C, M, Z) where R is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C<M>=Z or C<!M>=Z, returning the result in R, which is sparse or
// hypersparse.  The input matrix C is not modified.  Effectively, this
// computes R=C and then R<M>=Z or R<!M>=Z.  If the C_replace descriptor is
// enabled, then C has already been cleared, and is an empty (but non-NULL)
// matrix.

// phase1: does not compute R itself, but just counts the # of entries in each
// vector of R.  Fine tasks compute the # of entries in their slice of a
// single vector of R, and the results are cumsum'd.

// phase2: computes R, using the counts computed by phase1.

// C is sparse or hypersparse.  M and Z can have any sparsity structure.

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      sparse          sparse
        //      sparse  bitmap      sparse          sparse
        //      sparse  full        sparse          sparse

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  sparse      sparse          sparse
        //      sparse  sparse      bitmap          sparse
        //      sparse  sparse      full            sparse
        //      sparse  bitmap      sparse          sparse
        //      sparse  full        sparse          sparse

// FUTURE:: add special cases for C==Z, C==M, and Z==M aliases

//------------------------------------------------------------------------------
// R(i,j) = Z(i,j) when Z is sparse or hypersparse
//------------------------------------------------------------------------------

#undef GB_COPY_Z
#if defined ( GB_PHASE_1_OF_2 )
    #define GB_COPY_Z(i)                                        \
    {                                                           \
        rjnz++ ;                                                \
    }
#elif defined ( GB_ISO_MASKER )
    #define GB_COPY_Z(i)                                        \
    {                                                           \
        GB_ISET (Ri, pR, i) ;                                   \
        pR++ ;                                                  \
    }
#else
    #define GB_COPY_Z(i)                                        \
    {                                                           \
        GB_ISET (Ri, pR, i) ;                                   \
        GB_COPY_Z_TO_R (Rx, pR, Zx, pZ, Z_iso, rsize) ;         \
        pR++ ;                                                  \
    }
#endif

//------------------------------------------------------------------------------
// R(i,j) = Z(i,j) when Z is bitmap or full
//------------------------------------------------------------------------------

#undef GB_COPY_Z_BITMAP_OR_FULL
#if defined ( GB_PHASE_1_OF_2 )
    #define GB_COPY_Z_BITMAP_OR_FULL                            \
    {                                                           \
        rjnz += GBb_Z (Zb, pZ_start + i - iZ_first) ;           \
    }
#elif defined ( GB_ISO_MASKER )
    #define GB_COPY_Z_BITMAP_OR_FULL                            \
    {                                                           \
        int64_t pZ = pZ_start + i - iZ_first ;                  \
        if (GBb_Z (Zb, pZ))                                     \
        {                                                       \
            GB_ISET (Ri, pR, i) ;                               \
            pR++ ;                                              \
        }                                                       \
    }
#else
    #define GB_COPY_Z_BITMAP_OR_FULL                            \
    {                                                           \
        int64_t pZ = pZ_start + i - iZ_first ;                  \
        if (GBb_Z (Zb, pZ))                                     \
        {                                                       \
            GB_ISET (Ri, pR, i) ;                               \
            GB_COPY_Z_TO_R (Rx, pR, Zx, pZ, Z_iso, rsize) ;     \
            pR++ ;                                              \
        }                                                       \
    }
#endif

//------------------------------------------------------------------------------
// R(i,j) = C(i,j)
//------------------------------------------------------------------------------

#undef GB_COPY_C
#if defined ( GB_PHASE_1_OF_2 )
    #define GB_COPY_C(i)                                        \
    {                                                           \
        rjnz++ ;                                                \
    }
#elif defined ( GB_ISO_MASKER )
    #define GB_COPY_C(i)                                        \
    {                                                           \
        GB_ISET (Ri, pR, i) ;                                   \
        pR++ ;                                                  \
    }
#else
    #define GB_COPY_C(i)                                        \
    {                                                           \
        GB_ISET (Ri, pR, i) ;                                   \
        GB_COPY_C_TO_R (Rx, pR, Cx, pC, C_iso, rsize) ;         \
        pR++ ;                                                  \
    }
#endif

//------------------------------------------------------------------------------
// template for R = masker (C, M, Z) when R is sparse or hypersparse
//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase1: count entries in each C(:,j)
    // phase2: compute C
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;

    #pragma omp parallel for num_threads(R_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < R_ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast == -1) ;
        int64_t len ;
        if (fine_task)
        { 
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
            len = TaskList [taskid].len ;
        }
        else
        { 
            // a coarse task operates on one or more whole vectors
            len = vlen ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get j, the kth vector of R
            //------------------------------------------------------------------

            int64_t j = GBh_R (Rh, k) ;

            #if defined ( GB_PHASE_1_OF_2 )
            int64_t rjnz = 0 ;
            #else
            int64_t pR, pR_end ;
            if (fine_task)
            { 
                // A fine task computes a slice of R(:,j)
                pR     = TaskList [taskid  ].pC ;
                pR_end = TaskList [taskid+1].pC ;
                ASSERT (GB_IGET (Rp, k) <= pR) ;
                ASSERT (pR <= pR_end) ;
                ASSERT (pR_end <= GB_IGET (Rp, k+1)) ;
            }
            else
            { 
                // The vectors of R are never sliced for a coarse task.
                pR     = GB_IGET (Rp, k) ;
                pR_end = GB_IGET (Rp, k+1) ;
            }
            int64_t rjnz = pR_end - pR ;
            if (rjnz == 0)
            {
                continue ;
            }
            #endif

            //------------------------------------------------------------------
            // get C(:,j)
            //------------------------------------------------------------------

            int64_t pC = -1, pC_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Ci,Cx [pC...pC_end-1], which is
                // a subset of the vector C(:,j)
                pC     = TaskList [taskid].pA ;
                pC_end = TaskList [taskid].pA_end ;
            }
            else
            {
                // A coarse task operates on the entire vector C(:,j)
                int64_t kC = (R_to_C == NULL) ? j : R_to_C [k] ;
                if (kC >= 0)
                { 
                    pC     = GB_IGET (Cp, kC) ;
                    pC_end = GB_IGET (Cp, kC+1) ;
                }
            }

            int64_t cjnz = pC_end - pC ;        // nnz in C(:,j) for this slice
            bool cdense = (cjnz == len) && (cjnz > 0) ;

            #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DEBUG )
            // get the first index in C(:,j) for this vector
            int64_t iC_first = -1 ;
            if (cjnz > 0) iC_first = GB_IGET (Ci, pC) ;
            #endif

            #ifdef GB_DEBUG
            int64_t iC_last = -1 ;
            if (cjnz > 0) iC_last  = GB_IGET (Ci, pC_end-1) ;
            #endif

            //------------------------------------------------------------------
            // get Z(:,j)
            //------------------------------------------------------------------

            int64_t pZ = -1, pZ_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Zi,Zx [pZ...pZ_end-1], which is
                // a subset of the vector Z(:,j)
                pZ     = TaskList [taskid].pB ;
                pZ_end = TaskList [taskid].pB_end ;
            }
            else
            {
                // A coarse task operates on the entire vector Z(:,j)
                int64_t kZ = (R_to_Z == NULL) ? j : R_to_Z [k] ;
                if (kZ >= 0)
                { 
                    pZ     = GBp_Z (Zp, kZ, vlen) ;
                    pZ_end = GBp_Z (Zp, kZ+1, vlen) ;
                }
            }

            int64_t zjnz = pZ_end - pZ ;        // nnz in Z(:,j) for this slice
            int64_t pZ_start = pZ ;
            bool zdense = (zjnz == len) && (zjnz > 0) ;

            int64_t iZ_first = -1 ;
            #ifdef GB_DEBUG
            int64_t iZ_last = -1 ;
            #endif
            if (zjnz > 0)
            {
                iZ_first = GBi_Z (Zi, pZ, vlen) ;
                #ifdef GB_DEBUG
                iZ_last  = GBi_Z (Zi, pZ_end-1, vlen) ;
                #endif
            }

            //------------------------------------------------------------------
            // get M(:,j)
            //------------------------------------------------------------------

            int64_t pM = -1, pM_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Mi,Mx [pM...pM_end-1], which is
                // a subset of the vector M(:,j)
                pM     = TaskList [taskid].pM ;
                pM_end = TaskList [taskid].pM_end ;
            }
            else
            {
                // A coarse task operates on the entire vector M (:,j)
                int64_t kM = (R_to_M == NULL) ? j : R_to_M [k] ;
                if (kM >= 0)
                { 
                    pM     = GBp_M (Mp, kM, vlen) ;
                    pM_end = GBp_M (Mp, kM+1, vlen) ;
                }
            }

            int64_t mjnz = pM_end - pM ;    // nnz (M (:,j))
            bool mdense = (mjnz == len) && (mjnz > 0) ;

            // get the first index in M(:,j) for this vector
            int64_t iM_first = -1 ;
            int64_t pM_first = pM ;
            if (mjnz > 0) iM_first = GBi_M (Mi, pM_first, vlen) ;

            //------------------------------------------------------------------
            // R(:,j) = masker (C (:,j), M (:,j), Z (:,j))
            //------------------------------------------------------------------

            if (GB_Z_IS_BITMAP || GB_Z_IS_FULL)
            {

                //--------------------------------------------------------------
                // Method01: Z is bitmap or full; M is sparse or hypersparse
                //--------------------------------------------------------------

                //      ------------------------------------------
                //      C       <M> =        Z              R
                //      ------------------------------------------

                //      sparse  sparse      bitmap          sparse
                //      sparse  sparse      full            sparse

                // M is sparse or hypersparse, and not complemented.
                // Otherwise, R is bitmap and not computed here, but in
                // GB_bitmap_masker_template instead.

                ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                ASSERT (!GB_MASK_COMP) ;

                // 2-way merge of C(:,j) and M(:,j) and direct lookup of Z

                while (pC < pC_end && pM < pM_end)
                {
                    
                    int64_t iC = GB_IGET (Ci, pC) ;
                    int64_t iM = GB_IGET (Mi, pM) ;

                    if (iC < iM)
                    { 
                        // C(i,j) is present but M(i,j) is not
                        // R(i,j) = C(i,j)
                        GB_COPY_C (iC) ;
                        pC++ ;
                    }
                    else if (iC > iM)
                    { 
                        // M(i,j) is present but C(i,j) is not
                        int64_t i = iM ;
                        bool mij = GB_MCAST (Mx, pM, msize) ;
                        if (mij)
                        {
                            // R(i,j) = Z(i,j)
                            GB_COPY_Z_BITMAP_OR_FULL ;
                        }
                        pM++ ;
                    }
                    else
                    {
                        // both C(i,j) and M(i,j) are present
                        int64_t i = iM ;
                        bool mij = GB_MCAST (Mx, pM, msize) ;
                        if (mij)
                        { 
                            // R(i,j) = Z(i,j)
                            GB_COPY_Z_BITMAP_OR_FULL ;
                        }
                        else
                        { 
                            // R(i,j) = C(i,j)
                            GB_COPY_C (i) ;
                        }
                        pC++ ;
                        pM++ ;
                    }
                }

                // if M(:,j) is exhausted ; continue scanning all of C(:,j)
                #if defined ( GB_PHASE_1_OF_2 )
                rjnz += (pC_end - pC) ;
                #else
                for ( ; pC < pC_end ; pC++)
                { 
                    // C(i,j) is present but M(i,j) is not
                    int64_t i = GB_IGET (Ci, pC) ;
                    GB_COPY_C (i) ;
                }
                #endif

                // if C(:,j) is exhausted ; continue scanning all of M(:,j)
                for ( ; pM < pM_end ; pM++)
                {
                    // M(i,j) is present but C(i,j) is not
                    int64_t i = GB_IGET (Mi, pM) ;
                    bool mij = GB_MCAST (Mx, pM, msize) ;
                    if (mij)
                    { 
                        // R(i,j) = Z(i,j)
                        GB_COPY_Z_BITMAP_OR_FULL ;
                    }
                }

            }
            else if (mjnz == 0)
            {

                //--------------------------------------------------------------
                // Z is sparse or hypersparse, M(:,j) is empty
                //--------------------------------------------------------------

                //      ------------------------------------------
                //      C       <!M> =       Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse

                //      ------------------------------------------
                //      C       <M> =        Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse

                // Z must be sparse or hypersparse
                ASSERT (GB_IS_SPARSE (Z) || GB_IS_HYPERSPARSE (Z)) ;

                if (!GB_MASK_COMP)
                { 

                    //----------------------------------------------------------
                    // Method02: M(:,j) is empty and not complemented
                    //----------------------------------------------------------

                    // R(:,j) = C(:,j), regardless of Z(:,j)
                    #if defined ( GB_PHASE_1_OF_2 )
                    rjnz = cjnz ;
                    #else
                    ASSERT (rjnz == cjnz) ;
                    for (int64_t k = 0 ; k < cjnz ; k++)
                    { 
                        int64_t i = GB_IGET (Ci, pC + k) ;
                        GB_ISET (Ri, pR + k, i) ;
                    }
                    #ifndef GB_ISO_MASKER
                    GB_COPY_C_TO_R_RANGE (Rx, pR, Cx, pC, C_iso, rsize, cjnz) ;
                    #endif
                    #endif

                }
                else
                { 

                    //----------------------------------------------------------
                    // Method03: M(:,j) is empty and complemented
                    //----------------------------------------------------------

                    // R(:,j) = Z(:,j), regardless of C(:,j)
                    #if defined ( GB_PHASE_1_OF_2 )
                    rjnz = zjnz ;
                    #else
                    ASSERT (rjnz == zjnz) ;
                    for (int64_t k = 0 ; k < zjnz ; k++)
                    { 
                        int64_t i = GB_IGET (Zi, pZ + k) ;
                        GB_ISET (Ri, pR + k, i) ;
                    }
                    #ifndef GB_ISO_MASKER
                    GB_COPY_Z_TO_R_RANGE (Rx, pR, Zx, pZ, Z_iso, rsize, zjnz) ;
                    #endif
                    #endif
                }

            }
            else if (cdense && zdense)
            {

                //--------------------------------------------------------------
                // Method03: C(:,j) and Z(:,j) dense: thus R(:,j) dense
                //--------------------------------------------------------------

                //      ------------------------------------------
                //      C       <!M> =       Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse
                //      sparse  bitmap      sparse          sparse
                //      sparse  full        sparse          sparse

                //      ------------------------------------------
                //      C       <M> =        Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse
                //      sparse  bitmap      sparse          sparse
                //      sparse  full        sparse          sparse

                // Both C(:,j) and Z(:,j) are dense (that is, all entries
                // present), but both C and Z are stored in a sparse or
                // hypersparse sparsity structure.  M has any sparsity.

                ASSERT (GB_IS_SPARSE (Z) || GB_IS_HYPERSPARSE (Z)) ;

                ASSERT (cjnz == zjnz) ;
                ASSERT (iC_first == iZ_first) ;
                ASSERT (iC_last  == iZ_last ) ;
                #if defined ( GB_PHASE_1_OF_2 )
                rjnz = cjnz ;
                #else
                ASSERT (rjnz == cjnz) ;
                for (int64_t p = 0 ; p < cjnz ; p++)
                {
                    int64_t i = p + iC_first ;
                    GB_ISET (Ri, pR + p, i) ;   // Ri [pR + p] = i ;
                    int64_t iM = (pM < pM_end) ? GBi_M (Mi,pM,vlen) : INT64_MAX;
                    bool mij = false ;
                    if (i == iM)
                    { 
                        mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                        pM++ ;
                    }
                    if (GB_MASK_COMP) mij = !mij ;
                    #ifndef GB_ISO_MASKER
                    if (mij)
                    { 
                        // R(i,j) = Z (i,j)
                        GB_COPY_Z_TO_R (Rx, pR+p, Zx, pZ+p, Z_iso, rsize) ;
                    }
                    else
                    { 
                        // R(i,j) = C (i,j)
                        GB_COPY_C_TO_R (Rx, pR+p, Cx, pC+p, C_iso, rsize) ;
                    }
                    #endif
                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // Method04: 2-way merge of C(:,j) and Z(:,j)
                //--------------------------------------------------------------

                // Z is sparse or hypersparse; M has any sparsity structure
                ASSERT (GB_IS_SPARSE (Z) || GB_IS_HYPERSPARSE (Z)) ;

                //--------------------------------------------------------------
                // Z is sparse or hypersparse, M has any sparsity
                //--------------------------------------------------------------

                //      ------------------------------------------
                //      C       <!M> =       Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse
                //      sparse  bitmap      sparse          sparse
                //      sparse  full        sparse          sparse

                //      ------------------------------------------
                //      C       <M> =        Z              R
                //      ------------------------------------------

                //      sparse  sparse      sparse          sparse
                //      sparse  bitmap      sparse          sparse
                //      sparse  full        sparse          sparse

                while (pC < pC_end && pZ < pZ_end)
                {

                    //----------------------------------------------------------
                    // get the next i for R(:,j)
                    //----------------------------------------------------------

                    int64_t iC = GB_IGET (Ci, pC) ;
                    int64_t iZ = GB_IGET (Zi, pZ) ;
                    int64_t i = GB_IMIN (iC, iZ) ;

                    //----------------------------------------------------------
                    // get M(i,j)
                    //----------------------------------------------------------

                    bool mij = false ;

                    if (mdense)
                    { 

                        //------------------------------------------------------
                        // Method04a: M(:,j) is dense
                        //------------------------------------------------------

                        // mask is dense, lookup M(i,j)
                        // iM_first == Mi [pM_first]
                        // iM_first + delta == Mi [pM_first + delta]
                        // let i = iM_first + delta
                        // let pM = pM_first + delta
                        // then delta = i - iM_first
                        pM = pM_first + (i - iM_first) ;
                        ASSERT (i == GBi_M (Mi, pM, vlen)) ;
                        mij = GBb_M (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                        // increment pM for the wrapup phase below
                        pM++ ;

                    }
                    else
                    {

                        //------------------------------------------------------
                        // Method04b: M(:,j) is sparse
                        //------------------------------------------------------

                        // Use GB_split_binary_search so that pM can be used in
                        // the for loop with index pM in the wrapup phase.
                        ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                        int64_t pright = pM_end - 1 ;
                        bool found ;
                        found = GB_split_binary_search (i, Mi, GB_Mi_IS_32,
                            &pM, &pright) ;
                        if (found)
                        { 
                            ASSERT (i == GB_IGET (Mi, pM)) ;
                            mij = GB_MCAST (Mx, pM, msize) ;
                            // increment pM for the wrapup phase below
                            pM++ ;
                        }
                    }

                    if (GB_MASK_COMP) mij = !mij ;

                    //----------------------------------------------------------
                    // R(i,j) = C(i,j) or Z(i,j)
                    //----------------------------------------------------------

                    if (iC < iZ)
                    { 
                        // C(i,j) is present but Z(i,j) is not
                        if (!mij) GB_COPY_C (i) ;
                        pC++ ;
                    }
                    else if (iC > iZ)
                    { 
                        // Z(i,j) is present but C(i,j) is not
                        if (mij) GB_COPY_Z (i) ;
                        pZ++ ;
                    }
                    else
                    {
                        // both C(i,j) and Z(i,j) are present
                        if (mij)
                        { 
                            GB_COPY_Z (iC) ;
                        }
                        else
                        { 
                            GB_COPY_C (iC) ;
                        }
                        pC++ ;
                        pZ++ ;
                    }
                }

                //--------------------------------------------------------------
                // Method04: wrapup: C or Z are exhausted, or initially empty
                //--------------------------------------------------------------

                cjnz = pC_end - pC ;    // nnz (C(:,j)) remaining
                zjnz = pZ_end - pZ ;    // nnz (Z(:,j)) remaining
                mjnz = pM_end - pM ;    // nnz (M(:,j)) remaining

                if (cjnz == 0)
                {

                    //----------------------------------------------------------
                    // C(:,j) is empty
                    //----------------------------------------------------------

                    if (!GB_MASK_COMP)
                    {

                        //------------------------------------------------------
                        // mask is not complemented
                        //------------------------------------------------------

                        if (mdense)
                        {

                            //--------------------------------------------------
                            // Method04c: M(:,j) is dense
                            //--------------------------------------------------

                            for ( ; pZ < pZ_end ; pZ++)
                            { 
                                int64_t i = GB_IGET (Zi, pZ) ;
                                // mask is dense, lookup M(i,j)
                                pM = pM_first + (i - iM_first) ;
                                ASSERT (i == GBi_M (Mi, pM, vlen)) ;
                                bool mij = GBb_M (Mb, pM) &&
                                           GB_MCAST (Mx, pM, msize) ;
                                if (mij) GB_COPY_Z (i) ;
                            }

                        }
                        else if (zjnz > 32 * mjnz)
                        {

                            //--------------------------------------------------
                            // Method04d: Z(:,j) is much denser than M(:,j)
                            //--------------------------------------------------

                            // This loop requires pM to start at the first
                            // entry in M(:,j) that has not yet been handled.

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pM < pM_end ; pM++)
                            {
                                if (GB_MCAST (Mx, pM, msize))
                                { 
                                    int64_t i = GB_IGET (Mi, pM) ;
                                    int64_t pright = pZ_end - 1 ;
                                    bool found ;
                                    found = GB_binary_search (i, Zi,GB_Zi_IS_32,
                                        &pZ, &pright) ;
                                    if (found) GB_COPY_Z (i) ;
                                }
                            }

                        }
                        else if (mjnz > 32 * zjnz)
                        {

                            //--------------------------------------------------
                            // Method04e: M(:,j) is much denser than Z(:,j)
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pZ < pZ_end ; pZ++)
                            { 
                                // wrapup, C now empty, M(:,j) much denser than
                                // Z(:,j); C, M and Z all sparse
                                int64_t i = GB_IGET (Zi, pZ) ;
                                bool mij = false ;
                                int64_t pright = pM_end - 1 ;
                                bool found ;
                                found = GB_binary_search (i, Mi, GB_Mi_IS_32,
                                    &pM, &pright) ;
                                if (found) mij = GB_MCAST (Mx, pM, msize) ;
                                if (mij) GB_COPY_Z (i) ;
                            }

                        }
                        else
                        {

                            //--------------------------------------------------
                            // Method04f: M(:,j) and Z(:,j) about same # entries
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            while (pM < pM_end && pZ < pZ_end)
                            {
                                int64_t iM = GB_IGET (Mi, pM) ;
                                int64_t i = GB_IGET (Zi, pZ) ;
                                if (iM < i)
                                { 
                                    // M(i,j) exists but not Z(i,j)
                                    pM++ ;
                                }
                                else if (i < iM)
                                { 
                                    // Z(i,j) exists but not M(i,j)
                                    pZ++ ;
                                }
                                else
                                { 
                                    // both M(i,j) and Z(i,j) exist
                                    if (GB_MCAST (Mx, pM, msize)) GB_COPY_Z (i);
                                    pM++ ;
                                    pZ++ ;
                                }
                            }
                        }

                    }
                    else
                    {

                        //------------------------------------------------------
                        // complemented mask, and C(:,j) empty
                        //------------------------------------------------------

                        if (mdense)
                        {

                            //--------------------------------------------------
                            // Method04g: M(:,j) is dense
                            //--------------------------------------------------

                            for ( ; pZ < pZ_end ; pZ++)
                            { 
                                int64_t i = GB_IGET (Zi, pZ) ;
                                // mask is dense, lookup M(i,j)
                                pM = pM_first + (i - iM_first) ;
                                ASSERT (i == GBi_M (Mi, pM, vlen)) ;
                                bool mij = GBb_M (Mb, pM) &&
                                           GB_MCAST (Mx, pM, msize) ;
                                if (!mij) GB_COPY_Z (i) ; // mask complemented
                            }
                        }
                        else
                        {

                            //--------------------------------------------------
                            // Method04h: M(:,j) is sparse
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pZ < pZ_end ; pZ++)
                            { 
                                int64_t i = GB_IGET (Zi, pZ) ;
                                bool mij = false ;
                                int64_t pright = pM_end - 1 ;
                                bool found ;
                                found = GB_binary_search (i, Mi, GB_Mi_IS_32,
                                    &pM, &pright) ;
                                if (found) mij = GB_MCAST (Mx, pM, msize) ;
                                if (!mij) GB_COPY_Z (i) ; // mask complemented
                            }
                        }
                    }

                }
                else if (zjnz == 0)
                {

                    //----------------------------------------------------------
                    // Z(:,j) is empty
                    //----------------------------------------------------------

                    if (GB_MASK_COMP)
                    {

                        //------------------------------------------------------
                        // mask is complemented
                        //------------------------------------------------------

                        if (mdense)
                        {

                            //--------------------------------------------------
                            // Method04i: M(:,j) is dense
                            //--------------------------------------------------

                            for ( ; pC < pC_end ; pC++)
                            { 
                                int64_t i = GB_IGET (Ci, pC) ;
                                // mask is dense, lookup M(i,j)
                                pM = pM_first + (i - iM_first) ;
                                ASSERT (i == GBi_M (Mi, pM, vlen)) ;
                                bool mij = GBb_M (Mb, pM) &&
                                           GB_MCAST (Mx, pM, msize) ;
                                if (mij) GB_COPY_C (i) ;
                            }

                        }
                        else if (cjnz > 32 * mjnz)
                        {

                            //--------------------------------------------------
                            // Method04j: C(:,j) is much denser than M(:,j)
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pM < pM_end ; pM++)
                            {
                                if (GB_MCAST (Mx, pM, msize))
                                { 
                                    int64_t i = GB_IGET (Mi, pM) ;
                                    int64_t pright = pC_end - 1 ;
                                    bool found ;
                                    found = GB_binary_search (i, Ci,GB_Ci_IS_32,
                                        &pC, &pright) ;
                                    if (found) GB_COPY_C (i) ;
                                }
                            }

                        }
                        else if (mjnz > 32 * cjnz)
                        {

                            //--------------------------------------------------
                            // Method04k: M(:,j) is much denser than C(:,j)
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pC < pC_end ; pC++)
                            { 
                                int64_t i = GB_IGET (Ci, pC) ;
                                bool mij = false ;
                                int64_t pright = pM_end - 1 ;
                                bool found ;
                                found = GB_binary_search (i, Mi, GB_Mi_IS_32,
                                    &pM, &pright) ;
                                if (found) mij = GB_MCAST (Mx, pM, msize) ;
                                if (mij) GB_COPY_C (i) ;
                            }

                        }
                        else
                        {

                            //--------------------------------------------------
                            // Method04l: M(:,j) and C(:,j) about same # entries
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            while (pM < pM_end && pC < pC_end)
                            {
                                int64_t iM = GB_IGET (Mi, pM) ;
                                int64_t i = GB_IGET (Ci, pC) ;
                                if (iM < i)
                                { 
                                    // M(i,j) exists but not C(i,j)
                                    pM++ ;
                                }
                                else if (i < iM)
                                { 
                                    // C(i,j) exists but not M(i,j)
                                    pC++ ;
                                }
                                else
                                { 
                                    // both M(i,j) and C(i,j) exist
                                    if (GB_MCAST (Mx, pM, msize)) GB_COPY_C (i);
                                    pM++ ;
                                    pC++ ;
                                }
                            }
                        }

                    }
                    else
                    {

                        //------------------------------------------------------
                        // non-complemented mask, and Z(:,j) empty
                        //------------------------------------------------------

                        if (mdense)
                        {

                            //--------------------------------------------------
                            // Method04m: M(:,j) is dense
                            //--------------------------------------------------

                            for ( ; pC < pC_end ; pC++)
                            { 
                                int64_t i = GB_IGET (Ci, pC) ;
                                // mask is dense, lookup M(i,j)
                                pM = pM_first + (i - iM_first) ;
                                ASSERT (i == GBi_M (Mi, pM, vlen)) ;
                                bool mij = GBb_M (Mb, pM) &&
                                           GB_MCAST (Mx, pM, msize) ;
                                if (!mij) GB_COPY_C (i) ;
                            }
                        }
                        else
                        {

                            //--------------------------------------------------
                            // Method04n: M(:,j) is sparse
                            //--------------------------------------------------

                            ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
                            for ( ; pC < pC_end ; pC++)
                            { 
                                int64_t i = GB_IGET (Ci, pC) ;
                                // M(i,j) false if not present
                                bool mij = false ; 
                                int64_t pright = pM_end - 1 ;
                                bool found ;
                                found = GB_binary_search (i, Mi, GB_Mi_IS_32,
                                    &pM, &pright) ;
                                if (found) mij = GB_MCAST (Mx, pM, msize) ;
                                if (!mij) GB_COPY_C (i) ;
                            }
                        }
                    }
                }

                #if defined ( GB_PHASE_2_OF_2 )
                ASSERT (pR == pR_end) ;
                #endif
            }

            //------------------------------------------------------------------
            // final count of nnz (R(:,j))
            //------------------------------------------------------------------

            #if defined ( GB_PHASE_1_OF_2 )
            if (fine_task)
            { 
                TaskList [taskid].pC = rjnz ;
            }
            else
            { 
                GB_ISET (Rp, k, rjnz) ; // Rp [k] = rjnz ;
            }
            #endif
        }
    }
}

