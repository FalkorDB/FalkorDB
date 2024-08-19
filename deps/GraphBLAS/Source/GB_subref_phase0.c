//------------------------------------------------------------------------------
// GB_subref_phase0: find vectors of C = A(I,J) and determine I,J properties
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

#include "GB_subref.h"

#define GB_Ai(p) GBI_UNFLIP (Ai, p, avlen)

//------------------------------------------------------------------------------
// GB_find_Ap_start_end
//------------------------------------------------------------------------------

// Find pA and pA_end so that Ai,Ax [pA:pA_end-1] contains the vector
// A(imin:imax,kA).  If A(:,kA) is dense, [pA:pA_end-1] is the entire dense
// vector (it is not trimmed).  Otherwise, if A(imin:imax,kA) is empty, then
// pA and pA_end are set to -1 to denote an empty list.  The resulting pointers
// are then returned in Ap_start [kC] and Ap_end [kC].

static inline void GB_find_Ap_start_end
(
    // input, not modified
    const int64_t kA,
    const int64_t *restrict Ap,
    const int64_t *restrict Ai,
    const int64_t avlen,
    const int64_t imin,
    const int64_t imax,
    const int64_t kC,
    const int64_t nzombies,
    // output: Ap_start [kC] and Ap_end [kC]:
    int64_t *restrict Ap_start,
    int64_t *restrict Ap_end
)
{

    //--------------------------------------------------------------------------
    // get A(:,kA)
    //--------------------------------------------------------------------------

    int64_t pA     = GBP (Ap, kA, avlen) ;
    int64_t pA_end = GBP (Ap, kA+1, avlen) ;
    int64_t ajnz = pA_end - pA ;

    //--------------------------------------------------------------------------
    // trim it to A(imin:imax,kA)
    //--------------------------------------------------------------------------

    if (ajnz == avlen)
    { 

        //----------------------------------------------------------------------
        // A (:,kA) is dense; use pA and pA_end as-is
        //----------------------------------------------------------------------

        ;

    }
    else if (ajnz == 0 || GB_Ai (pA) > imax || GB_Ai (pA_end-1) < imin)
    { 

        //----------------------------------------------------------------------
        // intersection of A(:,kA) and imin:imax is empty
        //----------------------------------------------------------------------

        pA = -1 ;
        pA_end = -1 ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A (:,kA) is sparse, with at least one entry
        //----------------------------------------------------------------------

        // trim the leading part of A(:,kA)
        if (GB_Ai (pA) < imin)
        { 
            bool found, is_zombie ;
            int64_t pright = pA_end - 1 ;
            GB_SPLIT_BINARY_SEARCH_ZOMBIE (imin, Ai,
                pA, pright, found, nzombies, is_zombie) ;
        }

        // trim the trailing part of A (:,kA)
        if (imin == imax)
        {
            if (GB_Ai (pA) == imin)
            { 
                // found the the single entry A (i,kA)
                pA_end = pA + 1 ;
            }
            else
            { 
                // A (i,kA) has not been found
                pA = -1 ;
                pA_end = -1 ;
            }
        }
        else if (imax < GB_Ai (pA_end-1))
        { 
            bool found, is_zombie ;
            int64_t pleft = pA ;
            int64_t pright = pA_end - 1 ;
            GB_SPLIT_BINARY_SEARCH_ZOMBIE (imax, Ai,
                pleft, pright, found, nzombies, is_zombie) ;
            pA_end = (found) ? (pleft + 1) : pleft ;
        }

        #ifdef GB_DEBUG
        ajnz = pA_end - pA ;
        if (ajnz > 0 && Ap != NULL)
        {
            // A(imin:imax,kA) is now in Ai [pA:pA_end-1]
            ASSERT (GB_IMPLIES (Ap [kA] < pA,  GB_Ai (pA-1) < imin)) ;
            ASSERT (GB_IMPLIES (pA_end < Ap [kA+1], imax < GB_Ai (pA_end))) ;
            ASSERT (imin <= GB_Ai (pA)) ;
            ASSERT (GB_Ai (pA_end-1) <= imax) ;
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The result [pA:pA_end-1] defines the range of entries that need to be
    // accessed for constructing C(:,kC).

    Ap_start [kC] = pA ;
    Ap_end   [kC] = pA_end ;
}

//------------------------------------------------------------------------------
// GB_subref_phase0
//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE           \
{                                   \
    GB_WERK_POP (Count, int64_t) ;  \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_FREE (&Ch, Ch_size) ;                    \
    GB_FREE_WORK (&Ap_start, Ap_start_size) ;   \
    GB_FREE_WORK (&Ap_end, Ap_end_size) ;       \
}

GrB_Info GB_subref_phase0
(
    // output
    int64_t *restrict *p_Ch,         // Ch = C->h hyperlist, or NULL standard
    size_t *p_Ch_size,
    int64_t *restrict *p_Ap_start,   // A(:,kA) starts at Ap_start [kC]
    size_t *p_Ap_start_size,
    int64_t *restrict *p_Ap_end,     // ... and ends at Ap_end [kC] - 1
    size_t *p_Ap_end_size,
    int64_t *p_Cnvec,       // # of vectors in C
    bool *p_need_qsort,     // true if C must be sorted
    int *p_Ikind,           // kind of I
    int64_t *p_nI,          // length of I
    int64_t Icolon [3],     // for GB_RANGE, GB_STRIDE
    int64_t *p_nJ,          // length of J
    // input, not modified
    const GrB_Matrix A,
    const GrB_Index *I,     // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,       // length of I, or special
    const GrB_Index *J,     // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,       // length of J, or special
//  const bool must_sort,   // true if C must be returned sorted
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A for subref phase 0", GB0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;    // GB_bitmap_subref is used instead

    ASSERT (p_Ch != NULL) ;
    ASSERT (p_Ap_start != NULL) ;
    ASSERT (p_Ap_end != NULL) ;
    ASSERT (p_Cnvec != NULL) ;
    ASSERT (p_nJ != NULL) ;
    ASSERT (p_Ikind != NULL) ;
    ASSERT (p_nI != NULL) ;
    ASSERT (Icolon != NULL) ;
    ASSERT (I != NULL) ;
    ASSERT (J != NULL) ;

    GrB_Info info ;
    GB_WERK_DECLARE (Count, int64_t) ;
    int64_t *restrict Ch       = NULL ; size_t Ch_size = 0 ;
    int64_t *restrict Ap_start = NULL ; size_t Ap_start_size = 0 ;
    int64_t *restrict Ap_end   = NULL ; size_t Ap_end_size = 0 ;

    (*p_Ch        ) = NULL ;
    (*p_Ap_start  ) = NULL ;
    (*p_Ap_end    ) = NULL ;
    (*p_Cnvec     ) = 0 ;
    (*p_need_qsort) = false ;
    (*p_Ikind     ) = 0 ;
    (*p_nI        ) = 0 ;
    (*p_nJ        ) = 0 ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t *restrict Ap = A->p ;   // Ap (but not A->p) may be trimmed
    int64_t *restrict Ah = A->h ;   // Ah (but not A->h) may be trimmed
    int64_t *restrict Ai = A->i ;
    int64_t anvec = A->nvec ;       // may be trimmed
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;
    int64_t nzombies = A->nzombies ;

    //--------------------------------------------------------------------------
    // check the properties of I and J
    //--------------------------------------------------------------------------

    // C = A(I,J) so I is in range 0:avlen-1 and J is in range 0:avdim-1
    int64_t nI, nJ, Jcolon [3] ;
    int Ikind, Jkind ;
    GB_ijlength (I, ni, avlen, &nI, &Ikind, Icolon) ;
    GB_ijlength (J, nj, avdim, &nJ, &Jkind, Jcolon) ;

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;

    info = GB_ijproperties (I, ni, nI, avlen, &Ikind, Icolon,
        &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // I invalid or out of memory
        return (info) ;
    }

    info = GB_ijproperties (J, nj, nJ, avdim, &Jkind, Jcolon,
        &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // J invalid or out of memory
        return (info) ;
    }

    bool need_qsort = I_unsorted ;

    //--------------------------------------------------------------------------
    // determine if C is empty
    //--------------------------------------------------------------------------

    bool C_empty = (nI == 0 || nJ == 0) ;
    bool A_is_hyper = (Ah != NULL) ;

    //--------------------------------------------------------------------------
    // trim the hyperlist of A for (J = jbegin:jend case only)
    //--------------------------------------------------------------------------

    // Ah, Ap, and anvec are modified to include just the vectors in range
    // jmin:jmax, inclusive.  A itself is not modified, just the Ah and Ap
    // pointers, and the scalar anvec.  If J is ":", then jmin is zero and
    // jmax is avdim-1, so there is nothing to trim from Ah.  If C is empty,
    // then Ah and Ap will not be accessed at all, so this can be skipped.

    if (!C_empty && A_is_hyper && Jkind == GB_RANGE)
    {

        //----------------------------------------------------------------------
        // trim the leading end of Ah so that it starts with jmin:...
        //----------------------------------------------------------------------

        if (jmin > 0)
        { 
            bool found ;
            int64_t kleft = 0 ;
            int64_t kright = anvec-1 ;
            GB_SPLIT_BINARY_SEARCH (jmin, Ah, kleft, kright, found) ;
            Ah += kleft ;
            Ap += kleft ;
            anvec -= kleft ;
        }

        //----------------------------------------------------------------------
        // trim the trailing end of Ah so that it ends with ..:jmax
        //----------------------------------------------------------------------

        if (jmax < avdim-1)
        { 
            bool found ;
            int64_t kleft = 0 ;
            int64_t kright = anvec-1 ;
            GB_SPLIT_BINARY_SEARCH (jmax, Ah, kleft, kright, found) ;
            anvec = (found) ? (kleft + 1) : kleft ;
        }

        // Ah has been trimmed
        ASSERT (GB_IMPLIES (anvec > 0, jmin <= Ah [0] && Ah [anvec-1] <= jmax));
    }

    // Ah may now be empty, after being trimmed
    C_empty = C_empty || (anvec == 0) ;

    //--------------------------------------------------------------------------
    // build the hyper_hash, if needed
    //--------------------------------------------------------------------------

    bool J_is_all_or_range = (Jkind == GB_ALL || Jkind == GB_RANGE) ;
    bool J_is_long_stride = (Jkind == GB_STRIDE && anvec < nJ * 64) ;

    bool use_hyper_hash = !C_empty && A_is_hyper &&
            !J_is_all_or_range && !J_is_long_stride &&
            (A->Y != NULL || nJ > anvec) ;
    if (use_hyper_hash)
    { 
        GB_OK (GB_hyper_hash_build (A, Werk)) ;
    }

    const int64_t *restrict A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *restrict A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *restrict A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    #define NTASKS_PER_THREAD 8
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = 1, ntasks = 1 ;
    int ntasks_max = nthreads_max * NTASKS_PER_THREAD ;

    #define GB_GET_NTHREADS_AND_NTASKS(work)                            \
    {                                                                   \
        nthreads = GB_nthreads (work, chunk, nthreads_max) ;            \
        ntasks = (nthreads == 1) ? 1 : (NTASKS_PER_THREAD * nthreads) ; \
        ntasks = GB_IMIN (ntasks, work) ;                               \
        ntasks = GB_IMAX (ntasks, 1) ;                                  \
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Count, ntasks_max+1, int64_t) ;
    if (Count == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // compute Cnvec and determine the format of Ch
    //--------------------------------------------------------------------------

    // Ch is an explicit or implicit array of size Cnvec <= nJ.  jC = Ch [kC]
    // if C(:,jC) is the (kC)th vector of C.  If NULL, then C is standard, and
    // jC == kC.  jC is in the range 0 to nJ-1.

    int64_t Cnvec = 0 ;

    int64_t jbegin = Jcolon [GxB_BEGIN] ;
    int64_t jinc   = Jcolon [GxB_INC  ] ;

    if (C_empty)
    { 

        //----------------------------------------------------------------------
        // C is an empty hypersparse matrix
        //----------------------------------------------------------------------

        ;

    }
    else if (!A_is_hyper)
    { 

        //----------------------------------------------------------------------
        // both C and A are standard matrices
        //----------------------------------------------------------------------

        Cnvec = nJ ;
        GB_GET_NTHREADS_AND_NTASKS (nJ) ;

    }
    else if (J_is_all_or_range) // (Jkind == GB_ALL || Jkind == GB_RANGE)
    { 

        //----------------------------------------------------------------------
        // J is ":" or jbegin:jend
        //----------------------------------------------------------------------

        // For the case where J is jbegin:jend, Ah has been trimmed (see above).
        // Ch is a shifted copy of the trimmed Ah, of length Cnvec = anvec.  
        // so kA = kC, and jC = Ch [kC] = jA - jmin.  Ap has also been trimmed.

        Cnvec = anvec ;
        ASSERT (Cnvec <= nJ) ;
        GB_GET_NTHREADS_AND_NTASKS (anvec) ;

    }
    else if (J_is_long_stride) // (Jkind == GB_STRIDE && anvec < nJ * 64)
    {

        //----------------------------------------------------------------------
        // J is jbegin:jinc:jend, but J is large
        //----------------------------------------------------------------------

        // The case for Jkind == GB_STRIDE can be done by either this method,
        // or the one below.  This takes O(anvec) time, and the one below
        // takes O(nj*log2(anvec)), so use this method if anvec < nj * 64.

        // Ch is a list of length Cnvec, where Cnvec is the length of
        // the intersection of Ah and jbegin:jinc:jend.

        // count the length of Ch
        Cnvec = 0 ;

        GB_GET_NTHREADS_AND_NTASKS (anvec) ;

        // scan all of Ah and check each entry if it appears in J
        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t kA_start, kA_end, my_Cnvec = 0 ;
            GB_PARTITION (kA_start, kA_end, anvec,
                (jinc > 0) ? tid : (ntasks-tid-1), ntasks) ;
            for (int64_t kA = kA_start ; kA < kA_end ; kA++)
            {
                int64_t jA = Ah [kA] ;
                if (GB_ij_is_in_list (J, nJ, jA, GB_STRIDE, Jcolon))
                { 
                    my_Cnvec++ ;
                }
            }
            Count [tid] = my_Cnvec ;
        }

        GB_cumsum (Count, ntasks, NULL, 1, NULL) ;
        Cnvec = Count [ntasks] ;

    }
    else // Jkind == GB_LIST or GB_STRIDE
    {

        //----------------------------------------------------------------------
        // J is an explicit list, or jbegin:jinc:end
        //----------------------------------------------------------------------

        // Ch is an explicit list: the intersection of Ah and J

        // count the length of Ch
        Cnvec = 0 ;

        GB_GET_NTHREADS_AND_NTASKS (nJ) ;

        // scan all of J and check each entry if it appears in Ah

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jC_start, jC_end ;
            GB_PARTITION (jC_start, jC_end, nJ, tid, ntasks) ;
            int64_t my_Cnvec = 0 ;
            for (int64_t jC = jC_start ; jC < jC_end ; jC++)
            {
                int64_t jA = GB_ijlist (J, jC, Jkind, Jcolon) ;
                bool found ;
                int64_t kA = 0 ;
                if (use_hyper_hash)
                { 
                    // find jA using the hyper_hash
                    int64_t ignore1, ignore2 ;
                    kA = GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx,
                        A_hash_bits, jA, &ignore1, &ignore2) ;
                    found = (kA >= 0) ;
                }
                else
                { 
                    // find jA using binary search
                    int64_t kright = anvec-1 ;
                    GB_BINARY_SEARCH (jA, Ah, kA, kright, found) ;
                }
                if (found)
                { 
                    my_Cnvec++ ;
                }
            }
            Count [tid] = my_Cnvec ;
        }

        GB_cumsum (Count, ntasks, NULL, 1, NULL) ;
        Cnvec = Count [ntasks] ;
    }

    //--------------------------------------------------------------------------
    // allocate Ch, Ap_start, and Ap_end
    //--------------------------------------------------------------------------

    C_empty = C_empty || (Cnvec == 0) ;

    // C is hypersparse if A is hypersparse, or if C is empty
    bool C_is_hyper = A_is_hyper || C_empty ;

    if (C_is_hyper)
    {
        Ch = GB_MALLOC (Cnvec, int64_t, &Ch_size) ;
        if (Ch == NULL)
        { 
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    if (Cnvec > 0)
    {
        Ap_start = GB_MALLOC_WORK (Cnvec, int64_t, &Ap_start_size) ;
        Ap_end   = GB_MALLOC_WORK (Cnvec, int64_t, &Ap_end_size) ;
        if (Ap_start == NULL || Ap_end == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // create Ch, Ap_start, and Ap_end
    //--------------------------------------------------------------------------

    // For the (kC)th vector of C, which corresponds to the (kA)th vector of A,
    // pA = Ap_start [kC] and pA_end = Ap_end [kC] are pointers to the range
    // of entries in A(imin:imax,kA).

    if (C_empty)
    { 

        //----------------------------------------------------------------------
        // C is an empty hypersparse matrix
        //----------------------------------------------------------------------

        ;

    }
    else if (!A_is_hyper)
    {

        //----------------------------------------------------------------------
        // both C and A are standard matrices
        //----------------------------------------------------------------------

        int64_t jC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (jC = 0 ; jC < nJ ; jC++)
        { 
            int64_t jA = GB_ijlist (J, jC, Jkind, Jcolon) ;
            GB_find_Ap_start_end (jA, Ap, Ai, avlen, imin, imax,
                jC, nzombies, Ap_start, Ap_end) ;
        }

    }
    else if (J_is_all_or_range) // (Jkind == GB_ALL || Jkind == GB_RANGE)
    {

        //----------------------------------------------------------------------
        // J is ":" or jbegin:jend
        //----------------------------------------------------------------------

        // C and A are both hypersparse.  Ch is a shifted copy of the trimmed
        // Ah, of length Cnvec = anvec.  so kA = kC.  Ap has also been trimmed.

        int64_t kC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (kC = 0 ; kC < Cnvec ; kC++)
        { 
            int64_t kA = kC ;
            int64_t jA = Ah [kA] ;
            int64_t jC = jA - jmin ;
            Ch [kC] = jC ;
            GB_find_Ap_start_end (kA, Ap, Ai, avlen, imin, imax,
                kC, nzombies, Ap_start, Ap_end) ;
        }

    }
    else if (J_is_long_stride) // (Jkind == GB_STRIDE && anvec < nJ * 64)
    {

        //----------------------------------------------------------------------
        // J is jbegin:jinc:jend where jinc may be positive or negative
        //----------------------------------------------------------------------

        // C and A are both hypersparse.  Ch is constructed by scanning all
        // vectors in Ah [0..anvec-1] and checking if they appear in the
        // jbegin:jinc:jend sequence.

        if (jinc > 0)
        {
            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < ntasks ; tid++)
            {
                int64_t kA_start, kA_end ;
                GB_PARTITION (kA_start, kA_end, anvec, tid, ntasks) ;
                int64_t kC = Count [tid] ;
                for (int64_t kA = kA_start ; kA < kA_end ; kA++)
                {
                    int64_t jA = Ah [kA] ;
                    if (GB_ij_is_in_list (J, nJ, jA, GB_STRIDE, Jcolon))
                    { 
                        int64_t jC = (jA - jbegin) / jinc ;
                        Ch [kC] = jC ;
                        GB_find_Ap_start_end (kA, Ap, Ai, avlen, imin, imax,
                            kC, nzombies, Ap_start, Ap_end) ;
                        kC++ ;
                    }
                }
            }
        }
        else
        {
            int tid;
            #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < ntasks ; tid++)
            {
                int64_t kA_start, kA_end ;
                GB_PARTITION (kA_start, kA_end, anvec, ntasks-tid-1, ntasks) ;
                int64_t kC = Count [tid] ;
                for (int64_t kA = kA_end-1 ; kA >= kA_start ; kA--)
                {
                    int64_t jA = Ah [kA] ;
                    if (GB_ij_is_in_list (J, nJ, jA, GB_STRIDE, Jcolon))
                    { 
                        int64_t jC = (jA - jbegin) / jinc ;
                        Ch [kC] = jC ;
                        GB_find_Ap_start_end (kA, Ap, Ai, avlen, imin, imax,
                            kC, nzombies, Ap_start, Ap_end) ;
                        kC++ ;
                    }
                }
            }
        }

    }
    else // Jkind == GB_LIST or GB_STRIDE
    {

        //----------------------------------------------------------------------
        // J is an explicit list, or jbegin:jinc:jend
        //----------------------------------------------------------------------

        // C and A are both hypersparse.  Ch is constructed by scanning the
        // list J, or the entire jbegin:jinc:jend sequence.  Each vector is
        // then found in Ah, via binary search.

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jC_start, jC_end ;
            GB_PARTITION (jC_start, jC_end, nJ, tid, ntasks) ;
            int64_t kC = Count [tid] ;
            for (int64_t jC = jC_start ; jC < jC_end ; jC++)
            {
                int64_t jA = GB_ijlist (J, jC, Jkind, Jcolon) ;
                bool found ;
                int64_t kA = 0 ;
                if (use_hyper_hash)
                { 
                    // find jA using the hyper_hash
                    int64_t ignore1, ignore2 ;
                    kA = GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx,
                        A_hash_bits, jA, &ignore1, &ignore2) ;
                    found = (kA >= 0) ;
                }
                else
                { 
                    // find jA using binary search
                    int64_t kright = anvec-1 ;
                    GB_BINARY_SEARCH (jA, Ah, kA, kright, found) ;
                }
                if (found)
                { 
                    ASSERT (jA == Ah [kA]) ;
                    Ch [kC] = jC ;
                    GB_find_Ap_start_end (kA, Ap, Ai, avlen, imin, imax,
                        kC, nzombies, Ap_start, Ap_end) ;
                    kC++ ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    for (int64_t kC = 0 ; kC < Cnvec ; kC++)
    {
        // jC is the (kC)th vector of C = A(I,J)
        int64_t jC = GBH (Ch, kC) ;
        int64_t jA = GB_ijlist (J, jC, Jkind, Jcolon) ;
        // jA is the corresponding (kA)th vector of A.
        int64_t kA = 0 ;
        int64_t pright = A->nvec - 1 ;
        int64_t pA_start_all, pA_end_all ;
        bool found = GB_lookup (A->h != NULL,   // for debug only
            A->h, A->p, A->vlen, &kA, pright, jA, &pA_start_all, &pA_end_all) ;
        if (found && A->h != NULL)
        {
            ASSERT (jA == A->h [kA]) ;
        }
        int64_t pA      = Ap_start [kC] ;
        int64_t pA_end  = Ap_end   [kC] ;
        int64_t ajnz = pA_end - pA ;
        if (ajnz == avlen)
        {
            // A(:,kA) is dense; Ai [pA:pA_end-1] is the entire vector.
            // C(:,kC) will have exactly nI entries.
            ASSERT (pA     == pA_start_all) ;
            ASSERT (pA_end == pA_end_all  ) ;
            ;
        }
        else if (ajnz > 0)
        {
            // A(imin:imax,kA) has at least one entry, in Ai [pA:pA_end-1]
            ASSERT (imin <= GB_Ai (pA)) ;
            ASSERT (GB_Ai (pA_end-1) <= imax) ;
            ASSERT (pA_start_all <= pA && pA < pA_end && pA_end <= pA_end_all) ;
        }
        else
        {
            // A(imin:imax,kA) and C(:,kC) are empty
            ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    (*p_Ch        ) = Ch ;          (*p_Ch_size) = Ch_size ;
    (*p_Ap_start  ) = Ap_start ;    (*p_Ap_start_size) = Ap_start_size ;
    (*p_Ap_end    ) = Ap_end ;      (*p_Ap_end_size) = Ap_end_size ;
    (*p_Cnvec     ) = Cnvec ;
    (*p_need_qsort) = need_qsort ;
    (*p_Ikind     ) = Ikind ;
    (*p_nI        ) = nI ;
    (*p_nJ        ) = nJ ;
    return (GrB_SUCCESS) ;
}

