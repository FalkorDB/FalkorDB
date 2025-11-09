//------------------------------------------------------------------------------
// GB_add_phase0: find vectors of C to compute for C=A+B or C<M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The eWise add of two matrices, C=A+B, C<M>=A+B, or C<!M>=A+B starts with
// this phase, which determines which vectors of C need to be computed.
// This phase is also used for GB_masker, and for GB_SUBASSIGN_TWO_SLICE.

// On input, A and B are the two matrices being added, and M is the optional
// mask matrix (not complemented).  The complemented mask is handed in GB_mask,
// not here.

// On output, an integer (Cnvec) a boolean (Ch_to_Mh) and up to 3 arrays are
// returned, either NULL or of size Cnvec.  Let n = A->vdim be the vector
// dimension of A, B, M and C.

//      Ch:  the list of vectors to compute.  If not NULL, Ch [k] = j is the
//      kth vector in C to compute, which will become the hyperlist C->h of C.
//      Note that some of these vectors may turn out to be empty, because of
//      the mask, or because the vector j appeared in A or B, but is empty.
//      It is pruned at the end of GB_add_phase2.  If Ch is NULL then it is an
//      implicit list of size n, and Ch [k] == k for all k = 0:n-1.  In this
//      case, C will be a sparse matrix, not hypersparse.  Thus, the kth
//      vector is j = GBh (Ch, k).

//      Ch is freed by GB_add if phase1 fails.  phase2 either frees it or
//      transplants it into C, if C is hypersparse.

//      Ch_is_Mh:  true if the mask M is present, hypersparse, and not
//      complemented, false otherwise.  In this case Ch is a deep copy of Mh.
//      Only GB_add uses this option; it is not used by GB_masker or
//      GB_SUBASSIGN_TWO_SLICE (Ch_is_Mh is always false in this case).  This
//      is determined by passing in p_Ch_is_Mh as a NULL or non-NULL pointer.

//      C_to_A:  if A is hypersparse, then C_to_A [k] = kA if the kth vector,
//      j = GBh (Ch, k) appears in A, as j = Ah [kA].  If j does not appear in
//      A, then C_to_A [k] = -1.  If A is not hypersparse, then C_to_A is
//      returned as NULL.

//      C_to_B:  if B is hypersparse, then C_to_B [k] = kB if the kth vector,
//      j = GBh (Ch, k) appears in B, as j = Bh [kB].  If j does not appear in
//      B, then C_to_B [k] = -1.  If B is not hypersparse, then C_to_B is
//      returned as NULL.

//      C_to_M:  if M is hypersparse, and Ch_is_Mh is false, then C_to_M [k] =
//      kM if the kth vector, j = GBh (Ch, k) appears in M, as j = Mh [kM].  If
//      j does not appear in M, then C_to_M [k] = -1.  If M is not hypersparse,
//      then C_to_M is returned as NULL.

// M, A, B: any sparsity structure (hypersparse, sparse, bitmap, or full)
// C: not present here, but its sparsity structure is finalized, via the
// input/output parameter C_sparsity.

#define GB_FREE_WORKSPACE           \
{                                   \
    GB_WERK_POP (Work, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_MEMORY (&Ch, Ch_size) ;                \
    GB_FREE_MEMORY (&C_to_M, C_to_M_size) ;   \
    GB_FREE_MEMORY (&C_to_A, C_to_A_size) ;   \
    GB_FREE_MEMORY (&C_to_B, C_to_B_size) ;   \
    GB_FREE_WORKSPACE ;                     \
}

#include "add/GB_add.h"

//------------------------------------------------------------------------------
// GB_allocate_result
//------------------------------------------------------------------------------

static inline bool GB_allocate_result
(
    int64_t Cnvec,
    void **Ch_handle,                size_t *Ch_size_handle, size_t cjsize,
    int64_t *restrict *C_to_M_handle, size_t *C_to_M_size_handle,
    int64_t *restrict *C_to_A_handle, size_t *C_to_A_size_handle,
    int64_t *restrict *C_to_B_handle, size_t *C_to_B_size_handle
)
{
    bool ok = true ;
    if (Ch_handle != NULL)
    { 
        (*Ch_handle) = GB_MALLOC_MEMORY (Cnvec, cjsize, Ch_size_handle) ;
        ok = (*Ch_handle != NULL) ;
    }
    if (C_to_M_handle != NULL)
    { 
        (*C_to_M_handle) = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t),
            C_to_M_size_handle) ;
        ok = ok && (*C_to_M_handle != NULL) ;
    }
    if (C_to_A_handle != NULL)
    { 
        *C_to_A_handle = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t),
            C_to_A_size_handle) ;
        ok = ok && (*C_to_A_handle != NULL) ;
    }
    if (C_to_B_handle != NULL)
    { 
        *C_to_B_handle = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t),
            C_to_B_size_handle) ;
        ok = ok && (*C_to_B_handle != NULL) ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_add_phase0:  find the vectors of C for C<M>=A+B
//------------------------------------------------------------------------------

//  GrB_Info GB_add_phase0          // find vectors in C for C=A+B or C<M>=A+B
//  (
//      int64_t *p_Cnvec,           // # of vectors to compute in C
//      void **Ch_handle,           // Ch: size Cnvec, or NULL
//      size_t *Ch_size_handle,              // size of Ch in bytes
//      int64_t *restrict *C_to_M_handle,    // C_to_M: size Cnvec, or NULL
//      size_t *C_to_M_size_handle,          // size of C_to_M in bytes
//      int64_t *restrict *C_to_A_handle,    // C_to_A: size Cnvec, or NULL
//      size_t *C_to_A_size_handle,          // size of C_to_A in bytes
//      int64_t *restrict *C_to_B_handle,    // C_to_B: size Cnvec, or NULL
//      size_t *C_to_B_size_handle,          // size of C_to_A in bytes
//      bool *p_Ch_is_Mh,           // if true, then Ch == Mh
//      bool *p_Cp_is_32,           // if true, Cp is 32-bit; else 64-bit
//      bool *p_Cj_is_32,           // if true, Ch is 32-bit; else 64-bit
//      bool *p_Ci_is_32,           // if true, Ci is 32-bit; else 64-bit
//      int *C_sparsity,            // sparsity structure of C
//      const GrB_Matrix M,         // optional mask, may be NULL; not compl.
//      const GrB_Matrix A,         // first input matrix
//      const GrB_Matrix B,         // second input matrix
//      GB_Werk Werk
//  )

GB_CALLBACK_ADD_PHASE0_PROTO (GB_add_phase0)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // M, A, and B can be jumbled for this phase, but not phase1 or phase2

    GrB_Info info ;
    ASSERT (p_Cnvec != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    ASSERT (C_to_A_handle != NULL) ;
    ASSERT (C_to_B_handle != NULL) ;
    ASSERT (p_Cp_is_32 != NULL) ;
    ASSERT (p_Cj_is_32 != NULL) ;
    ASSERT (p_Ci_is_32 != NULL) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for add phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (M)) ;

    ASSERT_MATRIX_OK (A, "A for add phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for add phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (A->vdim == B->vdim) ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim)) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // initializations and check for quick return
    //--------------------------------------------------------------------------

    (*p_Cnvec) = 0 ;
    (*Ch_handle) = NULL ;
    if (C_to_M_handle != NULL)
    { 
        (*C_to_M_handle) = NULL ;
    }
    (*C_to_A_handle) = NULL ;
    (*C_to_B_handle) = NULL ;
    if (p_Ch_is_Mh != NULL)
    { 
        (*p_Ch_is_Mh) = false ;
    }

    if ((*C_sparsity) == GxB_BITMAP || (*C_sparsity) == GxB_FULL)
    { 
        // nothing to do in phase0 for C bitmap or full
        (*p_Cnvec) = A->vdim ;  // not needed; to be consistent with GB_emult
        (*p_Cp_is_32) = false ;
        (*p_Cj_is_32) = false ;
        (*p_Ci_is_32) = false ;
        return (GrB_SUCCESS) ;
    }

    GB_MDECL (Ch, , u) ; size_t Ch_size = 0 ;

    int64_t *restrict C_to_M = NULL ; size_t C_to_M_size = 0 ;
    int64_t *restrict C_to_A = NULL ; size_t C_to_A_size = 0 ;
    int64_t *restrict C_to_B = NULL ; size_t C_to_B_size = 0 ;

    GB_WERK_DECLARE (Work, int64_t) ;
    int ntasks = 0 ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = 1 ;      // nthreads depends on Cnvec, computed below

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    int64_t Cnvec ;

    int64_t n = A->vdim ;
    int64_t Anvec = A->nvec ;
    void *Ap = A->p ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    const bool A_is_hyper = (Ah != NULL) ;
    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;
    const int64_t anz = GB_nnz (A) ;

    int64_t Bnvec = B->nvec ;
    void *Bp = B->p ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    const bool B_is_hyper = (Bh != NULL) ;
    const bool Bp_is_32 = B->p_is_32 ;
    const bool Bj_is_32 = B->j_is_32 ;
    const int64_t bnz = GB_nnz (A) ;

    int64_t Mnvec = (M == NULL) ? 0 : M->nvec ;
    void *Mp = (M == NULL) ? NULL : M->p ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool Mp_is_32 = (M == NULL) ? false : M->p_is_32 ;
    const bool Mj_is_32 = (M == NULL) ? false : M->j_is_32 ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, anz + bnz, A->vlen, A->vdim, Werk) ;
    (*p_Cp_is_32) = Cp_is_32 ;
    (*p_Cj_is_32) = Cj_is_32 ;
    (*p_Ci_is_32) = Ci_is_32 ;
    size_t cjsize = (Cj_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    GB_Type_code cjcode = (Cj_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code mjcode = (Mj_is_32) ? GB_UINT32_code : GB_UINT64_code ;

    // For GB_add, if M is present, hypersparse, and not complemented, then C
    // will be hypersparse, and it will have the same set of vectors as M (Ch
    // will contain the same content as Mh).  For GB_masker, Ch is never equal
    // to Mh.
    bool Ch_is_Mh = (p_Ch_is_Mh != NULL) && (M != NULL && M_is_hyper) ;

    //--------------------------------------------------------------------------
    // find the set union of the non-empty vectors of A and B
    //--------------------------------------------------------------------------

    if (Ch_is_Mh)
    {

        //----------------------------------------------------------------------
        // C and M are hypersparse, with the same vectors as the hypersparse M
        //----------------------------------------------------------------------

        (*C_sparsity) = GxB_HYPERSPARSE ;
        ASSERT (M_is_hyper) ;
        Cnvec = Mnvec ;
        nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;

        if (!GB_allocate_result (Cnvec,
            &Ch, &Ch_size, cjsize,
            NULL, NULL,
            (A_is_hyper) ? (&C_to_A) : NULL, &C_to_A_size,
            (B_is_hyper) ? (&C_to_B) : NULL, &C_to_B_size))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_IPTR (Ch, Cj_is_32) ;

        // copy Mh into Ch.  Ch is Mh so C_to_M is not needed.
        GB_cast_int (Ch, cjcode, Mh, mjcode, Mnvec, nthreads) ;

        // construct the mapping from C to A and B, if they are hypersparse
        if (A_is_hyper || B_is_hyper)
        {

            // create the A->Y and B->Y hyper_hashes
            GB_OK (GB_hyper_hash_build (A, Werk)) ;
            GB_OK (GB_hyper_hash_build (B, Werk)) ;

            const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
            const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
            const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
            const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

            const void *B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
            const void *B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
            const void *B_Yx = (B->Y == NULL) ? NULL : B->Y->x ;
            const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;

            int64_t k ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (k = 0 ; k < Cnvec ; k++)
            {
                int64_t j = GB_IGET (Ch, k) ;   // j = Ch [k]
                if (A_is_hyper)
                { 
                    // C_to_A [k] = kA if Ah [kA] == j and A(:,j) is non-empty
                    int64_t pA, pA_end ;
                    int64_t kA = GB_hyper_hash_lookup (Ap_is_32, Aj_is_32,
                        Ah, Anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
                        j, &pA, &pA_end) ;
                    C_to_A [k] = (pA < pA_end) ? kA : -1 ;
                }
                if (B_is_hyper)
                { 
                    // C_to_B [k] = kB if Bh [kB] == j and B(:,j) is non-empty
                    int64_t pB, pB_end ;
                    int64_t kB = GB_hyper_hash_lookup (Bp_is_32, Bj_is_32,
                        Bh, Bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
                        j, &pB, &pB_end) ;
                    C_to_B [k] = (pB < pB_end) ? kB : -1 ;
                }
            }
        }

    }
    else if (A_is_hyper && B_is_hyper)
    {

        //----------------------------------------------------------------------
        // A and B are hypersparse: C will be hypersparse
        //----------------------------------------------------------------------

        // Ch is the set union of Ah and Bh.  This is handled with a parallel
        // merge, since Ah and Bh are both sorted lists.

        (*C_sparsity) = GxB_HYPERSPARSE ;

        //----------------------------------------------------------------------
        // create the tasks to construct Ch
        //----------------------------------------------------------------------

        double work = GB_IMIN (Anvec + Bnvec, n) ;
        nthreads = GB_nthreads (work, chunk, nthreads_max) ;

        ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;
        ntasks = GB_IMIN (ntasks, work) ;

        // allocate workspace
        GB_WERK_PUSH (Work, 3*(ntasks+1), int64_t) ;
        if (Work == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        int64_t *restrict kA_start = Work ;
        int64_t *restrict kB_start = Work + (ntasks+1) ;
        int64_t *restrict kC_start = Work + (ntasks+1)*2 ;

        kA_start [0] = (Anvec == 0) ? -1 : 0 ;
        kB_start [0] = (Bnvec == 0) ? -1 : 0 ;
        kA_start [ntasks] = (Anvec == 0) ? -1 : Anvec ;
        kB_start [ntasks] = (Bnvec == 0) ? -1 : Bnvec ;

        for (int taskid = 1 ; taskid < ntasks ; taskid++)
        { 
            // create tasks: A and B are both hyper
            double target_work = ((ntasks-taskid) * work) / ntasks ;
            GB_slice_vector (NULL, NULL,
                &(kA_start [taskid]), &(kB_start [taskid]),
                0, 0, NULL, false,          // Mi not present
                0, Anvec, Ah, Aj_is_32,     // Ah, explicit list
                0, Bnvec, Bh, Bj_is_32,     // Bh, explicit list
                n,                          // Ah and Bh have dimension n
                target_work) ;
        }

        //----------------------------------------------------------------------
        // count the entries in Ch for each task
        //----------------------------------------------------------------------

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule (dynamic,1)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {
            // merge Ah and Bh into Ch
            int64_t kA = kA_start [taskid] ;
            int64_t kB = kB_start [taskid] ;
            int64_t kA_end = kA_start [taskid+1] ;
            int64_t kB_end = kB_start [taskid+1] ;
            int64_t kC = 0 ;
            for ( ; kA < kA_end && kB < kB_end ; kC++)
            {
                int64_t jA = GB_IGET (Ah, kA) ;
                int64_t jB = GB_IGET (Bh, kB) ;
                if (jA < jB)
                { 
                    // jA appears in A but not B
                    kA++ ;
                }
                else if (jB < jA)
                { 
                    // jB appears in B but not A
                    kB++ ;
                }
                else
                { 
                    // j = jA = jB appears in both A and B
                    kA++ ;
                    kB++ ;
                }
            }
            kC_start [taskid] = kC + (kA_end - kA) + (kB_end - kB) ;
        }

        //----------------------------------------------------------------------
        // cumulative sum of entries in Ch for each task
        //----------------------------------------------------------------------

        GB_cumsum1_64 ((uint64_t *) kC_start, ntasks) ;
        Cnvec = kC_start [ntasks] ;

        //----------------------------------------------------------------------
        // allocate the result: Ch and the mappings C_to_[MAB]
        //----------------------------------------------------------------------

        // C will be hypersparse, so Ch is allocated.  The mask M is ignored
        // for computing Ch.  Ch is the set union of Ah and Bh.

        if (!GB_allocate_result (Cnvec,
            &Ch, &Ch_size, cjsize,
            (M_is_hyper) ? (&C_to_M) : NULL, &C_to_M_size,
            &C_to_A, &C_to_A_size,
            &C_to_B, &C_to_B_size))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_IPTR (Ch, Cj_is_32) ;

        //----------------------------------------------------------------------
        // compute the result: Ch and the mappings C_to_[AB]
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule (dynamic,1)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {
            // merge Ah and Bh into Ch
            int64_t kA = kA_start [taskid] ;
            int64_t kB = kB_start [taskid] ;
            int64_t kC = kC_start [taskid] ;
            int64_t kA_end = kA_start [taskid+1] ;
            int64_t kB_end = kB_start [taskid+1] ;

            // merge Ah and Bh into Ch
            for ( ; kA < kA_end && kB < kB_end ; kC++)
            {
                int64_t jA = GB_IGET (Ah, kA) ;
                int64_t jB = GB_IGET (Bh, kB) ;
                if (jA < jB)
                { 
                    // append jA to Ch
                    GB_ISET (Ch, kC, jA) ;  // Ch [kC] = jA
                    C_to_A [kC] = kA++ ;
                    C_to_B [kC] = -1 ;      // jA does not appear in B
                }
                else if (jB < jA)
                { 
                    // append jB to Ch
                    GB_ISET (Ch, kC, jB) ;  // Ch [kC] = jB ;
                    C_to_A [kC] = -1 ;      // jB does not appear in A
                    C_to_B [kC] = kB++ ;
                }
                else
                { 
                    // j appears in both A and B; append it to Ch
                    GB_ISET (Ch, kC, jA) ;  // Ch [kC] = jA
                    C_to_A [kC] = kA++ ;
                    C_to_B [kC] = kB++ ;
                }
            }
            if (kA < kA_end)
            {
                // B is exhausted but A is not
                for ( ; kA < kA_end ; kA++, kC++)
                { 
                    // append jA to Ch
                    int64_t jA = GB_IGET (Ah, kA) ;
                    GB_ISET (Ch, kC, jA) ;  // Ch [kC] = jA
                    C_to_A [kC] = kA ;
                    C_to_B [kC] = -1 ;
                }
            }
            else if (kB < kB_end)
            {
                // A is exhausted but B is not
                for ( ; kB < kB_end ; kB++, kC++)
                { 
                    // append jB to Ch
                    int64_t jB = GB_IGET (Bh, kB) ;
                    GB_ISET (Ch, kC, jB) ;  // Ch [kC] = jB ;
                    C_to_A [kC] = -1 ;
                    C_to_B [kC] = kB ;
                }
            }
            ASSERT (kC == kC_start [taskid+1]) ;
        }

        //----------------------------------------------------------------------
        // check result via a sequential merge
        //----------------------------------------------------------------------

        #ifdef GB_DEBUG
        // merge Ah and Bh into Ch
        int64_t kA = 0 ;
        int64_t kB = 0 ;
        int64_t kC = 0 ;
        for ( ; kA < Anvec && kB < Bnvec ; kC++)
        {
            int64_t jA = GB_IGET (Ah, kA) ;
            int64_t jB = GB_IGET (Bh, kB) ;
            if (jA < jB)
            {
                // append jA to Ch
                ASSERT (GB_IGET (Ch, kC) == jA) ;
                ASSERT (C_to_A [kC] == kA) ; kA++ ;
                ASSERT (C_to_B [kC] == -1) ;      // jA does not appear in B
            }
            else if (jB < jA)
            {
                // append jB to Ch
                ASSERT (GB_IGET (Ch, kC) == jB) ;
                ASSERT (C_to_A [kC] == -1) ;       // jB does not appear in A
                ASSERT (C_to_B [kC] == kB) ; kB++ ;
            }
            else
            {
                // j appears in both A and B; append it to Ch
                ASSERT (GB_IGET (Ch, kC) == jA) ;
                ASSERT (C_to_A [kC] == kA) ; kA++ ;
                ASSERT (C_to_B [kC] == kB) ; kB++ ;
            }
        }
        if (kA < Anvec)
        {
            // B is exhausted but A is not
            for ( ; kA < Anvec ; kA++, kC++)
            {
                // append jA to Ch
                int64_t jA = GB_IGET (Ah, kA) ;
                ASSERT (GB_IGET (Ch, kC) == jA) ;
                ASSERT (C_to_A [kC] == kA) ;
                ASSERT (C_to_B [kC] == -1) ;
            }
        }
        else if (kB < Bnvec)
        {
            // A is exhausted but B is not
            for ( ; kB < Bnvec ; kB++, kC++)
            {
                // append jB to Ch
                int64_t jB = GB_IGET (Bh, kB) ;
                ASSERT (GB_IGET (Ch, kC) == jB) ;
                ASSERT (C_to_A [kC] == -1) ;
                ASSERT (C_to_B [kC] == kB) ;
            }
        }
        ASSERT (kC == Cnvec) ;
        #endif

    }
    else if (A_is_hyper && !B_is_hyper)
    {

        //----------------------------------------------------------------------
        // A is hypersparse, B is not hypersparse
        //----------------------------------------------------------------------

        // C will be sparse.  Construct the C_to_A mapping.

        Cnvec = n ;
        nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;

        if (!GB_allocate_result (Cnvec,
            NULL, NULL, 0,
            (M_is_hyper) ? (&C_to_M) : NULL, &C_to_M_size,
            &C_to_A, &C_to_A_size,
            NULL, NULL))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < n ; j++)
        { 
            C_to_A [j] = -1 ;
        }

        // scatter Ah into C_to_A
        int64_t kA ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (kA = 0 ; kA < Anvec ; kA++)
        { 
            int64_t jA = GB_IGET (Ah, kA) ;
            C_to_A [jA] = kA ;
        }

    }
    else if (!A_is_hyper && B_is_hyper)
    {

        //----------------------------------------------------------------------
        // A is not hypersparse, B is hypersparse
        //----------------------------------------------------------------------

        // C will be sparse.  Construct the C_to_B mapping.

        Cnvec = n ;
        nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;

        if (!GB_allocate_result (Cnvec,
            NULL, NULL, 0,
            (M_is_hyper) ? (&C_to_M) : NULL, &C_to_M_size,
            NULL, NULL,
            &C_to_B, &C_to_B_size))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < n ; j++)
        { 
            C_to_B [j] = -1 ;
        }

        // scatter Bh into C_to_B
        int64_t kB ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (kB = 0 ; kB < Bnvec ; kB++)
        { 
            int64_t jB = GB_IGET (Bh, kB) ;
            C_to_B [jB] = kB ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A and B are both non-hypersparse
        //----------------------------------------------------------------------

        // C will be sparse
        Cnvec = n ;
        nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;

        if (!GB_allocate_result (Cnvec,
            NULL, NULL, 0,
            (M_is_hyper) ? (&C_to_M) : NULL, &C_to_M_size,
            NULL, NULL,
            NULL, NULL))
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // construct C_to_M if needed, if M is hypersparse
    //--------------------------------------------------------------------------

    if (C_to_M != NULL)
    {
        ASSERT (M_is_hyper) ;
        if (Ch != NULL)
        {
            // C is hypersparse

            // create the M->Y hyper_hash
            GB_OK (GB_hyper_hash_build (M, Werk)) ;

            const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
            const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
            const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
            const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;

            int64_t k ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (k = 0 ; k < Cnvec ; k++)
            { 
                int64_t j = GB_IGET (Ch, k) ;
                // C_to_M [k] = kM if Mh [kM] == j and M(:,j) is non-empty
                int64_t pM, pM_end ;
                int64_t kM = GB_hyper_hash_lookup (Mp_is_32, Mj_is_32,
                    Mh, Mnvec, Mp, M_Yp, M_Yi, M_Yx, M_hash_bits,
                    j, &pM, &pM_end) ;
                C_to_M [k] = (pM < pM_end) ? kM : -1 ;
            }
        }
        else
        {
            // C is sparse
            int64_t j ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (j = 0 ; j < n ; j++)
            { 
                C_to_M [j] = -1 ;
            }
            // scatter Mh into C_to_M
            int64_t kM ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (kM = 0 ; kM < Mnvec ; kM++)
            { 
                int64_t jM = GB_IGET (Mh, kM) ;
                C_to_M [jM] = kM ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_Cnvec) = Cnvec ;
    (*Ch_handle) = Ch ;             (*Ch_size_handle) = Ch_size ;
    if (C_to_M_handle != NULL)
    { 
        (*C_to_M_handle) = C_to_M ; (*C_to_M_size_handle) = C_to_M_size ;
    }
    (*C_to_A_handle) = C_to_A ;     (*C_to_A_size_handle) = C_to_A_size ;
    (*C_to_B_handle) = C_to_B ;     (*C_to_B_size_handle) = C_to_B_size ;
    if (p_Ch_is_Mh != NULL)
    { 
        // return Ch_is_Mh to GB_add.  For GB_masker, Ch is never Mh.
        (*p_Ch_is_Mh) = Ch_is_Mh ;
    }

    //--------------------------------------------------------------------------
    // The code below describes what the output contains:
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // the mappings are only constructed when C is sparse or hypersparse
    ASSERT ((*C_sparsity) == GxB_SPARSE || (*C_sparsity) == GxB_HYPERSPARSE) ;
    ASSERT (A != NULL) ;        // A and B are always present
    ASSERT (B != NULL) ;
    GB_IPTR (Ch, Cj_is_32) ;
    int64_t jlast = -1 ;
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {
        // C(:,j) is in the list, as the kth vector
        int64_t j ;
        if (Ch == NULL)
        {
            // C will be constructed as sparse
            ASSERT ((*C_sparsity) == GxB_SPARSE) ;
            j = k ;
        }
        else
        {
            // C will be constructed as hypersparse
            ASSERT ((*C_sparsity) == GxB_HYPERSPARSE) ;
            j = GB_IGET (Ch, k) ;
        }

        // vectors j in Ch are sorted, and in the range 0:n-1
        ASSERT (j >= 0 && j < n) ;
        ASSERT (j > jlast) ;
        jlast = j ;

        // see if A (:,j) exists
        if (C_to_A != NULL)
        {
            // A is hypersparse
            ASSERT (A_is_hyper) ;
            int64_t kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                int64_t jA = GB_IGET (Ah, kA) ;
                ASSERT (j == jA) ;
            }
        }
        else
        {
            // A is not hypersparse
            // C_to_A exists only if A is hypersparse
            ASSERT (!A_is_hyper) ;
        }

        // see if B (:,j) exists
        if (C_to_B != NULL)
        {
            // B is hypersparse
            ASSERT (B_is_hyper) ;
            int64_t kB = C_to_B [k] ;
            ASSERT (kB >= -1 && kB < B->nvec) ;
            if (kB >= 0)
            {
                int64_t jB = GB_IGET (Bh, kB) ;
                ASSERT (j == jB) ;
            }
        }
        else
        {
            // B is not hypersparse
            // C_to_B exists only if B is hypersparse
            ASSERT (!B_is_hyper) ;
        }

        // see if M (:,j) exists
        if (Ch_is_Mh)
        {
            // Ch is the same as Mh
            ASSERT (M != NULL) ;
            ASSERT (M_is_hyper) ;
            ASSERT (Ch != NULL) ;
            ASSERT (Mh != NULL) ;
            ASSERT (GB_IGET (Ch, k) == GB_IGET (Mh, k)) ;
            ASSERT (C_to_M == NULL) ;
        }
        else if (C_to_M != NULL)
        {
            // M is present and hypersparse
            ASSERT (M != NULL) ;
            ASSERT (M_is_hyper) ;
            int64_t kM = C_to_M [k] ;
            ASSERT (kM >= -1 && kM < M->nvec) ;
            if (kM >= 0)
            {
                int64_t jM = GB_IGET (Mh, kM) ;
                ASSERT (j == jM) ;
            }
        }
        else
        {
            // M is not present, or present and sparse, bitmap or full
            ASSERT (M == NULL || !M_is_hyper) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

