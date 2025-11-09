//------------------------------------------------------------------------------
// GB_emult_08_phase0: find vectors of C to compute for C=A.*B or C<M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The eWise multiply of two matrices, C=A.*B, C<M>=A.*B, or C<!M>=A.*B starts
// with this phase, which determines which vectors of C need to be computed.

// On input, A and B are the two matrices being ewise multiplied, and M is the
// optional mask matrix.  If present, it is not complemented.

// The M, A, and B matrices are sparse or hypersparse.  C will be sparse
// (if Ch is returned NULL) or hypersparse (if Ch is returned non-NULL).

//      Ch: the vectors to compute in C.  Not allocated, either NULL if C is
//      not hypersparse, or shallow and equal to A->h, B->h, or M->h.  Ch is
//      never allocated.

//      C_to_A:  if A is hypersparse, and Ch is not A->h, then C_to_A [k] = kA
//      if the kth vector j = Ch [k] is equal to Ah [kA].  If j does not appear
//      in A, then C_to_A [k] = -1. C is always hypersparse in this case.
//      Otherwise, C_to_A is returned as NULL.

//      C_to_B:  if B is hypersparse, and Ch is not B->h, then C_to_B [k] = kB
//      if the kth vector j = Ch [k] is equal to Bh [kB].  If j does not appear
//      in B, then C_to_B [k] = -1.  C is always hypersparse in this case.
//      Otherwise, C_to_B is returned as NULL.

//      C_to_M:  if M is hypersparse, and Ch is not M->h, then C_to_M [k] = kM
//      if the kth vector j = Ch [k] is equal to Mh [kM].  If j does not appear
//      in M, then C_to_M [k] = -1.  C is always hypersparse in this case.
//      Otherwise, C_to_M is returned as NULL.

// FUTURE:: exploit A==M, B==M, and A==B aliases

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_MEMORY (&C_to_M, C_to_M_size) ;       \
    GB_FREE_MEMORY (&C_to_A, C_to_A_size) ;       \
    GB_FREE_MEMORY (&C_to_B, C_to_B_size) ;       \
}

#include "emult/GB_emult.h"

GrB_Info GB_emult_08_phase0     // find vectors in C for C=A.*B or C<M>=A.*B
(
    int64_t *p_Cnvec,           // # of vectors to compute in C
    const void **Ch_handle,     // Ch is M->h, A->h, B->h, or NULL
    size_t *Ch_size_handle,
    int64_t *restrict *C_to_M_handle,    // C_to_M: size Cnvec, or NULL
    size_t *C_to_M_size_handle,
    int64_t *restrict *C_to_A_handle,    // C_to_A: size Cnvec, or NULL
    size_t *C_to_A_size_handle,
    int64_t *restrict *C_to_B_handle,    // C_to_B: size Cnvec, or NULL
    size_t *C_to_B_size_handle,
    bool *p_Cp_is_32,           // if true, Cp is 32-bit; else 64-bit
    bool *p_Cj_is_32,           // if true, Ch is 32-bit; else 64-bit
    bool *p_Ci_is_32,           // if true, Ci is 32-bit; else 64-bit
    int *C_sparsity,            // sparsity structure of C
    // original input:
    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // M, A, and B can be jumbled for this phase

    GrB_Info info ;
    ASSERT (p_Cnvec != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    ASSERT (Ch_size_handle != NULL) ;
    ASSERT (p_Cp_is_32 != NULL) ;
    ASSERT (p_Cj_is_32 != NULL) ;
    ASSERT (p_Ci_is_32 != NULL) ;
    ASSERT (C_to_A_handle != NULL) ;
    ASSERT (C_to_B_handle != NULL) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for emult phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (M)) ;

    ASSERT_MATRIX_OK (A, "A for emult phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for emult phase0", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // pattern not accessed
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (A->vdim == B->vdim) ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim)) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    (*p_Cnvec) = 0 ;          
    (*Ch_handle) = NULL ;
    (*Ch_size_handle) = 0 ;
    if (C_to_M_handle != NULL)
    { 
        (*C_to_M_handle) = NULL ;
    }
    (*C_to_A_handle) = NULL ;
    (*C_to_B_handle) = NULL ;

    ASSERT ((*C_sparsity) == GxB_SPARSE || (*C_sparsity) == GxB_HYPERSPARSE) ;

    GB_MDECL (Ch, , u) ; size_t Ch_size = 0 ;

    int64_t *restrict C_to_M = NULL ; size_t C_to_M_size = 0 ;
    int64_t *restrict C_to_A = NULL ; size_t C_to_A_size = 0 ;
    int64_t *restrict C_to_B = NULL ; size_t C_to_B_size = 0 ;

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    int64_t n = A->vdim ;

    int64_t Anvec = A->nvec ;
    void *Ah = A->h ;
    bool A_is_hyper = (Ah != NULL) ;

    int64_t Bnvec = B->nvec ;
    void *Bh = B->h ;
    bool B_is_hyper = (Bh != NULL) ;

    int64_t Mnvec = 0 ;
    void *Mh = NULL ;
    bool M_is_hyper = false ;

    if (M != NULL)
    { 
        Mnvec = M->nvec ;
        Mh = M->h ;
        M_is_hyper = (Mh != NULL) ;
    }

    //--------------------------------------------------------------------------
    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    //--------------------------------------------------------------------------

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    int64_t anz = GB_nnz (A) ;
    int64_t bnz = GB_nnz (B) ;
    int64_t cnz = GB_IMIN (anz, bnz) ;
    if (M != NULL && !Mask_comp)
    {
        int64_t mnz = GB_nnz (M) ;
        cnz = GB_IMIN (cnz, mnz) ;
    }
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, cnz, A->vlen, A->vdim, Werk) ;

    //--------------------------------------------------------------------------
    // determine if C is sparse or hypersparse, and find its hyperlist
    //--------------------------------------------------------------------------

    int64_t Cnvec ;

    if (M != NULL)
    {

        //----------------------------------------------------------------------
        // 8 cases to consider: A, B, M can each be hyper or sparse
        //----------------------------------------------------------------------

        // Mask is present and not complemented

        if (A_is_hyper)
        {
            if (B_is_hyper)
            {
                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (1) A hyper, B hyper, M hyper: C hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Bh, Ah

                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    Cnvec = GB_IMIN (Anvec, Bnvec) ;
                    Cnvec = GB_IMIN (Cnvec, Mnvec) ;
                    if (Cnvec == Anvec)
                    { 
                        Ch = Ah ; Ch_size = A->h_size ;
                        Cj_is_32 = A->j_is_32 ;
                    }
                    else if (Cnvec == Bnvec)
                    { 
                        Ch = Bh ; Ch_size = B->h_size ;
                        Cj_is_32 = B->j_is_32 ;
                    }
                    else // (Cnvec == Mnvec)
                    { 
                        Ch = Mh ; Ch_size = M->h_size ;
                        Cj_is_32 = M->j_is_32 ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // (2) A hyper, B hyper, M sparse: C hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Ah, Bh
                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    if (Anvec <= Bnvec)
                    { 
                        Ch = Ah ; Ch_size = A->h_size ;
                        Cj_is_32 = A->j_is_32 ;
                        Cnvec = Anvec ;
                    }
                    else
                    { 
                        Ch = Bh ; Ch_size = B->h_size ;
                        Cj_is_32 = B->j_is_32 ;
                        Cnvec = Bnvec ;
                    }
                }

            }
            else
            {

                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (3) A hyper, B sparse, M hyper: C hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Ah
                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    if (Anvec <= Mnvec)
                    { 
                        Ch = Ah ; Ch_size = A->h_size ;
                        Cj_is_32 = A->j_is_32 ;
                        Cnvec = Anvec ;
                    }
                    else
                    { 
                        Ch = Mh ; Ch_size = M->h_size ;
                        Cj_is_32 = M->j_is_32 ;
                        Cnvec = Mnvec ;
                    }

                }
                else
                { 

                    //----------------------------------------------------------
                    // (4) A hyper, B sparse, M sparse: C hyper
                    //----------------------------------------------------------

                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    Ch = Ah ; Ch_size = A->h_size ;
                    Cj_is_32 = A->j_is_32 ;
                    Cnvec = Anvec ;
                }
            }

        }
        else
        {

            if (B_is_hyper)
            {
                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (5) A sparse, B hyper, M hyper: C hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Bh
                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    if (Bnvec <= Mnvec)
                    { 
                        Ch = Bh ; Ch_size = B->h_size ;
                        Cj_is_32 = B->j_is_32 ;
                        Cnvec = Bnvec ;
                    }
                    else
                    { 
                        Ch = Mh ; Ch_size = M->h_size ;
                        Cj_is_32 = M->j_is_32 ;
                        Cnvec = Mnvec ;
                    }

                }
                else
                { 

                    //----------------------------------------------------------
                    // (6) A sparse, B hyper, M sparse: C hyper
                    //----------------------------------------------------------

                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    Ch = Bh ; Ch_size = B->h_size ;
                    Cj_is_32 = B->j_is_32 ;
                    Cnvec = Bnvec ;

                }
            }
            else
            {

                if (M_is_hyper)
                { 

                    //----------------------------------------------------------
                    // (7) A sparse, B sparse, M hyper: C hyper
                    //----------------------------------------------------------

                    (*C_sparsity) = GxB_HYPERSPARSE ;
                    Ch = Mh ; Ch_size = M->h_size ;
                    Cj_is_32 = M->j_is_32 ;
                    Cnvec = Mnvec ;

                }
                else
                { 

                    //----------------------------------------------------------
                    // (8) A sparse, B sparse, M sparse: C sparse
                    //----------------------------------------------------------

                    (*C_sparsity) = GxB_SPARSE ;
                    Ch = NULL ;
                    Cnvec = n ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // 4 cases to consider:  A, B can be hyper or sparse
        //----------------------------------------------------------------------

        // Mask is not present, or present and complemented.

        if (A_is_hyper)
        {
            if (B_is_hyper)
            {

                //--------------------------------------------------------------
                // (1) A hyper, B hyper:  C hyper
                //--------------------------------------------------------------

                // Ch = smaller of Ah, Bh
                (*C_sparsity) = GxB_HYPERSPARSE ;
                if (Anvec <= Bnvec)
                { 
                    Ch = Ah ; Ch_size = A->h_size ;
                    Cj_is_32 = A->j_is_32 ;
                    Cnvec = Anvec ;
                }
                else
                { 
                    Ch = Bh ; Ch_size = B->h_size ;
                    Cj_is_32 = B->j_is_32 ;
                    Cnvec = Bnvec ;
                }

            }
            else
            { 

                //--------------------------------------------------------------
                // (2) A hyper, B sparse: C hyper
                //--------------------------------------------------------------

                (*C_sparsity) = GxB_HYPERSPARSE ;
                Ch = Ah ; Ch_size = A->h_size ;
                Cj_is_32 = A->j_is_32 ;
                Cnvec = Anvec ;

            }

        }
        else
        {

            if (B_is_hyper)
            { 

                //--------------------------------------------------------------
                // (3) A sparse, B hyper: C hyper
                //--------------------------------------------------------------

                (*C_sparsity) = GxB_HYPERSPARSE ;
                Ch = Bh ; Ch_size = B->h_size ;
                Cj_is_32 = B->j_is_32 ;
                Cnvec = Bnvec ;

            }
            else
            { 

                //--------------------------------------------------------------
                // (4) A sparse, B sparse: C sparse
                //--------------------------------------------------------------

                (*C_sparsity) = GxB_SPARSE ;
                Ch = NULL ;
                Cnvec = n ;
            }
        }
    }

    GB_IPTR (Ch, Cj_is_32) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (Cnvec, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // construct C_to_M mapping
    //--------------------------------------------------------------------------

    if (M_is_hyper && Ch != Mh)
    {
        // allocate C_to_M
        ASSERT (Ch != NULL) ;
        C_to_M = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t), &C_to_M_size) ;
        if (C_to_M == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // create the M->Y hyper_hash
        GB_OK (GB_hyper_hash_build (M, Werk)) ;

        const void *Mp = M->p ;
        const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
        const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
        const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
        const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;

        // compute C_to_M
        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < Cnvec ; k++)
        { 
            int64_t pM, pM_end ;
            int64_t j = GB_IGET (Ch, k) ;
            int64_t kM = GB_hyper_hash_lookup (M->p_is_32, M->j_is_32,
                Mh, Mnvec, Mp, M_Yp, M_Yi, M_Yx, M_hash_bits, j, &pM, &pM_end) ;
            C_to_M [k] = (pM < pM_end) ? kM : -1 ;
        }
    }

    //--------------------------------------------------------------------------
    // construct C_to_A mapping
    //--------------------------------------------------------------------------

    if (A_is_hyper && Ch != Ah)
    {
        // allocate C_to_A
        ASSERT (Ch != NULL) ;
        C_to_A = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t), &C_to_A_size) ;
        if (C_to_A == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // create the A->Y hyper_hash
        GB_OK (GB_hyper_hash_build (A, Werk)) ;

        const void *Ap = A->p ;
        const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
        const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
        const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
        const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

        // compute C_to_A
        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < Cnvec ; k++)
        { 
            int64_t pA, pA_end ;
            int64_t j = GB_IGET (Ch, k) ;
            int64_t kA = GB_hyper_hash_lookup (A->p_is_32, A->j_is_32,
                Ah, Anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits, j, &pA, &pA_end) ;
            C_to_A [k] = (pA < pA_end) ? kA : -1 ;
        }
    }

    //--------------------------------------------------------------------------
    // construct C_to_B mapping
    //--------------------------------------------------------------------------

    if (B_is_hyper && Ch != Bh)
    {
        // allocate C_to_B
        ASSERT (Ch != NULL) ;
        C_to_B = GB_MALLOC_MEMORY (Cnvec, sizeof (int64_t), &C_to_B_size) ;
        if (C_to_B == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // create the B->Y hyper_hash
        GB_OK (GB_hyper_hash_build (B, Werk)) ;

        const void *Bp = B->p ;
        const void *B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
        const void *B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
        const void *B_Yx = (B->Y == NULL) ? NULL : B->Y->x ;
        const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;

        // compute C_to_B
        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < Cnvec ; k++)
        { 
            int64_t pB, pB_end ;
            int64_t j = GB_IGET (Ch, k) ;
            int64_t kB = GB_hyper_hash_lookup (B->p_is_32, B->j_is_32,
                Bh, Bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits, j, &pB, &pB_end) ;
            C_to_B [k] = (pB < pB_end) ? kB : -1 ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_Cnvec) = Cnvec ;
    (*Ch_handle) = Ch ;
    (*Ch_size_handle) = Ch_size ;
    (*p_Cp_is_32) = Cp_is_32 ;
    (*p_Cj_is_32) = Cj_is_32 ;
    (*p_Ci_is_32) = Ci_is_32 ;
    if (C_to_M_handle != NULL)
    {
        (*C_to_M_handle) = C_to_M ;
        (*C_to_M_size_handle) = C_to_M_size ;
    }
    (*C_to_A_handle) = C_to_A ; (*C_to_A_size_handle) = C_to_A_size ;
    (*C_to_B_handle) = C_to_B ; (*C_to_B_size_handle) = C_to_B_size ;

    //--------------------------------------------------------------------------
    // The code below describes what the output contains:
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    ASSERT (A != NULL) ;        // A and B are always present
    ASSERT (B != NULL) ;
    GB_IDECL (Ah, const, u) ; GB_IPTR (Ah, A->j_is_32) ;
    GB_IDECL (Bh, const, u) ; GB_IPTR (Bh, B->j_is_32) ;
    bool Mj_is_32 = (M == NULL) ? false : M->j_is_32 ;
    GB_IDECL (Mh, const, u) ; GB_IPTR (Mh, Mj_is_32) ;
    int64_t jlast = -1 ;
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        // C(:,j) is in the list, as the kth vector
        int64_t j ;
        if (Ch == NULL)
        {
            // C will be constructed as sparse
            j = k ;
        }
        else
        {
            // C will be constructed as hypersparse
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
            ASSERT (A_is_hyper)
            int64_t kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                int64_t jA = GB_IGET (Ah, kA) ; // OK: A is hyper
                ASSERT (j == jA) ;
            }
        }
        else if (A_is_hyper)
        {
            // A is hypersparse, and Ch is a shallow copy of A->h
            ASSERT (Ch == Ah) ;
            ASSERT (Cj_is_32 == A->j_is_32) ;
        }

        // see if B (:,j) exists
        if (C_to_B != NULL)
        {
            // B is hypersparse
            ASSERT (B_is_hyper)
            int64_t kB = C_to_B [k] ;
            ASSERT (kB >= -1 && kB < B->nvec) ;
            if (kB >= 0)
            {
                int64_t jB = GB_IGET (Bh, kB) ; // OK: B is hyper
                ASSERT (j == jB) ;
            }
        }
        else if (B_is_hyper)
        {
            // A is hypersparse, and Ch is a shallow copy of A->h
            ASSERT (Ch == Bh) ;
            ASSERT (Cj_is_32 == B->j_is_32) ;
        }

        // see if M (:,j) exists
        if (Ch != NULL && M != NULL && Ch == Mh)
        {
            // Ch is the same as Mh
            ASSERT (C_to_M == NULL) ;
            ASSERT (Cj_is_32 == M->j_is_32) ;
        }
        else if (C_to_M != NULL)
        {
            // M is present and hypersparse
            ASSERT (M != NULL) ;
            ASSERT (Mh != NULL) ;
            ASSERT (M_is_hyper) ;
            int64_t kM = C_to_M [k] ;
            ASSERT (kM >= -1 && kM < M->nvec) ;
            if (kM >= 0)
            {
                int64_t jM = GB_IGET (Mh, kM) ; // OK: M is hyper
                ASSERT (j == jM) ;
            }
        }
        else
        {
            // M is not present, or in sparse form
            ASSERT (M == NULL || Mh == NULL) ;
        }
    }
    #endif

    return (GrB_SUCCESS) ;
}

