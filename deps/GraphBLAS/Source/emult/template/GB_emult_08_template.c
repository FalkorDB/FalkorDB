//------------------------------------------------------------------------------
// GB_emult_08_template:  phase1 and phase2 for C=A.*B, C<M>=A.*B, C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C=A.*B, C<M>=A.*B, or C<!M>=A.*B, where C is sparse/hypersparse.

// M, A, and B can have any sparsity structure.  If both A and B are full, then
// GB_add is used instead (this is the only case where C can be full).

// phase1: does not compute C itself, but just counts the # of entries in each
// vector of C.  Fine tasks compute the # of entries in their slice of a
// single vector of C, and the results are cumsum'd.

// phase2: computes C, using the counts computed by phase1.

// No input matrix can be jumbled, and C is constructed as unjumbled.

// The following cases are handled:

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse  (method: 8bcd)

        //      ------------------------------------------
        //      C       <M>=        A       .*      B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse  (method: 8e)
        //      sparse  bitmap      sparse          sparse  (method: 8fgh)
        //      sparse  full        sparse          sparse  (method: 8fgh)
        //      sparse  sparse      sparse          bitmap  (9  (8e) or 2)
        //      sparse  sparse      sparse          full    (9  (8e) or 2)
        //      sparse  sparse      bitmap          sparse  (10 (8e) or 3)
        //      sparse  sparse      full            sparse  (10 (8e) or 3)

        //      ------------------------------------------
        //      C       <!M>=       A       .*      B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse  (8bcd: M later)
        //      sparse  bitmap      sparse          sparse  (method: 8fgh)
        //      sparse  full        sparse          sparse  (method: 8fgh)

// Methods 9 and 10 are not yet implemented, and are currently handled by this
// Method 8 instead.  See GB_emult_sparsity for this decision.
// "M later" means that C<!M>=A.*B is being computed, but the mask is not
// handled here; insteadl T=A.*B is computed and C<!M>=T is done later.

{

    // iB_first is unused if the operator is FIRST or PAIR

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t *restrict Ab = A->b ;
    const int64_t vlen = A->vlen ;

    #ifdef GB_JIT_KERNEL
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_full   GB_A_IS_FULL
    #else
    const bool Ai_is_32 = A->i_is_32 ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_full = GB_IS_FULL (A) ;
    #define GB_Ai_IS_32 Ai_is_32
    #endif

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t *restrict Bb = B->b ;

    #ifdef GB_JIT_KERNEL
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_full   GB_B_IS_FULL
    #else
    const bool Bi_is_32 = B->i_is_32 ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_full = GB_IS_FULL (B) ;
    #define GB_Bi_IS_32 Bi_is_32
    #endif

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;

    #ifdef GB_JIT_KERNEL
    #define M_is_hyper  GB_M_IS_HYPER
    #define M_is_sparse GB_M_IS_SPARSE
    #define M_is_bitmap GB_M_IS_BITMAP
    #define M_is_full   GB_M_IS_FULL
    #define M_is_sparse_or_hyper (GB_M_IS_SPARSE || GB_M_IS_HYPER)
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #define M_is_present (!GB_NO_MASK)
    #else
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    const bool M_is_present = (M != NULL) ;
    #endif

    size_t msize = 0 ;
    if (M_is_present)
    { 
        Mb = M->b ;
        Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    #ifndef GB_EMULT_08_PHASE
    #define GB_EMULT_08_PHASE 2
    #endif

    #if ( GB_EMULT_08_PHASE == 2 )
    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;
    #endif
    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif
    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    #endif

    //--------------------------------------------------------------------------
    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B: C is sparse or hypersparse
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < C_ntasks ; taskid++)
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
            // get j, the kth vector of C
            //------------------------------------------------------------------

            int64_t j = GBh_C (Ch, k) ;

            #if ( GB_EMULT_08_PHASE == 1 )
            int64_t cjnz = 0 ;
            #else
            int64_t pC, pC_end ;
            if (fine_task)
            { 
                // A fine task computes a slice of C(:,j)
                pC     = TaskList [taskid  ].pC ;
                pC_end = TaskList [taskid+1].pC ;
                ASSERT (GB_IGET (Cp, k) <= pC) ;
                ASSERT (pC <= pC_end) ;
                ASSERT (pC_end <= GB_IGET (Cp, k+1)) ;
            }
            else
            { 
                // The vectors of C are never sliced for a coarse task.
                pC     = GB_IGET (Cp, k) ;
                pC_end = GB_IGET (Cp, k+1) ;
            }
            int64_t cjnz = pC_end - pC ;
            if (cjnz == 0) continue ;
            #endif

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA = -1, pA_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Ai,Ax [pA...pA_end-1], which is
                // a subset of the vector A(:,j)
                pA     = TaskList [taskid].pA ;
                pA_end = TaskList [taskid].pA_end ;
            }
            else
            {
                // A coarse task operates on the entire vector A (:,j)
                int64_t kA = ((void *) Ch == (void *) Ah) ? k :
                            ((C_to_A == NULL) ? j : C_to_A [k]) ;
                if (kA >= 0)
                { 
                    pA     = GBp_A (Ap, kA, vlen) ;
                    pA_end = GBp_A (Ap, kA+1, vlen) ;
                }
            }

            int64_t ajnz = pA_end - pA ;        // nnz in A(:,j) for this slice
            int64_t pA_start = pA ;
            bool adense = (ajnz == len) ;

            // get the first and last indices in A(:,j) for this vector
            int64_t iA_first = -1 ;
            if (ajnz > 0)
            { 
                iA_first = GBi_A (Ai, pA, vlen) ;
            }
            #if ( GB_EMULT_08_PHASE == 1 ) || defined ( GB_DEBUG )
            int64_t iA_last = -1 ;
            if (ajnz > 0)
            { 
                iA_last  = GBi_A (Ai, pA_end-1, vlen) ;
            }
            #endif

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t pB = -1, pB_end = -1 ;
            if (fine_task)
            { 
                // A fine task operates on Bi,Bx [pB...pB_end-1], which is
                // a subset of the vector B(:,j)
                pB     = TaskList [taskid].pB ;
                pB_end = TaskList [taskid].pB_end ;
            }
            else
            {
                // A coarse task operates on the entire vector B (:,j)
                int64_t kB = ((void *) Ch == (void *) Bh) ? k :
                            ((C_to_B == NULL) ? j : C_to_B [k]) ;
                if (kB >= 0)
                { 
                    pB     = GBp_B (Bp, kB, vlen) ;
                    pB_end = GBp_B (Bp, kB+1, vlen) ;
                }
            }

            int64_t bjnz = pB_end - pB ;        // nnz in B(:,j) for this slice
            int64_t pB_start = pB ;
            bool bdense = (bjnz == len) ;

            // get the first and last indices in B(:,j) for this vector
            int64_t iB_first = -1 ;
            if (bjnz > 0)
            { 
                iB_first = GBi_B (Bi, pB, vlen) ;
            }
            #if ( GB_EMULT_08_PHASE == 1 ) || defined ( GB_DEBUG )
            int64_t iB_last = -1 ;
            if (bjnz > 0)
            { 
                iB_last  = GBi_B (Bi, pB_end-1, vlen) ;
            }
            #endif

            //------------------------------------------------------------------
            // C(:,j)<optional mask> = A (:,j) .* B (:,j) or subvector
            //------------------------------------------------------------------

            #if ( GB_EMULT_08_PHASE == 1 )
            if (ajnz == 0 || bjnz == 0)
            { 
                // Method8(a): A(:,j) and/or B(:,j) are empty
                ;
            }
            else if (iA_last < iB_first || iB_last < iA_first)
            { 
                // Method8(a): intersection of A(:,j) and B(:,j) is empty
                // the last entry of A(:,j) comes before the first entry
                // of B(:,j), or visa versa
                ;
            }
            else
            #endif

            #ifdef GB_JIT_KERNEL
            {
                #if GB_NO_MASK
                {
                    // C=A.*B, all matrices sparse/hyper
                    #include "template/GB_emult_08bcd.c"
                }
                #elif (GB_M_IS_SPARSE || GB_M_IS_HYPER)
                {
                    // C<M>=A.*B, C and M are sparse/hyper
                    // either A or B are sparse/hyper
                    #include "template/GB_emult_08e.c"
                }
                #else
                {
                    // C<#M>=A.*B; C, A and B are sparse/hyper; M is bitmap/full
                    #include "template/GB_emult_08fgh.c"
                }
                #endif
            }
            #else
            {
                if (M == NULL)
                {
                    // C=A.*B, all matrices sparse/hyper
                    #include "template/GB_emult_08bcd.c"
                }
                else if (M_is_sparse_or_hyper)
                {
                    // C<M>=A.*B, C and M are sparse/hyper
                    // either A or B are sparse/hyper
                    #include "template/GB_emult_08e.c"
                }
                else
                {
                    // C<#M>=A.*B; C, A and B are sparse/hyper; M is bitmap/full
                    #include "template/GB_emult_08fgh.c"
                }
            }
            #endif

            //------------------------------------------------------------------
            // final count of nnz (C (:,j))
            //------------------------------------------------------------------

            #if ( GB_EMULT_08_PHASE == 1 )
            if (fine_task)
            { 
                TaskList [taskid].pC = cjnz ;
            }
            else
            { 
                GB_ISET (Cp, k, cjnz) ;     // Cp [k] = cjnz ;
            }
            #endif
        }
    }
}

