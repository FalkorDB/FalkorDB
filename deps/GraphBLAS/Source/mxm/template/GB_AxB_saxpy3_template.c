//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy3_template.c computes C=A*B for any semiring and matrix types,
// where C is sparse or hypersparse.

#include "include/GB_unused.h"

//------------------------------------------------------------------------------
// template code for C=A*B via the saxpy3 method
//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;     // shadowed below
    ASSERT (Cp != NULL) ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;
    #ifndef GB_JIT_KERNEL
    const bool Ci_is_32 = C->i_is_32 ;
    #define GB_Ci_IS_32 Ci_is_32
    #endif

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE_U (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t *restrict Bb = B->b ;
    const int64_t bvlen = B->vlen ;
    #ifdef GB_JIT_KERNEL
    #define B_iso GB_B_ISO
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_sparse_or_hyper (GB_B_IS_SPARSE || GB_B_IS_HYPER)
    #else
    const bool B_iso = B->iso ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_sparse_or_hyper = B_is_sparse || B_is_hyper ;
    #endif

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE_U (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t *restrict Ab = A->b ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_bitmap GB_A_IS_BITMAP
    #else
    const bool A_iso = A->iso ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;
    const bool Ai_is_32 = A->i_is_32 ;
    #define GB_Ap_IS_32 Ap_is_32
    #define GB_Aj_IS_32 Aj_is_32
    #define GB_Ai_IS_32 Ai_is_32
    #endif
    const bool A_jumbled = A->jumbled ;
    const bool A_ok_for_binary_search = 
        ((A_is_sparse || A_is_hyper) && !A_jumbled) ;

    const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    #if ( !GB_NO_MASK )
    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE_U (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    #ifdef GB_JIT_KERNEL
    #define M_is_hyper  GB_M_IS_HYPER
    #define M_is_bitmap GB_M_IS_BITMAP
    #else
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool Mp_is_32 = M->p_is_32 ;
    const bool Mj_is_32 = M->j_is_32 ;
    #define GB_Mp_IS_32 Mp_is_32
    #define GB_Mj_IS_32 Mj_is_32
    #endif
    const bool M_jumbled = GB_JUMBLED (M) ;
    size_t msize = M->type->size ;
    int64_t mnvec = M->nvec ;
    int64_t mvlen = M->vlen ;
    // get the M hyper_hash
    const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
    const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
    const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
    const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;
    #endif

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif

    //==========================================================================
    // phase2: numeric work for fine tasks
    //==========================================================================

    // Coarse tasks: nothing to do in phase2.
    // Fine tasks: compute nnz (C(:,j)), and values in Hx via atomics.

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = SaxpyTasks [taskid].vector ;
        int team_size = SaxpyTasks [taskid].team_size ;
        uint64_t hash_size = SaxpyTasks [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int64_t pB     = SaxpyTasks [taskid].start ;
        int64_t pB_end = SaxpyTasks [taskid].end + 1 ;
        int64_t j = GBh_B (Bh, kk) ;

        GB_GET_T_FOR_SECONDJ ;

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_C_TYPE *restrict Hx = (GB_C_TYPE *) SaxpyTasks [taskid].Hx ;
        #endif

        #if GB_IS_PLUS_FC32_MONOID
        float  *restrict Hx_real = (float *) Hx ;
        float  *restrict Hx_imag = Hx_real + 1 ;
        #elif GB_IS_PLUS_FC64_MONOID
        double *restrict Hx_real = (double *) Hx ;
        double *restrict Hx_imag = Hx_real + 1 ;
        #endif

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase2: fine Gustavson task
            //------------------------------------------------------------------

            // Hf [i] == 0: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.
            //      M(i,j) is 0, or M is not present.
            //      if M: Hf [i] stays equal to 0 (or 3 if locked)
            //      if !M, or no M: C(i,j) is a new entry seen for 1st time

            // Hf [i] == 1: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.  M is present.
            //      M(i,j) is 1. (either M or !M case)
            //      if M: C(i,j) is a new entry seen for the first time.
            //      if !M: Hf [i] stays equal to 1 (or 3 if locked)

            // Hf [i] == 2: unlocked, i has been seen in C(:,j).
            //      Hx [i] is initialized.  This case is independent of M.

            // Hf [i] == 3: locked.  Hx [i] cannot be accessed.

            int8_t *restrict
                Hf = (int8_t *restrict) SaxpyTasks [taskid].Hf ;

            #if ( GB_NO_MASK )
            { 
                // phase2: fine Gustavson task, C(:,j)=A*B(:,j)
                #include "template/GB_AxB_saxpy3_fineGus_phase2.c"
            }
            #elif ( !GB_MASK_COMP )
            { 
                // phase2: fine Gustavson task, C(:,j)<M(:,j)>=A*B(:,j)
                #include "template/GB_AxB_saxpy3_fineGus_M_phase2.c"
            }
            #else
            { 
                // phase2: fine Gustavson task, C(:,j)<!M(:,j)>=A*B(:,j)
                #include "template/GB_AxB_saxpy3_fineGus_notM_phase2.c"
            }
            #endif

        }
        else
        {

            //------------------------------------------------------------------
            // phase2: fine hash task
            //------------------------------------------------------------------

            // Each hash entry Hf [hash] splits into two parts, (h,f).  f
            // is in the 2 least significant bits.  h is 62 bits, and is
            // the 1-based index i of the C(i,j) entry stored at that
            // location in the hash table.

            // If M is present (M or !M), and M(i,j)=1, then (i+1,1)
            // has been inserted into the hash table, in phase0.

            // Given Hf [hash] split into (h,f)

            // h == 0, f == 0: unlocked and unoccupied.
            //                  note that if f=0, h must be zero too.

            // h == i+1, f == 1: unlocked, occupied by M(i,j)=1.
            //                  C(i,j) has not been seen, or is ignored.
            //                  Hx is not initialized.  M is present.
            //                  if !M: this entry will be ignored in C.

            // h == i+1, f == 2: unlocked, occupied by C(i,j).
            //                  Hx is initialized.  M is no longer
            //                  relevant.

            // h == (anything), f == 3: locked.

            uint64_t *restrict Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf;
            uint64_t hash_bits = (hash_size-1) ;

            #if ( GB_NO_MASK )
            { 

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)=A*B(:,j)
                //--------------------------------------------------------------

                // no mask present, or mask ignored
                #undef GB_CHECK_MASK_ij
                #include "template/GB_AxB_saxpy3_fineHash_phase2.c"

            }
            #elif ( !GB_MASK_COMP )
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_in_place)
                {
                    // M is bitmap/as-if-full, thus not scattered into Hf
                    if (M_is_bitmap && Mask_struct)
                    { 
                        // M is bitmap and structural
                        const int8_t *restrict Mjb = Mb + pM_start ;
                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            if (!Mjb [i]) continue ;
                        #include "template/GB_AxB_saxpy3_fineHash_phase2.c"
                    }
                    else
                    { 
                        // M is bitmap/dense
                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            const int64_t pM = pM_start + i ;           \
                            GB_GET_M_ij (pM) ;                          \
                            if (!mij) continue ;
                        #include "template/GB_AxB_saxpy3_fineHash_phase2.c"
                    }
                }
                else
                { 
                    // M(:,j) is sparse and scattered into Hf
                    #include "template/GB_AxB_saxpy3_fineHash_M_phase2.c"
                }

            }
            #else
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<!M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_in_place)
                {
                    // M is bitmap/as-if-full, thus not scattered into Hf
                    if (M_is_bitmap && Mask_struct)
                    { 
                        // M is bitmap and structural
                        const int8_t *restrict Mjb = Mb + pM_start ;
                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            if (Mjb [i]) continue ;
                        #include "template/GB_AxB_saxpy3_fineHash_phase2.c"
                    }
                    else
                    { 
                        // M is bitmap/dense
                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            const int64_t pM = pM_start + i ;           \
                            GB_GET_M_ij (pM) ;                          \
                            if (mij) continue ;
                        #include "template/GB_AxB_saxpy3_fineHash_phase2.c"
                    }
                }
                else
                {
                    // M(:,j) is sparse/hyper and scattered into Hf
                    #include "template/GB_AxB_saxpy3_fineHash_notM_phase2.c"
                }
            }
            #endif
        }
    }

    //==========================================================================
    // phase3/phase4: count nnz(C(:,j)) for fine tasks, cumsum of Cp
    //==========================================================================

    // C->p may be revised by GB_AxB_saxpy3_cumsum, from 32-bit to 64-bit.

    GrB_Info info ;
    info = GB_AxB_saxpy3_cumsum (C, SaxpyTasks, nfine, chunk, nthreads, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //==========================================================================
    // phase5: numeric phase for coarse tasks, gather for fine tasks
    //==========================================================================

{

    // Cp may have started as 32-bit but might now be 64-bit, depending on the
    // problem size.  Use the new C->p for phase5.  For the JIT kernel, the
    // size of C->p is no longer known at compile time.  All kernels (JIT,
    // PreJIT, Factory, and generic) must thus use the ternary operator in the
    // GB_Cp_IGET macro.  The definitions of Cp, Cp32, and Cp64 intentionally
    // shadow the definitions above, by GB_Cp_DECLARE (...) ;

    const void *Cp = C->p ;
    const uint32_t *restrict Cp32 = C->p_is_32 ? Cp : NULL ;
    const uint64_t *restrict Cp64 = C->p_is_32 ? NULL : Cp ;
    ASSERT (Cp != NULL) ;

    #define GB_Cp_IGET(k) (Cp32 ? Cp32 [k] : Cp64 [k])

    // C is iso for the ANY_PAIR semiring, and non-iso otherwise

    // allocate Ci and Cx
    int64_t cnz = GB_Cp_IGET (cnvec) ;
    info = GB_bix_alloc (C, cnz, GxB_SPARSE, false, true,
        GB_IS_ANY_PAIR_SEMIRING) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        // note the C->p and C->h are not freed if GB_bix_alloc fails
        ASSERT (C->p != NULL) ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    C->nvals = cnz ;

    for (int64_t kk = 0 ; kk < cnvec ; kk++)
    {
        int64_t pC_start = GB_Cp_IGET (kk) ;
        int64_t pC_end = GB_Cp_IGET (kk+1) ;
        ASSERT (pC_start >= 0) ;
        ASSERT (pC_start <= pC_end) ;
        ASSERT (pC_end <= cnz) ;
    }

    GB_Ci_DECLARE_U (Ci, ) ; GB_Ci_PTR (Ci, C) ;
    #if ( !GB_IS_ANY_PAIR_SEMIRING )
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    bool C_jumbled = false ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) reduction(||:C_jumbled)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_C_TYPE *restrict Hx = (GB_C_TYPE *) SaxpyTasks [taskid].Hx ;
        #endif
        uint64_t hash_size = SaxpyTasks [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool task_C_jumbled = false ;

        if (taskid < nfine)
        {

            //------------------------------------------------------------------
            // fine task: gather pattern and values
            //------------------------------------------------------------------

            int64_t kk = SaxpyTasks [taskid].vector ;
            int team_size = SaxpyTasks [taskid].team_size ;
            int leader    = SaxpyTasks [taskid].leader ;
            int my_teamid = taskid - leader ;
            int64_t pC = GB_Cp_IGET (kk) ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // Hf [i] == 2 if C(i,j) is an entry in C(:,j)
                int8_t *restrict
                    Hf = (int8_t *restrict) SaxpyTasks [taskid].Hf ;
                int64_t cjnz = GB_Cp_IGET (kk+1) - pC ;
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;
                if (cjnz == cvlen)
                {
                    // C(:,j) is dense
                    for (int64_t i = istart ; i < iend ; i++)
                    { 
                        GB_ISET (Ci, pC + i, i) ;   // Ci [pC + i] = i ;
                    }
                    // copy Hx [istart:iend-1] into Cx [pC+istart:pC+iend-1]
                    GB_CIJ_MEMCPY (pC + istart, istart, iend - istart) ;
                }
                else
                {
                    // C(:,j) is sparse
                    pC += SaxpyTasks [taskid].my_cjnz ;
                    for (uint64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i] == 2)
                        { 
                            GB_CIJ_GATHER (pC, i) ;     // Cx [pC] = Hx [i]
                            GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
                            pC++ ;
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: fine hash task, C=A*B, C<M>=A*B, C<!M>=A*B
                //--------------------------------------------------------------

                // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
                // and the index i of the entry is (Hf [hash] >> 2) - 1.

                uint64_t *restrict
                    Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ;
                pC += SaxpyTasks [taskid].my_cjnz ;
                for (uint64_t hash = mystart ; hash < myend ; hash++)
                {
                    uint64_t hf = Hf [hash] ;
                    if ((hf & 3) == 2)
                    { 
                        uint64_t i = (hf >> 2) - 1 ;    // found C(i,j) in hash
                        GB_ISET (Ci, pC, i) ;       // Ci [pC] = i ;
                        GB_CIJ_GATHER (pC, hash) ;  // Cx [pC] = Hx [hash]
                        pC++ ;
                    }
                }
                task_C_jumbled = true ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // numeric coarse task: compute C(:,kfirst:klast)
            //------------------------------------------------------------------

            uint64_t *restrict
                Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf ;
            int64_t kfirst = SaxpyTasks [taskid].start ;
            int64_t klast = SaxpyTasks [taskid].end ;
            int64_t nk = klast - kfirst + 1 ;
            uint64_t mark = 2*nk + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: coarse Gustavson task
                //--------------------------------------------------------------

                #if !defined ( GB_GENERIC ) && !GB_IS_ANY_PAIR_SEMIRING
                // declare the monoid identity value, for GB_COMPUTE_DENSE_C_j,
                // needed only for the 3 kinds of coarseGus_*_phase5 below.
                // Not used for generic case, nor for (any,pair) semiring.
                GB_DECLARE_IDENTITY_CONST (zidentity) ;
                #endif

                #if ( GB_NO_MASK )
                { 
                    // phase5: coarse Gustavson task, C=A*B
                    #include "template/GB_AxB_saxpy3_coarseGus_noM_phase5.c"
                }
                #elif ( !GB_MASK_COMP )
                { 
                    // phase5: coarse Gustavson task, C<M>=A*B
                    #include "template/GB_AxB_saxpy3_coarseGus_M_phase5.c"
                }
                #else
                { 
                    // phase5: coarse Gustavson task, C<!M>=A*B
                    #include "template/GB_AxB_saxpy3_coarseGus_notM_phase5.c"
                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: coarse hash task
                //--------------------------------------------------------------

                uint64_t *restrict Hi = SaxpyTasks [taskid].Hi ;
                uint64_t hash_bits = (hash_size-1) ;

                #if ( GB_NO_MASK )
                { 

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // no mask present, or mask ignored (see below)
                    #undef GB_CHECK_MASK_ij
                    #include "template/GB_AxB_saxpy3_coarseHash_phase5.c"

                }
                #elif ( !GB_MASK_COMP )
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_in_place)
                    {
                        // M is bitmap/as-if-full, thus not scattered into Hf
                        if (M_is_bitmap && Mask_struct)
                        { 
                            // M is bitmap and structural
                            #define GB_MASK_IS_BITMAP_AND_STRUCTURAL
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                if (!Mjb [i]) continue ;
                            #include "template/GB_AxB_saxpy3_coarseHash_phase5.c"
                        }
                        else
                        { 
                            // M is bitmap/dense
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                const int64_t pM = pM_start + i ;           \
                                GB_GET_M_ij (pM) ;                          \
                                if (!mij) continue ;
                            #include "template/GB_AxB_saxpy3_coarseHash_phase5.c"
                        }
                    }
                    else
                    { 
                        // M is sparse and scattered into Hf
                        #include "template/GB_AxB_saxpy3_coarseHash_M_phase5.c"
                    }

                }
                #else
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<!M>=A*B
                    //---------------------------------------------------------- 

                    if (M_in_place)
                    {
                        // M is bitmap/as-if-full, thus not scattered into Hf
                        if (M_is_bitmap && Mask_struct)
                        { 
                            // M is bitmap and structural
                            #define GB_MASK_IS_BITMAP_AND_STRUCTURAL
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                if (Mjb [i]) continue ;
                            #include "template/GB_AxB_saxpy3_coarseHash_phase5.c"
                        }
                        else
                        { 
                            // M is bitmap/dense
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                const int64_t pM = pM_start + i ;           \
                                GB_GET_M_ij (pM) ;                          \
                                if (mij) continue ;
                            #include "template/GB_AxB_saxpy3_coarseHash_phase5.c"
                        }
                    }
                    else
                    { 
                        // M is sparse and scattered into Hf
                        #include "template/GB_AxB_saxpy3_coarseHash_notM_phase5.c"
                    }
                }
                #endif
            }
        }
        C_jumbled = C_jumbled || task_C_jumbled ;
    }

    //--------------------------------------------------------------------------
    // log the state of C->jumbled
    //--------------------------------------------------------------------------

    C->jumbled = C_jumbled ;    // C is jumbled if any task left it jumbled
}
}

#undef GB_NO_MASK
#undef GB_MASK_COMP

