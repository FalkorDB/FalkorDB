//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic_template: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Symbolic analysis for C=A*B, C<M>=A*B or C<!M>=A*B, via GB_AxB_saxpy3.
// Coarse tasks compute nnz (C (:,j)) for each of their vectors j.  Fine tasks
// just scatter the mask M into the hash table.  This phase does not depend on
// the semiring, nor does it depend on the type of C, A, or B.  It does access
// the values of M, if the mask matrix M is present and not structural.

// If B is hypersparse, C must also be hypersparse.
// Otherwise, C must be sparse.

// The sparsity of A and B are #defined' constants for this method,
// as is the 3 cases of the mask (no M, M, or !M).

#include "mxm/GB_AxB_saxpy3.h"
#include "mxm/include/GB_mxm_shared_definitions.h"
#include "mxm/include/GB_AxB_saxpy3_template.h"
#include "include/GB_unused.h"

#define GB_META16
#include "mxm/include/GB_meta16_definitions.h"

void GB_EVAL2 (GB (AxB_saxpy3_sym), GB_MASK_A_B_SUFFIX)
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    #if ( !GB_NO_MASK )
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    #endif
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
)
{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    const int64_t cvlen = C->vlen ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE_U (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t *restrict Bb = B->b ;
    const int64_t bvlen = B->vlen ;

    ASSERT (GB_B_IS_SPARSE == GB_IS_SPARSE (B)) ;
    ASSERT (GB_B_IS_HYPER  == GB_IS_HYPERSPARSE (B)) ;
    ASSERT (GB_B_IS_BITMAP == GB_IS_BITMAP (B)) ;
    ASSERT (GB_B_IS_FULL   == GB_IS_FULL   (B)) ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE_U (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t *restrict Ab = A->b ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_jumbled = A->jumbled ;

    ASSERT (GB_A_IS_SPARSE == GB_IS_SPARSE (A)) ;
    ASSERT (GB_A_IS_HYPER  == GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_A_IS_BITMAP == GB_IS_BITMAP (A)) ;
    ASSERT (GB_A_IS_FULL   == GB_IS_FULL   (A)) ;
    const bool Ai_is_32 = A->i_is_32 ;
    #define GB_Ai_IS_32 Ai_is_32

    #if GB_A_IS_HYPER
    const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;
    #define GB_Ap_IS_32 Ap_is_32
    #define GB_Aj_IS_32 Aj_is_32
    #endif

    #if ( !GB_NO_MASK )
    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;
    GB_Mi_DECLARE_U (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = M->b ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    size_t  msize = M->type->size ;
    int64_t mnvec = M->nvec ;
    int64_t mvlen = M->vlen ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_jumbled = GB_JUMBLED (M) ;
    // get the M hyper_hash
    const void *M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;
    const void *M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;
    const void *M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;
    const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;
    const bool Mp_is_32 = M->p_is_32 ;
    const bool Mj_is_32 = M->j_is_32 ;
    #define GB_Mp_IS_32 Mp_is_32
    #define GB_Mj_IS_32 Mj_is_32
    #endif

    //==========================================================================
    // phase1: count nnz(C(:,j)) for coarse tasks, scatter M for fine tasks
    //==========================================================================

    // At this point, all of Hf [...] is zero, for all tasks.
    // Hi and Hx are not initialized.

    int taskid ;
    #if defined ( GCC_PPC_BUG ) && ( !GB_NO_MASK )
    bool punt = false ;
    #endif

    #pragma omp parallel for num_threads(nthreads) schedule(static,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        uint64_t hash_size = SaxpyTasks [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;

        if (taskid < nfine)
        {

            //------------------------------------------------------------------
            // no work for fine tasks in phase1 if M is not present
            //------------------------------------------------------------------

            #if ( !GB_NO_MASK )
            {

                //--------------------------------------------------------------
                // get the task descriptor
                //--------------------------------------------------------------

                int64_t kk = SaxpyTasks [taskid].vector ;
                int64_t bjnz = (Bp == NULL) ? bvlen :
                    (GB_IGET (Bp, kk+1) - GB_IGET (Bp, kk)) ;
                // no work to do if B(:,j) is empty
                if (bjnz == 0) continue ;

                // partition M(:,j)
                GB_GET_M_j ;        // get M(:,j)

                int team_size = SaxpyTasks [taskid].team_size ;
                int leader    = SaxpyTasks [taskid].leader ;
                int my_teamid = taskid - leader ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, mjnz, my_teamid, team_size) ;
                mystart += pM_start ;
                myend   += pM_start ;

                if (use_Gustavson)
                { 

                    //----------------------------------------------------------
                    // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                    //----------------------------------------------------------

                    // Scatter the values of M(:,j) into Hf.  No atomics needed
                    // since all indices i in M(;,j) are unique.  Do not
                    // scatter the mask if M(:,j) is a dense vector, since in
                    // that case the numeric phase accesses M(:,j) directly,
                    // not via Hf.

                    if (mjnz > 0)
                    { 
                        int8_t *restrict
                            Hf = (int8_t *restrict) SaxpyTasks [taskid].Hf ;
                        GB_SCATTER_M_j (mystart, myend, 1) ;
                    }

                }
                else if (!M_in_place)
                { 

                    //----------------------------------------------------------
                    // phase1: fine hash task, C<M>=A*B or C<!M>=A*B (parallel)
                    //----------------------------------------------------------

                    #ifdef GCC_PPC_BUG
                    // This fine hash task is skipped and done below, outside
                    // of this parallel region.
                    GB_ATOMIC_WRITE
                    punt = true ;
                    #else
                    #include "mxm/factory/GB_AxB_saxpy3_symbolic_fine_template.c"
                    #endif
                }
            }
            #endif

        }
        else
        {

            //------------------------------------------------------------------
            // coarse tasks: compute nnz in each vector of A*B(:,kfirst:klast)
            //------------------------------------------------------------------

            uint64_t *restrict
                Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf ;
            int64_t kfirst = SaxpyTasks [taskid].start ;
            int64_t klast  = SaxpyTasks [taskid].end ;
            uint64_t mark = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1: coarse Gustavson task
                //--------------------------------------------------------------

                #if ( GB_NO_MASK )
                { 
                    // phase1: coarse Gustavson task, C=A*B
                    #include "mxm/template/GB_AxB_saxpy3_coarseGus_noM_phase1.c"
                }
                #elif ( !GB_MASK_COMP )
                { 
                    // phase1: coarse Gustavson task, C<M>=A*B
                    #include "mxm/template/GB_AxB_saxpy3_coarseGus_M_phase1.c"
                }
                #else
                { 
                    // phase1: coarse Gustavson task, C<!M>=A*B
                    #include "mxm/template/GB_AxB_saxpy3_coarseGus_notM_phase1.c"
                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // phase1: coarse hash task
                //--------------------------------------------------------------

                uint64_t *restrict Hi = SaxpyTasks [taskid].Hi ;
                uint64_t hash_bits = (hash_size-1) ;

                #if ( GB_NO_MASK )
                { 

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    #undef GB_CHECK_MASK_ij
                    #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"

                }
                #elif ( !GB_MASK_COMP )
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_in_place)
                    { 

                        //------------------------------------------------------
                        // M(:,j) is dense.  M is not scattered into Hf.
                        //------------------------------------------------------

                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (!mij) continue ;

                        switch (msize)
                        {
                            default:
                            case GB_1BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint8_t
                                #undef  M_SIZE
                                #define M_SIZE 1
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_2BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint16_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_4BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint32_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_8BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint64_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_16BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint64_t
                                #undef  M_SIZE
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (!mij) continue ;
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                        }

                    }
                    else
                    { 

                        //------------------------------------------------------
                        // M is sparse and scattered into Hf
                        //------------------------------------------------------
                        
                        #include "mxm/template/GB_AxB_saxpy3_coarseHash_M_phase1.c"
                    }

                }
                #else
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    if (M_in_place)
                    {

                        //------------------------------------------------------
                        // M(:,j) is dense.  M is not scattered into Hf.
                        //------------------------------------------------------

                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (mij) continue ;

                        switch (msize)
                        {
                            default:
                            case GB_1BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint8_t
                                #undef  M_SIZE
                                #define M_SIZE 1
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_2BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint16_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_4BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint32_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_8BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint64_t
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                            case GB_16BYTE : 
                                #undef  M_TYPE
                                #define M_TYPE uint64_t
                                #undef  M_SIZE
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (mij) continue ;
                                #include "mxm/template/GB_AxB_saxpy3_coarseHash_phase1.c"
                                break ;
                        }

                    }
                    else
                    { 

                        //------------------------------------------------------
                        // M is sparse and scattered into Hf
                        //------------------------------------------------------

                        #include "mxm/template/GB_AxB_saxpy3_coarseHash_notM_phase1.c"
                    }
                }
                #endif
            }
        }
    }

    //==========================================================================
    // phase 1: punt, for gcc on Power8 or s390x
    //==========================================================================

    // A single thread is used to do all fine hash tasks, when M_in_place
    // is false.

    #if defined ( GCC_PPC_BUG ) && ( !GB_NO_MASK )
    if (punt)
    {
        for (taskid = 0 ; taskid < nfine ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            uint64_t hash_size = SaxpyTasks [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;

            if (!use_Gustavson && !M_in_place)
            {

                //--------------------------------------------------------------
                // get the task descriptor
                //--------------------------------------------------------------

                int64_t kk = SaxpyTasks [taskid].vector ;
                int64_t bjnz = (Bp == NULL) ? bvlen :
                    (GB_IGET (Bp, kk+1) - GB_IGET (Bp, kk)) ;
                // no work to do if B(:,j) is empty
                if (bjnz == 0) continue ;

                // partition M(:,j)
                GB_GET_M_j ;        // get M(:,j)

                int team_size = SaxpyTasks [taskid].team_size ;
                int leader    = SaxpyTasks [taskid].leader ;
                int my_teamid = taskid - leader ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, mjnz, my_teamid, team_size) ;
                mystart += pM_start ;
                myend   += pM_start ;

                //----------------------------------------------------------
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B (1 thread)
                //----------------------------------------------------------

                #include "mxm/factory/GB_AxB_saxpy3_symbolic_fine_template.c"
            }
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // check result for phase1 for fine tasks
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    #if ( !GB_NO_MASK )
    {
        for (taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t kk = SaxpyTasks [taskid].vector ;
            ASSERT (kk >= 0 && kk < B->nvec) ;
            int64_t bjnz = (Bp == NULL) ? bvlen :
                    (GB_IGET (Bp, kk+1) - GB_IGET (Bp, kk)) ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;
            uint64_t hash_size = SaxpyTasks [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            int leader = SaxpyTasks [taskid].leader ;
            if (leader != taskid) continue ;
            GB_GET_M_j ;        // get M(:,j)
            if (mjnz == 0) continue ;
            int64_t mjcount2 = 0 ;
            int64_t mjcount = 0 ;
            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                GB_GET_M_ij (pM) ;                  // get M(i,j)
                if (mij) mjcount++ ;
            }
            if (use_Gustavson)
            {
                // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                int8_t *restrict
                    Hf = (int8_t *restrict) SaxpyTasks [taskid].Hf ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij (pM) ;               // get M(i,j)
                    uint64_t i = GBi_M (Mi, pM, mvlen) ;
                    ASSERT (Hf [i] == mij) ;
                }
                for (int64_t i = 0 ; i < cvlen ; i++)
                {
                    ASSERT (Hf [i] == 0 || Hf [i] == 1) ;
                    if (Hf [i] == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
            else if (!M_in_place)
            {
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1
                uint64_t *restrict
                    Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf ;
                uint64_t hash_bits = (hash_size-1) ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij (pM) ;              // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    uint64_t i = GBi_M (Mi, pM, mvlen) ;
                    uint64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    int64_t probe = 0 ;
                    for (GB_HASH (i))
                    {
                        uint64_t hf = Hf [hash] ;
                        if (hf == i_mine) 
                        {
                            mjcount2++ ;
                            break ;
                        }
                        ASSERT (hf != 0) ;
                        probe++ ;
                        ASSERT (probe < cvlen) ;
                    }
                }
                ASSERT (mjcount == mjcount2) ;
                mjcount2 = 0 ;
                for (uint64_t hash = 0 ; hash < hash_size ; hash++)
                {
                    uint64_t hf = Hf [hash] ;
                    uint64_t h = (hf >> 2) ; // empty (0), or a 1-based index
                    uint64_t f = (hf & 3) ;  // 0 if empty or 1 if occupied
                    if (f == 1) ASSERT (h >= 1 && h <= cvlen) ;
                    ASSERT (hf == 0 || f == 1) ;
                    if (f == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
        }
    }
    #endif
    #endif
}

