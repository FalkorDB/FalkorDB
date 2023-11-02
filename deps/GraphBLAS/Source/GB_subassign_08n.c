//------------------------------------------------------------------------------
// GB_subassign_08n: C(I,J)<M> += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 08n: C(I,J)<M> += A ; no S

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none

// C not bitmap; C can be full since no zombies are inserted in that case.
// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.
// M, A: not bitmap; Method 08s is used instead if M or A are bitmap.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// GB_PHASE1_ACTION
//------------------------------------------------------------------------------

// action to take for phase 1 when A(i,j) exists and M(i,j)=1
#define GB_PHASE1_ACTION                                                    \
{                                                                           \
    if (cjdense)                                                            \
    {                                                                       \
        /* direct lookup of C(iC,jC) */                                     \
        GB_iC_DENSE_LOOKUP ;                                                \
        /* ----[C A 1] or [X A 1]------------------------------- */         \
        /* [C A 1]: action: ( =C+A ): apply accum                */         \
        /* [X A 1]: action: ( undelete ): zombie lives           */         \
        GB_withaccum_C_A_1_matrix ;                                         \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* binary search for C(iC,jC) in C(:,jC) */                         \
        GB_iC_BINARY_SEARCH ;                                               \
        if (cij_found)                                                      \
        {                                                                   \
            /* ----[C A 1] or [X A 1]--------------------------- */         \
            /* [C A 1]: action: ( =C+A ): apply accum            */         \
            /* [X A 1]: action: ( undelete ): zombie lives       */         \
            GB_withaccum_C_A_1_matrix ;                                     \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* ----[. A 1]-------------------------------------- */         \
            /* [. A 1]: action: ( insert )                       */         \
            task_pending++ ;                                                \
        }                                                                   \
    }                                                                       \
}

//------------------------------------------------------------------------------
// GB_PHASE2_ACTION
//------------------------------------------------------------------------------

// action to take for phase 2 when A(i,j) exists and M(i,j)=1
#define GB_PHASE2_ACTION                                                    \
{                                                                           \
    ASSERT (!cjdense) ;                                                     \
    {                                                                       \
        /* binary search for C(iC,jC) in C(:,jC) */                         \
        GB_iC_BINARY_SEARCH ;                                               \
        if (!cij_found)                                                     \
        {                                                                   \
            /* ----[. A 1]-------------------------------------- */         \
            /* [. A 1]: action: ( insert )                       */         \
            GB_PENDING_INSERT_aij ;                                         \
        }                                                                   \
    }                                                                       \
}

//------------------------------------------------------------------------------
// GB_subassign_08n: C(I,J)<M> += A ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_08n
(
    GrB_Matrix C,
    // input:
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (M)) ;    // Method 08s is used if M is bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;    // Method 08s is used if A is bitmap
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_MATRIX_WAIT_IF_JUMBLED (C) ;
    GB_MATRIX_WAIT_IF_JUMBLED (M) ;
    GB_MATRIX_WAIT_IF_JUMBLED (A) ;

    GB_GET_C ;      // C must not be bitmap
    int64_t zorig = C->nzombies ;
    const int64_t Cnvec = C->nvec ;
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    GB_GET_C_HYPER_HASH ;
    GB_GET_MASK ;
    GB_GET_ACCUM_MATRIX ;
    const int64_t *Ah = A->h ;

    //--------------------------------------------------------------------------
    // Method 08n: C(I,J)<M> += A ; no S
    //--------------------------------------------------------------------------

    // Time: Close to optimal. Omega (sum_j (min (nnz (A(:,j)), nnz (M(:,j)))),
    // since only the intersection of A.*M needs to be considered.  If either
    // M(:,j) or A(:,j) are very sparse compared to the other, then the shorter
    // is traversed with a linear-time scan and a binary search is used for the
    // other.  If the number of nonzeros is comparable, a linear-time scan is
    // used for both.  Once two entries M(i,j)=1 and A(i,j) are found with the
    // same index i, the entry A(i,j) is accumulated or inserted into C.

    // The algorithm is very much like the eWise multiplication of A.*M, so the
    // parallel scheduling relies on GB_emult_08_phase0 and GB_ewise_slice.

    //--------------------------------------------------------------------------
    // Parallel: slice the eWiseMult of Z=A.*M (Method 08n only)
    //--------------------------------------------------------------------------

    // Method 08n only.  If C is sparse, it is sliced for a fine task, so that
    // it can do a binary search via GB_iC_BINARY_SEARCH.  But if C(:,jC) is
    // dense, C(:,jC) is not sliced, so the fine task must do a direct lookup
    // via GB_iC_DENSE_LOOKUP.  Otherwise a race condition will occur.
    // The Z matrix is not constructed, except for its hyperlist (Zh_shallow)
    // and mapping to A and M.

    // No matrix (C, M, or A) can be bitmap.  C, M, A can be sparse/hyper/full,
    // in any combination.

    int64_t Znvec ;
    const int64_t *restrict Zh_shallow = NULL ;
    GB_OK (GB_subassign_08n_slice (
        &TaskList, &TaskList_size, &ntasks, &nthreads,
        &Znvec, &Zh_shallow, &Z_to_A, &Z_to_A_size, &Z_to_M, &Z_to_M_size,
        C, I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
        A, M, Werk)) ;
    GB_ALLOCATE_NPENDING_WERK ;

    //--------------------------------------------------------------------------
    // phase 1: undelete zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR_PHASE1 ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Zh_shallow, k) ;
            GB_GET_EVEC (pA, pA_end, pA, pA_end, Ap, Ah, j, k, Z_to_A, Avlen) ;
            GB_GET_EVEC (pM, pM_end, pB, pB_end, Mp, Mh, j, k, Z_to_M, Mvlen) ;

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t mjnz = pM_end - pM ;
            if (ajnz == 0 || mjnz == 0) continue ;
            int64_t iA_first = GBI (Ai, pA, Avlen) ;
            int64_t iA_last  = GBI (Ai, pA_end-1, Avlen) ;
            int64_t iM_first = GBI (Mi, pM, Mvlen) ;
            int64_t iM_last  = GBI (Mi, pM_end-1, Mvlen) ;
            if (iA_last < iM_first || iM_last < iA_first) continue ;
            int64_t pM_start = pM ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC (fine_task, taskid) ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;


            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += A(:,j) ; no S
            //------------------------------------------------------------------

            if (ajnz > 32 * mjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than M(:,j)
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBI (Mi, pM, Mvlen) ;
                        // find iA in A(:,j)
                        int64_t pright = pA_end - 1 ;
                        bool found ;
                        // FUTURE::: exploit dense A(:,j)
                        GB_BINARY_SEARCH (iA, Ai, pA, pright, found) ;
                        if (found) GB_PHASE1_ACTION ;
                    }
                }

            }
            else if (mjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // M(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // FUTURE::: exploit dense mask
                bool mjdense = false ;

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (mij) GB_PHASE1_ACTION ;
                }

            }
            else
            {

                //----------------------------------------------------------
                // A(:,j) and M(:,j) have about the same # of entries
                //----------------------------------------------------------

                // linear-time scan of A(:,j) and M(:,j)

                while (pA < pA_end && pM < pM_end)
                {
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    int64_t iM = GBI (Mi, pM, Mvlen) ;
                    if (iA < iM)
                    { 
                        // A(i,j) exists but not M(i,j)
                        GB_NEXT (A) ;
                    }
                    else if (iM < iA)
                    { 
                        // M(i,j) exists but not A(i,j)
                        GB_NEXT (M) ;
                    }
                    else
                    { 
                        // both A(i,j) and M(i,j) exist
                        if (GB_MCAST (Mx, pM, msize)) GB_PHASE1_ACTION ;
                        GB_NEXT (A) ;
                        GB_NEXT (M) ;
                    }
                }
            }
        }

        GB_PHASE1_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    GB_PENDING_CUMSUM ;
    zorig = C->nzombies ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(&&:pending_sorted)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR_PHASE2 ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Zh_shallow, k) ;
            GB_GET_EVEC (pA, pA_end, pA, pA_end, Ap, Ah, j, k, Z_to_A, Avlen) ;
            GB_GET_EVEC (pM, pM_end, pB, pB_end, Mp, Mh, j, k, Z_to_M, Mvlen) ;

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t mjnz = pM_end - pM ;
            if (ajnz == 0 || mjnz == 0) continue ;
            int64_t iA_first = GBI (Ai, pA, Avlen) ;
            int64_t iA_last  = GBI (Ai, pA_end-1, Avlen) ;
            int64_t iM_first = GBI (Mi, pM, Mvlen) ;
            int64_t iM_last  = GBI (Mi, pM_end-1, Mvlen) ;
            if (iA_last < iM_first || iM_last < iA_first) continue ;
            int64_t pM_start = pM ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC (fine_task, taskid) ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;
            if (cjdense) continue ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += A(:,j) ; no S
            //------------------------------------------------------------------

            if (ajnz > 32 * mjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than M(:,j)
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {
                    if (GB_MCAST (Mx, pM, msize))
                    { 
                        int64_t iA = GBI (Mi, pM, Mvlen) ;
                        // find iA in A(:,j)
                        int64_t pright = pA_end - 1 ;
                        bool found ;
                        // FUTURE::: exploit dense A(:,j)
                        GB_BINARY_SEARCH (iA, Ai, pA, pright, found) ;
                        if (found) GB_PHASE2_ACTION ;
                    }
                }

            }
            else if (mjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // M(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // FUTURE::: exploit dense mask
                bool mjdense = false ;

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (mij) GB_PHASE2_ACTION ;
                }

            }
            else
            {

                //----------------------------------------------------------
                // A(:,j) and M(:,j) have about the same # of entries
                //----------------------------------------------------------

                // linear-time scan of A(:,j) and M(:,j)

                while (pA < pA_end && pM < pM_end)
                {
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    int64_t iM = GBI (Mi, pM, Mvlen) ;
                    if (iA < iM)
                    { 
                        // A(i,j) exists but not M(i,j)
                        GB_NEXT (A) ;
                    }
                    else if (iM < iA)
                    { 
                        // M(i,j) exists but not A(i,j)
                        GB_NEXT (M) ;
                    }
                    else
                    { 
                        // both A(i,j) and M(i,j) exist
                        if (GB_MCAST (Mx, pM, msize)) GB_PHASE2_ACTION ;
                        GB_NEXT (A) ;
                        GB_NEXT (M) ;
                    }
                }
            }
        }

        GB_PHASE2_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

