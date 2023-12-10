//------------------------------------------------------------------------------
// GB_subassign_12_and_20: C(I,J)<M or !M,repl> += A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 12: C(I,J)<M,repl> += A ; using S
// Method 20: C(I,J)<!M,repl> += A ; using S

// M:           present
// Mask_comp:   true or false
// C_replace:   true
// accum:       present
// A:           matrix
// S:           constructed

// C: not bitmap: use GB_bitmap_assign instead
// M, A: any sparsity structure.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

GrB_Info GB_subassign_12_and_20
(
    GrB_Matrix C,
    // input:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, !M, else use M
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A

    //--------------------------------------------------------------------------
    // S = C(I,J)
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_CLEAR_STATIC_HEADER (S, &S_header) ;
    GB_OK (GB_subassign_symbolic (S, C, I, ni, J, nj, true, Werk)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_JUMBLED (M) ;
    GB_MATRIX_WAIT_IF_JUMBLED (A) ;

    GB_GET_C ;      // C must not be bitmap
    GB_GET_MASK ;
    GB_GET_MASK_HYPER_HASH ;
    GB_GET_S ;
    GB_GET_ACCUM_MATRIX ;

    //--------------------------------------------------------------------------
    // Method 12: C(I,J)<M,repl> += A ; using S
    // Method 20: C(I,J)<!M,repl> += A ; using S
    //--------------------------------------------------------------------------

    // Time: all entries in S+A must be traversed, so Omega(nnz(S)+nnz(A)) is
    // required.  All cases of the mask (0, 1, or not present) must be
    // considered, because of the C_replace descriptor being true.

    //--------------------------------------------------------------------------
    // Parallel: A+S (Methods 02, 04, 09, 10, 11, 12, 14, 16, 18, 20)
    //--------------------------------------------------------------------------

    if (A_is_bitmap)
    { 
        // all of IxJ must be examined
        GB_SUBASSIGN_IXJ_SLICE ;
    }
    else
    { 
        // traverse all A+S
        GB_SUBASSIGN_TWO_SLICE (A, S) ;
    }

    //--------------------------------------------------------------------------
    // phase 1: create zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    if (A_is_bitmap)
    {

        //----------------------------------------------------------------------
        // phase1: A is bitmap
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:nzombies)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_IXJ_TASK_DESCRIPTOR_PHASE1 (iA_start, iA_end) ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t j = kfirst ; j <= klast ; j++)
            {

                //--------------------------------------------------------------
                // get S(iA_start:iA_end,j)
                //--------------------------------------------------------------

                GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;
                int64_t pA_start = j * Avlen ;

                //--------------------------------------------------------------
                // get M(:,j)
                //--------------------------------------------------------------

                int64_t pM_start, pM_end ;
                GB_LOOKUP_VECTOR (pM_start, pM_end, M, j) ;
                bool mjdense = (pM_end - pM_start) == Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(iA_start:iA_end,j) and A(ditto,j)
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {
                    int64_t pA = pA_start + iA ;
                    bool Sfound = (pS < pS_end) && (GBI (Si, pS, Svlen) == iA) ;
                    bool Afound = Ab [pA] ;

                    if (Sfound && !Afound)
                    {
                        // S (i,j) is present but A (i,j) is not
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        if (!mij)
                        { 
                            // ----[C . 0] or [X . 0]---------------------------
                            // [X . 0]: action: ( X ): still a zombie
                            // [C . 0]: C_repl: action: ( delete ): now zombie
                            GB_C_S_LOOKUP ;
                            GB_DELETE_ENTRY ;
                        }
                        GB_NEXT (S) ;
                    }
                    else if (!Sfound && Afound)
                    {
                        // S (i,j) is not present, A (i,j) is present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
                    }
                    else if (Sfound && Afound)
                    {
                        // both S (i,j) and A (i,j) present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        GB_C_S_LOOKUP ;
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): A to C no accum
                            // [C A 1]: action: ( =C+A ): apply accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_withaccum_C_A_1_matrix ;
                        }
                        else
                        { 
                            // ----[C A 0] or [X A 0]---------------------------
                            // [X A 0]: action: ( X ): still a zombie
                            // [C A 0]: C_repl: action: ( delete ): now zombie
                            GB_DELETE_ENTRY ;
                        }
                        GB_NEXT (S) ;
                    }
                }
            }
            GB_PHASE1_TASK_WRAPUP ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // phase1: A is hypersparse, sparse, or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:nzombies)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_TASK_DESCRIPTOR_PHASE1 ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // get A(:,j) and S(:,j)
                //--------------------------------------------------------------

                int64_t j = GBH (Zh, k) ;
                GB_GET_MAPPED (pA, pA_end, pA, pA_end, Ap, j, k, Z_to_X, Avlen);
                GB_GET_MAPPED (pS, pS_end, pB, pB_end, Sp, j, k, Z_to_S, Svlen);

                //--------------------------------------------------------------
                // get M(:,j)
                //--------------------------------------------------------------

                int64_t pM_start, pM_end ;
                GB_LOOKUP_VECTOR (pM_start, pM_end, M, j) ;
                bool mjdense = (pM_end - pM_start) == Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(:,j) and A(:,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                // int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

                // while both list S (:,j) and A (:,j) have entries
                while (pS < pS_end && pA < pA_end)
                {
                    int64_t iS = GBI (Si, pS, Svlen) ;
                    int64_t iA = GBI (Ai, pA, Avlen) ;

                    if (iS < iA)
                    {
                        // S (i,j) is present but A (i,j) is not
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iS) ;
                        if (Mask_comp) mij = !mij ;
                        if (!mij)
                        { 
                            // ----[C . 0] or [X . 0]---------------------------
                            // [X . 0]: action: ( X ): still a zombie
                            // [C . 0]: C_repl: action: ( delete ): now zombie
                            GB_C_S_LOOKUP ;
                            GB_DELETE_ENTRY ;
                        }
                        GB_NEXT (S) ;
                    }
                    else if (iA < iS)
                    {
                        // S (i,j) is not present, A (i,j) is present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
                        GB_NEXT (A) ;
                    }
                    else
                    {
                        // both S (i,j) and A (i,j) present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        GB_C_S_LOOKUP ;
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): A to C no accum
                            // [C A 1]: action: ( =C+A ): apply accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_withaccum_C_A_1_matrix ;
                        }
                        else
                        { 
                            // ----[C A 0] or [X A 0]---------------------------
                            // [X A 0]: action: ( X ): still a zombie
                            // [C A 0]: C_repl: action: ( delete ): now zombie
                            GB_DELETE_ENTRY ;
                        }
                        GB_NEXT (S) ;
                        GB_NEXT (A) ;
                    }
                }

                // while list S (:,j) has entries.  List A (:,j) exhausted.
                while (pS < pS_end)
                {
                    int64_t iS = GBI (Si, pS, Svlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iS) ;
                    if (Mask_comp) mij = !mij ;
                    if (!mij)
                    { 
                        // ----[C . 0] or [X . 0]-------------------------------
                        // [X . 0]: action: ( X ): still a zombie
                        // [C . 0]: C_repl: action: ( delete ): becomes zombie
                        GB_C_S_LOOKUP ;
                        GB_DELETE_ENTRY ;
                    }
                    GB_NEXT (S) ;
                }

                // while list A (:,j) has entries.  List S (:,j) exhausted.
                while (pA < pA_end)
                {
                    // S (i,j) is not present, A (i,j) is present
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (Mask_comp) mij = !mij ;
                    if (mij)
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        task_pending++ ;
                    }
                    GB_NEXT (A) ;
                }
            }

            GB_PHASE1_TASK_WRAPUP ;
        }
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    GB_PENDING_CUMSUM ;

    if (A_is_bitmap)
    {

        //----------------------------------------------------------------------
        // phase2: A is bitmap
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(&&:pending_sorted)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_IXJ_TASK_DESCRIPTOR_PHASE2 (iA_start, iA_end) ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t j = kfirst ; j <= klast ; j++)
            {

                //--------------------------------------------------------------
                // get S(iA_start:iA_end,j)
                //--------------------------------------------------------------

                GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;
                int64_t pA_start = j * Avlen ;

                //--------------------------------------------------------------
                // get M(:,j)
                //--------------------------------------------------------------

                int64_t pM_start, pM_end ;
                GB_LOOKUP_VECTOR (pM_start, pM_end, M, j) ;
                bool mjdense = (pM_end - pM_start) == Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(iA_start:iA_end,j) and A(ditto,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {
                    int64_t pA = pA_start + iA ;
                    bool Sfound = (pS < pS_end) && (GBI (Si, pS, Svlen) == iA) ;
                    bool Afound = Ab [pA] ;
                    if (!Sfound && Afound)
                    {
                        // S (i,j) is not present, A (i,j) is present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                            GB_PENDING_INSERT_aij ;
                        }
                    }
                    else if (Sfound)
                    { 
                        // S (i,j) present
                        GB_NEXT (S) ;
                    }
                }
            }
            GB_PHASE2_TASK_WRAPUP ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // phase2: A is hypersparse, sparse, or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(&&:pending_sorted)
        for (taskid = 0 ; taskid < ntasks ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            GB_GET_TASK_DESCRIPTOR_PHASE2 ;

            //------------------------------------------------------------------
            // compute all vectors in this task
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // get A(:,j) and S(:,j)
                //--------------------------------------------------------------

                int64_t j = GBH (Zh, k) ;
                GB_GET_MAPPED (pA, pA_end, pA, pA_end, Ap, j, k, Z_to_X, Avlen);
                GB_GET_MAPPED (pS, pS_end, pB, pB_end, Sp, j, k, Z_to_S, Svlen);

                //--------------------------------------------------------------
                // get M(:,j)
                //--------------------------------------------------------------

                int64_t pM_start, pM_end ;
                GB_LOOKUP_VECTOR (pM_start, pM_end, M, j) ;
                bool mjdense = (pM_end - pM_start) == Mvlen ;

                //--------------------------------------------------------------
                // do a 2-way merge of S(:,j) and A(:,j)
                //--------------------------------------------------------------

                // jC = J [j] ; or J is a colon expression
                int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

                // while both list S (:,j) and A (:,j) have entries
                while (pS < pS_end && pA < pA_end)
                {
                    int64_t iS = GBI (Si, pS, Svlen) ;
                    int64_t iA = GBI (Ai, pA, Avlen) ;

                    if (iS < iA)
                    { 
                        // S (i,j) is present but A (i,j) is not
                        GB_NEXT (S) ;
                    }
                    else if (iA < iS)
                    {
                        // S (i,j) is not present, A (i,j) is present
                        GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                        if (Mask_comp) mij = !mij ;
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                            GB_PENDING_INSERT_aij ;
                        }
                        GB_NEXT (A) ;
                    }
                    else
                    { 
                        // both S (i,j) and A (i,j) present
                        GB_NEXT (S) ;
                        GB_NEXT (A) ;
                    }
                }

                // while list A (:,j) has entries.  List S (:,j) exhausted.
                while (pA < pA_end)
                {
                    // S (i,j) is not present, A (i,j) is present
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iA) ;
                    if (Mask_comp) mij = !mij ;
                    if (mij)
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                        GB_PENDING_INSERT_aij ;
                    }
                    GB_NEXT (A) ;
                }
            }

            GB_PHASE2_TASK_WRAPUP ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

