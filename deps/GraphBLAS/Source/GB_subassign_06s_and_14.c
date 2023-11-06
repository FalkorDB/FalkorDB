//------------------------------------------------------------------------------
// GB_subassign_06s_and_14: C(I,J)<M or !M> = A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 06s: C(I,J)<M> = A ; using S
// Method 14:  C(I,J)<!M> = A ; using S

// M:           present
// Mask_comp:   true or false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           constructed

// C: not bitmap or full: use GB_bitmap_assign instead
// M, A: any sparsity structure.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

GrB_Info GB_subassign_06s_and_14
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

    ASSERT_MATRIX_OK (C, "C input for Method 06s/14", GB0) ;
    ASSERT_MATRIX_OK (M, "M input for Method 06s/14", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for Method 06s/14", GB0) ;
    ASSERT_MATRIX_OK (S, "S constructed for Method 06s/14", GB0) ;

    GB_GET_C ;      // C must not be bitmap
    GB_GET_MASK ;
    GB_GET_MASK_HYPER_HASH ;
    GB_GET_A ;
    GB_GET_S ;
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 06s: C(I,J)<M> = A ; using S
    //--------------------------------------------------------------------------

    // Time: O((nnz(A)+nnz(S))*log(m)) where m is the # of entries in a vector
    // of M, not including the time to construct S=C(I,J).  If A, S, and M
    // are similar in sparsity, then this method can perform well.  If M is
    // very sparse, Method 06n should be used instead.  Method 06s is selected
    // if nnz (A) < nnz (M) or if M is bitmap.

    //--------------------------------------------------------------------------
    // Method 14: C(I,J)<!M> = A ; using S
    //--------------------------------------------------------------------------

    // Time: Close to optimal.  Omega(nnz(S)+nnz(A)) is required, and the
    // sparsity of !M cannot be exploited.  The time taken is
    // O((nnz(A)+nnz(S))*log(m)) where m is the # of entries in a vector of M.

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
        // phase1: A is bitmap TODO: this is SLOW! for method 06s
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
                        if (mij)
                        { 
                            // ----[C . 1] or [X . 1]---------------------------
                            // [C . 1]: action: ( delete ): becomes zombie
                            // [X . 1]: action: ( X ): still zombie
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
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): A to C no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_C_S_LOOKUP ;
                            GB_noaccum_C_A_1_matrix ;
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
                        if (mij)
                        { 
                            // ----[C . 1] or [X . 1]---------------------------
                            // [C . 1]: action: ( delete ): becomes zombie
                            // [X . 1]: action: ( X ): still zombie
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
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =A ): A to C no accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_C_S_LOOKUP ;
                            GB_noaccum_C_A_1_matrix ;
                        }
                        GB_NEXT (S) ;
                        GB_NEXT (A) ;
                    }
                }

                // while list S (:,j) has entries.  List A (:,j) exhausted.
                while (pS < pS_end)
                {
                    // S (i,j) is present but A (i,j) is not
                    int64_t iS = GBI (Si, pS, Svlen) ;
                    GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP (iS) ;
                    if (Mask_comp) mij = !mij ;
                    if (mij)
                    { 
                        // ----[C . 1] or [X . 1]-------------------------------
                        // [C . 1]: action: ( delete ): becomes zombie
                        // [X . 1]: action: ( X ): still zombie
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

