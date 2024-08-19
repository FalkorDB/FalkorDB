//------------------------------------------------------------------------------
// GB_subassign_02: C(I,J) = A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 02: C(I,J) = A ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           constructed

// C: not bitmap or full: use GB_bitmap_assign instead
// A: any sparsity structure.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

GrB_Info GB_subassign_02
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
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;
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

    GB_MATRIX_WAIT_IF_JUMBLED (A) ;

    GB_GET_C ;      // C must not be bitmap
    GB_GET_A ;
    GB_GET_S ;
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 02: C(I,J) = A ; using S
    //--------------------------------------------------------------------------

    // Time: Optimal.  All entries in A+S must be examined, so the work is
    // Omega (nnz(A)+nnz(S)).

    // Method 02 and Method 04 are somewhat similar.  They differ on how C is
    // modified when the entry is present in S but not A.

    // TODO: phase2 of Method 02 and 04 are identical and could be
    // done in a single function.

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
                // do a 2-way merge of S(iA_start:iA_end,j) and A(ditto,j)
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {
                    int64_t pA = pA_start + iA ;
                    bool Sfound = (pS < pS_end) && (GBI (Si, pS, Svlen) == iA) ;
                    bool Afound = Ab [pA] ;
                    if (Sfound && !Afound)
                    { 
                        // ----[C . 1] or [X . 1]-------------------------------
                        // S (i,j) is present but A (i,j) is not
                        // [C . 1]: action: ( delete ): becomes zombie
                        // [X . 1]: action: ( X ): still a zombie
                        GB_C_S_LOOKUP ;
                        GB_DELETE_ENTRY ;
                        GB_NEXT (S) ;
                    }
                    else if (!Sfound && Afound)
                    { 
                        // ----[. A 1]------------------------------------------
                        // S (i,j) is not present, A (i,j) is present
                        // [. A 1]: action: ( insert )
                        task_pending++ ;
                    }
                    else if (Sfound && Afound)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // both S (i,j) and A (i,j) present
                        // [C A 1]: action: ( =A ): copy A into C, no accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_C_S_LOOKUP ;
                        GB_noaccum_C_A_1_matrix ;
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
                        // ----[C . 1] or [X . 1]-------------------------------
                        // S (i,j) is present but A (i,j) is not
                        // [C . 1]: action: ( delete ): becomes zombie
                        // [X . 1]: action: ( X ): still a zombie
                        GB_C_S_LOOKUP ;
                        GB_DELETE_ENTRY ;
                        GB_NEXT (S) ;
                    }
                    else if (iA < iS)
                    { 
                        // ----[. A 1]------------------------------------------
                        // S (i,j) is not present, A (i,j) is present
                        // [. A 1]: action: ( insert )
                        task_pending++ ;
                        GB_NEXT (A) ;
                    }
                    else
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // both S (i,j) and A (i,j) present
                        // [C A 1]: action: ( =A ): copy A into C, no accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_C_S_LOOKUP ;
                        GB_noaccum_C_A_1_matrix ;
                        GB_NEXT (S) ;
                        GB_NEXT (A) ;
                    }
                }

                // while list S (:,j) has entries.  List A (:,j) exhausted.
                while (pS < pS_end)
                { 
                    // ----[C . 1] or [X . 1]-----------------------------------
                    // S (i,j) is present but A (i,j) is not
                    // [C . 1]: action: ( delete ): becomes zombie
                    // [X . 1]: action: ( X ): still a zombie
                    GB_C_S_LOOKUP ;
                    GB_DELETE_ENTRY ;
                    GB_NEXT (S) ;
                }

                // List A (:,j) has entries.  List S (:,j) exhausted.
                task_pending += (pA_end - pA) ;
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
                        // ----[. A 1]------------------------------------------
                        // S (i,j) is not present, A (i,j) is present
                        // [. A 1]: action: ( insert )
                        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                        GB_PENDING_INSERT_aij ;
                        GB_NEXT (A) ;
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
                        GB_NEXT (S) ;
                    }
                    else if (iA < iS)
                    { 
                        // ----[. A 1]------------------------------------------
                        // S (i,j) is not present, A (i,j) is present
                        // [. A 1]: action: ( insert )
                        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                        GB_PENDING_INSERT_aij ;
                        GB_NEXT (A) ;
                    }
                    else
                    { 
                        GB_NEXT (S) ;
                        GB_NEXT (A) ;
                    }
                }

                // ignore the remainder of S (:,j)

                // while list A (:,j) has entries.  List S (:,j) exhausted.
                while (pA < pA_end)
                { 
                    // ----[. A 1]----------------------------------------------
                    // S (i,j) is not present, A (i,j) is present
                    // [. A 1]: action: ( insert )
                    int64_t iA = GBI (Ai, pA, Avlen) ;
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    GB_PENDING_INSERT_aij ;
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

