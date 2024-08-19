//------------------------------------------------------------------------------
// GB_subassign_19: C(I,J)<!M,repl> += scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 19: C(I,J)<!M,repl> += scalar ; using S

// M:           present
// Mask_comp:   true
// C_replace:   true
// accum:       present
// A:           scalar
// S:           constructed

// C: not bitmap
// M: not bitmap

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

GrB_Info GB_subassign_19
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
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ; ASSERT (!GB_IS_FULL (C)) ;
    ASSERT (!GB_any_aliased (C, M)) ;   // NO ALIAS of C==M

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

    GB_GET_C ;      // C must not be bitmap
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    const int64_t Cnvec = C->nvec ;
    GB_GET_MASK ;
    GB_GET_MASK_HYPER_HASH ;
    GB_GET_S ;
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 19: C(I,J)<!M,repl> += scalar ; using S
    //--------------------------------------------------------------------------

    // Time: Close to optimal; must visit all IxJ, so Omega(|I|*|J|) is
    // required.  The sparsity of !M cannot be exploited.

    // Methods 13, 15, 17, and 19 are very similar.

    //--------------------------------------------------------------------------
    // Parallel: all IxJ (Methods 01, 03, 13, 15, 17, 19)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_IXJ_SLICE ;

    //--------------------------------------------------------------------------
    // phase 1: create zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_IXJ_TASK_DESCRIPTOR_PHASE1 (iA_start, iA_end) ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

            //------------------------------------------------------------------
            // get S(iA_start:end,j) and M(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;
            GB_LOOKUP_VECTOR_FOR_IXJ (M, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M,repl> += scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {

                //--------------------------------------------------------------
                // Get the indices at the top of each list.
                //--------------------------------------------------------------

                int64_t iS = (pS < pS_end) ? GBI (Si, pS, Svlen) : INT64_MAX ;
                int64_t iM = (pM < pM_end) ? GBI (Mi, pM, Mvlen) : INT64_MAX ;

                //--------------------------------------------------------------
                // find the smallest index of [iS iA iM] (always iA)
                //--------------------------------------------------------------

                int64_t i = iA ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                if (i == iM)
                { 
                    // mij = (bool) M [pM]
                    mij = GBB (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    GB_NEXT (M) ;
                }
                else
                { 
                    // mij not present, implicitly false
                    ASSERT (i < iM) ;
                    mij = false ;
                }

                // complement the mask entry mij since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // accumulate the entry
                //--------------------------------------------------------------

                if (i == iS)
                {
                    ASSERT (i == iA) ;
                    {
                        // both S (i,j) and A (i,j) present
                        GB_C_S_LOOKUP ;
                        if (mij)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =C+A ): apply accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_withaccum_C_A_1_scalar ;
                        }
                        else
                        { 
                            // ----[C A 0] or [X A 0]---------------------------
                            // [X A 0]: action: ( X ): still a zombie
                            // [C A 0]: C_repl: action: ( delete ): zombie
                            GB_DELETE_ENTRY ;
                        }
                        GB_NEXT (S) ;
                    }
                }
                else
                {
                    ASSERT (i == iA) ;
                    {
                        // S (i,j) is not present, A (i,j) is present
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            task_pending++ ;
                        }
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

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(&&:pending_sorted)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_IXJ_TASK_DESCRIPTOR_PHASE2 (iA_start, iA_end) ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

            //------------------------------------------------------------------
            // get S(iA_start:end,j) and M(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;
            GB_LOOKUP_VECTOR_FOR_IXJ (M, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M,repl> += scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {

                //--------------------------------------------------------------
                // Get the indices at the top of each list.
                //--------------------------------------------------------------

                int64_t iS = (pS < pS_end) ? GBI (Si, pS, Svlen) : INT64_MAX ;
                int64_t iM = (pM < pM_end) ? GBI (Mi, pM, Mvlen) : INT64_MAX ;

                //--------------------------------------------------------------
                // find the smallest index of [iS iA iM] (always iA)
                //--------------------------------------------------------------

                int64_t i = iA ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                if (i == iM)
                { 
                    // mij = (bool) M [pM]
                    mij = GBB (Mb, pM) && GB_MCAST (Mx, pM, msize) ;
                    GB_NEXT (M) ;
                }
                else
                { 
                    // mij not present, implicitly false
                    ASSERT (i < iM) ;
                    mij = false ;
                }

                // complement the mask entry mij since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // accumulate the entry
                //--------------------------------------------------------------

                if (i == iS)
                {
                    ASSERT (i == iA) ;
                    { 
                        GB_NEXT (S) ;
                    }
                }
                else
                {
                    ASSERT (i == iA) ;
                    {
                        // S (i,j) is not present, A (i,j) is present
                        if (mij)
                        { 
                            // ----[. A 1]--------------------------------------
                            // [. A 1]: action: ( insert )
                            int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                            GB_PENDING_INSERT (scalar) ;
                        }
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

