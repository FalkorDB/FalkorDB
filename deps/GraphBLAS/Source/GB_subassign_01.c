//------------------------------------------------------------------------------
// GB_subassign_01: C(I,J) = scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 01: C(I,J) = scalar ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           constructed

// C: not bitmap

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

GrB_Info GB_subassign_01
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
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ;

    //--------------------------------------------------------------------------
    // S = C(I,J)
    //--------------------------------------------------------------------------

    GB_EMPTY_TASKLIST ;
    GB_CLEAR_STATIC_HEADER (S, &S_header) ;
    GB_OK (GB_subassign_symbolic (S, C, I, ni, J, nj, true, Werk)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;      // C must not be bitmap
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    const int64_t Cnvec = C->nvec ;
    GB_GET_SCALAR ;
    GB_GET_S ;
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 01: C(I,J) = scalar ; using S
    //--------------------------------------------------------------------------

    // Time: Optimal; must visit all IxJ, so Omega(|I|*|J|) is required.

    // Entries in S are found and the corresponding entry in C replaced with
    // the scalar.  The traversal of S is identical to the traversal of M in
    // Method 4.

    // Method 01 and Method 03 are very similar.

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
            // get S(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC) = scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {
                bool found = (pS < pS_end) && (GBI (Si, pS, Svlen) == iA) ;
                if (!found)
                { 
                    // ----[. A 1]----------------------------------------------
                    // S (i,j) is not present, the scalar is present
                    // [. A 1]: action: ( insert )
                    task_pending++ ;
                }
                else
                { 
                    // ----[C A 1] or [X A 1]-----------------------------------
                    // both S (i,j) and A (i,j) present
                    // [C A 1]: action: ( =A ): scalar to C, no accum
                    // [X A 1]: action: ( undelete ): zombie lives
                    GB_C_S_LOOKUP ;
                    GB_noaccum_C_A_1_scalar ;
                    GB_NEXT (S) ;
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
            // get S(iA_start:end,j)
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_FOR_IXJ (S, iA_start) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC) = scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {
                bool found = (pS < pS_end) && (GBI (Si, pS, Svlen) == iA) ;
                if (!found)
                { 
                    // ----[. A 1]----------------------------------------------
                    // S (i,j) is not present, the scalar is present
                    // [. A 1]: action: ( insert )
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    GB_PENDING_INSERT (scalar) ;
                }
                else
                { 
                    // both S (i,j) and A (i,j) present
                    GB_NEXT (S) ;
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

