//------------------------------------------------------------------------------
// GB_subassign_08n_slice: slice the entries and vectors for GB_subassign_08n
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs a set of tasks to compute C for GB_subassign_08n, based on
// slicing two input matrices (A and M).  Fine tasks must also find their
// location in their vector C(:,jC).

// This method is used only by GB_subassign_08n.  New zombies cannot be
// created, since no entries are deleted.  Old zombies can be brought back to
// life, however.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   +   A   -       08n:  C(I,J)<M> += A, no S

// C, M, A: not bitmap.  C can be full.

// If C is bitmap, then GB_bitmap_assign_M_accum is used instead.
// If M or A are bitmap, but C is sparse or hyper, then Method 08s is used
// instead (which handles both M and A as bitmap).  As a result, this method
// does not need to consider the bitmap case for C, M, or A.

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"
#include "GB_emult.h"
// Npending is set to NULL by the GB_EMPTY_TASKLIST macro, but unused here.
#include "GB_unused.h"

GrB_Info GB_subassign_08n_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs, of size max_ntasks
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    int64_t *p_Znvec,               // # of vectors to compute in Z
    const int64_t *restrict *Zh_handle,  // Zh_shallow is A->h, M->h, or NULL
    int64_t *restrict *Z_to_A_handle,    // Z_to_A: size Znvec, or NULL
    size_t *Z_to_A_size_handle,
    int64_t *restrict *Z_to_M_handle,    // Z_to_M: size Znvec, or NULL
    size_t *Z_to_M_size_handle,
    // input:
    const GrB_Matrix C,             // output matrix C
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix A,             // matrix to slice
    const GrB_Matrix M,             // matrix to slice
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (M)) ;    // Method 08n is not used for M bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;    // Method 08n is not used for A bitmap

    GB_EMPTY_TASKLIST
    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;
    ASSERT_MATRIX_OK (C, "C for 08n_slice", GB0) ;
    ASSERT_MATRIX_OK (M, "M for 08n_slice", GB0) ;
    ASSERT_MATRIX_OK (A, "A for 08n_slice", GB0) ;

    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_JUMBLED (A)) ;

    ASSERT (p_Znvec != NULL) ;
    ASSERT (Zh_handle != NULL) ;
    ASSERT (Z_to_A_handle != NULL) ;
    ASSERT (Z_to_M_handle != NULL) ;

    (*p_TaskList  ) = NULL ;
    (*p_TaskList_size) = 0 ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;

    (*p_Znvec      ) = 0 ;
    (*Zh_handle    ) = NULL ;
    (*Z_to_A_handle) = NULL ;
    (*Z_to_M_handle) = NULL ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int64_t *restrict Ci = C->i ;
    int64_t nzombies = C->nzombies ;
    const int64_t Cnvec = C->nvec ;
    const int64_t Cvlen = C->vlen ;
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Cp = C->p ;
    const bool C_is_hyper = (Ch != NULL) ;
    GB_GET_C_HYPER_HASH ;

    const int64_t *restrict Mp = M->p ;
    const int64_t *restrict Mh = M->h ;
    const int64_t *restrict Mi = M->i ;
    const int64_t Mvlen = M->vlen ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t Avlen = A->vlen ;

    //--------------------------------------------------------------------------
    // construct fine/coarse tasks for eWise multiply of A.*M
    //--------------------------------------------------------------------------

    // Compare with the first part of GB_emult for A.*B.  Note that M in this
    // function takes the place of B in GB_emult.

    int64_t Znvec ;
    const int64_t *restrict Zh_shallow = NULL ;
    int Z_sparsity = GxB_SPARSE ;
    GB_OK (GB_emult_08_phase0 (&Znvec, &Zh_shallow, &Zh_size, NULL, NULL,
        &Z_to_A, &Z_to_A_size, &Z_to_M, &Z_to_M_size, &Z_sparsity, NULL, A, M,
        Werk)) ;

    // Z is still sparse or hypersparse, not bitmap or full
    ASSERT (Z_sparsity == GxB_SPARSE || Z_sparsity == GxB_HYPERSPARSE) ;

    GB_OK (GB_ewise_slice (
        &TaskList, &TaskList_size, &ntasks, &nthreads,
        Znvec, Zh_shallow, NULL, Z_to_A, Z_to_M, false,
        NULL, A, M, Werk)) ;

    //--------------------------------------------------------------------------
    // slice C(:,jC) for each fine task
    //--------------------------------------------------------------------------

    // Each fine task that operates on C(:,jC) must be limited to just its
    // portion of C(:,jC).  Otherwise, one task could bring a zombie to life,
    // at the same time another is attempting to do a binary search on that
    // entry.  This is safe as long as a 64-bit integer read/write is always
    // atomic, but there is no gaurantee that this is true for all
    // architectures.  Note that GB_subassign_08n cannot create new zombies.

    // This work could be done in parallel, but each task does at most 2 binary
    // searches.  The total work for all the binary searches will likely be
    // small.  So do the work with a single thread.

    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR ;

        //----------------------------------------------------------------------
        // do the binary search for this fine task
        //----------------------------------------------------------------------

        if (fine_task)
        {

            //------------------------------------------------------------------
            // get A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t k = kfirst ;
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

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_LOOKUP_VECTOR_jC (false, 0) ;
            bool cjdense = (pC_end - pC_start == Cvlen) ;

            //------------------------------------------------------------------
            // slice C(:,jC) for this fine task
            //------------------------------------------------------------------

            if (cjdense)
            { 
                // do not slice C(:,jC) if it is dense
                TaskList [taskid].pC     = pC_start ;
                TaskList [taskid].pC_end = pC_end ;
            }
            else
            { 
                // find where this task starts and ends in C(:,jC)
                int64_t iA_start = GB_IMIN (iA_first, iM_first) ;
                int64_t iC1 = GB_ijlist (I, iA_start, Ikind, Icolon) ;
                int64_t iA_end = GB_IMAX (iA_last, iM_last) ;
                int64_t iC2 = GB_ijlist (I, iA_end, Ikind, Icolon) ;

                // If I is an explicit list, it must be already sorted
                // in ascending order, and thus iC1 <= iC2.  If I is
                // GB_ALL or GB_STRIDE with inc >= 0, then iC1 < iC2.
                // But if inc < 0, then iC1 > iC2.  iC_start and iC_end
                // are used for a binary search bracket, so iC_start <=
                // iC_end must hold.
                int64_t iC_start = GB_IMIN (iC1, iC2) ;
                int64_t iC_end   = GB_IMAX (iC1, iC2) ;

                // this task works on Ci,Cx [pC:pC_end-1]
                int64_t pleft = pC_start ;
                int64_t pright = pC_end - 1 ;
                bool found, is_zombie ;
                GB_SPLIT_BINARY_SEARCH_ZOMBIE (iC_start, Ci, pleft, pright,
                    found, nzombies, is_zombie) ;
                TaskList [taskid].pC = pleft ;

                pleft = pC_start ;
                pright = pC_end - 1 ;
                GB_SPLIT_BINARY_SEARCH_ZOMBIE (iC_end, Ci, pleft, pright,
                    found, nzombies, is_zombie) ;
                TaskList [taskid].pC_end = (found) ? (pleft+1) : pleft ;
            }

            ASSERT (TaskList [taskid].pC <= TaskList [taskid].pC_end) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_TaskList  ) = TaskList ;
    (*p_TaskList_size) = TaskList_size ;
    (*p_ntasks    ) = ntasks ;
    (*p_nthreads  ) = nthreads ;

    (*p_Znvec      ) = Znvec ;
    (*Zh_handle    ) = Zh_shallow ;
    (*Z_to_A_handle) = Z_to_A ; (*Z_to_A_size_handle) = Z_to_A_size ;
    (*Z_to_M_handle) = Z_to_M ; (*Z_to_M_size_handle) = Z_to_M_size ;

    return (GrB_SUCCESS) ;
}

