//------------------------------------------------------------------------------
// GB_subassign_08n_slice: slice the entries and vectors for GB_subassign_08n
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

#include "assign/GB_subassign_methods.h"
#include "emult/GB_emult.h"
// Npending is set to NULL by the GB_EMPTY_TASKLIST macro, but unused here.
#include "include/GB_unused.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#if 0
GrB_Info GX_subassign_08n_slice                                             \
(                                                                           \
    /* output: */                                                           \
    GB_task_struct **p_TaskList,    /* size max_ntasks */                   \
    size_t *p_TaskList_size,        /* size of TaskList */                  \
    int *p_ntasks,                  /* # of tasks constructed */            \
    int *p_nthreads,                /* # of threads to use */               \
    int64_t *p_Znvec,               /* # of vectors to compute in Z */      \
    const void **Zh_handle,         /* Zh is A->h, M->h, or NULL */         \
    int64_t **Z_to_A_handle,        /* Z_to_A: size Znvec, or NULL */       \
    size_t *Z_to_A_size_handle,                                             \
    int64_t **Z_to_M_handle,        /* Z_to_M: size Znvec, or NULL */       \
    size_t *Z_to_M_size_handle,                                             \
    bool *Zj_is_32_handle,                                                  \
    /* input: */                                                            \
    const GrB_Matrix C,         /* output matrix C */                       \
    const void *I,              /* I index list */                          \
    const bool I_is_32,                                                     \
    const int64_t nI,                                                       \
    const int Ikind,                                                        \
    const int64_t Icolon [3],                                               \
    const void *J,              /* J index list */                          \
    const bool J_is_32,                                                     \
    const int64_t nJ,                                                       \
    const int Jkind,                                                        \
    const int64_t Jcolon [3],                                               \
    const GrB_Matrix A,         /* matrix to slice */                       \
    const GrB_Matrix M,         /* matrix to slice */                       \
    GB_Werk Werk                                                            \
)
#endif

GB_CALLBACK_SUBASSIGN_08N_SLICE_PROTO (GB_subassign_08n_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix S = NULL ;           // not constructed
    GB_EMPTY_TASKLIST

    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_BITMAP (M)) ;    // Method 08n is not used for M bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;    // Method 08n is not used for A bitmap

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

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    void *Ch = C->h ;
    void *Ci = C->i ;
    const bool may_see_zombies = (C->nzombies > 0) ;
    const int64_t Cnvec = C->nvec ;
    const int64_t Cvlen = C->vlen ;
    const bool C_is_hyper = (Ch != NULL) ;
    const bool Cp_is_32 = C->p_is_32 ;
    const bool Cj_is_32 = C->j_is_32 ;
    const bool Ci_is_32 = C->i_is_32 ;
    GB_GET_C_HYPER_HASH ;

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    void *Mh = M->h ;
    const int64_t Mvlen = M->vlen ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    void *Ah = A->h ;
    const int64_t Avlen = A->vlen ;

    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;
    GB_IDECL (J, const, u) ; GB_IPTR (J, J_is_32) ;

    //--------------------------------------------------------------------------
    // construct fine/coarse tasks for eWise multiply of A.*M
    //--------------------------------------------------------------------------

    // Compare with the first part of GB_emult for A.*B.  Note that M in this
    // function takes the place of B in GB_emult.

    int64_t Znvec ;
    GB_MDECL (Zh_shallow, const, u) ;

    bool Zp_is_32, Zj_is_32, Zi_is_32 ;

    int Z_sparsity = GxB_SPARSE ;
    GB_OK (GB_emult_08_phase0 (&Znvec, &Zh_shallow, &Zh_size, NULL, NULL,
        &Z_to_A, &Z_to_A_size, &Z_to_M, &Z_to_M_size,
        &Zp_is_32, &Zj_is_32, &Zi_is_32,
        &Z_sparsity, NULL, false, A, M, Werk)) ;

    // Z is still sparse or hypersparse, not bitmap or full
    ASSERT (Z_sparsity == GxB_SPARSE || Z_sparsity == GxB_HYPERSPARSE) ;

    GB_OK (GB_ewise_slice (
        &TaskList, &TaskList_size, &ntasks, &nthreads,
        Znvec, Zh_shallow, Zj_is_32, NULL, Z_to_A, Z_to_M, false,
        NULL, A, M, Werk)) ;

    GB_IPTR (Zh_shallow, Zj_is_32) ;

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
            int64_t j = GBh (Zh_shallow, k) ;

            // A fine task operates on a slice of A(:,k)
            int64_t pA     = TaskList [taskid].pA ;
            int64_t pA_end = TaskList [taskid].pA_end ;

            // A fine task operates on a slice of M(:,k)
            int64_t pM     = TaskList [taskid].pB ;
            int64_t pM_end = TaskList [taskid].pB_end ;

            //------------------------------------------------------------------
            // quick checks for empty intersection of A(:,j) and M(:,j)
            //------------------------------------------------------------------

            int64_t ajnz = pA_end - pA ;
            int64_t mjnz = pM_end - pM ;
            if (ajnz == 0 || mjnz == 0) continue ;
            int64_t iA_first = GBi_A (Ai, pA, Avlen) ;
            int64_t iA_last  = GBi_A (Ai, pA_end-1, Avlen) ;
            int64_t iM_first = GBi_M (Mi, pM, Mvlen) ;
            int64_t iM_last  = GBi_M (Mi, pM_end-1, Mvlen) ;
            if (iA_last < iM_first || iM_last < iA_first) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            // lookup jC in C
            // jC = J [j] ; or J is ":" or jbegin:jend or jbegin:jinc:jend
            int64_t jC = GB_IJLIST (J, j, Jkind, Jcolon) ;
            int64_t pC_start, pC_end ;
            GB_LOOKUP_VECTOR_C (jC, pC_start, pC_end) ;

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
                int64_t iC1 = GB_IJLIST (I, iA_start, Ikind, Icolon) ;
                int64_t iA_end = GB_IMAX (iA_last, iM_last) ;
                int64_t iC2 = GB_IJLIST (I, iA_end, Ikind, Icolon) ;

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
                GB_split_binary_search_zombie (iC_start, Ci, Ci_is_32,
                    &pleft, &pright, may_see_zombies, &is_zombie) ;
                TaskList [taskid].pC = pleft ;

                pleft = pC_start ;
                pright = pC_end - 1 ;
                found = GB_split_binary_search_zombie (iC_end, Ci, Ci_is_32,
                    &pleft, &pright, may_see_zombies, &is_zombie) ;
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
    (*Zj_is_32_handle) = Zj_is_32 ;

    return (GrB_SUCCESS) ;
}

