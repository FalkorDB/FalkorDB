//------------------------------------------------------------------------------
// GB_add_phase1: # entries in C=A+B, C<M or !M>=A+B (C is sparse/hyper)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_add_phase1 counts the number of entries in each vector of C, for C=A+B,
// C<M>=A+B, or C<!M>=A+B and then does a cumulative sum to find Cp.
// GB_add_phase1 is preceded by GB_add_phase0, which finds the non-empty
// vectors of C.  If the mask M is sparse, it is not complemented; only a
// bitmap or full M is complemented.

// C is sparse or hypersparse, as determined by GB_add_sparsity.  M, A, and B
// can have any sparsity structure, but only a specific set of cases will be
// used (see the list in Template/GB_add_sparse_template.c).

// Cp is constructed here, and either freed by phase2, or transplanted into C.

#include "GB_add.h"
#include "GB_unused.h"

GrB_Info GB_add_phase1                  // count nnz in each C(:,j)
(
    int64_t **Cp_handle,                // output of size Cnvec+1
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint
    // tasks from phase0b:
    GB_task_struct *restrict TaskList,   // array of structs
    const int C_ntasks,                 // # of tasks
    const int C_nthreads,               // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    const int64_t *restrict Ch,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,                // if true, then Ch == M->h
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Cnvec_nonempty != NULL) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for add phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    ASSERT_MATRIX_OK (A, "A for add phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT_MATRIX_OK (B, "B for add phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (A->vdim == B->vdim) ;

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    (*Cp_handle) = NULL ;
    int64_t *restrict Cp = NULL ; size_t Cp_size = 0 ;
    Cp = GB_CALLOC (GB_IMAX (2, Cnvec+1), int64_t, &Cp_size) ;
    if (Cp == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of C
    //--------------------------------------------------------------------------

    // for the "easy mask" condition:
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_B = GB_all_aliased (M, B) ;

    #define GB_ADD_PHASE 1
    #include "GB_add_template.c"

    //--------------------------------------------------------------------------
    // cumulative sum of Cp and fine tasks in TaskList
    //--------------------------------------------------------------------------

    GB_task_cumsum (Cp, Cnvec, Cnvec_nonempty, TaskList, C_ntasks, C_nthreads,
        Werk) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Cp_handle) = Cp ;
    (*Cp_size_handle) = Cp_size ;
    return (GrB_SUCCESS) ;
}

