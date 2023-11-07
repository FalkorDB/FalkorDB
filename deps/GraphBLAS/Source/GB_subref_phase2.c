//------------------------------------------------------------------------------
// GB_subref_phase2: find # of entries in C=A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed, but GB_subref_method has 6 or 7 variants.

// GB_subref_phase2 counts the number of entries in each vector of C, for
// C=A(I,J) and then does a cumulative sum to find Cp.

// Cp is either freed by phase2, or transplanted into C.

#include "GB_subref.h"

GrB_Info GB_subref_phase2               // count nnz in each C(:,j)
(
    // computed by phase2:
    int64_t **Cp_handle,                // output of size Cnvec+1
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    // tasks from phase1:
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads to use
    const int64_t *Mark,                // for I inverse buckets, size A->vlen
    const int64_t *Inext,               // for I inverse buckets, size nI
    const int64_t nduplicates,          // # of duplicates, if I inverted
    // analysis from phase0:
    const int64_t *restrict Ap_start,
    const int64_t *restrict Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    // original input:
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const bool symbolic,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Cp_size_handle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for subref phase2", GB0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;    // GB_bitmap_subref is used instead

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    (*Cp_handle) = NULL ;
    (*Cp_size_handle) = 0 ;
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

    #define GB_ANALYSIS_PHASE
    if (symbolic)
    { 
        #define GB_SYMBOLIC
        // symbolic extraction must handle zombies
        const int64_t nzombies = A->nzombies ;
        #include "GB_subref_template.c"
    }
    else
    { 
        // iso and non-iso numeric extraction do not see zombies
        ASSERT (!GB_ZOMBIES (A)) ;
        #include "GB_subref_template.c"
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Cp and fine tasks in TaskList
    //--------------------------------------------------------------------------

    GB_task_cumsum (Cp, Cnvec, Cnvec_nonempty, TaskList, ntasks, nthreads,
        Werk) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Cp_handle) = Cp ;
    (*Cp_size_handle) = Cp_size ;
    return (GrB_SUCCESS) ;
}

