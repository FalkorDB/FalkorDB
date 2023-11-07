//------------------------------------------------------------------------------
// GB_masker_phase1: find # of entries in R = masker (C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// GB_masker_phase1 counts the number of entries in each vector of R, for R =
// masker (C,M,Z), and then does a cumulative sum to find Cp.  GB_masker_phase1
// is preceded by GB_add_phase0, which finds the non-empty vectors of R.  This
// phase is done entirely in parallel.

// R, M, C, and Z can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  All cases of the mask M are handled: present and not
// complemented, and present and complemented.  The mask is always present for
// R=masker(C,M,Z).

// Rp is either freed by phase2, or transplanted into R.

#include "GB_mask.h"
#include "GB_unused.h"

GrB_Info GB_masker_phase1           // count nnz in each R(:,j)
(
    // computed by phase1:
    int64_t **Rp_handle,            // output of size Rnvec+1
    size_t *Rp_size_handle,
    int64_t *Rnvec_nonempty,        // # of non-empty vectors in R
    // tasks from phase1a:
    GB_task_struct *restrict TaskList,       // array of structs
    const int R_ntasks,               // # of tasks
    const int R_nthreads,             // # of threads to use
    // analysis from phase0:
    const int64_t Rnvec,
    const int64_t *restrict Rh,
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    // original input:
    const GrB_Matrix M,             // required mask
    const bool Mask_comp,           // if true, then M is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix C,
    const GrB_Matrix Z,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Rp_handle != NULL) ;
    ASSERT (Rp_size_handle != NULL) ;
    ASSERT (Rnvec_nonempty != NULL) ;

    ASSERT_MATRIX_OK (M, "M for mask phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    ASSERT_MATRIX_OK (C, "C for mask phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ; 
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ; 

    ASSERT_MATRIX_OK (Z, "Z for mask phase1", GB0) ;
    ASSERT (!GB_ZOMBIES (Z)) ; 
    ASSERT (!GB_JUMBLED (Z)) ;
    ASSERT (!GB_PENDING (Z)) ; 

    ASSERT (!GB_IS_BITMAP (C)) ;    // not used if C is bitmap

    ASSERT (C->vdim == Z->vdim && C->vlen == Z->vlen) ;
    ASSERT (C->vdim == M->vdim && C->vlen == M->vlen) ;

    int64_t *restrict Rp = NULL ; size_t Rp_size = 0 ;
    (*Rp_handle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    Rp = GB_CALLOC (GB_IMAX (2, Rnvec+1), int64_t, &Rp_size) ;
    if (Rp == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of R
    //--------------------------------------------------------------------------

    #define GB_PHASE_1_OF_2
    #include "GB_masker_template.c"

    //--------------------------------------------------------------------------
    // cumulative sum of Rp and fine tasks in TaskList
    //--------------------------------------------------------------------------

    GB_task_cumsum (Rp, Rnvec, Rnvec_nonempty, TaskList, R_ntasks, R_nthreads,
        Werk) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Rp_handle) = Rp ;
    (*Rp_size_handle) = Rp_size ;
    return (GrB_SUCCESS) ;
}

