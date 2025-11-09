//------------------------------------------------------------------------------
// GB_masker_phase1: find # of entries in R = masker (C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_masker_phase1 counts the number of entries in each vector of R, for R =
// masker (C,M,Z), and then does a cumulative sum to find Cp.  GB_masker_phase1
// is preceded by GB_add_phase0, which finds the non-empty vectors of R.  This
// phase is done entirely in parallel.

// R can be sparse or hypersparse, as determined by GB_add_phase0.  M, C, and Z
// can have any sparsity format.  All cases of the mask M are handled: present
// and not complemented, and present and complemented.  The mask is always
// present for R=masker(C,M,Z).

// Rp is either freed by phase2, or transplanted into R.

#include "mask/GB_mask.h"
#include "jitifyer/GB_stringify.h"
#include "include/GB_masker_shared_definitions.h"

#define GB_FREE_ALL GB_FREE_MEMORY (&Rp, Rp_size) ;

GrB_Info GB_masker_phase1           // count nnz in each R(:,j)
(
    // computed by phase1:
    void **Rp_handle,               // vector pointers for R
    size_t *Rp_size_handle,
    int64_t *Rnvec_nonempty,        // # of non-empty vectors in R
    // tasks from phase1a:
    GB_task_struct *restrict TaskList,       // array of structs
    const int R_ntasks,               // # of tasks
    const int R_nthreads,             // # of threads to use
    // analysis from phase0:
    const int64_t Rnvec,
    const void *Rh,
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    const bool Rp_is_32,
    const bool Rj_is_32,
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

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    (*Rp_handle) = NULL ;
    void *Rp = NULL ; size_t Rp_size = 0 ;
    size_t rpsize = (Rp_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    Rp = GB_CALLOC_MEMORY (GB_IMAX (2, Rnvec+1), rpsize, &Rp_size) ;
    if (Rp == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of R
    //--------------------------------------------------------------------------

    // via the JIT kernel
    GrB_Info info = GB_masker_phase1_jit (
        // computed by phase1:
        Rp,                         // output of size Rnvec+1
        Rnvec_nonempty,             // # of non-empty vectors in R
        // tasks from phase1a:
        TaskList,                   // array of structs
        R_ntasks,                   // # of tasks
        R_nthreads,                 // # of threads to use
        // analysis from phase0:
        Rnvec, Rh, R_to_M, R_to_C, R_to_Z, Rp_is_32, Rj_is_32,
        // original input:
        M, Mask_comp, Mask_struct, C, Z) ;

    if (info == GrB_NO_VALUE)
    { 
        // via the generic kernel
        GBURBLE ("(generic masker) ") ;
        #define GB_PHASE_1_OF_2
        #include "mask/template/GB_masker_template.c"
        info = GrB_SUCCESS ;
    }

    GB_OK (info) ;

    //--------------------------------------------------------------------------
    // cumulative sum of Rp and fine tasks in TaskList
    //--------------------------------------------------------------------------

    GB_task_cumsum (Rp, Rp_is_32, Rnvec, Rnvec_nonempty, TaskList, R_ntasks,
        R_nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Rp_handle) = Rp ;
    (*Rp_size_handle) = Rp_size ;
    return (GrB_SUCCESS) ;
}

