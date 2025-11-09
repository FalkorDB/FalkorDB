//------------------------------------------------------------------------------
// GB_mask: definitions for GB_mask and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MASK_H
#define GB_MASK_H
#include "GB.h"
#include "math/GB_math.h"

GrB_Info GB_mask                // C<M> = Z
(
    GrB_Matrix C_result,        // both input C and result matrix
    const GrB_Matrix M,         // optional mask matrix, can be NULL
    GrB_Matrix *Zhandle,        // Z = results of computation, perhaps shallow.
                                // Z is freed when done.
    const bool C_replace,       // true if clear(C) to be done first
    const bool Mask_comp,       // true if M is to be complemented
    const bool Mask_struct,     // if true, use the only structure of M
    GB_Werk Werk
) ;

GrB_Info GB_masker          // R = masker (C, M, Z)
(
    GrB_Matrix R,           // output matrix, static header
    const bool R_is_csc,    // format of output matrix R
    const GrB_Matrix M,     // required input mask
    const bool Mask_comp,   // descriptor for M
    const bool Mask_struct, // if true, use the only structure of M
    const GrB_Matrix C,     // input C matrix
    const GrB_Matrix Z,     // input Z matrix
    GB_Werk Werk
) ;

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
) ;

GrB_Info GB_masker_phase2           // phase2 for R = masker (C,M,Z)
(
    GrB_Matrix R,                   // output matrix, static header
    const bool R_is_csc,            // format of output matrix R
    // from phase1:
    void **Rp_handle,               // vector pointers for R
    size_t Rp_size,
    const int64_t Rnvec_nonempty,   // # of non-empty vectors in R
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,     // array of structs
    const int R_ntasks,             // # of tasks
    const int R_nthreads,           // # of threads to use
    // analysis from phase0:
    const int64_t Rnvec,
    void **Rh_handle,               // R->h hyperlist
    size_t Rh_size,
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    const bool Rp_is_32,
    const bool Rj_is_32,
    const bool Ri_is_32,
    const int R_sparsity,
    // original input:
    const GrB_Matrix M,             // required mask
    const bool Mask_comp,           // if true, then M is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix C,
    const GrB_Matrix Z,
    GB_Werk Werk
) ;

int GB_masker_sparsity      // return the sparsity structure for R
(
    // input:
    const GrB_Matrix C,     // input C matrix
    const GrB_Matrix M,     // mask for C, always present
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix Z      // input Z matrix
) ;

#endif

