//------------------------------------------------------------------------------
// GB_subref.h: definitions for GB_subref_* functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBREF_H
#define GB_SUBREF_H
#include "ij/GB_ij.h"
#include "extract/include/GB_subref_method.h"

GrB_Info GB_subref              // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix C,               // output matrix, static header
    // input, not modified
    bool C_iso,                 // if true, return C as iso, regardless of A
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t ni,           // length of I, or special
    const void *J,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool J_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase0
(
    // output
    void **p_Ch,            // Ch = C->h hyperlist, or NULL
    bool *p_Cj_is_32,       // if true, C->h is 32-bit; else 64-bit
    bool *p_Ci_is_32,       // if true, C->i is 32-bit; else 64-bit
    size_t *p_Ch_size,
    void **p_Ap_start,      // A(:,kA) starts at Ap_start [kC]
    size_t *p_Ap_start_size,
    void **p_Ap_end,        // ... and ends at Ap_end [kC] - 1
    size_t *p_Ap_end_size,
    int64_t *p_Cnvec,       // # of vectors in C
    bool *p_need_qsort,     // true if C must be sorted
    int *p_Ikind,           // kind of I
    int64_t *p_nI,          // length of I
    int64_t Icolon [3],     // for GB_RANGE, GB_STRIDE
    int64_t *p_nJ,          // length of J
    // input, not modified
    const GrB_Matrix A,
    const void *I,          // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,     // if true, I is 32-bit; else 64-bit
    const int64_t ni,       // length of I, or special
    const void *J,          // index list for C = A(I,J), or GrB_ALL, etc.
    const bool J_is_32,     // if true, I is 32-bit; else 64-bit
    const int64_t nj,       // length of J, or special
    GB_Werk Werk
) ;

GrB_Info GB_I_inverse           // invert the I list for C=A(I,:)
(
    const void *I,              // list of indices, duplicates OK
    const bool I_is_32,         // if true, I is 32-bit; else 64 bit
    int64_t nI,                 // length of I
    int64_t avlen,              // length of the vectors of A
    // outputs:
    GrB_Matrix *R_handle,       // R = inverse (I)
    GB_Werk Werk
) ;

GrB_Info GB_subref_slice    // phase 1 of GB_subref
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,              // # of tasks constructed
    int *p_nthreads,            // # of threads for subref operation
    bool *p_post_sort,          // true if a final post-sort is needed
    GrB_Matrix *R_handle,       // R = inverse (I), if needed
    uint64_t **p_Cwork,         // workspace of size max(2,C->nvec+1)
    size_t *p_Cwork_size,
    // from phase0:
    const void *Ap_start,       // location of A(imin:imax,kA)
    const void *Ap_end,
    const int64_t Cnvec,        // # of vectors of C
    const bool need_qsort,      // true if C must be sorted
    const int Ikind,            // GB_ALL, GB_RANGE, GB_STRIDE or GB_LIST
    const int64_t nI,           // length of I
    const int64_t Icolon [3],   // for GB_RANGE and GB_STRIDE
    // original input:
    const int64_t avlen,        // A->vlen
    const int64_t anz,          // nnz (A)
    const bool Ap_is_32,        // if true, Ap_start/end are 32-bit; else 64
    const void *I,
    const bool I_is_32,         // if true, I is 32-bit; else 64 bit
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase2               // count nnz in each C(:,j)
(
    // computed by phase2:
    void **Cp_handle,                   // output of size Cnvec+1
    bool *p_Cp_is_32,                   // if true, Cp is 32-bit; else 64 bit
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    // tasks from phase1:
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads to use
    const GrB_Matrix R,                 // R = inverse (I), if needed
    uint64_t **p_Cwork,                 // workspace of size max(2,C->nvec+1)
    size_t Cwork_size,
    // analysis from phase0:
    const void *Ap_start,
    const void *Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // original input:
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const bool symbolic,
    GB_Werk Werk
) ;

GrB_Info GB_subref_phase3   // C=A(I,J)
(
    GrB_Matrix C,               // output matrix, static header
    // from phase2:
    void **Cp_handle,           // vector pointers for C
    const bool Cp_is_32,        // if true, Cp is 32-bit; else 64-bit
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // from phase1:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int ntasks,                           // # of tasks
    const int nthreads,                         // # of threads to use
    const bool post_sort,               // true if post-sort needed
    const GrB_Matrix R,                 // R = inverse (I), if needed
    // from phase0:
    void **Ch_handle,
    const bool Cj_is_32,        // if true, C->h is 32-bit; else 64-bit
    const bool Ci_is_32,        // if true, C->i is 32-bit; else 64-bit
    size_t Ch_size,
    const void *Ap_start,
    const void *Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // from GB_subref:
    const GrB_Type ctype,       // type of C to create
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // iso value of C
    // original input:
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,
    const void *I,
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const bool symbolic,
    GB_Werk Werk
) ;

GrB_Info GB_bitmap_subref       // C = A(I,J): either symbolic or numeric
(
    // output:
    GrB_Matrix C,               // output matrix, static header
    // inputs, not modified:
    const GrB_Type ctype,       // type of C to create
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // scalar value of C, if iso
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t ni,           // length of I, or special
    const void *J,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool J_is_32,         // if true, J is 32-bit; else 64-bit
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
) ;

#endif

