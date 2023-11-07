//------------------------------------------------------------------------------
// GB_emult.h: definitions for GB_emult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EMULT_H
#define GB_EMULT_H
#include "GB.h"
#include "GB_bitmap_assign_methods.h"

#define GB_EMULT_METHOD1_ADD 1      /* use GB_add instead of emult */
#define GB_EMULT_METHOD2     2      /* use GB_emult_02 */
#define GB_EMULT_METHOD3     3      /* use GB_emult_03 */
#define GB_EMULT_METHOD4     4      /* use GB_emult_04 */
#define GB_EMULT_METHOD5     5      /* use GB_emult_bitmap Method5 */
#define GB_EMULT_METHOD6     6      /* use GB_emult_bitmap Method6 */
#define GB_EMULT_METHOD7     7      /* use GB_emult_bitmap Method7 */
#define GB_EMULT_METHOD8     8      /* use GB_emult_08_phase[012] */
#define GB_EMULT_METHOD9     9      /* use GB_emult_08_phase[012] for now */
#define GB_EMULT_METHOD10    10     /* use GB_emult_08_phase[012] for now */

GrB_Info GB_emult           // C=A.*B or C<M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL.  Not complemented
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use the !M
    bool *mask_applied,
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Werk Werk
) ;

int GB_emult_sparsity       // return the sparsity structure for C
(
    // output:
    bool *apply_mask,       // if true then mask will be applied by GB_emult
    int *ewise_method,      // method to use
    // input:
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B      // input B matrix
) ;

GrB_Info GB_emult_08_phase0     // find vectors in C for C=A.*B or C<M>=A.*B
(
    int64_t *p_Cnvec,           // # of vectors to compute in C
    const int64_t *restrict *Ch_handle,  // Ch is M->h, A->h, B->h, or NULL
    size_t *Ch_size_handle,
    int64_t *restrict *C_to_M_handle,    // C_to_M: size Cnvec, or NULL
    size_t *C_to_M_size_handle,
    int64_t *restrict *C_to_A_handle,    // C_to_A: size Cnvec, or NULL
    size_t *C_to_A_size_handle,
    int64_t *restrict *C_to_B_handle,    // C_to_B: size Cnvec, or NULL
    size_t *C_to_B_size_handle,
    int *C_sparsity,            // sparsity structure of C
    // original input:
    const GrB_Matrix M,         // optional mask, may be NULL
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
) ;

GrB_Info GB_emult_08_phase1                 // count nnz in each C(:,j)
(
    // computed by phase1:
    int64_t **Cp_handle,                    // output of size Cnvec+1
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,                // # of non-empty vectors in C
    // tasks from phase1a:
    GB_task_struct *restrict TaskList,   // array of structs
    const int C_ntasks,                     // # of tasks
    const int C_nthreads,                   // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    const int64_t *restrict Ch,          // Ch is NULL, or shallow pointer
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
) ;

GrB_Info GB_emult_08_phase2             // C=A.*B or C<M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    // from phase1:
    int64_t **Cp_handle,    // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList, // array of structs
    const int C_ntasks,                         // # of tasks
    const int C_nthreads,                       // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    const int64_t *restrict Ch,
    size_t Ch_size,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const int C_sparsity,
    // from GB_emult_sparsity:
    const int ewise_method,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
) ;

GrB_Info GB_emult_02        // C=A.*B when A is sparse/hyper, B bitmap/full
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix (sparse/hyper)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    GrB_BinaryOp op,        // op to perform C = op (A,B)
    GB_Werk Werk
) ;

GrB_Info GB_emult_02_phase1 // symbolic analysis for GB_emult_02 and GB_emult_03
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads,
    // workspace:
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    // output:
    int64_t *Cp_kfirst,
    GB_Werk Werk
) ;

GrB_Info GB_emult_03        // C=A.*B when A is bitmap/full, B sparse/hyper
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix (sparse/hyper)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    GrB_BinaryOp op,        // op to perform C = op (A,B)
    GB_Werk Werk
) ;

GrB_Info GB_emult_04        // C<M>=A.*B, M sparse/hyper, A and B bitmap/full
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // sparse/hyper, not NULL
    const bool Mask_struct, // if true, use the only structure of M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix (bitmap/full)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Werk Werk
) ;

GrB_Info GB_emult_bitmap    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B
(
    GrB_Matrix C,           // output matrix, static header
    const int ewise_method,
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix (bitmap/full)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Werk Werk
) ;

bool GB_emult_iso           // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    GrB_Matrix B,           // input matrix
    GrB_BinaryOp op         // binary operator
) ;

GrB_Info GB_emult_generic       // generic emult
(
    // input/output:
    GrB_Matrix C,           // output matrix, static header
    // input:
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,  // array of structs
    const int C_ntasks,                         // # of tasks
    const int C_nthreads,                       // # of threads to use
    // analysis from phase0:
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const int C_sparsity,
    // from GB_emult_sparsity:
    const int ewise_method,
    // from GB_emult_04, GB_emult_03, GB_emult_02:
    const int64_t *restrict Cp_kfirst,
    // to slice M, A, and/or B,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads,
    const int64_t *B_ek_slicing, const int B_ntasks, const int B_nthreads,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B
) ;

#endif

