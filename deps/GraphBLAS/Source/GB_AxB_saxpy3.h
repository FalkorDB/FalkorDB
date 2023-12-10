//------------------------------------------------------------------------------
// GB_AxB_saxpy3.h: definitions for C=A*B saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy3 method uses a mix of Gustavson's method and the Hash method,
// combining the two for any given C=A*B computation.

#ifndef GB_AXB_SAXPY3_H
#define GB_AXB_SAXPY3_H

#include "GB.h"

GrB_Info GB_AxB_saxpy3              // C = A*B using Gustavson+Hash
(
    GrB_Matrix C,                   // output matrix, static header
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    int C_sparsity,                 // construct C as sparse or hypersparse
    const GrB_Matrix M_input,       // optional mask matrix
    const bool Mask_comp_input,     // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    GrB_Desc_Value AxB_method,      // Default, Gustavson, or Hash
    const int do_sort,              // if nonzero, try to sort in saxpy3
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_saxpy3task_struct: task descriptor for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// A coarse task computes C(:,j1:j2) = A*B(:,j1:j2), for a contiguous set of
// vectors j1:j2.  A coarse taskid is denoted by SaxpyTasks [taskid].vector ==
// -1, kfirst = SaxpyTasks [taskid].start, and klast = SaxpyTasks [taskid].end,
// and where j1 = GBH (Bh, kstart) and likewise for j2.  No summation is needed
// for the final result of each coarse task.

// A fine taskid computes A*B(k1:k2,j) for a single vector C(:,j), for a
// contiguous range k1:k2, where kk = Tasklist[taskid].vector (which is >= 0),
// k1 = Bi [SaxpyTasks [taskid].start], k2 = Bi [SaxpyTasks [taskid].end].  It
// sums its computations in a hash table shared by all fine tasks that compute
// C(:,j), via atomics.  The vector index j is GBH (Bh, kk).

// Both tasks use a hash table allocated uniquely for the task, in Hi, Hf, and
// Hx.  The size of the hash table is determined by the maximum # of flops
// needed to compute any vector in C(:,j1:j2) for a coarse task, or the entire
// computation of the single vector in a fine task.  For the Hash method, the
// table has a size that is twice the smallest a power of 2 larger than the
// flop count.  If this size is a significant fraction of C->vlen, then the
// Hash method is not used, and Gustavson's method is used, with the hash size
// is set to C->vlen.

#include "GB_saxpy3task_struct.h"

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_flopcount:  compute flops for GB_AxB_saxpy3
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3_flopcount
(
    int64_t *Mwork,             // amount of work to handle the mask M
    int64_t *Bflops,            // size B->nvec+1 and all zero
    const GrB_Matrix M,         // optional mask matrix
    const bool Mask_comp,       // if true, mask is complemented
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_cumsum: cumulative sum of C->p for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// GB_CALLBACK_SAXPY3_CUMSUM_PROTO (GB_AxB_saxpy3_cumsum) ;

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_slice_balanced: create balanced parallel tasks for saxpy3
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3_slice_balanced
(
    // inputs
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    GrB_Desc_Value AxB_method,      // Default, Gustavson, or Hash
    bool builtin_semiring,          // if true, semiring is builtin
    // outputs
    GB_saxpy3task_struct **SaxpyTasks_handle,
    size_t *SaxpyTasks_size_handle,
    bool *apply_mask,               // if true, apply M during sapxy3
    bool *M_in_place,               // if true, use M in-place
    int *ntasks,                    // # of tasks created (coarse and fine)
    int *nfine,                     // # of fine tasks created
    int *nthreads,                  // # of threads to use
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_slice_quick: create a single sequential task for saxpy3
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3_slice_quick
(
    // inputs
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    // outputs
    GB_saxpy3task_struct **SaxpyTasks_handle,
    size_t *SaxpyTasks_size_handle,
    int *ntasks,                    // # of tasks created (coarse and fine)
    int *nfine,                     // # of fine tasks created
    int *nthreads,                  // # of threads to use
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

void GB_AxB_saxpy3_symbolic
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_comp,       // M complemented, or not
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_bh       // C=A*B, A is bitmap, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_bs       // C = A*B, A is bitmap, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_fh       // C = A*B, A is full, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_fs       // C = A*B, A is full, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_hb       // C = A*B, A is hypersparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_hf       // C = A*B, A is hypersparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_hh       // C = A*B, A is hypersparse, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_hs       // C = A*B, A is hypersparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_sb       // C = A*B, A is sparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_sf       // C = A*B, A is sparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_sh       // C = A*B, A is sparse, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_ss       // C = A*B, A is sparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mbb      // C<M> = A*B, A is bitmap, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mbf      // C<M> = A*B, A is bitmap, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mbh      // C<M> = A*B, A is bitmap, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mbs      // C<M> = A*B, A is bitmap, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mfb      // C<M> = A*B, A is full, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mff      // C<M> = A*B, A is full, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mfh      // C<M> = A*B, A is full, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mfs      // C<M> = A*B, A is full, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mhb      // C<M> = A*B, A is hypersparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mhf      // C<M> = A*B, A is hypersparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mhh      // C<M> = A*B, A and B are hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mhs      // C<M> = A*B, A is hypersparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_msb      // C<M> = A*B, A is sparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_msf      // C<M> = A*B, A is sparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_msh      // C<M> = A*B, A is sparse, B is hyperparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_mss      // C<M> = A*B, A is sparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nbh      // C<!M> = A*B, A is bitmap, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nbs      // C<!M> = A*B, A is bitmap, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nfh      // C<!M> = A*B, A is full, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nfs      // C<!M> = A*B, A is full, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nhb      // C<!M> = A*B, A is hypersparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nhf      // C<!M> = A*B, A is hypersparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nhh      // C<!M> = A*B, A and B  re hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nhs      // C<!M> = A*B, A is hypersparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nsb      // C<!M> = A*B, A is sparse, B is bitmap
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nsf      // C<!M> = A*B, A is sparse, B is full
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nsh      // C<!M> = A*B, A is sparse, B is hypersparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

void GB_AxB_saxpy3_sym_nss      // C<!M> = A*B, A is sparse, B is sparse
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
) ;

#endif

