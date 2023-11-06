//------------------------------------------------------------------------------
// GB_AxB_saxpy.h: definitions for GB_AxB_saxpy and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_AXB_SAXPY_H
#define GB_AXB_SAXPY_H
#include "GB.h"
#include "GB_AxB_saxpy3.h"

//------------------------------------------------------------------------------
// GB_AxB_saxpy
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy               // C = A*B using Gustavson/Hash/Bitmap
(
    GrB_Matrix C,                   // output, static header
    GrB_Matrix C_in,                // original input matrix
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    bool *done_in_place,            // if true, C was computed in-place 
    const GrB_Desc_Value AxB_method,
    const int do_sort,              // if nonzero, try to sort in saxpy3
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_AxB_saxpy_sparsity: determine the sparsity of C
//------------------------------------------------------------------------------

void GB_AxB_saxpy_sparsity          // determine C_sparsity and method to use
(
    // output:
    int *C_sparsity,                // sparsity structure of C
    int *saxpy_method,              // saxpy method to use
    // input:
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input A matrix
    const GrB_Matrix B              // input B matrix
) ;

//------------------------------------------------------------------------------
// saxpy4: C+=A*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy4              // C += A*B
(
    GrB_Matrix C,                   // users input/output matrix
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B and accum
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *done_in_place,            // if true, saxpy4 has computed the result
    GB_Werk Werk
) ;

void GB_AxB_saxpy4_tasks
(
    // output
    int *p_ntasks,                  // # of tasks to use
    int *p_nthreads,                // # of threads to use
    int *p_nfine_tasks_per_vector,  // # of tasks per vector (fine case only)
    bool *p_use_coarse_tasks,       // if true, use coarse tasks
    bool *p_use_atomics,            // if true, use atomics
    // input
    int64_t anz,                    // # of entries in A (sparse or hyper)
    int64_t bnz,                    // # of entries held in B
    int64_t bvdim,                  // # of vectors of B (bitmap or full)
    int64_t cvlen                   // # of vectors of C (bitmap or full)
) ;

//------------------------------------------------------------------------------
// saxpy5: C+=A*B where A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy5              // C += A*B
(
    GrB_Matrix C,                   // users input/output matrix
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B and accum
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *done_in_place,            // if true, saxpy5 has computed the result
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// saxbit:
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxbit        // C = A*B where C is bitmap
(
    GrB_Matrix C,                   // output matrix, static header
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Werk Werk
) ;

#endif

