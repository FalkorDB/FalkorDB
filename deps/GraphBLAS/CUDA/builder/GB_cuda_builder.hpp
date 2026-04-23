//------------------------------------------------------------------------------
// GB_cuda_builder.hpp: CPU definitions for CUDA builder operations
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_SELECT_H
#define GB_CUDA_SELECT_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_builder_jit
(
    // output:
    GrB_Matrix *Thandle,
    // input:
    const GrB_Type ttype,   // type of output matrix T
    const int64_t vlen,     // length of each vector of T
    const int64_t vdim,     // number of vectors in T
    const bool is_csc,      // true if T is CSC, false if CSR
    const bool is_matrix,   // true if T a GrB_Matrix, false if vector
    const GB_void *Key_input,  // present if Key_in preloaded
    const GB_void *I,       // original indices, size nvals
    const GB_void *J,       // original indices, size nvals
    const GB_void *X,       // array of values of tuples, size nvals,
                            // or size 1 if X is iso
    const bool X_iso,       // true if X is iso
    const int64_t nvals,    // number of tuples
    GrB_BinaryOp dup,       // binary function to assemble duplicates,
                            // if NULL use the SECOND operator to
                            // keep the most recent duplicate.
    const GrB_Type xtype,   // the type of X
    bool I_is_32,       // true if I is 32 bit, false if 64
    bool J_is_32,       // true if J is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32,      // true if T->i is built as 32 bit, false if 64
    bool known_no_duplicates,   // true if tuples known to have no duplicates
    bool known_sorted,          // true if tuples known to be sorted on input
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

