//------------------------------------------------------------------------------
// GB_extract.h: definitions for GB_extract
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EXTRACT_H
#define GB_EXTRACT_H
#include "GB.h"

GrB_Info GB_extract                 // C<M> = accum (C, A(I,J))
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C matrix descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // mask descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // A matrix descriptor
    const void *Rows,               // row indices
    const bool Rows_is_32,          // if true, Rows is 32-bit; else 64-bit
    const uint64_t nRows_in,        // number of row indices
    const void *Cols,               // column indices
    const bool Cols_is_32,          // if true, Rows is 32-bit; else 64-bit
    const uint64_t nCols_in,        // number of column indices
    GB_Werk Werk
) ;

#endif

