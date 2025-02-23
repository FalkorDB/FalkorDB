//------------------------------------------------------------------------------
// GB_assign.h: definitions for GB_assign and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ASSIGN_H
#define GB_ASSIGN_H
#include "GB.h"
#include "math/GB_math.h"

GrB_Info GB_assign                  // C<M>(I,J) += A or A'
(
    GrB_Matrix C,                   // input/output matrix for results
    bool C_replace,                 // descriptor for C
    const GrB_Matrix M_in,          // optional mask for C
    const bool Mask_comp,           // true if mask is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const bool M_transpose,         // true if the mask should be transposed
    const GrB_BinaryOp accum,       // optional accum for accum(C,T)
    const GrB_Matrix A_in,          // input matrix
    const bool A_transpose,         // true if A is transposed
    const void *I,                  // row indices
    const bool I_is_32,             // if true, I is 32-bit; else 64-bit
    const uint64_t nI_in,           // number of row indices
    const void *J,                  // column indices
    const bool J_is_32,             // if true, J is 32-bit; else 64-bit
    const uint64_t nJ_in,           // number of column indices
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GB_Type_code scalar_code, // type code of scalar to expand
    int assign_kind,                // row, col, or matrix/vector assign
    GB_Werk Werk
) ;

GrB_Info GB_assign_scalar           // C<M>(I,J) += x
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // accum for Z=accum(C(I,J),T)
    const void *scalar,             // scalar to assign to C(I,J)
    const GB_Type_code scalar_code, // type code of scalar to assign
    const void *I,                  // row indices
    const bool I_is_32,             // if true, I is 32-bit; else 64-bit
    const uint64_t nI,              // number of row indices
    const void *J,                  // column indices
    const bool J_is_32,             // if true, J is 32-bit; else 64-bit
    const uint64_t nJ,              // number of column indices
    const GrB_Descriptor desc,      // descriptor for C and M
    GB_Werk Werk
) ;

GrB_Info GB_Matrix_assign_scalar    // C<Mask>(I,J) = accum (C(I,J),s)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    const GrB_Scalar scalar,        // scalar to assign to C(I,J)
    const void *I,                  // row indices
    const bool I_is_32,
    uint64_t ni,                    // number of row indices
    const void *J,                  // column indices
    const bool J_is_32,
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc,
    GB_Werk Werk
) ;

GrB_Info GB_Vector_assign_scalar    // w<Mask>(I) = accum (w(I),s)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar scalar,        // scalar to assign to w(I)
    const void *I,                  // row indices
    const bool I_is_32,
    uint64_t ni,                    // number of row indices
    const GrB_Descriptor desc,      // descriptor for w and Mask
    GB_Werk Werk
) ;

#endif

