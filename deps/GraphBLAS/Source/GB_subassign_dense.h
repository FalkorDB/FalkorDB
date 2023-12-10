//------------------------------------------------------------------------------
// GB_subassign_dense.h: definitions for dense subassign methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBASSIGN_DENSE_H
#define GB_SUBASSIGN_DENSE_H

#include "GB_ek_slice.h"

//------------------------------------------------------------------------------
// GB_subassign_23: C(:,:) += A where C is dense and A is sparse or dense
//------------------------------------------------------------------------------

GrB_Info GB_subassign_23      // C += A; C is dense, A is sparse or dense
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_22: C(:,:) += scalar where C is dense
//------------------------------------------------------------------------------

GrB_Info GB_subassign_22      // C += x where C is dense and x is a scalar
(
    GrB_Matrix C,                   // input/output matrix
    const void *scalar,             // input scalar
    const GrB_Type scalar_type,           // type of the input scalar
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_05d: C(:,:)<M> = scalar ; C is dense
//------------------------------------------------------------------------------

GrB_Info GB_subassign_05d
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_06d: C(:,:)<A> = A ; C is dense
//------------------------------------------------------------------------------

GrB_Info GB_subassign_06d
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const bool Mask_struct,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_24: C = A
//------------------------------------------------------------------------------

GrB_Info GB_subassign_24    // C = A, copy A into an existing matrix C
(
    GrB_Matrix C,           // output matrix to modify
    const GrB_Matrix A,     // input matrix to copy
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_25: C<M> = A ; C is empty, A is dense, M is structural
//------------------------------------------------------------------------------

GrB_Info GB_subassign_25
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

#endif

