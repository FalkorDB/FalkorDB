//------------------------------------------------------------------------------
// GB_reduce.h: definitions for GB_reduce
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_REDUCE_H
#define GB_REDUCE_H
#include "GB.h"

GrB_Info GB_reduce_to_scalar    // s = reduce_to_scalar (A)
(
    void *c,                    // result scalar
    const GrB_Type ctype,       // the type of scalar, c
    const GrB_BinaryOp accum,   // for c = accum(c,s)
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_Werk Werk
) ;

GrB_Info GB_reduce_to_vector        // C<M> = accum (C,reduce(A))
(
    GrB_Matrix C,                   // input/output for results, size n-by-1
    const GrB_Matrix M,             // optional M for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C,T)
    const GrB_Monoid monoid,        // reduce monoid for T=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Werk Werk
) ;

void GB_reduce_to_scalar_iso        // s = reduce (A) where A is iso
(
    GB_void *restrict s,            // output scalar of type reduce->op->ztype
    GrB_Monoid monoid,              // monoid to use for the reduction
    GrB_Matrix A                    // matrix to reduce
) ;

void GB_reduce_worker_iso
(
    GB_void *restrict s,            // output scalar
    GxB_binary_function freduce,    // reduction function
    GB_void *restrict a,            // iso value of A
    uint64_t n,                     // number of entries in A to reduce
    size_t zsize                    // size of s and a
) ;

GrB_Info GB_Scalar_reduce
(
    GrB_Scalar S,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    GB_Werk Werk
) ;

#endif

