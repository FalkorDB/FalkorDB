//------------------------------------------------------------------------------
// GB_build.h: definitions for GB_build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_BUILD_H
#define GB_BUILD_H
#include "GB.h"

GrB_Info GB_build               // build matrix
(
    GrB_Matrix C,               // matrix to build
    const void *I,              // row indices of tuples
    const void *J,              // col indices of tuples (NULL for vector)
    const void *X,              // values, size 1 if iso
    const uint64_t nvals,       // number of tuples
    const GrB_BinaryOp dup,     // binary op to assemble duplicates (or NULL)
    const GrB_Type xtype,       // type of X array
    const bool is_matrix,       // true if C is a matrix, false if GrB_Vector
    const bool X_iso,           // if true the C is iso and X has size 1 entry
    bool I_is_32,               // if true, I is 32-bit; else 64-bit
    bool J_is_32,               // if true, J is 32-bit; else 64-bit
    GB_Werk Werk
) ;

GrB_Info GB_builder                 // build a matrix from tuples
(
    GrB_Matrix T,                   // matrix to build, static or dynamic header
    const GrB_Type ttype,           // type of output matrix T
    const int64_t vlen,             // length of each vector of T
    const int64_t vdim,             // number of vectors in T
    const bool is_csc,              // true if T is CSC, false if CSR
    void **I_work_handle,           // for (i,k) or (j,i,k) tuples
    size_t *I_work_size_handle,
    void **J_work_handle,           // for (j,i,k) tuples
    size_t *J_work_size_handle,
    GB_void **S_work_handle,        // array of values of tuples, size ijslen,
                                    // or size 1 if S is iso
    size_t *S_work_size_handle,
    bool known_sorted,              // true if tuples known to be sorted
    bool known_no_duplicates,       // true if tuples known to not have dupl
    int64_t ijslen,                 // size of I_work and J_work arrays
    const bool is_matrix,           // true if T a GrB_Matrix, false if vector
    const void *restrict I_input,   // original indices, size nvals
    const void *restrict J_input,   // original indices, size nvals
    const GB_void *restrict S_input,// array of values of tuples, size nvals,
                                    // or size 1 if S_input or S_work are iso
    const bool S_iso,               // true if S_input or S_work are iso
    const int64_t nvals,            // number of tuples, and size of K_work
    GrB_BinaryOp dup,               // binary function to assemble duplicates,
                                    // if NULL use the SECOND operator to
                                    // keep the most recent duplicate.
    const GrB_Type stype,           // the type of S_work or S_input
    bool do_burble,                 // if true, then burble is allowed
    GB_Werk Werk,
    bool I_is_32,       // true if I (I_work or I_input) is 32 bit, false if 64
    bool J_is_32,       // true if J (J_work or J_input) is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32       // true if T->i is built as 32 bit, false if 64
) ;

#endif

