//------------------------------------------------------------------------------
// GB_mask_very_sparse.h: determine if a mask is very sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MASK_VERY_SPARSE_H
#define GB_MASK_VERY_SPARSE_H

// GB_MASK_VERY_SPARSE is true if C<M>=A+B, C<M>=A.*B or C<M>=accum(C,T) is
// being computed, and the mask M is very sparse compared with A and B.
// If any matrix is NULL, GB_nnz returns zero.

#define GB_MASK_VERY_SPARSE(mfactor,M,A,B) \
    (((double) ((mfactor) * GB_nnz (M))) < ((double) (GB_nnz (A) + GB_nnz (B))))

#endif

