//------------------------------------------------------------------------------
// GB_jit_kernel_concat_sparse: concatenate A into a sparse matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA,A_iso) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_SPARSE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_SPARSE_PROTO (GB_jit_kernel)
{
    #include "GB_concat_sparse_template.c"
    return (GrB_SUCCESS) ;
}

