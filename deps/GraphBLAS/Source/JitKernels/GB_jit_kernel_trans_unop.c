//------------------------------------------------------------------------------
// GB_jit_kernel_trans_unop.c: C = op (A') for unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: C=op(A') is used only for unary ops; extend it to index unary ops

// cij = op (aij)
#define GB_APPLY_OP(pC,pA) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_UNOP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_UNOP_PROTO (GB_jit_kernel)
{
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
}

