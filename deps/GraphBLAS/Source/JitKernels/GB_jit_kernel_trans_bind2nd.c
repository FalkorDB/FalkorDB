//------------------------------------------------------------------------------
// GB_jit_kernel_trans_bind2nd.c: Cx = op (A',x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij, y)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)                      \
{                                               \
    GB_DECLAREA (aij) ;                         \
    GB_GETA (aij, Ax, pA, false) ;              \
    GB_EWISEOP (Cx, pC, aij, y, 0, 0) ;         \
}

GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_BIND2ND_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_BIND2ND_PROTO (GB_jit_kernel)
{
    GB_Y_TYPE y = (*((const GB_Y_TYPE *) y_input)) ;
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
}

