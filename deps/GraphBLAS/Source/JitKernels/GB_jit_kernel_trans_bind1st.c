//------------------------------------------------------------------------------
// GB_jit_kernel_trans_bind1st.c: Cx = op (x,A')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (x, aij)
#define GB_APPLY_OP(pC,pA)                      \
{                                               \
    GB_DECLAREB (aij) ;                         \
    GB_GETB (aij, Ax, pA, false) ;              \
    GB_EWISEOP (Cx, pC, x, aij, 0, 0) ;         \
}

GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_BIND1ST_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_TRANS_BIND1ST_PROTO (GB_jit_kernel)
{
    #define GB_BIND_1ST
    GB_X_TYPE x = (*((const GB_X_TYPE *) x_input)) ;
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
}

