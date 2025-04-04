//------------------------------------------------------------------------------
// GB_jit_kernel_convert_b2s.c: convert sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(Cx,pC,Ax,pA) GB_UNOP (Cx, pC, Ax, pA, 0, , , )

GB_JIT_GLOBAL GB_JIT_KERNEL_CONVERT_B2S_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_CONVERT_B2S_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #include "template/GB_convert_b2s_template.c"
    return (GrB_SUCCESS) ;
}

