//------------------------------------------------------------------------------
// GB_jit_kernel_convert_s2b.c: convert sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(Axnew,pnew,Ax,p) GB_UNOP (Axnew, pnew, Ax, p, A_iso, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_CONVERT_S2B_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_CONVERT_S2B_PROTO (GB_jit_kernel)
{
    #include "GB_convert_s2b_template.c"
    return (GrB_SUCCESS) ;
}

