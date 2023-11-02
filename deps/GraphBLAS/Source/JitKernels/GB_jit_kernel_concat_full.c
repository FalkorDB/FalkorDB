//------------------------------------------------------------------------------
// GB_jit_kernel_concat_full: concatenate a full A into a full matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA,A_iso) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_FULL_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_CONCAT_FULL_PROTO (GB_jit_kernel)
{
    #include "GB_concat_full_template.c"
    return (GrB_SUCCESS) ;
}

