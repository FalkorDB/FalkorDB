//------------------------------------------------------------------------------
// GB_jit_kernel_colscale.c: C=A*D matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_COLSCALE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_COLSCALE_PROTO (GB_jit_kernel)
{
    #include "GB_colscale_template.c"
    return (GrB_SUCCESS) ;
}

