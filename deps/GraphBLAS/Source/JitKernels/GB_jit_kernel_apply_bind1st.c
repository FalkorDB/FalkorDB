//------------------------------------------------------------------------------
// GB_jit_kernel_apply_bind1st.c: Cx = op (x,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel)
{
    #include "GB_apply_bind1st_template.c"
    return (GrB_SUCCESS) ;
}

