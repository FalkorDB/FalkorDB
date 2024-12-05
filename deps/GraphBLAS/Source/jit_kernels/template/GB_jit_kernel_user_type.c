//------------------------------------------------------------------------------
// GB_jit_kernel_user_type.c: JIT kernel for a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_USER_TYPE_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_USER_TYPE_PROTO (GB_jit_kernel)
{
    (*user_type_size) = sizeof (GB_USER_TYPE) ;
    (*defn) = GB_USER_TYPE_DEFN ;
    return (GrB_SUCCESS) ;
}

