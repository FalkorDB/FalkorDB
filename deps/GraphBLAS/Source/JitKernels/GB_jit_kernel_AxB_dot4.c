//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot4.c: JIT kernel for C+=A'*B dot4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B: dot product, C is full, dot4 method

GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT4_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_AXB_DOT4_PROTO (GB_jit_kernel)
{
    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    #endif

    #include "GB_AxB_dot4_meta.c"
    return (GrB_SUCCESS) ;
}

