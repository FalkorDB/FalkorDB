//------------------------------------------------------------------------------
// GB_jit_kernel_iso_expand.c: expand an iso scalar into an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_ISO_EXPAND_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_ISO_EXPAND_PROTO (GB_jit_kernel)
{
    GB_A_TYPE *restrict Z = (GB_A_TYPE *) X ;
    GB_A_TYPE x = (* ((GB_A_TYPE *) scalar)) ;
    int64_t p ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < n ; p++)
    {
        Z [p] = x ;
    }
    return (GrB_SUCCESS) ;
}

