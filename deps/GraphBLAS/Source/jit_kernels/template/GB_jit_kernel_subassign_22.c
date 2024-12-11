//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_22.c: C += y where C is dense, y is a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Unlike most jit kernels for assign, the input scalar is already typecast
// to the op->ytype.

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    // get callback functions
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_ek_slice) ;
    GB_GET_CALLBACK (GB_werk_pop) ;
    GB_GET_CALLBACK (GB_werk_push) ;

    GB_Y_TYPE ywork = (*((GB_Y_TYPE *) scalar)) ;
    #include "template/GB_subassign_22_template.c"
    return (GrB_SUCCESS) ;
}

