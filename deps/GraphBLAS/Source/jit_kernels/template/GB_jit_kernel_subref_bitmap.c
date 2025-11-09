//------------------------------------------------------------------------------
// GB_jit_kernel_subref_bitmap.c: A = C(I,J) where C and A are bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_subref_method.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                     \
{                                                       \
    GB_FREE_MEMORY (&TaskList_IxJ, TaskList_IxJ_size) ;   \
}

GB_JIT_GLOBAL GB_JIT_KERNEL_BITMAP_SUBREF_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_BITMAP_SUBREF_PROTO (GB_jit_kernel)
{
    // get callback functions
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_subassign_IxJ_slice) ;

    // declare the TaskList_IxJ workspace
    GrB_Info info = GrB_SUCCESS ;
    GB_task_struct *TaskList_IxJ = NULL ; size_t TaskList_IxJ_size = 0 ;
    int ntasks_IxJ = 0, nthreads_IxJ = 0 ;

    // get C and A
    #ifdef GB_C_IS_BITMAP
    const int8_t *restrict Ab = A->b ;
          int8_t *restrict Cb = C->b ;
    #endif
    const int64_t vlen = A->vlen ;
    #define GB_COPY_ENTRY(pC,pA) Cx [pC] = Ax [pA] ;
    const GB_C_TYPE *restrict Ax = (GB_C_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    const GB_I_TYPE *restrict I = I_input ;
    const GB_J_TYPE *restrict J = J_input ;

    #include "template/GB_bitmap_subref_template.c"
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

