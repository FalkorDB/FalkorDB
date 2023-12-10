//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_06d.c:  C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 06d: C(:,:)<A> = A ; no S, C is dense, M and A are aliased

// M:           present, and aliased to A
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           matrix, and aliased to M
// S:           none
// I:           NULL
// J:           NULL

// C must be bitmap or as-if-full.  No entries are deleted and thus no zombies
// are introduced into C.  C can be hypersparse, sparse, bitmap, or full, and
// its sparsity structure does not change.  If C is hypersparse, sparse, or
// full, then the pattern does not change (all entries are present, and this
// does not change), and these cases can all be treated the same (as if full).
// If C is bitmap, new entries can be inserted into the bitmap C->b.

// C and A can have any sparsity structure.

GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    GB_ek_slice_f GB_ek_slice = my_callback->GB_ek_slice_func ;
    GB_werk_pop_f GB_werk_pop = my_callback->GB_werk_pop_func ;
    GB_werk_push_f GB_werk_push = my_callback->GB_werk_push_func ;
    GB_memset_f GB_memset = my_callback->GB_memset_func ;
    #endif

    #include "GB_subassign_06d_template.c"
    return (GrB_SUCCESS) ;
}

