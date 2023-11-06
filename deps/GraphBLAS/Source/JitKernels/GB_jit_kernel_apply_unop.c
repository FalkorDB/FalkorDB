//------------------------------------------------------------------------------
// GB_jit_kernel_apply_unop.c: Cx = op (A) for unary or index unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#if GB_DEPENDS_ON_I

    // cij = op (aij)
    #define GB_APPLY_OP(p)                          \
    {                                               \
        int64_t i = GBI_A (Ai, p, avlen) ;          \
        GB_UNOP (Cx, p, Ax, p, A_iso, i, j, y) ;    \
    }

#else

    // cij = op (aij)
    #define GB_APPLY_OP(p) GB_UNOP (Cx, p, Ax, p, A_iso, i, j, y)

#endif

GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_out ;
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    #if GB_A_IS_BITMAP
    int8_t *restrict Ab = A->b ;
    #endif
    GB_A_NHELD (anz) ;      // int64_t anz = GB_nnz_held (A) ;
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = (*((GB_Y_TYPE *) ythunk)) ;
    #endif

    #if GB_DEPENDS_ON_J
    {
        const int64_t *restrict Ap = A->p ;
        const int64_t *restrict Ah = A->h ;
        const int64_t *restrict Ai = A->i ;
        int64_t avlen = A->vlen ;
        #include "GB_apply_unop_ijp.c"
    }
    #else
    {
        #include "GB_apply_unop_ip.c"
    }

    #endif
    return (GrB_SUCCESS) ;
}

