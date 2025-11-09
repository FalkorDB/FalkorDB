//------------------------------------------------------------------------------
// GB_jit_kernel_apply_unop.c: Cx = op (A) for unary or index unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#if GB_DEPENDS_ON_I

    // cij = op (aij)
    #define GB_APPLY_OP(pC,pA)                      \
    {                                               \
        int64_t i = GBi_A (Ai, pA, avlen) ;         \
        GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y) ;  \
    }

#else

    // cij = op (aij)
    #define GB_APPLY_OP(pC,pA) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

#endif

GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_out ;
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    int8_t *restrict Ab = A->b ;
    GB_A_NHELD (anz) ;      // int64_t anz = GB_nnz_held (A) ;
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = (*((GB_Y_TYPE *) ythunk)) ;
    #endif

    #if GB_DEPENDS_ON_J
    {
        GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
        GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
        GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
        int64_t avlen = A->vlen ;
        #include "template/GB_apply_unop_ijp_template.c"
    }
    #else
    {
        #include "template/GB_apply_unop_ip_template.c"
    }
    #endif
    return (GrB_SUCCESS) ;
}

