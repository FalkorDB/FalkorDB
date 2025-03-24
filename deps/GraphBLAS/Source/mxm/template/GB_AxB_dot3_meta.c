//------------------------------------------------------------------------------
// GB_AxB_dot3_meta: C<M>=A'*B via dot products, where C is sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template is #include'd in 3 ways to construct:
//  * a generic method: mxm/factory/GB_AxB_dot_generic.c
//  * a Factory method: FactoryKernels/GB_AxB_*, the Adot3B method
//  * a JIT kernel: jit_kernels/GB_jit_kernel_AxB_dot3.c

#define GB_DOT3
#define GB_DOT3_PHASE2

#include "include/GB_unused.h"
#include "include/GB_AxB_dot_cij.h"

// GB_DOT_ALWAYS_SAVE_CIJ: C(i,j) = cij
#if GB_CIJ_CHECK

    #define GB_DOT_ALWAYS_SAVE_CIJ              \
    {                                           \
        cij_exists = true ;                     \
        /* Cx [pC] = cij */                     \
        GB_PUTC (cij, Cx, pC) ;                 \
        GB_ISET (Ci, pC, i) ; /* Ci [pC] = i */ \
    }

#else

    #define GB_DOT_ALWAYS_SAVE_CIJ              \
    {                                           \
        /* Cx [pC] = cij */                     \
        GB_PUTC (cij, Cx, pC) ;                 \
        GB_ISET (Ci, pC, i) ; /* Ci [pC] = i */ \
    }

#endif

// GB_DOT_SAVE_CIJ: C(i,j) = cij, if it exists
#define GB_DOT_SAVE_CIJ                         \
{                                               \
    if (GB_CIJ_EXISTS)                          \
    {                                           \
        /* Cx [pC] = cij */                     \
        GB_PUTC (cij, Cx, pC) ;                 \
        GB_ISET (Ci, pC, i) ; /* Ci [pC] = i */ \
    }                                           \
}

{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    // C and M have the same sparsity pattern (both are sparse or hyper),
    // except entries of C may become zombies.  M is not complemented.

    int64_t nzombies = 0 ;

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;

    GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
    const int64_t cvlen = C->vlen ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t bnvec = B->nvec ;

    #ifdef GB_JIT_KERNEL
    // B matrix properties fixed in the jit kernel
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_iso GB_B_ISO
    #else
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_iso = B->iso ;
    const bool Bp_is_32 = B->p_is_32 ;
    const bool Bj_is_32 = B->j_is_32 ;
    const bool Bi_is_32 = B->i_is_32 ;
    #define GB_Bp_IS_32 Bp_is_32
    #define GB_Bj_IS_32 Bj_is_32
    #define GB_Bi_IS_32 Bi_is_32
    #endif

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t anvec = A->nvec ;

    #ifdef GB_JIT_KERNEL
    // A matrix properties fixed in the jit kernel
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_iso GB_A_ISO
    #else
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_iso = A->iso ;
    const bool Ap_is_32 = A->p_is_32 ;
    const bool Aj_is_32 = A->j_is_32 ;
    const bool Ai_is_32 = A->i_is_32 ;
    #define GB_Ap_IS_32 Ap_is_32
    #define GB_Aj_IS_32 Aj_is_32
    #define GB_Ai_IS_32 Ai_is_32
    #endif

    const void *A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const void *A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const void *A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    const void *B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
    const void *B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
    const void *B_Yx = (B->Y == NULL) ? NULL : B->Y->x ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
    #if !GB_IS_ANY_PAIR_SEMIRING
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    const int64_t vlen = A->vlen ;
    ASSERT (A->vlen == B->vlen) ;

    #ifdef GB_JIT_KERNEL
    #define Mask_struct GB_MASK_STRUCT
    #endif

    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const size_t mvlen = M->vlen ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;

    //--------------------------------------------------------------------------
    // C<M> = A'*B via dot products, where C and M are both sparse/hyper
    //--------------------------------------------------------------------------

    // 4 possible cases of the mask are handled:

    // M can be sparse or hyper, and always present
    // M can be structural or valued
    // M is not complemented

    // The other 12 cases of the mask, and the one no-mask case, are handled
    // by dot2.

    #ifdef GB_JIT_KERNEL
    {
        GB_DECLARE_TERMINAL_CONST (zterminal) ;
        #define GB_META16
        #include "include/GB_meta16_definitions.h"
        #include "template/GB_AxB_dot3_template.c"
    }
    #else
    {
        const bool M_is_sparse = GB_IS_SPARSE (M) ;
        ASSERT (M_is_sparse || GB_IS_HYPERSPARSE (M)) ;
        if (M_is_sparse && Mask_struct && A_is_sparse && B_is_sparse)
        { 
            // special case: M is present, sparse, structural, and not
            // complemented, and A and B are sparse
            #define GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "template/GB_AxB_dot3_template.c"
            #undef  GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
        }
        else
        { 
            // general case
            const size_t msize = M->type->size ;
            #include "template/GB_meta16_factory.c"
        }
    }
    #endif

    C->nzombies = nzombies ;
}

#undef GB_DOT_ALWAYS_SAVE_CIJ
#undef GB_DOT_SAVE_CIJ

#undef GB_DOT3
#undef GB_DOT3_PHASE2

