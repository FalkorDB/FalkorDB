//------------------------------------------------------------------------------
// GB_AxB_dot3_meta: C<M>=A'*B via dot products, where C is sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This template is #include'd in 3 ways to construct:
//  * a generic method: Template/GB_AxB_dot_generic.c)
//  * a Factory method: FactoryKernels/GB_AxB_*, the Adot3B method
//  * a JIT kernel: JitKernels/GB_jit_kernel_AxB_dot3.c

#define GB_DOT3
#define GB_DOT3_PHASE2

#include "GB_unused.h"
#include "GB_AxB_dot_cij.h"

// GB_DOT_ALWAYS_SAVE_CIJ: C(i,j) = cij
#if GB_CIJ_CHECK

    #define GB_DOT_ALWAYS_SAVE_CIJ      \
    {                                   \
        cij_exists = true ;             \
        /* Cx [pC] = cij */             \
        GB_PUTC (cij, Cx, pC) ;         \
        Ci [pC] = i ;                   \
    }

#else

    #define GB_DOT_ALWAYS_SAVE_CIJ      \
    {                                   \
        /* Cx [pC] = cij */             \
        GB_PUTC (cij, Cx, pC) ;         \
        Ci [pC] = i ;                   \
    }

#endif

// GB_DOT_SAVE_CIJ: C(i,j) = cij, if it exists
#define GB_DOT_SAVE_CIJ                 \
{                                       \
    if (GB_CIJ_EXISTS)                  \
    {                                   \
        /* Cx [pC] = cij */             \
        GB_PUTC (cij, Cx, pC) ;         \
        Ci [pC] = i ;                   \
    }                                   \
}

{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    // C and M have the same sparsity pattern (both are sparse or hyper),
    // except entries of C may become zombies.  M is not complemented.

    int64_t nzombies = 0 ;

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    const int64_t *restrict Cp = C->p ;
    const int64_t *restrict Ch = C->h ;
    int64_t  *restrict Ci = C->i ;
    const int64_t cvlen = C->vlen ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t *restrict Bi = B->i ;
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
    #endif

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t *restrict Ai = A->i ;
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
    #endif

    const int64_t *restrict A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *restrict A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *restrict A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    const int64_t *restrict B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
    const int64_t *restrict B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
    const int64_t *restrict B_Yx = (B->Y == NULL) ? NULL : B->Y->x ;
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

    const int64_t *restrict Mi = M->i ;
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
        #include "GB_meta16_definitions.h"
        #include "GB_AxB_dot3_template.c"
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
            #include "GB_AxB_dot3_template.c"
            #undef  GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
        }
        else
        { 
            // general case
            const size_t msize = M->type->size ;
            #include "GB_meta16_factory.c"
        }
    }
    #endif

    C->nzombies = nzombies ;
}

#undef GB_DOT_ALWAYS_SAVE_CIJ
#undef GB_DOT_SAVE_CIJ

#undef GB_DOT3
#undef GB_DOT3_PHASE2

