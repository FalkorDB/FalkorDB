//------------------------------------------------------------------------------
// GB_AxB_dot4_meta:  C+=A'*B via dot products, where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B where C is a dense matrix and computed in-place.  The monoid of the
// semiring matches the accum operator, and the type of C matches the ztype of
// accum.  That is, no typecasting can be done with C.

// This method is not used for the generic case with memcpy's and function
// pointers.  It is only used for pre-generated and JIT kernels.

// The matrix C is the user input matrix.  C is not iso on output, but might
// iso on input, in which case the input iso scalar is cinput, and C->x has
// been expanded to non-iso.  If A and/or B are hypersparse, the iso value of C
// has been expanded, so that C->x is initialized.  Otherwise, C->x is not
// initialized.  Instead, each entry is initialized by the iso value in
// the GB_GET4C(cij,p) macro.  A and/or B can be iso.

#define GB_DOT4

// cij += A(k,i) * B(k,j)
#undef  GB_DOT
#define GB_DOT(k,pA,pB)                                                     \
{                                                                           \
    GB_IF_TERMINAL_BREAK (cij, zterminal) ; /* break if cij == zterminal */ \
    GB_DECLAREA (aki) ;                                                     \
    GB_GETA (aki, Ax, pA, A_iso) ;          /* aki = A(k,i) */              \
    GB_DECLAREB (bkj) ;                                                     \
    GB_GETB (bkj, Bx, pB, B_iso) ;          /* bkj = B(k,j) */              \
    GB_MULTADD (cij, aki, bkj, i, k, j) ;   /* cij += aki * bkj */          \
}

{ 

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t cvlen = C->vlen ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t *restrict Bb = B->b ;
    const int64_t vlen = B->vlen ;
    const int64_t bvdim = B->vdim ;

    #ifdef GB_JIT_KERNEL
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_iso GB_B_ISO
    #else
    const bool B_is_hyper  = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_iso = B->iso ;
    const bool Bi_is_32 = B->i_is_32 ;
    #define GB_Bi_IS_32 Bi_is_32
    #endif

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t *restrict Ab = A->b ;
    const int64_t avdim = A->vdim ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (A->vdim == C->vlen) ;

    #ifdef GB_JIT_KERNEL
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_iso GB_A_ISO
    #else
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_iso = A->iso ;
    const bool Ai_is_32 = A->i_is_32 ;
    #define GB_Ai_IS_32 Ai_is_32
    #endif

    #if GB_IS_ANY_MONOID
    #error "dot4 not supported for ANY monoids"
    #endif

    GB_DECLARE_TERMINAL_CONST (zterminal) ;

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    // get the Cx [0] iso input scalar, if C was iso in GB_AxB_dot4
    #ifdef GB_JIT_KERNEL
    #define C_in_iso GB_C_IN_ISO
    #endif
    GB_DECLARE_IDENTITY_CONST (zidentity) ;
    const GB_C_TYPE cinput = (C_in_iso) ? Cx [0] : zidentity ;

    int ntasks = naslice * nbslice ;

    //--------------------------------------------------------------------------
    // C += A'*B
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    #define  GB_META16
    #include "include/GB_meta16_definitions.h"
    #include "template/GB_AxB_dot4_template.c"
    #else
    #include "template/GB_meta16_factory.c"
    #endif
}

#undef GB_DOT
#undef GB_DOT4

