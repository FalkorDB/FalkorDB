//------------------------------------------------------------------------------
// GB_emult_08_meta:  phase1 and phase2 for C=A.*B, C<M>=A.*B, C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C=A.*B, C<M>=A.*B, or C<!M>=A.*B. 

// C is sparse or hypersparse.  M, A, and B can have any sparsity structure.
// If both A and B are full, then GB_add is used instead (this is the only case
// where C can be full).

// phase1: does not compute C itself, but just counts the # of entries in each
// vector of C.  Fine tasks compute the # of entries in their slice of a
// single vector of C, and the results are cumsum'd.

// phase2: computes C, using the counts computed by phase1.

{

    // iB_first is unused if the operator is FIRST or PAIR
    #include "GB_unused.h"

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;

    #ifdef GB_JIT_KENEL
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_full   GB_A_IS_FULL
    #else
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_full = GB_IS_FULL (A) ;
    #endif

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t *restrict Bi = B->i ;

    #ifdef GB_JIT_KENEL
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_full   GB_B_IS_FULL
    #else
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_full = GB_IS_FULL (B) ;
    #endif

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int8_t  *restrict Mb = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;

    #ifdef GB_JIT_KENEL
    #define M_is_hyper  GB_M_IS_HYPER
    #define M_is_sparse GB_M_IS_SPARSE
    #define M_is_bitmap GB_M_IS_BITMAP
    #define M_is_full   GB_M_IS_FULL
    #define M_is_sparse_or_hyper (GB_M_IS_SPARSE || GB_M_IS_HYPER)
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #define M_is_present (!GB_NO_MASK)
    #else
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    const bool M_is_present = (M != NULL) ;
    #endif

    size_t msize = 0 ;
    if (M_is_present)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    #ifndef GB_EMULT_08_PHASE
    #define GB_EMULT_08_PHASE 2
    #endif

    #if ( GB_EMULT_08_PHASE == 2 )
    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;
    #endif
    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif
    const int64_t  *restrict Cp = C->p ;
    const int64_t  *restrict Ch = C->h ;
          int64_t  *restrict Ci = C->i ;
    #endif

    //--------------------------------------------------------------------------
    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B: C is sparse or hypersparse
    //--------------------------------------------------------------------------

    #if ( GB_EMULT_08_PHASE == 1 )
    { 
        // phase1: symbolic phase
        #include "GB_emult_08_template.c"
    }
    #else
    { 
        // phase2: numerical phase
        #include "GB_emult_08_template.c"
    }
    #endif
}

