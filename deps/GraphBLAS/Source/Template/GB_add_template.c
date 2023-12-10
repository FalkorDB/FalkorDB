//------------------------------------------------------------------------------
// GB_add_template:  phase1 and phase2 for C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C=A+B, C<M>=A+B, or C<!M>=A+B, for eWiseAdd or eWiseUnion.

// phase1: does not compute C itself, but just counts the # of entries in each
// vector of C.  Fine tasks compute the # of entries in their slice of a
// single vector of C, and the results are cumsum'd.

// phase2: computes C, using the counts computed by phase1.

// for eWiseUnion:
//      #define GB_IS_EWISEUNION 1
//      if A(i,j) is not present: C(i,j) = alpha + B(i,j)
//      if B(i,j) is not present: C(i,j) = A(i,j) + beta
// for eWiseAdd:
//      #define GB_IS_EWISEUNION 0
//      if A(i,j) is not present: C(i,j) = B(i,j)
//      if B(i,j) is not present: C(i,j) = A(i,j)

{

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    int taskid ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;

    #ifdef GB_JIT_KERNEL
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_full   GB_A_IS_FULL
    #define A_iso       GB_A_ISO
    #else
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_full = GB_IS_FULL (A) ;
    // unlike GB_emult, both A and B may be iso
    const bool A_iso = A->iso ;
    #endif

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t *restrict Bi = B->i ;

    #ifdef GB_JIT_KERNEL
    #define B_is_hyper  GB_B_IS_HYPER
    #define B_is_sparse GB_B_IS_SPARSE
    #define B_is_bitmap GB_B_IS_BITMAP
    #define B_is_full   GB_B_IS_FULL
    #define B_iso       GB_B_ISO
    #else
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_full = GB_IS_FULL (B) ;
    const bool B_iso = B->iso ;
    #endif

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int8_t  *restrict Mb = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;
    size_t msize = 0 ;

    #ifdef GB_JIT_KERNEL
    #define M_is_hyper  GB_M_IS_HYPER
    #define M_is_sparse GB_M_IS_SPARSE
    #define M_is_bitmap GB_M_IS_BITMAP
    #define M_is_full   GB_M_IS_FULL
    #define M_is_sparse_or_hyper (GB_M_IS_SPARSE || GB_M_IS_HYPER)
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #else
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    #endif

    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    //--------------------------------------------------------------------------
    // phase 2 definitions
    //--------------------------------------------------------------------------

    // phase 1 is used only by GB_add_phase1, and it does not depend on the
    // data type or operator, so there is only one copy of that method.  phase
    // 2 is used by GB_add_phase2, via the factory kernels, the JIT kernels,
    // and the generic kernel.

    #ifndef GB_ADD_PHASE
    #define GB_ADD_PHASE 2
    #endif

    #if ( GB_ADD_PHASE == 2 )

        #ifdef GB_JIT_KERNEL
        ASSERT (!C->iso) ;
        #endif

        #ifdef GB_ISO_ADD
        ASSERT (C->iso) ;
        #else
        const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
        const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
              GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
        ASSERT (!C->iso) ;
        #endif

        const int64_t  *restrict Cp = C->p ;
        const int64_t  *restrict Ch = C->h ;
              int8_t   *restrict Cb = C->b ;
              int64_t  *restrict Ci = C->i ;
        GB_C_NHELD (cnz) ;      // const int64_t cnz = GB_nnz_held (C) ;

    #endif

    //--------------------------------------------------------------------------
    // C=A+B, C<M>=A+B, or C<!M>=A+B: 3 cases for the sparsity of C
    //--------------------------------------------------------------------------

    #if ( GB_ADD_PHASE == 1 )

        // phase1: symbolic phase
        // C is sparse or hypersparse (never bitmap or full)
        #include "GB_add_sparse_template.c"

    #else

        // phase2: numerical phase

        #ifdef GB_POSITIONAL_OP
            // op doesn't depend aij, bij, alpha_scalar, or beta_scalar
            #define GB_LOAD_A(aij, Ax,pA,A_iso)
            #define GB_LOAD_B(bij, Bx,pB,B_iso)
        #else
            #define GB_LOAD_A(aij, Ax,pA,A_iso) \
                GB_DECLAREA (aij) ;             \
                GB_GETA (aij, Ax,pA,A_iso)
            #define GB_LOAD_B(bij, Bx,pB,B_iso) \
                GB_DECLAREB (bij) ;             \
                GB_GETB (bij, Bx,pB,B_iso)
        #endif

        #ifdef GB_JIT_KERNEL
        {
            #if GB_C_IS_SPARSE || GB_C_IS_HYPER
            {
                #include "GB_add_sparse_template.c"
            }
            #elif GB_C_IS_BITMAP
            {
                #include "GB_add_bitmap_template.c"
            }
            #else
            {
                #include "GB_add_full_template.c"
            }
            #endif
        }
        #else
        {
            if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
            { 
                // C is sparse or hypersparse
                #include "GB_add_sparse_template.c"
            }
            else if (C_sparsity == GxB_BITMAP)
            { 
                // C is bitmap (phase2 only)
                #include "GB_add_bitmap_template.c"
            }
            else
            { 
                // C is full (phase2 only), and not iso
                ASSERT (C_sparsity == GxB_FULL) ;
                #include "GB_add_full_template.c"
            }
        }
        #endif

    #endif
}

#undef GB_ISO_ADD
#undef GB_LOAD_A
#undef GB_LOAD_B
#undef GB_IS_EWISEUNION


