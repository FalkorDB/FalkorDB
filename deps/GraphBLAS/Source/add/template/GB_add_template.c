//------------------------------------------------------------------------------
// GB_add_template:  phase1 and phase2 for C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int8_t *restrict Ab = A->b ;
    const int64_t vlen = A->vlen ;

    #ifdef GB_JIT_KERNEL
    #define A_is_bitmap GB_A_IS_BITMAP
    #else
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    // unlike GB_emult, both A and B may be iso
    const bool Ai_is_32 = A->i_is_32 ;
    #define GB_Ai_IS_32 Ai_is_32
    #endif

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const int8_t *restrict Bb = B->b ;

    #ifdef GB_JIT_KERNEL
    #define B_is_bitmap GB_B_IS_BITMAP
    #else
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool Bi_is_32 = B->i_is_32 ;
    #define GB_Bi_IS_32 Bi_is_32
    #endif

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    GB_Mi_DECLARE (Mi, const) ; GB_Mi_PTR (Mi, M) ;
    const int8_t *restrict Mb = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;
    size_t msize = 0 ;

    #ifdef GB_JIT_KERNEL
    #define M_is_hyper  GB_M_IS_HYPER
    #define M_is_sparse GB_M_IS_SPARSE
    #define M_is_sparse_or_hyper (GB_M_IS_SPARSE || GB_M_IS_HYPER)
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #else
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    #endif

    if (M != NULL)
    { 
        Mb = M->b ;
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
        #define A_is_full   GB_A_IS_FULL
        #define B_is_full   GB_B_IS_FULL
        #else
        const bool A_is_full = GB_IS_FULL (A) ;
        const bool B_is_full = GB_IS_FULL (B) ;
        #endif

        #ifdef GB_ISO_ADD
        ASSERT (C->iso) ;
        #else
        const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
        const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
              GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
        ASSERT (!C->iso) ;
        #ifdef GB_JIT_KERNEL
        #define A_iso GB_A_ISO
        #define B_iso GB_B_ISO
        #else
        const bool A_iso = A->iso ;
        const bool B_iso = B->iso ;
        #endif
        #endif

        GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
        GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
        GB_Mh_DECLARE (Mh, const) ; GB_Mh_PTR (Mh, M) ;

        GB_Cp_DECLARE (Cp, const) ; GB_Cp_PTR (Cp, C) ;
        GB_Ch_DECLARE (Ch, const) ; GB_Ch_PTR (Ch, C) ;
        GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;
        int8_t *restrict Cb = C->b ;
        GB_C_NHELD (cnz) ;      // const int64_t cnz = GB_nnz_held (C) ;

    #endif


    //--------------------------------------------------------------------------
    // C=A+B, C<M>=A+B, or C<!M>=A+B: 3 cases for the sparsity of C
    //--------------------------------------------------------------------------

    #if ( GB_ADD_PHASE == 1 )

        // phase1: symbolic phase
        // C is sparse or hypersparse (never bitmap or full)
        #include "template/GB_add_sparse_template.c"

    #else

        // phase2: numerical phase

        #ifdef GB_BUILTIN_POSITIONAL_OP
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
                #include "template/GB_add_sparse_template.c"
            }
            #elif GB_C_IS_BITMAP
            {
                #include "template/GB_add_bitmap_template.c"
            }
            #else
            {
                #include "template/GB_add_full_template.c"
            }
            #endif
        }
        #else
        {
            if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
            { 
                // C is sparse or hypersparse
                #include "template/GB_add_sparse_template.c"
            }
            else if (C_sparsity == GxB_BITMAP)
            { 
                // C is bitmap (phase2 only)
                #include "template/GB_add_bitmap_template.c"
            }
            else
            { 
                // C is full (phase2 only), and not iso
                ASSERT (C_sparsity == GxB_FULL) ;
                #include "template/GB_add_full_template.c"
            }
        }
        #endif

    #endif
}

#undef GB_ISO_ADD
#undef GB_LOAD_A
#undef GB_LOAD_B
#undef GB_IS_EWISEUNION

