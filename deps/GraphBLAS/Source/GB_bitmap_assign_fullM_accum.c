//------------------------------------------------------------------------------
// GB_bitmap_assign_fullM_accum:  assign to C bitmap, M is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M>(I,J) += A           assign
// C(I,J)<M> += A           subassign

// C<M,repl>(I,J) += A      assign
// C(I,J)<M,repl> += A      subassign

// C<!M>(I,J) += A          assign
// C(I,J)<!M> += A          subassign

// C<!M,repl>(I,J) += A     assign
// C(I,J)<!M,repl> += A     subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// JIT: needed.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_bitmap_assign_fullM_accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
    const GrB_Index *I,         // I index list
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,         // mask matrix, which is present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBURBLE_BITMAP_ASSIGN ("bit1", M, Mask_comp, accum,
        Ikind, Jkind, assign_kind) ;
    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M full, accum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M full, accum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, M full, accum", GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_BITMAP ;           // C must be bitmap
    GB_GET_M
    GB_GET_A_AND_SCALAR
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // to get the effective value of the mask entry mij
    //--------------------------------------------------------------------------

    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ Mask_comp ;

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (A == NULL)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M>(I,J) += scalar
        //----------------------------------------------------------------------

        // for all IxJ
        //  get the effective value of the mask, via GB_GET_MIJ:
        //      for row assign: get mij = m(jC,0)
        //      for col assign: get mij = m(iC,0)
        //      for assign: get mij = M(iC,jC)
        //      for subassign: get mij = M(i,j)
        //      if complemented: mij = !mij
        //  if mij == 1:
        //      if Cb(p) == 0
        //          Cx(p) = scalar
        //          Cb(p) = 1       // C(iC,jC) is now present, insert
        //      else // if Cb(p) == 1:
        //          Cx(p) += scalar // C(iC,jC) still present, updated

        // if C FULL: no change, just cb = GBB (Cb,pC)

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)                          \
        {                                                   \
            int64_t pM = GB_GET_pM ;                        \
            GB_GET_MIJ (mij, pM) ;                          \
            if (mij)                                        \
            {                                               \
                int8_t cb = Cb [pC] ;                       \
                if (cb == 0)                                \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
                    Cb [pC] = 1 ;                           \
                    task_cnvals++ ;                         \
                }                                           \
                else /* (cb == 1) */                        \
                {                                           \
                    /* Cx [pC] += scalar */                 \
                    GB_ACCUMULATE_scalar (Cx, pC, ywork) ;  \
                }                                           \
            }                                               \
        }

        ASSERT (assign_kind == GB_ASSIGN || assign_kind == GB_SUBASSIGN) ;

        switch (assign_kind)
        {
            case GB_ASSIGN : 
                // C<M>(I,J) += scalar where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> += scalar where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM pA
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;
            default: ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<M or !M>(I,J) += A
        //----------------------------------------------------------------------

        // for all entries aij in A (A can be hyper, sparse, bitmap, or full)
        //     get the effective value of the mask, via GB_GET_MIJ:
        //         for row assign: get mij = m(jC,0)
        //         for col assign: get mij = m(iC,0)
        //         for assign: get mij = M(iC,jC)
        //         for subassign: get mij = M(i,j)
        //         if complemented: mij = !mij
        //     if mij == 1:
        //         if Cb(p) == 0
        //             Cx(p) = aij
        //             Cb(p) = 1       // C(iC,jC) is now present, insert
        //             task_cnvals++
        //         else // if Cb(p) == 1:
        //             Cx(p) += aij    // C(iC,jC) still present, updated

        // if C FULL: no change, just cb = GBB (Cb,pC)

        #define GB_AIJ_WORK(pC,pA)                                      \
        {                                                               \
            int64_t pM = GB_GET_pM ;                                    \
            GB_GET_MIJ (mij, pM) ;                                      \
            if (mij)                                                    \
            {                                                           \
                int8_t cb = Cb [pC] ;                                   \
                if (cb == 0)                                            \
                {                                                       \
                    /* Cx [pC] = Ax [pA] */                             \
                    GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                    Cb [pC] = 1 ;                                       \
                    task_cnvals++ ;                                     \
                }                                                       \
                else /* (cb == 1) */                                    \
                {                                                       \
                    /* Cx [pC] += Ax [pA] */                            \
                    GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork) ;  \
                }                                                       \
            }                                                           \
        }

        switch (assign_kind)
        {
            case GB_ROW_ASSIGN : 
                // C<m>(i,J) += A where m is a 1-by-C->vdim row vector
                #undef  GB_GET_pM
                #define GB_GET_pM jC
                #include "GB_bitmap_assign_A_template.c"
                break ;
            case GB_COL_ASSIGN : 
                // C<m>(I,j) += A where m is a C->vlen-by-1 column vector
                #undef  GB_GET_pM
                #define GB_GET_pM iC
                #include "GB_bitmap_assign_A_template.c"
                break ;
            case GB_ASSIGN : 
                // C<M>(I,J) += A where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "GB_bitmap_assign_A_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> += A where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM (iA + jA * nI)
                #include "GB_bitmap_assign_A_template.c"
                break ;
            default: ;
        }
    }

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    { 
        // if C FULL: use two passes: first pass checks if any
        // entry must be deleted.  If none: do nothing.  Else:  change C
        // to full and do 2nd pass as below.

        // for row assign: for all entries in C(i,:)
        // for col assign: for all entries in C(:,j)
        // for assign: for all entries in C(:,:)
        // for subassign: for all entries in C(I,J)
        //      get effective value mij of the mask via GB_GET_MIJ
        //      if mij == 0 set Cb(p) = 0
        #define GB_CIJ_WORK(pC)             \
        {                                   \
            if (!mij)                       \
            {                               \
                int8_t cb = Cb [pC] ;       \
                Cb [pC] = 0 ;               \
                task_cnvals -= (cb == 1) ;  \
            }                               \
        }
        #include "GB_bitmap_assign_C_template.c"
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, M full, accum", GB0) ;
    return (GrB_SUCCESS) ;
}

