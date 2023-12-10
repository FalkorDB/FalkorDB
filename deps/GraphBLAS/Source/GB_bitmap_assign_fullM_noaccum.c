//------------------------------------------------------------------------------
// GB_bitmap_assign_fullM_noaccum:  assign to C bitmap, M is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M>(I,J) = A       assign
// C(I,J)<M> = A       subassign

// C<M,repl>(I,J) = A       assign
// C(I,J)<M,repl> = A       subassign

// C<!M>(I,J) = A       assign
// C(I,J)<!M> = A       subassign

// C<!M,repl>(I,J) = A       assign
// C(I,J)<!M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// JIT: needed.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_bitmap_assign_fullM_noaccum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    const bool C_replace,       // descriptor for C
    // inputs:
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
//  const GrB_BinaryOp accum,   // not present
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

    GBURBLE_BITMAP_ASSIGN ("bit2", M, Mask_comp, NULL,
        Ikind, Jkind, assign_kind) ;
    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M full, noaccum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M full, noaccum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, M full, noaccum", GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_BITMAP ;           // C must be bitmap
    GB_GET_M
    GB_GET_A_AND_SCALAR

    //--------------------------------------------------------------------------
    // to get the effective value of the mask entry mij
    //--------------------------------------------------------------------------

    #undef  GB_GET_MIJ
    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ Mask_comp ;

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    { 
        // if C FULL: use two passes: first pass checks if any
        // entry must be deleted.  If none: do nothing.  Else:  change C
        // to full and do 2nd pass as below.

        // for row assign: set Cb(i,:) to zero if mij == 0
        // for col assign: set Cb(:,j) to zero if mij == 0
        // for assign: set Cb(:,:) to zero if mij == 0
        // for subassign set Cb(I,J) to zero if mij == 0
        #undef  GB_CIJ_WORK
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
    // assignment phase
    //--------------------------------------------------------------------------

    if (A == NULL)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M>(I,J) = scalar
        //----------------------------------------------------------------------

        // if C FULL: no change, just cb = GBB (CB,pC)

        // for all entries in IxJ
        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)                      \
        {                                               \
            int64_t pM = GB_GET_pM ;                    \
            GB_GET_MIJ (mij, pM) ;                      \
            if (mij)                                    \
            {                                           \
                int8_t cb = Cb [pC] ;                   \
                /* Cx [pC] = scalar */                  \
                GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
                Cb [pC] = 1 ;                           \
                task_cnvals += (cb == 0) ;              \
            }                                           \
        }

        ASSERT (assign_kind == GB_ASSIGN || assign_kind == GB_SUBASSIGN) ;

        switch (assign_kind)
        {
            case GB_ASSIGN : 
                // C<M>(I,J) = scalar where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> = scalar where M has the same size as A
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
        // matrix assignment: C<M or !M>(I,J) = A
        //----------------------------------------------------------------------

        // assign A into C:

            //  for all entries aij in A
            //      get the effective value of the mask:
            //          for row assign: get mij = m(jC,0)
            //          for col assign: get mij = m(iC,0)
            //          for assign: get mij = M(iC,jC)
            //          for subassign: get mij = M(i,j)
            //          if complemented: mij = !mij
            //      if mij == 1:
            //          Cx(p) = aij     // C(iC,jC) inserted or updated
            //          Cb(p) = 4

        // clear entries from C that were not in A:

            // for all entries in IxJ
                // get the effective value of the mask
                // if mij == 1
                    // 0 -> 0
                    // 1 -> 0           delete because aij not present
                    // 4 -> 1

        // TODO: if A is bitmap or full, use a single pass

        #define GB_AIJ_WORK(pC,pA)                                  \
        {                                                           \
            int64_t pM = GB_GET_pM ;                                \
            GB_GET_MIJ (mij, pM) ;                                  \
            if (mij)                                                \
            {                                                       \
                int8_t cb = Cb [pC] ;                               \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                Cb [pC] = 4 ;                                       \
                task_cnvals += (cb == 0) ;                          \
            }                                                       \
        }

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)          \
        {                                   \
            int64_t pM = GB_GET_pM ;        \
            GB_GET_MIJ (mij, pM) ;          \
            if (mij)                        \
            {                               \
                int8_t cb = Cb [pC] ;       \
                Cb [pC] = (cb > 1) ;        \
                task_cnvals -= (cb == 1) ;  \
            }                               \
        }

        switch (assign_kind)
        {
            case GB_ROW_ASSIGN : 
                // C<m>(i,J) = A where m is a 1-by-C->vdim row vector
                #undef  GB_GET_pM
                #define GB_GET_pM jC
                #include "GB_bitmap_assign_A_template.c"
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_COL_ASSIGN : 
                // C<m>(I,j) = A where m is a C->vlen-by-1 column vector
                #undef  GB_GET_pM
                #define GB_GET_pM iC
                #include "GB_bitmap_assign_A_template.c"
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_ASSIGN : 
                // C<M>(I,J) = A where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "GB_bitmap_assign_A_template.c"
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_SUBASSIGN : 
                // C(I,J)<M> = A where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM (iA + jA * nI)
                #include "GB_bitmap_assign_A_template.c"
                #include "GB_bitmap_assign_IxJ_template.c"
                break ;

            default: ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, M full, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

