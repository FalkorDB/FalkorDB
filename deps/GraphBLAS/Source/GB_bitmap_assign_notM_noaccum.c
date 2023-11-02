//------------------------------------------------------------------------------
// GB_bitmap_assign_notM_noaccum:  assign to C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<!M>(I,J) = A       assign
// C(I,J)<!M> = A       subassign

// C<!M,repl>(I,J) = A       assign
// C(I,J)<!M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, hypersparse or sparse (not bitmap or full)
// Mask_comp:   true
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// JIT: needed.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

GrB_Info GB_bitmap_assign_notM_noaccum
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
    const GrB_Matrix M,         // mask matrix
//  const bool Mask_comp,       // true here, for !M only
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

    GBURBLE_BITMAP_ASSIGN ("bit8", M, true, NULL,
        Ikind, Jkind, assign_kind) ;
    ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, !M, noaccum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, !M, noaccum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, !M, noaccum", GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_BITMAP ;           // C must be bitmap
    GB_SLICE_M
    GB_GET_A_AND_SCALAR

    //--------------------------------------------------------------------------
    // scatter M into the bitmap of C
    //--------------------------------------------------------------------------

    // Cb [pC] += 2 for each entry M(i,j) in the mask
    GB_bitmap_M_scatter (C, I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
        M, Mask_struct, assign_kind, GB_BITMAP_M_SCATTER_PLUS_2,
        M_ek_slicing, M_ntasks, M_nthreads) ;

    // Cb (i,j) = 0:   cij not present, mij zero: can be modified
    // Cb (i,j) = 1:   cij present, mij zero: can be modified,
    //                      but delete if aij not present
    // Cb (i,j) = 2:   cij not present, mij == 1: do not modify
    // Cb (i,j) = 3:   cij present, mij == 1: do not modify

    //--------------------------------------------------------------------------
    // assign A into C
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 

        //----------------------------------------------------------------------
        // scalar assignment: C<!M>(I,J) = scalar
        //----------------------------------------------------------------------

        // for all IxJ
        #define GB_IXJ_WORK(pC,ignore)                  \
        {                                               \
            int8_t cb = Cb [pC] ;                       \
            if (cb <= 1)                                \
            {                                           \
                /* Cx [pC] = scalar */                  \
                GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
                Cb [pC] = 1 ;                           \
                task_cnvals += (cb == 0) ;              \
            }                                           \
            else if (C_replace)                         \
            {                                           \
                /* delete this entry */                 \
                Cb [pC] = 0 ;                           \
                task_cnvals -= (cb == 3) ;              \
            }                                           \
            else                                        \
            {                                           \
                /* keep this entry */                   \
                Cb [pC] = (cb == 3) ;                   \
            }                                           \
        }
        #include "GB_bitmap_assign_IxJ_template.c"

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<!M>(I,J) = A
        //----------------------------------------------------------------------

        // for all entries aij in A (A can be hyper, sparse, bitmap, or full)
        //     if Cb(p) == 0       // C(iC,jC) is now present, insert
        //         Cx(p) = aij     //
        //         Cb(p) = 4       // keep it
        //         task_cnvals++
        //     if Cb(p) == 1       // C(iC,jC) still present, updated
        //         Cx(p) = aij     //
        //         Cb(p) = 4       // keep it
        //     if Cb(p) == 2       // do nothing
        //     if Cb(p) == 3       // do nothing

        #define GB_AIJ_WORK(pC,pA)                                  \
        {                                                           \
            int8_t cb = Cb [pC] ;                                   \
            if (cb <= 1)                                            \
            {                                                       \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                Cb [pC] = 4 ;                                       \
                task_cnvals += (cb == 0) ;                          \
            }                                                       \
        }
        #include "GB_bitmap_assign_A_template.c"

        //----------------------------------------------------------------------
        // handle entries in IxJ
        //----------------------------------------------------------------------

        if (C_replace)
        { 
            // for all IxJ
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC,ignore)              \
            {                                           \
                int8_t cb = Cb [pC] ;                   \
                Cb [pC] = (cb == 4) ;                   \
                task_cnvals -= (cb == 1 || cb == 3) ;   \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        else
        { 
            // for all IxJ
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC,ignore)              \
            {                                           \
                int8_t cb = Cb [pC] ;                   \
                Cb [pC] = (cb == 4 || cb == 3) ;        \
                task_cnvals -= (cb == 1) ;              \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // handle entries outside of IxJ
    //--------------------------------------------------------------------------

    if (assign_kind == GB_SUBASSIGN)
    { 
        // see above.  no more work to do
    }
    else
    {
        #define GB_NO_SUBASSIGN_CASE
        if (C_replace)
        { 
            // for all entries in C.  Also clears M from C
            #define GB_CIJ_WORK(pC)                 \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = (cb == 1) ;               \
                task_cnvals -= (cb == 3) ;          \
            }
            #include "GB_bitmap_assign_C_template.c"
        }
        else
        { 
            // clear M from C
            // Cb [pC] %= 2 for each entry M(i,j) in the mask
            GB_bitmap_M_scatter (C, I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, Mask_struct, assign_kind, GB_BITMAP_M_SCATTER_MOD_2,
                M_ek_slicing, M_ntasks, M_nthreads) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, !M, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

