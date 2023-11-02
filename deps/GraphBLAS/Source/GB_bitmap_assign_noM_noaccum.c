//------------------------------------------------------------------------------
// GB_bitmap_assign_noM_noaccum:  assign to C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<>(I,J) = A             assign
// C(I,J)<> = A             subassign

// C<repl>(I,J) = A         assign
// C(I,J)<repl> = A         subassign

// C<!>(I,J) = A            assign
// C(I,J)<!> = A            subassign

// C<!,repl>(I,J) = A       assign
// C(I,J)<!,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   true or false
// Mask_struct: true or false (ignored)
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// If M is not present and Mask_comp is true, then an empty mask is
// complemented.  This case is handled by GB_assign_prep by calling this
// method with no matrix A, but with a scalar (which is unused).  However,
// for GB_ASSIGN, C<!,replace>(I,J)=anything clears all of C, regardless of
// I and J.  In that case, GB_assign_prep calls GB_clear instead.

// JIT: needed.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_bitmap_assign_noM_noaccum
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
//  const GrB_Matrix M,         // mask matrix, not present here
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

    GBURBLE_BITMAP_ASSIGN ("bit6", NULL, Mask_comp, NULL,
        Ikind, Jkind, assign_kind) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign: noM, noaccum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign: noM, noaccum", GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_BITMAP ;           // C must be bitmap
    GB_GET_A_AND_SCALAR

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    {
        if (assign_kind == GB_ASSIGN)
        { 
            // for assign: set all Cb(:,:) to zero
            GB_memset (Cb, 0, cnzmax, nthreads_max) ;
            cnvals = 0 ;
        }
        else
        { 
            // for row assign: set Cb(i,:) to zero
            // for col assign: set Cb(:,j) to zero
            // for subassign: set all Cb(I,J) to zero
            #define NO_ASSIGN_CASE
            #define GB_CIJ_WORK(pC)                 \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = 0 ;                       \
                task_cnvals -= (cb == 1) ;          \
            }
            #include "GB_bitmap_assign_C_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (!Mask_comp)
    {

        if (A == NULL)
        { 

            //------------------------------------------------------------------
            // scalar assignment: C(I,J) = scalar
            //------------------------------------------------------------------

            // for all IxJ
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC,ignore)              \
            {                                           \
                int8_t cb = Cb [pC] ;                   \
                /* Cx [pC] = scalar */                  \
                GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
                Cb [pC] = 1 ;                           \
                task_cnvals += (cb == 0) ;              \
            }
            #include "GB_bitmap_assign_IxJ_template.c"

        }
        else
        {

            //------------------------------------------------------------------
            // matrix assignment: C(I,J) = A
            //------------------------------------------------------------------

            if (!C_replace)
            { 
                // delete all entries in C(I,J)
                #undef  GB_IXJ_WORK
                #define GB_IXJ_WORK(pC,ignore)          \
                {                                       \
                    int8_t cb = Cb [pC] ;               \
                    Cb [pC] = 0 ;                       \
                    task_cnvals -= (cb == 1) ;          \
                }
                #include "GB_bitmap_assign_IxJ_template.c"
            }

            // for all entries aij in A (A hyper, sparse, bitmap, or full)
            //      Cx(p) = aij     // C(iC,jC) inserted or updated
            //      Cb(p) = 1

            #define GB_AIJ_WORK(pC,pA)                              \
            {                                                       \
                int8_t cb = Cb [pC] ;                               \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                Cb [pC] = 1 ;                                       \
            }
            #include "GB_bitmap_assign_A_template.c"

            cnvals += GB_nnz (A) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign: noM, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

