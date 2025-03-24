//------------------------------------------------------------------------------
// GB_bitmap_assign_6_template: C bitmap, no M, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
// I and J.  In that case, GB_assign_prep calls GB_clear instead of this
// method (see the dead code below).

// If C were full: entries can be deleted if C_replace is true,
// or if A is not full and missing at least one entry.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    {
        if (GB_ASSIGN_KIND == GB_ASSIGN)
        {
            // for assign: set all Cb(:,:) to zero
            // (this is currently dead code; see note above)
            GB_memset (Cb, 0, cnzmax, nthreads_max) ;
            cnvals = 0 ;
        }
        else
        { 
            // for row assign: set Cb(i,:) to zero
            // for col assign: set Cb(:,j) to zero
            // for subassign: set all Cb(I,J) to zero
            #define GB_CIJ_WORK(pC)                 \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = 0 ;                       \
                task_cnvals -= (cb == 1) ;          \
            }
            #define GB_NO_ASSIGN_CASE
            #include "template/GB_bitmap_assign_C_template.c"
            #undef GB_NO_ASSIGN_CASE
        }
    }

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (!GB_MASK_COMP)
    {

        if (GB_SCALAR_ASSIGN)
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
                GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                Cb [pC] = 1 ;                           \
                task_cnvals += (cb == 0) ;              \
            }
            #include "template/GB_bitmap_assign_IxJ_template.c"

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
                #include "template/GB_bitmap_assign_IxJ_template.c"
            }

            // for all entries aij in A (A hyper, sparse, bitmap, or full)
            //      Cx(p) = aij     // C(iC,jC) inserted or updated
            //      Cb(p) = 1

            #define GB_AIJ_WORK(pC,pA)                                      \
            {                                                               \
                int8_t cb = Cb [pC] ;                                       \
                /* Cx [pC] = Ax [pA] */                                     \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ;    \
                Cb [pC] = 1 ;                                               \
            }
            #include "template/GB_bitmap_assign_A_template.c"

            GB_A_NVALS (anz) ;
            cnvals += anz ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign: noM, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

