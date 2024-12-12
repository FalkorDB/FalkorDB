//------------------------------------------------------------------------------
// GB_bitmap_assign_5_template: C bitmap, no M, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<>(I,J) += A            assign
// C(I,J)<> += A            subassign

// C<repl>(I,J) += A        assign
// C(I,J)<repl> += A        subassign

// C<!>(I,J) += A           assign: no work to do
// C(I,J)<!> += A           subassign: no work to do

// C<!,repl>(I,J) += A      assign: just clear C(I,J) of all entries
// C(I,J)<!,repl> += A      subassign: just clear C(I,J) of all entries
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign (all the same)

// If Mask_comp is true, then an empty mask is complemented.  This case has
// already been handled by GB_assign_prep, which calls GB_bitmap_assign with a
// scalar (which is unused).

// If C were full: entries can be deleted only if C_replace is true.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (!GB_MASK_COMP)
    {

        //----------------------------------------------------------------------
        // C(I,J) += A or += scalar
        //----------------------------------------------------------------------

        if (GB_SCALAR_ASSIGN)
        { 

            //------------------------------------------------------------------
            // scalar assignment: C(I,J) += scalar
            //------------------------------------------------------------------

            // for all entries in IxJ
            #define GB_IXJ_WORK(pC,ignore)                  \
            {                                               \
                int8_t cb = Cb [pC] ;                       \
                if (cb == 0)                                \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ;   \
                    Cb [pC] = 1 ;                           \
                    task_cnvals++ ;                         \
                }                                           \
                else                                        \
                {                                           \
                    /* Cx [pC] += scalar */                 \
                    GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;  \
                }                                           \
            }
            #include "template/GB_bitmap_assign_IxJ_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // matrix assignment: C(I,J) += A
            //------------------------------------------------------------------

            // for all entries aij in A (A hyper, sparse, bitmap, or full)
            //        if Cb(p) == 0
            //            Cx(p) = aij
            //            Cb(p) = 1       // C(iC,jC) is now present, insert
            //        else // if Cb(p) == 1:
            //            Cx(p) += aij    // C(iC,jC) still present, updated
            //            task_cnvals++

            #define GB_AIJ_WORK(pC,pA)                                  \
            {                                                           \
                int8_t cb = Cb [pC] ;                                   \
                if (cb == 0)                                            \
                {                                                       \
                    /* Cx [pC] = Ax [pA] */                             \
                    GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ; \
                    Cb [pC] = 1 ;                                       \
                    task_cnvals++ ;                                     \
                }                                                       \
                else                                                    \
                {                                                       \
                    /* Cx [pC] += Ax [pA] */                            \
                    GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork, C_iso) ;  \
                }                                                       \
            }
            #include "template/GB_bitmap_assign_A_template.c"
        }
    }

#if 0
    else if (C_replace)
    {

        //----------------------------------------------------------------------
        // This case is handled by GB_assign_prep and is thus not needed here.
        //----------------------------------------------------------------------

        // mask not present yet complemented: C_replace phase only

        // for row assign: for all entries in C(i,:)
        // for col assign: for all entries in C(:,j)
        // for assign: for all entries in C(:,:)
        // for subassign: for all entries in C(I,J)
        //      M not present; so effective value of the mask is mij==0
        //      set Cb(p) = 0

        #define GB_CIJ_WORK(pC)         \
        {                               \
            int8_t cb = Cb [pC] ;       \
            Cb [pC] = 0 ;               \
            task_cnvals -= (cb == 1) ;  \
        }
        #include "template/GB_bitmap_assign_C_template.c"
    }
#endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, no M, accum", GB0) ;
    return (GrB_SUCCESS) ;
}

