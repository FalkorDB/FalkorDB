//------------------------------------------------------------------------------
// GB_bitmap_assign_7_template: C bitmap, !M sparse/hyper, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<!M>(I,J) += A       assign
// C(I,J)<!M> += A       subassign

// C<!M,repl>(I,J) += A       assign
// C(I,J)<!M,repl> += A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, hypersparse or sparse (not bitmap or full)
// Mask_comp:   true
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// If C were full: entries can be deleted only if C_replace is true.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_SLICE_M_FOR_BITMAP
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // scatter the mask M into C
    //--------------------------------------------------------------------------

    // Cb [pC] += 2 for each entry M(i,j) in the mask
    #undef  GB_MASK_WORK
    #define GB_MASK_WORK(pC) Cb [pC] += 2
    #define GB_NO_CNVALS
    #include "template/GB_bitmap_assign_M_template.c"
    #undef  GB_NO_CNVALS

    // the bitmap of C now contains:
    //    Cb (i,j) = 0:  mij == 0, cij not present, can be modified
    //    Cb (i,j) = 1:  mij == 0, cij present, can be modified
    //    Cb (i,j) = 2:  mij == 1, cij not present, do not modify
    //    Cb (i,j) = 3:  mij == 1, cij present, do not modify

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    { 

        //----------------------------------------------------------------------
        // scalar assignment: C<!M>(I,J) += scalar
        //----------------------------------------------------------------------

        // for all IxJ
        #define GB_IXJ_WORK(pC,ignore)                  \
        {                                               \
            int8_t cb = Cb [pC] ;                       \
            if (cb == 0)                                \
            {                                           \
                /* Cx [pC] = scalar  */                 \
                GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                Cb [pC] = 1 ;                           \
                task_cnvals++ ;                         \
            }                                           \
            else if (cb == 1)                           \
            {                                           \
                /* Cx [pC] += scalar */                 \
                GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;  \
            }                                           \
        }
        #include "template/GB_bitmap_assign_IxJ_template.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // matrix assignment: C<!M>(I,J) += A
        //----------------------------------------------------------------------

        // for all entries aij in A (A can be hyper, sparse, bitmap, or full)
        //     if Cb(p) == 0
        //         Cx(p) = aij
        //         Cb(p) = 1       // C(iC,jC) is now present, insert
        //         task_cnvals++
        //     if Cb(p) == 1
        //         Cx(p) += aij    // C(iC,jC) still present, updated
        //         Cb(p) still 1
        //     if Cb(p) == 2       // do nothing
        //     if Cb(p) == 3       // do nothing

        #define GB_AIJ_WORK(pC,pA)                                  \
        {                                                           \
            int8_t cb = Cb [pC] ;                                   \
            if (cb == 0)                                            \
            {                                                       \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ;   \
                Cb [pC] = 1 ;                                       \
                task_cnvals++ ;                                     \
            }                                                       \
            else if (cb == 1)                                       \
            {                                                       \
                /* Cx [pC] += Ax [pA] */                            \
                GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork, C_iso) ;  \
            }                                                       \
        }
        #include "template/GB_bitmap_assign_A_template.c"
    }

    //--------------------------------------------------------------------------
    // clear M from C and handle the C_replace phase
    //--------------------------------------------------------------------------

    if (!C_replace)
    { 
        // for each entry mij == 1
                // 2 -> 0
                // 3 -> 1       keep this entry
        // Cb [pC] -= 2 for each entry M(i,j) in the mask
        #undef  GB_MASK_WORK
        #define GB_MASK_WORK(pC) Cb [pC] -= 2
        #define GB_NO_CNVALS
        #include "template/GB_bitmap_assign_M_template.c"
        #undef  GB_NO_CNVALS
    }
    else
    { 
        // for each entry mij == 1
                // 2 -> 0
                // 3 -> 0       delete this entry
        #undef  GB_MASK_WORK
        #define GB_MASK_WORK(pC)                \
        {                                       \
            int8_t cb = Cb [pC] ;               \
            task_cnvals -= (cb == 3) ;          \
            Cb [pC] = 0 ;                       \
        }
        #include "template/GB_bitmap_assign_M_template.c"
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, !M, accum", GB0) ;
    return (GrB_SUCCESS) ;
}

