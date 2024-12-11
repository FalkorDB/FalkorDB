//------------------------------------------------------------------------------
// GB_bitmap_assign_8_template: C bitmap, !M sparse/hyper, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
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

// If C were full: entries can be deleted if C_replace is true,
// or if A is not full and missing at least one entry.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_SLICE_M_FOR_BITMAP

    //--------------------------------------------------------------------------
    // scatter M into the bitmap of C
    //--------------------------------------------------------------------------

    // Cb [pC] += 2 for each entry M(i,j) in the mask
    #undef  GB_MASK_WORK
    #define GB_MASK_WORK(pC) Cb [pC] += 2
    #define GB_NO_CNVALS
    #include "template/GB_bitmap_assign_M_template.c"
    #undef  GB_NO_CNVALS

    // the bitmap of C now contains:
    //    Cb (i,j) = 0:  mij == 0, cij not present, do not modify
    //    Cb (i,j) = 1:  mij == 0, cij present, do not modify
    //    Cb (i,j) = 2:  mij == 1, cij not present, can be modified
    //    Cb (i,j) = 3:  mij == 1, cij present, can be modified

    //--------------------------------------------------------------------------
    // assign A into C
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
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
                GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ;   \
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
        #include "template/GB_bitmap_assign_IxJ_template.c"

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
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ;   \
                Cb [pC] = 4 ;                                       \
                task_cnvals += (cb == 0) ;                          \
            }                                                       \
        }
        #include "template/GB_bitmap_assign_A_template.c"

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
            #include "template/GB_bitmap_assign_IxJ_template.c"
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
            #include "template/GB_bitmap_assign_IxJ_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // handle entries outside of IxJ
    //--------------------------------------------------------------------------

    if (GB_ASSIGN_KIND == GB_SUBASSIGN)
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
            #include "template/GB_bitmap_assign_C_template.c"
        }
        else
        { 
            // clear M from C
            // Cb [pC] %= 2 for each entry M(i,j) in the mask
            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] &= 1
            #define GB_NO_CNVALS
            #include "template/GB_bitmap_assign_M_template.c"
            #undef  GB_NO_CNVALS

        }
        #undef GB_NO_SUBASSIGN_CASE
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, !M, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

