//------------------------------------------------------------------------------
// GB_bitmap_subref_template: C = A(I,J) where A is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A(I,J), where C and A are bitmap/full, numeric and non-iso

{


    //--------------------------------------------------------------------------
    // C = A(I,J)
    //--------------------------------------------------------------------------

    int64_t cnvals = 0 ;

    if (GB_C_IS_BITMAP)
    { 

        //----------------------------------------------------------------------
        // C=A(I,J) non-iso numeric with A and C bitmap; both non-iso
        //----------------------------------------------------------------------

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pA,pC)                                  \
        {                                                           \
            int8_t ab = Ab [pA] ;                                   \
            Cb [pC] = ab ;                                          \
            if (ab)                                                 \
            {                                                       \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_ENTRY (pC, pA)                              \
                task_cnvals++ ;                                     \
            }                                                       \
        }
        #include "template/GB_bitmap_assign_IxJ_template.c"
        C->nvals = cnvals ;
    }
    else
    { 

        //----------------------------------------------------------------------
        // C=A(I,J) non-iso numeric with A and C full, both are non-iso
        //----------------------------------------------------------------------

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pA,pC)                                  \
        {                                                           \
            /* Cx [pC] = Ax [pA] */                                 \
            GB_COPY_ENTRY (pC, pA)                                  \
        }
        #define GB_NO_CNVALS
        #include "template/GB_bitmap_assign_IxJ_template.c"
        #undef  GB_NO_CNVALS
    }
}

