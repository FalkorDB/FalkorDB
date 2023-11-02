//------------------------------------------------------------------------------
// GB_select_positional_bitmap: C=select(A,thunk) when C is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// A is bitmap or as-if-full.  C is bitmap

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_positional_bitmap
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (GB_OPCODE_IS_POSITIONAL (opcode)) ;

    //--------------------------------------------------------------------------
    // positional operators when C is bitmap
    //--------------------------------------------------------------------------

    #define GB_A_TYPE GB_void
    #include "GB_select_shared_definitions.h"

    switch (opcode)
    {

        case GB_TRIL_idxunop_code      : 
            #define GB_TRIL_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_TRIU_idxunop_code      : 
            #define GB_TRIU_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_DIAG_idxunop_code      : 
            #define GB_DIAG_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_OFFDIAG_idxunop_code   : 
        case GB_DIAGINDEX_idxunop_code : 
            #define GB_OFFDIAG_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_ROWINDEX_idxunop_code  : 
            #define GB_ROWINDEX_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_ROWLE_idxunop_code     : 
            #define GB_ROWLE_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_ROWGT_idxunop_code     : 
            #define GB_ROWGT_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_COLINDEX_idxunop_code  : 
            #define GB_COLINDEX_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_COLLE_idxunop_code     : 
            #define GB_COLLE_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        case GB_COLGT_idxunop_code     : 
            #define GB_COLGT_SELECTOR
            #include "GB_select_bitmap_template.c"
            break ;

        default: ;
    }

    return (GrB_SUCCESS) ;
}

