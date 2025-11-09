//------------------------------------------------------------------------------
// GB_select_positional_bitmap: C=select(A,thunk) when C is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is bitmap or as-if-full.  C is bitmap

#include "select/GB_select.h"
#include "include/GB_unused.h"

GrB_Info GB_select_positional_bitmap
(
    // input/output:
    GrB_Matrix C,                   // C->b and C->nvals are computed
    // input:
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
    ASSERT (GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode)) ;
    ASSERT (GB_IS_BITMAP (C)) ;

    //--------------------------------------------------------------------------
    // positional operators when C is bitmap
    //--------------------------------------------------------------------------

    #define GB_A_TYPE GB_void
    #include "select/include/GB_select_shared_definitions.h"

    switch (opcode)
    {

        case GB_TRIL_idxunop_code      : 
            #define GB_TRIL_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_TRIU_idxunop_code      : 
            #define GB_TRIU_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_DIAG_idxunop_code      : 
            #define GB_DIAG_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_OFFDIAG_idxunop_code   : 
        case GB_DIAGINDEX_idxunop_code : 
            #define GB_OFFDIAG_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_ROWINDEX_idxunop_code  : 
            #define GB_ROWINDEX_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_ROWLE_idxunop_code     : 
            #define GB_ROWLE_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_ROWGT_idxunop_code     : 
            #define GB_ROWGT_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_COLINDEX_idxunop_code  : 
            #define GB_COLINDEX_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_COLLE_idxunop_code     : 
            #define GB_COLLE_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        case GB_COLGT_idxunop_code     : 
            #define GB_COLGT_SELECTOR
            #include "select/template/GB_select_bitmap_template.c"
            break ;

        default: ;
    }

    return (GrB_SUCCESS) ;
}

