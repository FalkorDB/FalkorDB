//------------------------------------------------------------------------------
// GB_positional_unop_ijflip: swap i and j in a unary positional op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Flip i and j to handle the CSR/CSC variations, for unary positional ops.

#include "GB.h"

GrB_UnaryOp GB_positional_unop_ijflip   // return flipped operator
(
    GrB_UnaryOp op                      // operator to flip
)
{

    ASSERT (op != NULL) ;

    if (op->ztype == GrB_INT64)
    {
        switch (op->opcode)
        {
            case GB_POSITIONI_unop_code  : return (GxB_POSITIONJ_INT64 ) ;
            case GB_POSITIONI1_unop_code : return (GxB_POSITIONJ1_INT64) ;
            case GB_POSITIONJ_unop_code  : return (GxB_POSITIONI_INT64 ) ;
            case GB_POSITIONJ1_unop_code : return (GxB_POSITIONI1_INT64) ;
            default: ;
        }
    }
    else
    {
        switch (op->opcode)
        {
            case GB_POSITIONI_unop_code  : return (GxB_POSITIONJ_INT32 ) ;
            case GB_POSITIONI1_unop_code : return (GxB_POSITIONJ1_INT32) ;
            case GB_POSITIONJ_unop_code  : return (GxB_POSITIONI_INT32 ) ;
            case GB_POSITIONJ1_unop_code : return (GxB_POSITIONI1_INT32) ;
            default: ;
        }
    }
    // non-positional op is returned unmodified
    return (op) ;
}

