//------------------------------------------------------------------------------
// GB_positional_binop_ijflip: swap i and j in a binary positional op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Flip i and j to handle the CSR/CSC variations, for ewise positional ops.
// This is not used for semirings.

#include "GB.h"

GrB_BinaryOp GB_positional_binop_ijflip // return flipped operator
(
    GrB_BinaryOp op                     // operator to flip
)
{

    ASSERT (op != NULL) ;

    if (op->ztype == GrB_INT64)
    {
        switch (op->opcode)
        {
            case GB_FIRSTI_binop_code   : return (GxB_FIRSTJ_INT64  ) ;
            case GB_FIRSTI1_binop_code  : return (GxB_FIRSTJ1_INT64 ) ;
            case GB_FIRSTJ_binop_code   : return (GxB_FIRSTI_INT64  ) ;
            case GB_FIRSTJ1_binop_code  : return (GxB_FIRSTI1_INT64 ) ;
            case GB_SECONDI_binop_code  : return (GxB_SECONDJ_INT64 ) ;
            case GB_SECONDI1_binop_code : return (GxB_SECONDJ1_INT64) ;
            case GB_SECONDJ_binop_code  : return (GxB_SECONDI_INT64 ) ;
            case GB_SECONDJ1_binop_code : return (GxB_SECONDI1_INT64) ;
            default: ;
        }
    }
    else
    {
        switch (op->opcode)
        {
            case GB_FIRSTI_binop_code   : return (GxB_FIRSTJ_INT32  ) ;
            case GB_FIRSTI1_binop_code  : return (GxB_FIRSTJ1_INT32 ) ;
            case GB_FIRSTJ_binop_code   : return (GxB_FIRSTI_INT32  ) ;
            case GB_FIRSTJ1_binop_code  : return (GxB_FIRSTI1_INT32 ) ;
            case GB_SECONDI_binop_code  : return (GxB_SECONDJ_INT32 ) ;
            case GB_SECONDI1_binop_code : return (GxB_SECONDJ1_INT32) ;
            case GB_SECONDJ_binop_code  : return (GxB_SECONDI_INT32 ) ;
            case GB_SECONDJ1_binop_code : return (GxB_SECONDI1_INT32) ;
            default: ;
        }
    }

    // non-positional op is returned unmodified
    return (op) ;
}

