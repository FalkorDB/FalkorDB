//------------------------------------------------------------------------------
// GB_positional_idxunop_ijflip: swap i and j in an index unary positional op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Flip i and j to handle the CSR/CSC variations, for index-unary positional
// ops.  The two operators GB_FLIPDIAG_INT[32,64] are only used internally
// and are not available to the user application.

#include "GB.h"

GrB_IndexUnaryOp GB_positional_idxunop_ijflip   // return flipped operator
(
    int64_t *ithunk,            // input/output: revised value of thunk
    GrB_IndexUnaryOp op         // operator to flip
)
{

    ASSERT (op != NULL) ;

    if (op->ztype == GrB_INT64)
    {
        switch (op->opcode)
        {

            case GB_ROWINDEX_idxunop_code  : 
                // i+thunk becomes j+thunk: no change to thunk
                return (GrB_COLINDEX_INT64) ;

            case GB_COLINDEX_idxunop_code  : 
                // j+thunk becomes i+thunk: no change to thunk
                return (GrB_ROWINDEX_INT64) ;

            case GB_DIAGINDEX_idxunop_code : 
                // j-(i+thunk) becomes i-(j+thunk): no change to thunk
                return (GxB_FLIPDIAGINDEX_INT64) ;

            default: ;
        }

    }
    else if (op->ztype == GrB_INT32)
    {

        switch (op->opcode)
        {

            case GB_ROWINDEX_idxunop_code  : 
                // i+thunk becomes j+thunk: no change to thunk
                return (GrB_COLINDEX_INT32) ;

            case GB_COLINDEX_idxunop_code  : 
                // j+thunk becomes i+thunk: no change to thunk
                return (GrB_ROWINDEX_INT32) ;

            case GB_DIAGINDEX_idxunop_code : 
                // j-(i+thunk) becomes i-(j+thunk): no change to thunk
                return (GxB_FLIPDIAGINDEX_INT32) ;

            default: ;
        }

    }
    else if (op->ztype == GrB_BOOL)
    {

        switch (op->opcode)
        {

            case GB_TRIL_idxunop_code      : 
                // (j <= (i+thunk)) becomes (i <= (j+thunk))
                // or (i-thunk) <= j which is j >= (i-thunk).
                // TRIL becomes TRIU and thunk must be negated.
                (*ithunk) = -(*ithunk) ;
                return (GrB_TRIU) ;

            case GB_TRIU_idxunop_code      : 
                // (j >= (i+thunk)) becomes (i >= (j+thunk))
                // or (i-thunk) >= j which is j <= (i-thunk).
                // TRIU becomes TRIL and thunk must be negated.
                (*ithunk) = -(*ithunk) ;
                return (GrB_TRIL) ;

            case GB_DIAG_idxunop_code      : 
            case GB_OFFDIAG_idxunop_code   : 
                // DIAG:    (j == (i+thunk))
                // OFFDIAG: (j != (i+thunk))
                // no change to DIAG and OFFDIAG, but negate the thunk
                (*ithunk) = -(*ithunk) ;
                return (op) ;

            case GB_COLLE_idxunop_code     : 
                // (j <= thunk) becomes (i <= thunk)
                return (GrB_ROWLE) ;

            case GB_COLGT_idxunop_code     : 
                // (j > thunk) becomes (i > thunk)
                return (GrB_ROWGT) ;

            case GB_ROWLE_idxunop_code     : 
                // (i <= thunk) becomes (j <= thunk)
                return (GrB_COLLE) ;

            case GB_ROWGT_idxunop_code     : 
                // (i > thunk) becomes (j > thunk)
                return (GrB_COLGT) ;

            default: ;
        }
    }

    // all other ops are not modified
    return (op) ;
}

