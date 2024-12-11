//------------------------------------------------------------------------------
// GB_positional_offset: return the offset of a positional operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

int64_t GB_positional_offset        // return the positional thunk
(
    GB_Opcode opcode,               // opcode of positional operator
    GrB_Scalar Thunk,               // thunk for idxunops, or NULL
    bool *depends_on_j              // if true, the op depends on j
)
{

    int64_t ithunk = 0 ;
    if (Thunk != NULL)
    { 
        // get the value of the thunk, for positional idxunops
        GB_cast_scalar (&ithunk, GB_INT64_code, Thunk->x, Thunk->type->code,
            Thunk->type->size) ;
    }

    bool j_dep = false ;

    switch (opcode)
    {

        // offset 0, does not depend on j
        case GB_POSITIONI_unop_code  : // z = position_i(A(i,j)) == i
        case GB_FIRSTI_binop_code    : // z = first_i(A(i,j),y) == i
        case GB_SECONDI_binop_code   : // z = second_i(x,A(i,j)) == i
            j_dep = false ;
            ithunk = 0 ;
            break ;

        // offset 1, does not depend on j
        case GB_POSITIONI1_unop_code : // z = position_i1(A(i,j)) == i+1
        case GB_FIRSTI1_binop_code   : // z = first_i1(A(i,j),y) == i+1
        case GB_SECONDI1_binop_code  : // z = second_i1(x,A(i,j)) == i+1
            j_dep = false ;
            ithunk = 1 ;
            break ;

        // offset 0, depends on j
        case GB_POSITIONJ_unop_code  : // z = position_j(A(i,j)) == j
        case GB_FIRSTJ_binop_code    : // z = first_j(A(i,j),y) == j
        case GB_SECONDJ_binop_code   : // z = second_j(x,A(i,j)) == j
            j_dep = true ;
            ithunk = 0 ;
            break ;

        // offset 1, depends on j
        case GB_POSITIONJ1_unop_code : // z = position_j1(A(i,j)) == j+1
        case GB_FIRSTJ1_binop_code   : // z = first_j1(A(i,j),y) == j+1
        case GB_SECONDJ1_binop_code  : // z = second_j1(x,A(i,j)) == j+1
            j_dep = true ;
            ithunk = 1 ;
            break ;

        // idxunops that do not depend on j
        case GB_ROWINDEX_idxunop_code  :   // (i+thunk): row index - thunk
        case GB_ROWLE_idxunop_code     :   // (i <= thunk): A (0:thunk,:)
        case GB_ROWGT_idxunop_code     :   // (i > thunk): A (thunk+1:nrows-1,:)
            j_dep = false ;
            break ;

        // idxunops that depend on j
        case GB_COLINDEX_idxunop_code  :   // (j+thunk): col index - thunk
        case GB_DIAGINDEX_idxunop_code :   // (j-(i+thunk)): diag index + thunk
        case GB_FLIPDIAGINDEX_idxunop_code :   // (i-(j+thunk)), internal use
        case GB_TRIL_idxunop_code      :   // (j <= (i+thunk)): tril (A,thunk)
        case GB_TRIU_idxunop_code      :   // (j >= (i+thunk)): triu (A,thunk)
        case GB_DIAG_idxunop_code      :   // (j == (i+thunk)): diag(A,thunk)
        case GB_OFFDIAG_idxunop_code   :   // (j != (i+thunk)): offdiag(A,thunk)
        case GB_COLLE_idxunop_code     :   // (j <= thunk): A (:,0:thunk)
        case GB_COLGT_idxunop_code     :   // (j > thunk): A (:,thunk+1:ncols-1)
            j_dep = true ;
            break ;

        // all other operators
        default: ;
    }

    if (depends_on_j != NULL)
    { 
        (*depends_on_j) = j_dep ;
    }

    return (ithunk) ;
}

