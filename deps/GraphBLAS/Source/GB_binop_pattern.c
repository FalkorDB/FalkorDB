//------------------------------------------------------------------------------
// GB_binop_pattern: determine if values of A and B will be used by C=op(A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Given the opcode of a operator z = op (x,y), and a flipxy flag, determine if
// C=op(A,B) will use just the pattern of A and/or B.  op(A,B) is the multiply
// operator for C=A*B, or the ewise operator for eWiseMult.

#include "GB.h"
#include "GB_binop.h"

void GB_binop_pattern
(
    // outputs:
    bool *A_is_pattern,     // true if A is pattern-only, because of the op
    bool *B_is_pattern,     // true if B is pattern-only, because of the op
    // inputs:
    const bool flipxy,      // if true,  z = op (b,a) will be computed
                            // if false, z = op (a,b) will be computed
    const GB_Opcode opcode  // opcode of binary op
)
{

    //--------------------------------------------------------------------------
    // determine A_is_pattern and B_is_pattern
    //--------------------------------------------------------------------------

    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;

    if (op_is_positional || op_is_pair)
    { 
        // mult (x,y) does not depend on the values of x or y
        (*A_is_pattern) = true ;
        (*B_is_pattern) = true ;
    }
    else if (flipxy)
    { 
        // z = mult (b,a) will be computed
        (*A_is_pattern) = op_is_first  ;
        (*B_is_pattern) = op_is_second ;
    }
    else
    { 
        // z = mult (a,b) will be computed
        (*A_is_pattern) = op_is_second ;
        (*B_is_pattern) = op_is_first  ;
    }
}

