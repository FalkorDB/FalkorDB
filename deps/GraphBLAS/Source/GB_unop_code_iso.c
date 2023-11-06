//------------------------------------------------------------------------------
// GB_unop_code_iso: determine if C = A, C = unop(A), or C = binop(A) is iso
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// C = A, C = unop(A), C = binop(A,scalar), or C = binop(scalar,A) is being
// computed, with A optionally transposed.  Determine if C is iso.

GB_iso_code GB_unop_code_iso    // return the iso status of C
(
    GrB_Matrix A,           // input matrix
    GB_Operator op,         // unary/idxunop/binop, if present
    bool binop_bind1st      // if true, C = binop(x,A), else C = binop(A,y)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    GB_Opcode opcode = GB_NOP_code ;
    if (op != NULL) opcode = op->opcode ;

    //--------------------------------------------------------------------------
    // positional ops or user-defined idxunops never result in an iso matrix
    //--------------------------------------------------------------------------

    // idxunops are either positional, valued, or user-defined.  Positional and
    // user-defined ops lead to a non-iso result even if the input is iso.  A
    // valued idxunop (such as GrB_VALUENE_FP32) has been renamed as a binary
    // op, with bind_1st false.  As a result, no remaining idxunops result in
    // an iso-valued output.

    if (GB_OPCODE_IS_POSITIONAL (opcode) || GB_IS_INDEXUNARYOP_CODE (opcode))
    { 
        // this is the only case where C is non-iso even if A is iso
        return (GB_NON_ISO) ;
    }

    //--------------------------------------------------------------------------
    // C = unop (A) or pair (...)
    //--------------------------------------------------------------------------

    if ((opcode == GB_ONE_unop_code) || (opcode == GB_PAIR_binop_code))
    { 
        // if unop is ONE or binop is PAIR, then C is iso, with a value
        // equal to 1
        return (GB_ISO_1) ;
    }

    //--------------------------------------------------------------------------
    // C = binop (scalar,A) or binop (A,scalar)
    //--------------------------------------------------------------------------

    if ((opcode == GB_ANY_binop_code) ||                      // C = any(...)
        (opcode == GB_FIRST_binop_code  &&  binop_bind1st) || // C = first(x,A)
        (opcode == GB_SECOND_binop_code && !binop_bind1st))   // C = second(A,y)
    { 
        // if binop is FIRST and binop_bind1st is true, or if binop is SECOND
        // and binop_bind1st is false, or if binop is ANY, then C is iso, with
        // a value equal to the scalar
        return (GB_ISO_S) ;
    }

    //--------------------------------------------------------------------------
    // C is iso if A is iso
    //--------------------------------------------------------------------------

    if (A->iso)
    {
        if (opcode == GB_NOP_code || opcode == GB_IDENTITY_unop_code)
        { 
            // C = (ctype) A
            return (GB_ISO_A) ;
        }
        else if (GB_IS_UNARYOP_CODE (opcode))
        { 
            // C = unop (A)
            return (GB_ISO_OP1_A) ;
        }
        else if (binop_bind1st)
        { 
            // C = binop (scalar, A)
            return (GB_ISO_OP2_SA) ;
        }
        else
        { 
            // C = binop (A, scalar)
            return (GB_ISO_OP2_AS) ;
        }
    }

    //--------------------------------------------------------------------------
    // otherwise, C is not iso
    //--------------------------------------------------------------------------

    return (GB_NON_ISO) ;
}

