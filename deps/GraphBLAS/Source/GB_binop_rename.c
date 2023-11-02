//------------------------------------------------------------------------------
// GB_binop_rename: rename a bound binary operator to its unary op equivalent
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A binary operator can be applied to a matrix with GrB_apply, but some of
// them can be remapped into equivalent unary operators.  Any idxunop can also
// be applied, and some of them can be remapped into binary ops (with bind2nd).

#include "GB.h"
#include "GB_binop.h"
#include "GB_unop.h"

void GB_binop_rename            // rename a bound binary op or an idxunop
(
    GB_Operator *op,            // operator to rename
    bool binop_bind1st
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (op != NULL && (*op) != NULL) ;
    GB_Opcode opcode = (*op)->opcode ;
    GrB_Type xtype = (*op)->xtype ;
    GB_Type_code xcode = (xtype == NULL) ? GB_ignore_code : xtype->code ;

    //--------------------------------------------------------------------------
    // rename a binary op or an idxunop
    //--------------------------------------------------------------------------

    if (GB_IS_BINARYOP_CODE (opcode))
    {

        //----------------------------------------------------------------------
        // rename a binary operator to its equivalent unary op
        //----------------------------------------------------------------------

        if (xcode == GB_BOOL_code)
        { 
            // for boolean: DIV becomes FIRST, RDIV becomes SECOND
            opcode = GB_boolean_rename (opcode) ;
        }

        if ((opcode == GB_FIRST_binop_code  && !binop_bind1st) ||
            (opcode == GB_SECOND_binop_code &&  binop_bind1st))
        { 

            //------------------------------------------------------------------
            // first(A,scalar) and second(scalar,A) become identity(A)
            //------------------------------------------------------------------

            switch (xcode)
            {
                case GB_BOOL_code    : (*op) = (GB_Operator) GrB_IDENTITY_BOOL   ; break ;
                case GB_INT8_code    : (*op) = (GB_Operator) GrB_IDENTITY_INT8   ; break ;
                case GB_INT16_code   : (*op) = (GB_Operator) GrB_IDENTITY_INT16  ; break ;
                case GB_INT32_code   : (*op) = (GB_Operator) GrB_IDENTITY_INT32  ; break ;
                case GB_INT64_code   : (*op) = (GB_Operator) GrB_IDENTITY_INT64  ; break ;
                case GB_UINT8_code   : (*op) = (GB_Operator) GrB_IDENTITY_UINT8  ; break ;
                case GB_UINT16_code  : (*op) = (GB_Operator) GrB_IDENTITY_UINT16 ; break ;
                case GB_UINT32_code  : (*op) = (GB_Operator) GrB_IDENTITY_UINT32 ; break ;
                case GB_UINT64_code  : (*op) = (GB_Operator) GrB_IDENTITY_UINT64 ; break ;
                case GB_FP32_code    : (*op) = (GB_Operator) GrB_IDENTITY_FP32   ; break ;
                case GB_FP64_code    : (*op) = (GB_Operator) GrB_IDENTITY_FP64   ; break ;
                case GB_FC32_code    : (*op) = (GB_Operator) GxB_IDENTITY_FC32   ; break ;
                case GB_FC64_code    : (*op) = (GB_Operator) GxB_IDENTITY_FC64   ; break ;
                default:;
            }

        }
        else if (opcode == GB_PAIR_binop_code)
        { 

            //------------------------------------------------------------------
            // pair(A,scalar) and pair(scalar,A) become one(A)
            //------------------------------------------------------------------

            (*op) = (GB_Operator) GB_unop_one (xcode) ;
        }

    }
    else if (GB_IS_INDEXUNARYOP_CODE (opcode))
    {

        //----------------------------------------------------------------------
        // rename an idxunop operator to its equivalent binary op
        //----------------------------------------------------------------------

        // All built-in GrB_IndexUnaryOps of the form GrB_VALUE* can be
        // remapped to equivalent binary ops, with bind 2nd.  After this
        // remapping, only the positional idxunops need to be handled in
        // GB_apply, as well as user-defined idxunops.

        ASSERT (!binop_bind1st) ;

        switch (opcode)
        {
            case GB_VALUENE_idxunop_code : // (aij != thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_NE_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_NE_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_NE_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_NE_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_NE_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_NE_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_NE_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_NE_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_NE_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_NE_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_NE_FP64   ; break ;
                    case GB_FC32_code    : (*op) = (GB_Operator) GxB_NE_FC32   ; break ;
                    case GB_FC64_code    : (*op) = (GB_Operator) GxB_NE_FC64   ; break ;
                    default:;
                }
                break ;

            case GB_VALUEEQ_idxunop_code : // (aij == thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_EQ_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_EQ_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_EQ_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_EQ_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_EQ_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_EQ_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_EQ_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_EQ_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_EQ_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_EQ_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_EQ_FP64   ; break ;
                    case GB_FC32_code    : (*op) = (GB_Operator) GxB_EQ_FC32   ; break ;
                    case GB_FC64_code    : (*op) = (GB_Operator) GxB_EQ_FC64   ; break ;
                    default:;
                }
                break ;

            case GB_VALUEGT_idxunop_code : // (aij > thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_GT_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_GT_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_GT_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_GT_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_GT_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_GT_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_GT_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_GT_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_GT_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_GT_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_GT_FP64   ; break ;
                    default:;
                }
                break ;

            case GB_VALUEGE_idxunop_code : // (aij >= thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_GE_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_GE_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_GE_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_GE_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_GE_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_GE_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_GE_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_GE_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_GE_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_GE_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_GE_FP64   ; break ;
                    default:;
                }
                break ;

            case GB_VALUELT_idxunop_code : // (aij < thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_LT_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_LT_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_LT_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_LT_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_LT_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_LT_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_LT_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_LT_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_LT_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_LT_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_LT_FP64   ; break ;
                    default:;
                }
                break ;

            case GB_VALUELE_idxunop_code : // (aij <= thunk)

                switch (xcode)
                {
                    case GB_BOOL_code    : (*op) = (GB_Operator) GrB_LE_BOOL   ; break ;
                    case GB_INT8_code    : (*op) = (GB_Operator) GrB_LE_INT8   ; break ;
                    case GB_INT16_code   : (*op) = (GB_Operator) GrB_LE_INT16  ; break ;
                    case GB_INT32_code   : (*op) = (GB_Operator) GrB_LE_INT32  ; break ;
                    case GB_INT64_code   : (*op) = (GB_Operator) GrB_LE_INT64  ; break ;
                    case GB_UINT8_code   : (*op) = (GB_Operator) GrB_LE_UINT8  ; break ;
                    case GB_UINT16_code  : (*op) = (GB_Operator) GrB_LE_UINT16 ; break ;
                    case GB_UINT32_code  : (*op) = (GB_Operator) GrB_LE_UINT32 ; break ;
                    case GB_UINT64_code  : (*op) = (GB_Operator) GrB_LE_UINT64 ; break ;
                    case GB_FP32_code    : (*op) = (GB_Operator) GrB_LE_FP32   ; break ;
                    case GB_FP64_code    : (*op) = (GB_Operator) GrB_LE_FP64   ; break ;
                    default:;
                }
                break ;

            default:;
        }
    }
}

