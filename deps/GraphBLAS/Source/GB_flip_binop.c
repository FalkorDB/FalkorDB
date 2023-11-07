//------------------------------------------------------------------------------
// GB_flip_binop:  flip a binary operator for GrB_mxm
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_binop.h"

GrB_BinaryOp GB_flip_binop  // flip a binary operator
(
    // input:
    GrB_BinaryOp op,        // binary operator to flip
    // input/output:
    bool *flipxy            // true on input, set to false if op is flipped
)
{

    //--------------------------------------------------------------------------
    // quick return if binary op is not flipped
    //--------------------------------------------------------------------------

    if (!(*flipxy))
    { 
        // op is not flipped
        return (op) ;
    }

    (*flipxy) = false ;     // set below to true if the op is not flipped

    //--------------------------------------------------------------------------
    // handle the general case: both ewise and mxm
    //--------------------------------------------------------------------------

    GB_Type_code xcode = op->xtype->code ;
    bool int32 = (xcode == GB_INT32_code) ;

    switch (op->opcode)
    {

        //----------------------------------------------------------------------
        // swap FIRST and SECOND
        //----------------------------------------------------------------------

        case GB_FIRST_binop_code  :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_SECOND_BOOL) ;
                case GB_INT8_code   : return (GrB_SECOND_INT8) ;
                case GB_INT16_code  : return (GrB_SECOND_INT16) ;
                case GB_INT32_code  : return (GrB_SECOND_INT32) ;
                case GB_INT64_code  : return (GrB_SECOND_INT64) ;
                case GB_UINT8_code  : return (GrB_SECOND_UINT8) ;
                case GB_UINT16_code : return (GrB_SECOND_UINT16) ;
                case GB_UINT32_code : return (GrB_SECOND_UINT32) ;
                case GB_UINT64_code : return (GrB_SECOND_UINT64) ;
                case GB_FP32_code   : return (GrB_SECOND_FP32) ;
                case GB_FP64_code   : return (GrB_SECOND_FP64) ;
                case GB_FC32_code   : return (GxB_SECOND_FC32) ;
                case GB_FC64_code   : return (GxB_SECOND_FC64) ;
                default: ;
            }
            break ;

        case GB_SECOND_binop_code :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_FIRST_BOOL) ;
                case GB_INT8_code   : return (GrB_FIRST_INT8) ;
                case GB_INT16_code  : return (GrB_FIRST_INT16) ;
                case GB_INT32_code  : return (GrB_FIRST_INT32) ;
                case GB_INT64_code  : return (GrB_FIRST_INT64) ;
                case GB_UINT8_code  : return (GrB_FIRST_UINT8) ;
                case GB_UINT16_code : return (GrB_FIRST_UINT16) ;
                case GB_UINT32_code : return (GrB_FIRST_UINT32) ;
                case GB_UINT64_code : return (GrB_FIRST_UINT64) ;
                case GB_FP32_code   : return (GrB_FIRST_FP32) ;
                case GB_FP64_code   : return (GrB_FIRST_FP64) ;
                case GB_FC32_code   : return (GxB_FIRST_FC32) ;
                case GB_FC64_code   : return (GxB_FIRST_FC64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap LT and GT
        //----------------------------------------------------------------------

        case GB_GT_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_LT_BOOL) ;
                case GB_INT8_code   : return (GrB_LT_INT8) ;
                case GB_INT16_code  : return (GrB_LT_INT16) ;
                case GB_INT32_code  : return (GrB_LT_INT32) ;
                case GB_INT64_code  : return (GrB_LT_INT64) ;
                case GB_UINT8_code  : return (GrB_LT_UINT8) ;
                case GB_UINT16_code : return (GrB_LT_UINT16) ;
                case GB_UINT32_code : return (GrB_LT_UINT32) ;
                case GB_UINT64_code : return (GrB_LT_UINT64) ;
                case GB_FP32_code   : return (GrB_LT_FP32) ;
                case GB_FP64_code   : return (GrB_LT_FP64) ;
                default: ;
            }
            break ;

        case GB_LT_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_GT_BOOL) ;
                case GB_INT8_code   : return (GrB_GT_INT8) ;
                case GB_INT16_code  : return (GrB_GT_INT16) ;
                case GB_INT32_code  : return (GrB_GT_INT32) ;
                case GB_INT64_code  : return (GrB_GT_INT64) ;
                case GB_UINT8_code  : return (GrB_GT_UINT8) ;
                case GB_UINT16_code : return (GrB_GT_UINT16) ;
                case GB_UINT32_code : return (GrB_GT_UINT32) ;
                case GB_UINT64_code : return (GrB_GT_UINT64) ;
                case GB_FP32_code   : return (GrB_GT_FP32) ;
                case GB_FP64_code   : return (GrB_GT_FP64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap LE and GE
        //----------------------------------------------------------------------

        case GB_GE_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_LE_BOOL) ;
                case GB_INT8_code   : return (GrB_LE_INT8) ;
                case GB_INT16_code  : return (GrB_LE_INT16) ;
                case GB_INT32_code  : return (GrB_LE_INT32) ;
                case GB_INT64_code  : return (GrB_LE_INT64) ;
                case GB_UINT8_code  : return (GrB_LE_UINT8) ;
                case GB_UINT16_code : return (GrB_LE_UINT16) ;
                case GB_UINT32_code : return (GrB_LE_UINT32) ;
                case GB_UINT64_code : return (GrB_LE_UINT64) ;
                case GB_FP32_code   : return (GrB_LE_FP32) ;
                case GB_FP64_code   : return (GrB_LE_FP64) ;
                default: ;
            }
            break ;

        case GB_LE_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_GE_BOOL) ;
                case GB_INT8_code   : return (GrB_GE_INT8) ;
                case GB_INT16_code  : return (GrB_GE_INT16) ;
                case GB_INT32_code  : return (GrB_GE_INT32) ;
                case GB_INT64_code  : return (GrB_GE_INT64) ;
                case GB_UINT8_code  : return (GrB_GE_UINT8) ;
                case GB_UINT16_code : return (GrB_GE_UINT16) ;
                case GB_UINT32_code : return (GrB_GE_UINT32) ;
                case GB_UINT64_code : return (GrB_GE_UINT64) ;
                case GB_FP32_code   : return (GrB_GE_FP32) ;
                case GB_FP64_code   : return (GrB_GE_FP64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap ISLT and ISGT
        //----------------------------------------------------------------------

        case GB_ISGT_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_ISLT_BOOL) ;
                case GB_INT8_code   : return (GxB_ISLT_INT8) ;
                case GB_INT16_code  : return (GxB_ISLT_INT16) ;
                case GB_INT32_code  : return (GxB_ISLT_INT32) ;
                case GB_INT64_code  : return (GxB_ISLT_INT64) ;
                case GB_UINT8_code  : return (GxB_ISLT_UINT8) ;
                case GB_UINT16_code : return (GxB_ISLT_UINT16) ;
                case GB_UINT32_code : return (GxB_ISLT_UINT32) ;
                case GB_UINT64_code : return (GxB_ISLT_UINT64) ;
                case GB_FP32_code   : return (GxB_ISLT_FP32) ;
                case GB_FP64_code   : return (GxB_ISLT_FP64) ;
                default: ;
            }
            break ;

        case GB_ISLT_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_ISGT_BOOL) ;
                case GB_INT8_code   : return (GxB_ISGT_INT8) ;
                case GB_INT16_code  : return (GxB_ISGT_INT16) ;
                case GB_INT32_code  : return (GxB_ISGT_INT32) ;
                case GB_INT64_code  : return (GxB_ISGT_INT64) ;
                case GB_UINT8_code  : return (GxB_ISGT_UINT8) ;
                case GB_UINT16_code : return (GxB_ISGT_UINT16) ;
                case GB_UINT32_code : return (GxB_ISGT_UINT32) ;
                case GB_UINT64_code : return (GxB_ISGT_UINT64) ;
                case GB_FP32_code   : return (GxB_ISGT_FP32) ;
                case GB_FP64_code   : return (GxB_ISGT_FP64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap ISLE and ISGE
        //----------------------------------------------------------------------

        case GB_ISGE_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_ISLE_BOOL) ;
                case GB_INT8_code   : return (GxB_ISLE_INT8) ;
                case GB_INT16_code  : return (GxB_ISLE_INT16) ;
                case GB_INT32_code  : return (GxB_ISLE_INT32) ;
                case GB_INT64_code  : return (GxB_ISLE_INT64) ;
                case GB_UINT8_code  : return (GxB_ISLE_UINT8) ;
                case GB_UINT16_code : return (GxB_ISLE_UINT16) ;
                case GB_UINT32_code : return (GxB_ISLE_UINT32) ;
                case GB_UINT64_code : return (GxB_ISLE_UINT64) ;
                case GB_FP32_code   : return (GxB_ISLE_FP32) ;
                case GB_FP64_code   : return (GxB_ISLE_FP64) ;
                default: ;
            }
            break ;

        case GB_ISLE_binop_code     :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_ISGE_BOOL) ;
                case GB_INT8_code   : return (GxB_ISGE_INT8) ;
                case GB_INT16_code  : return (GxB_ISGE_INT16) ;
                case GB_INT32_code  : return (GxB_ISGE_INT32) ;
                case GB_INT64_code  : return (GxB_ISGE_INT64) ;
                case GB_UINT8_code  : return (GxB_ISGE_UINT8) ;
                case GB_UINT16_code : return (GxB_ISGE_UINT16) ;
                case GB_UINT32_code : return (GxB_ISGE_UINT32) ;
                case GB_UINT64_code : return (GxB_ISGE_UINT64) ;
                case GB_FP32_code   : return (GxB_ISGE_FP32) ;
                case GB_FP64_code   : return (GxB_ISGE_FP64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap DIV and RDIV
        //----------------------------------------------------------------------

        case GB_DIV_binop_code  :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_RDIV_BOOL) ;
                case GB_INT8_code   : return (GxB_RDIV_INT8) ;
                case GB_INT16_code  : return (GxB_RDIV_INT16) ;
                case GB_INT32_code  : return (GxB_RDIV_INT32) ;
                case GB_INT64_code  : return (GxB_RDIV_INT64) ;
                case GB_UINT8_code  : return (GxB_RDIV_UINT8) ;
                case GB_UINT16_code : return (GxB_RDIV_UINT16) ;
                case GB_UINT32_code : return (GxB_RDIV_UINT32) ;
                case GB_UINT64_code : return (GxB_RDIV_UINT64) ;
                case GB_FP32_code   : return (GxB_RDIV_FP32) ;
                case GB_FP64_code   : return (GxB_RDIV_FP64) ;
                case GB_FC32_code   : return (GxB_RDIV_FC32) ;
                case GB_FC64_code   : return (GxB_RDIV_FC64) ;
                default: ;
            }
            break ;

        case GB_RDIV_binop_code :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_DIV_BOOL) ;
                case GB_INT8_code   : return (GrB_DIV_INT8) ;
                case GB_INT16_code  : return (GrB_DIV_INT16) ;
                case GB_INT32_code  : return (GrB_DIV_INT32) ;
                case GB_INT64_code  : return (GrB_DIV_INT64) ;
                case GB_UINT8_code  : return (GrB_DIV_UINT8) ;
                case GB_UINT16_code : return (GrB_DIV_UINT16) ;
                case GB_UINT32_code : return (GrB_DIV_UINT32) ;
                case GB_UINT64_code : return (GrB_DIV_UINT64) ;
                case GB_FP32_code   : return (GrB_DIV_FP32) ;
                case GB_FP64_code   : return (GrB_DIV_FP64) ;
                case GB_FC32_code   : return (GxB_DIV_FC32) ;
                case GB_FC64_code   : return (GxB_DIV_FC64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // swap MINUS and RMINUS
        //----------------------------------------------------------------------

        case GB_MINUS_binop_code  :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GxB_RMINUS_BOOL) ;
                case GB_INT8_code   : return (GxB_RMINUS_INT8) ;
                case GB_INT16_code  : return (GxB_RMINUS_INT16) ;
                case GB_INT32_code  : return (GxB_RMINUS_INT32) ;
                case GB_INT64_code  : return (GxB_RMINUS_INT64) ;
                case GB_UINT8_code  : return (GxB_RMINUS_UINT8) ;
                case GB_UINT16_code : return (GxB_RMINUS_UINT16) ;
                case GB_UINT32_code : return (GxB_RMINUS_UINT32) ;
                case GB_UINT64_code : return (GxB_RMINUS_UINT64) ;
                case GB_FP32_code   : return (GxB_RMINUS_FP32) ;
                case GB_FP64_code   : return (GxB_RMINUS_FP64) ;
                case GB_FC32_code   : return (GxB_RMINUS_FC32) ;
                case GB_FC64_code   : return (GxB_RMINUS_FC64) ;
                default: ;
            }
            break ;

        case GB_RMINUS_binop_code :

            switch (xcode)
            {
                case GB_BOOL_code   : return (GrB_MINUS_BOOL) ;
                case GB_INT8_code   : return (GrB_MINUS_INT8) ;
                case GB_INT16_code  : return (GrB_MINUS_INT16) ;
                case GB_INT32_code  : return (GrB_MINUS_INT32) ;
                case GB_INT64_code  : return (GrB_MINUS_INT64) ;
                case GB_UINT8_code  : return (GrB_MINUS_UINT8) ;
                case GB_UINT16_code : return (GrB_MINUS_UINT16) ;
                case GB_UINT32_code : return (GrB_MINUS_UINT32) ;
                case GB_UINT64_code : return (GrB_MINUS_UINT64) ;
                case GB_FP32_code   : return (GrB_MINUS_FP32) ;
                case GB_FP64_code   : return (GrB_MINUS_FP64) ;
                case GB_FC32_code   : return (GxB_MINUS_FC32) ;
                case GB_FC64_code   : return (GxB_MINUS_FC64) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // positional operators: flip for mxm methods, not for ewise (see above)
        //----------------------------------------------------------------------

        case GB_FIRSTI_binop_code       : 
            return (int32 ? GxB_SECONDJ_INT32 : GxB_SECONDJ_INT64) ;

        case GB_FIRSTJ_binop_code       : 
            return (int32 ? GxB_SECONDI_INT32 : GxB_SECONDI_INT64) ;

        case GB_FIRSTI1_binop_code      : 
            return (int32 ? GxB_SECONDJ1_INT32 : GxB_SECONDJ1_INT64) ;

        case GB_FIRSTJ1_binop_code      : 
            return (int32 ? GxB_SECONDI1_INT32 : GxB_SECONDI1_INT64) ;

        case GB_SECONDI_binop_code      : 
            return (int32 ? GxB_FIRSTJ_INT32 : GxB_FIRSTJ_INT64) ;

        case GB_SECONDJ_binop_code      : 
            return (int32 ? GxB_FIRSTI_INT32 : GxB_FIRSTI_INT64) ;

        case GB_SECONDI1_binop_code     : 
            return (int32 ? GxB_FIRSTJ1_INT32 : GxB_FIRSTJ1_INT64) ;

        case GB_SECONDJ1_binop_code     : 
            return (int32 ? GxB_FIRSTI1_INT32 : GxB_FIRSTI1_INT64) ;

        //----------------------------------------------------------------------
        // these operators are commutative; they are their own flipped ops:
        //----------------------------------------------------------------------

        case GB_ANY_binop_code          :
        case GB_PAIR_binop_code         :
        case GB_MIN_binop_code          :
        case GB_MAX_binop_code          :
        case GB_PLUS_binop_code         :
        case GB_TIMES_binop_code        :
        case GB_ISEQ_binop_code         :
        case GB_ISNE_binop_code         :
        case GB_LOR_binop_code          :
        case GB_LAND_binop_code         :
        case GB_LXOR_binop_code         :
        case GB_BOR_binop_code          :
        case GB_BAND_binop_code         :
        case GB_BXOR_binop_code         :
        case GB_BXNOR_binop_code        :
        case GB_EQ_binop_code           :
        case GB_NE_binop_code           :
        case GB_HYPOT_binop_code        :
            return (op) ;

        //----------------------------------------------------------------------
        // these operators are not commutative and do not have flipped ops:
        //----------------------------------------------------------------------

        // These are the only cases of built-in binary operators that are not
        // flipped.

        case GB_POW_binop_code          :
        case GB_BGET_binop_code         :
        case GB_BSET_binop_code         :
        case GB_BCLR_binop_code         :
        case GB_BSHIFT_binop_code       :
        case GB_ATAN2_binop_code        :
        case GB_FMOD_binop_code         :
        case GB_REMAINDER_binop_code    :
        case GB_COPYSIGN_binop_code     :
        case GB_LDEXP_binop_code        :
        case GB_CMPLX_binop_code        :
        case GB_USER_binop_code         :
        default: ;
    }

    //--------------------------------------------------------------------------
    // operator cannot be flipped
    //--------------------------------------------------------------------------

    (*flipxy) = true ;
    return (op) ;
}

