//------------------------------------------------------------------------------
// GB_mx_BinaryOp_to_Monoid: convert a binary op to a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

GrB_Monoid GB_mx_BinaryOp_to_Monoid // monoid, or NULL if error
(
    const GrB_BinaryOp add          // monoid operator
)
{

    if (add == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "monoid operator missing") ;
        return (NULL) ;
    }

    if (Complex != GxB_FC64) // && add->ztype == Complex)
    {
        // user-defined monoids
        if (add == Complex_plus ) return (Complex_plus_monoid ) ;
        if (add == Complex_times) return (Complex_times_monoid) ;
    }

    switch (add->opcode)
    {

        //----------------------------------------------------------------------
        case GB_MIN_binop_code     :
        //----------------------------------------------------------------------

            // 11 MIN monoids
            switch (add->xtype->code)
            {
                // bool case redudant with AND
                case GB_BOOL_code   : return (GxB_LAND_BOOL_MONOID    ) ;
                case GB_INT8_code   : return (GxB_MIN_INT8_MONOID     ) ;
                case GB_UINT8_code  : return (GxB_MIN_UINT8_MONOID    ) ;
                case GB_INT16_code  : return (GxB_MIN_INT16_MONOID    ) ;
                case GB_UINT16_code : return (GxB_MIN_UINT16_MONOID   ) ;
                case GB_INT32_code  : return (GxB_MIN_INT32_MONOID    ) ;
                case GB_UINT32_code : return (GxB_MIN_UINT32_MONOID   ) ;
                case GB_INT64_code  : return (GxB_MIN_INT64_MONOID    ) ;
                case GB_UINT64_code : return (GxB_MIN_UINT64_MONOID   ) ;
                case GB_FP32_code   : return (GxB_MIN_FP32_MONOID     ) ;
                case GB_FP64_code   : return (GxB_MIN_FP64_MONOID     ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type (MIN)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_binop_code     :
        //----------------------------------------------------------------------

            // 11 MAX monoids
            switch (add->xtype->code)
            {
                // bool case redundant with OR
                case GB_BOOL_code   : return (GxB_LOR_BOOL_MONOID     ) ;
                case GB_INT8_code   : return (GxB_MAX_INT8_MONOID     ) ;
                case GB_UINT8_code  : return (GxB_MAX_UINT8_MONOID    ) ;
                case GB_INT16_code  : return (GxB_MAX_INT16_MONOID    ) ;
                case GB_UINT16_code : return (GxB_MAX_UINT16_MONOID   ) ;
                case GB_INT32_code  : return (GxB_MAX_INT32_MONOID    ) ;
                case GB_UINT32_code : return (GxB_MAX_UINT32_MONOID   ) ;
                case GB_INT64_code  : return (GxB_MAX_INT64_MONOID    ) ;
                case GB_UINT64_code : return (GxB_MAX_UINT64_MONOID   ) ;
                case GB_FP32_code   : return (GxB_MAX_FP32_MONOID     ) ;
                case GB_FP64_code   : return (GxB_MAX_FP64_MONOID     ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type (MAX)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_binop_code    :
        //----------------------------------------------------------------------

            // 11 PLUS monoids
            switch (add->xtype->code)
            {
                // bool case redundant with OR
                case GB_BOOL_code   : return (GxB_LOR_BOOL_MONOID     ) ;
                case GB_INT8_code   : return (GxB_PLUS_INT8_MONOID    ) ;
                case GB_UINT8_code  : return (GxB_PLUS_UINT8_MONOID   ) ;
                case GB_INT16_code  : return (GxB_PLUS_INT16_MONOID   ) ;
                case GB_UINT16_code : return (GxB_PLUS_UINT16_MONOID  ) ;
                case GB_INT32_code  : return (GxB_PLUS_INT32_MONOID   ) ;
                case GB_UINT32_code : return (GxB_PLUS_UINT32_MONOID  ) ;
                case GB_INT64_code  : return (GxB_PLUS_INT64_MONOID   ) ;
                case GB_UINT64_code : return (GxB_PLUS_UINT64_MONOID  ) ;
                case GB_FP32_code   : return (GxB_PLUS_FP32_MONOID    ) ;
                case GB_FP64_code   : return (GxB_PLUS_FP64_MONOID    ) ;
                case GB_FC32_code   : return (GxB_PLUS_FC32_MONOID    ) ;
                case GB_FC64_code   : return (GxB_PLUS_FC64_MONOID    ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type (PLUS)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_binop_code   :
        //----------------------------------------------------------------------

            // 11 TIMES monoids
            switch (add->xtype->code)
            {
                // bool case redundant with AND
                case GB_BOOL_code   : return (GxB_LAND_BOOL_MONOID    ) ;
                case GB_INT8_code   : return (GxB_TIMES_INT8_MONOID   ) ;
                case GB_UINT8_code  : return (GxB_TIMES_UINT8_MONOID  ) ;
                case GB_INT16_code  : return (GxB_TIMES_INT16_MONOID  ) ;
                case GB_UINT16_code : return (GxB_TIMES_UINT16_MONOID ) ;
                case GB_INT32_code  : return (GxB_TIMES_INT32_MONOID  ) ;
                case GB_UINT32_code : return (GxB_TIMES_UINT32_MONOID ) ;
                case GB_INT64_code  : return (GxB_TIMES_INT64_MONOID  ) ;
                case GB_UINT64_code : return (GxB_TIMES_UINT64_MONOID ) ;
                case GB_FP32_code   : return (GxB_TIMES_FP32_MONOID   ) ;
                case GB_FP64_code   : return (GxB_TIMES_FP64_MONOID   ) ;
                case GB_FC32_code   : return (GxB_TIMES_FC32_MONOID   ) ;
                case GB_FC64_code   : return (GxB_TIMES_FC64_MONOID   ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type (TIMES)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ANY_binop_code   :
        //----------------------------------------------------------------------

            // 11 ANY monoids
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : return (GxB_ANY_BOOL_MONOID   ) ;
                case GB_INT8_code   : return (GxB_ANY_INT8_MONOID   ) ;
                case GB_UINT8_code  : return (GxB_ANY_UINT8_MONOID  ) ;
                case GB_INT16_code  : return (GxB_ANY_INT16_MONOID  ) ;
                case GB_UINT16_code : return (GxB_ANY_UINT16_MONOID ) ;
                case GB_INT32_code  : return (GxB_ANY_INT32_MONOID  ) ;
                case GB_UINT32_code : return (GxB_ANY_UINT32_MONOID ) ;
                case GB_INT64_code  : return (GxB_ANY_INT64_MONOID  ) ;
                case GB_UINT64_code : return (GxB_ANY_UINT64_MONOID ) ;
                case GB_FP32_code   : return (GxB_ANY_FP32_MONOID   ) ;
                case GB_FP64_code   : return (GxB_ANY_FP64_MONOID   ) ;
                case GB_FC32_code   : return (GxB_ANY_FC32_MONOID   ) ;
                case GB_FC64_code   : return (GxB_ANY_FC64_MONOID   ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type (ANY)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LOR_binop_code      :
        //----------------------------------------------------------------------

            // 2 OR boolean monoids
            // both GrB_LOR and GxB_LOR_BOOL (same opcode)
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : return (GxB_LOR_BOOL_MONOID     ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid (OR)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_binop_code     :
        //----------------------------------------------------------------------

            // 2 AND boolean monoids
            // both GrB_LAND and GxB_LAND_BOOL (same opcode)
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : return (GxB_LAND_BOOL_MONOID    ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid (AND)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_binop_code     : // GrB_LXOR and GxB_LXOR_BOOL (same opcode)
        //----------------------------------------------------------------------

            // 2 XOR boolean monoids
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : return (GxB_LXOR_BOOL_MONOID    ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid (XOR)") ;
                    return (NULL) ;
            }
            break ;


        //----------------------------------------------------------------------
        case GB_ISEQ_binop_code     : // both GrB_EQ_BOOL and GxB_ISEQ_BOOL
        case GB_EQ_binop_code       : // (different opcode)
        //----------------------------------------------------------------------

            // EQ and ISEQ boolean monoids
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : return (GxB_EQ_BOOL_MONOID      ) ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid (EQ)") ;
                    return (NULL) ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : return (GxB_BOR_UINT8_MONOID ) ;
                case GB_UINT16_code : return (GxB_BOR_UINT16_MONOID) ;
                case GB_UINT32_code : return (GxB_BOR_UINT32_MONOID) ;
                case GB_UINT64_code : return (GxB_BOR_UINT64_MONOID) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : return (GxB_BAND_UINT8_MONOID ) ;
                case GB_UINT16_code : return (GxB_BAND_UINT16_MONOID) ;
                case GB_UINT32_code : return (GxB_BAND_UINT32_MONOID) ;
                case GB_UINT64_code : return (GxB_BAND_UINT64_MONOID) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : return (GxB_BXOR_UINT8_MONOID ) ;
                case GB_UINT16_code : return (GxB_BXOR_UINT16_MONOID) ;
                case GB_UINT32_code : return (GxB_BXOR_UINT32_MONOID) ;
                case GB_UINT64_code : return (GxB_BXOR_UINT64_MONOID) ;
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : return (GxB_BXNOR_UINT8_MONOID ) ;
                case GB_UINT16_code : return (GxB_BXNOR_UINT16_MONOID) ;
                case GB_UINT32_code : return (GxB_BXNOR_UINT32_MONOID) ;
                case GB_UINT64_code : return (GxB_BXNOR_UINT64_MONOID) ;
                default: ;
            }
            break ;


        //----------------------------------------------------------------------
        default: 
        //----------------------------------------------------------------------

            mexWarnMsgIdAndTxt ("GB:warn", "unsupported add operator") ;
            return (NULL) ;
    }

    mexWarnMsgIdAndTxt ("GB:warn", "unknown monoid") ;
    return (NULL) ;
}

