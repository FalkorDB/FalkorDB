//------------------------------------------------------------------------------
// GB_mx_Monoid: construct a monoid from a built-in operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Also defines the identity of the monoid

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;
#define FREE_ALL ;

#include "GB_mex.h"

bool GB_mx_Monoid               // true if successful, false otherwise
(
    GrB_Monoid *handle,         // monoid to construct
    const GrB_BinaryOp add,     // monoid operator
    const bool malloc_debug     // true if malloc debug should be done
)
{

    GrB_Monoid M = NULL ;
    (*handle) = NULL ;
    if (add == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "monoid operator missing") ;
        return (false) ;
    }

    switch (add->opcode)
    {
        case GB_MIN_binop_code     :

            // 11 MIN monoids
            switch (add->xtype->code)
            {
                // bool case redundant with AND
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_  (&M, add, (bool    ) true)) ;        break ;
                case GB_INT8_code   : METHOD (GrB_Monoid_new_INT8_  (&M, add, (int8_t  ) INT8_MAX)) ;    break ;
                case GB_INT16_code  : METHOD (GrB_Monoid_new_INT16_ (&M, add, (int16_t ) INT16_MAX)) ;   break ;
                case GB_INT32_code  : METHOD (GrB_Monoid_new_INT32_ (&M, add, (int32_t ) INT32_MAX)) ;   break ;
                case GB_INT64_code  : METHOD (GrB_Monoid_new_INT64_ (&M, add, (int64_t ) INT64_MAX)) ;   break ;
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) UINT8_MAX)) ;   break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) UINT16_MAX)) ;  break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) UINT32_MAX)) ;  break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) UINT64_MAX)) ;  break ;
                case GB_FP32_code   : METHOD (GrB_Monoid_new_FP32_  (&M, add, (float   ) INFINITY)) ;    break ;
                case GB_FP64_code   : METHOD (GrB_Monoid_new_FP64_  (&M, add, (double  ) INFINITY)) ;    break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type for MIN") ;
                    return (false) ;
            }
            break ;

        case GB_MAX_binop_code     :

            // 11 MAX monoids
            switch (add->xtype->code)
            {
                // bool case redundant with OR
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_  (&M, add, (bool    ) false)) ;       break ;
                case GB_INT8_code   : METHOD (GrB_Monoid_new_INT8_  (&M, add, (int8_t  ) INT8_MIN)) ;    break ;
                case GB_INT16_code  : METHOD (GrB_Monoid_new_INT16_ (&M, add, (int16_t ) INT16_MIN)) ;   break ;
                case GB_INT32_code  : METHOD (GrB_Monoid_new_INT32_ (&M, add, (int32_t ) INT32_MIN)) ;   break ;
                case GB_INT64_code  : METHOD (GrB_Monoid_new_INT64_ (&M, add, (int64_t ) INT64_MIN)) ;   break ;
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0)) ;           break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0)) ;           break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0)) ;           break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0)) ;           break ;
                case GB_FP32_code   : METHOD (GrB_Monoid_new_FP32_  (&M, add, (float   ) -INFINITY)) ;   break ;
                case GB_FP64_code   : METHOD (GrB_Monoid_new_FP64_  (&M, add, (double  ) -INFINITY)) ;   break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type for MAX") ;
                    return (false) ;
            }
            break ;

        case GB_PLUS_binop_code    :

            // 13 PLUS monoids
            switch (add->xtype->code)
            {
                // bool case redundant with OR
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_  (&M, add, (bool    ) 0)) ;           break ;
                case GB_INT8_code   : METHOD (GrB_Monoid_new_INT8_  (&M, add, (int8_t  ) 0)) ;           break ;
                case GB_INT16_code  : METHOD (GrB_Monoid_new_INT16_ (&M, add, (int16_t ) 0)) ;           break ;
                case GB_INT32_code  : METHOD (GrB_Monoid_new_INT32_ (&M, add, (int32_t ) 0)) ;           break ;
                case GB_INT64_code  : METHOD (GrB_Monoid_new_INT64_ (&M, add, (int64_t ) 0)) ;           break ;
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0)) ;           break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0)) ;           break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0)) ;           break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0)) ;           break ;
                case GB_FP32_code   : METHOD (GrB_Monoid_new_FP32_  (&M, add, (float   ) 0)) ;           break ;
                case GB_FP64_code   : METHOD (GrB_Monoid_new_FP64_  (&M, add, (double  ) 0)) ;           break ;
                case GB_FC32_code   : METHOD (GxB_Monoid_new_FC32_  (&M, add, (GxB_CMPLXF(0,0)))) ;      break ;
                case GB_FC64_code   : METHOD (GxB_Monoid_new_FC64_  (&M, add, (GxB_CMPLX (0,0)))) ;      break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type for (PLUS)") ;
                    return (false) ;
            }
            break ;

        case GB_TIMES_binop_code   :

            // 13 TIMES monoids
            switch (add->xtype->code)
            {
                // bool case redundant with AND
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_  (&M, add, (bool    ) true)) ;        break ;
                case GB_INT8_code   : METHOD (GrB_Monoid_new_INT8_  (&M, add, (int8_t  ) 1)) ;           break ;
                case GB_INT16_code  : METHOD (GrB_Monoid_new_INT16_ (&M, add, (int16_t ) 1)) ;           break ;
                case GB_INT32_code  : METHOD (GrB_Monoid_new_INT32_ (&M, add, (int32_t ) 1)) ;           break ;
                case GB_INT64_code  : METHOD (GrB_Monoid_new_INT64_ (&M, add, (int64_t ) 1)) ;           break ;
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 1)) ;           break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 1)) ;           break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 1)) ;           break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 1)) ;           break ;
                case GB_FP32_code   : METHOD (GrB_Monoid_new_FP32_  (&M, add, (float   ) 1)) ;           break ;
                case GB_FP64_code   : METHOD (GrB_Monoid_new_FP64_  (&M, add, (double  ) 1)) ;           break ;
                case GB_FC32_code   : METHOD (GxB_Monoid_new_FC32_  (&M, add, (GxB_CMPLXF(1,0)))) ;      break ;
                case GB_FC64_code   : METHOD (GxB_Monoid_new_FC64_  (&M, add, (GxB_CMPLX (1,0)))) ;      break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type for (TIMES)") ;
                    return (false) ;
            }
            break ;

        case GB_ANY_binop_code   :

            // 13 ANY monoids
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL   (&M, add, (bool    ) false)) ;       break ;
                case GB_INT8_code   : METHOD (GrB_Monoid_new_INT8_  (&M, add, (int8_t  ) 0)) ;           break ;
                case GB_INT16_code  : METHOD (GrB_Monoid_new_INT16_ (&M, add, (int16_t ) 0)) ;           break ;
                case GB_INT32_code  : METHOD (GrB_Monoid_new_INT32_ (&M, add, (int32_t ) 0)) ;           break ;
                case GB_INT64_code  : METHOD (GrB_Monoid_new_INT64_ (&M, add, (int64_t ) 0)) ;           break ;
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0)) ;           break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0)) ;           break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0)) ;           break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0)) ;           break ;
                case GB_FP32_code   : METHOD (GrB_Monoid_new_FP32_  (&M, add, (float   ) 0)) ;           break ;
                case GB_FP64_code   : METHOD (GrB_Monoid_new_FP64_  (&M, add, (double  ) 0)) ;           break ;
                case GB_FC32_code   : METHOD (GxB_Monoid_new_FC32_  (&M, add, (GxB_CMPLXF(0,0)))) ;      break ;
                case GB_FC64_code   : METHOD (GxB_Monoid_new_FC64_  (&M, add, (GxB_CMPLX (0,0)))) ;      break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "unknown type for (ANY)") ;
                    return (false) ;
            }
            break ;

        case GB_LOR_binop_code      :

            // both GrB_LOR and GxB_LOR_BOOL (same opcode)
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_(&M, add, (bool    ) false)) ;        break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (OR)") ;
                    return (false) ;
            }
            break ;

        case GB_LAND_binop_code     :

            // both GrB_LAND and GxB_LAND_BOOL (same opcode)
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_(&M, add, (bool    ) true)) ;        break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (AND)") ;
                    return (false) ;
            }
            break ;

        case GB_LXOR_binop_code     :

            // both GrB_LXOR and GxB_LXOR_BOOL (same opcode)
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_(&M, add, (bool    ) false)) ;        break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (XOR)") ;
                    return (false) ;
            }
            break ;

        case GB_ISEQ_binop_code     :
        case GB_EQ_binop_code     :

            // both GrB_EQ_BOOL and GxB_ISEQ_BOOL (same opcode), also GrB_LXNOR
            switch (add->xtype->code)
            {
                case GB_BOOL_code   : METHOD (GrB_Monoid_new_BOOL_(&M, add, (bool    ) true)) ;         break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (EQ)") ;
                    return (false) ;
            }
            break ;

        case GB_BOR_binop_code     :

            // BOR monoids (bitwise or):
            // GxB_BOR_UINT8_MONOID,         // identity: 0   terminal: 0xFF
            // GxB_BOR_UINT16_MONOID,        // identity: 0   terminal: 0xFFFF
            // GxB_BOR_UINT32_MONOID,        // identity: 0   terminal: 0xFFFFFFFF
            // GxB_BOR_UINT64_MONOID,        // identity: 0   terminal: 0xFFFFFFFFFFFFFFFF

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0)) ;         break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0)) ;         break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0)) ;         break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0)) ;         break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (BOR)") ;
                    return (false) ;
            }
            break ;

        case GB_BAND_binop_code     :

            // BAND monoids (bitwise and):
            // GxB_BAND_UINT8_MONOID,        // identity: 0xFF               terminal: 0
            // GxB_BAND_UINT16_MONOID,       // identity: 0xFFFF             terminal: 0
            // GxB_BAND_UINT32_MONOID,       // identity: 0xFFFFFFFF         terminal: 0
            // GxB_BAND_UINT64_MONOID,       // identity: 0xFFFFFFFFFFFFFFFF terminal: 0

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0xFF)) ;               break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0xFFFF)) ;             break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0xFFFFFFFF)) ;         break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0xFFFFFFFFFFFFFFFF)) ; break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (BAND)") ;
                    return (false) ;
            }
            break ;

        case GB_BXOR_binop_code     :

            // BXOR monoids (bitwise xor):
            // GxB_BXOR_UINT8_MONOID,        // identity: 0
            // GxB_BXOR_UINT16_MONOID,       // identity: 0
            // GxB_BXOR_UINT32_MONOID,       // identity: 0
            // GxB_BXOR_UINT64_MONOID,       // identity: 0

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0)) ;         break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0)) ;         break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0)) ;         break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0)) ;         break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (BXOR)") ;
                    return (false) ;
            }
            break ;

        case GB_BXNOR_binop_code     :

            // BXNOR monoids (bitwise xnor):
            // GxB_BXNOR_UINT8_MONOID,       // identity: 0xFF
            // GxB_BXNOR_UINT16_MONOID,      // identity: 0xFFFF
            // GxB_BXNOR_UINT32_MONOID,      // identity: 0xFFFFFFFF
            // GxB_BXNOR_UINT64_MONOID ;     // identity: 0xFFFFFFFFFFFFFFFF

            switch (add->xtype->code)
            {
                case GB_UINT8_code  : METHOD (GrB_Monoid_new_UINT8_ (&M, add, (uint8_t ) 0xFF)) ;               break ;
                case GB_UINT16_code : METHOD (GrB_Monoid_new_UINT16_(&M, add, (uint16_t) 0xFFFF)) ;             break ;
                case GB_UINT32_code : METHOD (GrB_Monoid_new_UINT32_(&M, add, (uint32_t) 0xFFFFFFFF)) ;         break ;
                case GB_UINT64_code : METHOD (GrB_Monoid_new_UINT64_(&M, add, (uint64_t) 0xFFFFFFFFFFFFFFFF)) ; break ;
                default: 
                    mexWarnMsgIdAndTxt ("GB:warn", "invalid monoid for (BXNOR)") ;
                    return (false) ;
            }
            break ;

        default: 
            mexWarnMsgIdAndTxt ("GB:warn", "unsupported add operator") ;
            return (false) ;
    }

    ASSERT_MONOID_OK (M, "monoid", GB0) ;
    (*handle) = M ;
    return (true) ;
}

