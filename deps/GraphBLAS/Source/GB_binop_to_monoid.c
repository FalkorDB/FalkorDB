//------------------------------------------------------------------------------
// GB_binop_to_monoid: convert a binary op into its corresponding monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_binop.h"

GrB_Monoid GB_binop_to_monoid       // return the corresponding monoid, or NULL
(
    const GrB_BinaryOp op_in        // binary op to convert
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (op_in != NULL) ;

    //--------------------------------------------------------------------------
    // convert the binary op_in to its corresponding monoid
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (op_in, "binary op to convert to monoid", GB0) ;
    GrB_BinaryOp op = GB_boolean_rename_op (op_in) ;
    GB_Type_code zcode = op->ztype->code ;
    GB_Opcode opcode = op->opcode ;

    switch (opcode)
    {

        case GB_MIN_binop_code:

            switch (zcode)
            {
                // 10 MIN monoids: for 10 real types
                case GB_INT8_code   : return (GrB_MIN_MONOID_INT8    ) ;
                case GB_INT16_code  : return (GrB_MIN_MONOID_INT16   ) ;
                case GB_INT32_code  : return (GrB_MIN_MONOID_INT32   ) ;
                case GB_INT64_code  : return (GrB_MIN_MONOID_INT64   ) ;
                case GB_UINT8_code  : return (GrB_MIN_MONOID_UINT8   ) ;
                case GB_UINT16_code : return (GrB_MIN_MONOID_UINT16  ) ;
                case GB_UINT32_code : return (GrB_MIN_MONOID_UINT32  ) ;
                case GB_UINT64_code : return (GrB_MIN_MONOID_UINT64  ) ;
                case GB_FP32_code   : return (GrB_MIN_MONOID_FP32    ) ;
                case GB_FP64_code   : return (GrB_MIN_MONOID_FP64    ) ;
                default: ;
            }
            break ;

        case GB_MAX_binop_code:

            switch (zcode)
            {
                // 10 MAX monoids: for 10 real types
                case GB_INT8_code   : return (GrB_MAX_MONOID_INT8    ) ;
                case GB_INT16_code  : return (GrB_MAX_MONOID_INT16   ) ;
                case GB_INT32_code  : return (GrB_MAX_MONOID_INT32   ) ;
                case GB_INT64_code  : return (GrB_MAX_MONOID_INT64   ) ;
                case GB_UINT8_code  : return (GrB_MAX_MONOID_UINT8   ) ;
                case GB_UINT16_code : return (GrB_MAX_MONOID_UINT16  ) ;
                case GB_UINT32_code : return (GrB_MAX_MONOID_UINT32  ) ;
                case GB_UINT64_code : return (GrB_MAX_MONOID_UINT64  ) ;
                case GB_FP32_code   : return (GrB_MAX_MONOID_FP32    ) ;
                case GB_FP64_code   : return (GrB_MAX_MONOID_FP64    ) ;
                default: ;
            }
            break ;

        case GB_TIMES_binop_code:

            switch (zcode)
            {
                // 12 TIMES monoids: 10 real types, and 2 complex types
                case GB_INT8_code   : return (GrB_TIMES_MONOID_INT8  ) ;
                case GB_INT16_code  : return (GrB_TIMES_MONOID_INT16 ) ;
                case GB_INT32_code  : return (GrB_TIMES_MONOID_INT32 ) ;
                case GB_INT64_code  : return (GrB_TIMES_MONOID_INT64 ) ;
                case GB_UINT8_code  : return (GrB_TIMES_MONOID_UINT8 ) ;
                case GB_UINT16_code : return (GrB_TIMES_MONOID_UINT16) ;
                case GB_UINT32_code : return (GrB_TIMES_MONOID_UINT32) ;
                case GB_UINT64_code : return (GrB_TIMES_MONOID_UINT64) ;
                case GB_FP32_code   : return (GrB_TIMES_MONOID_FP32  ) ;
                case GB_FP64_code   : return (GrB_TIMES_MONOID_FP64  ) ;
                case GB_FC32_code   : return (GxB_TIMES_FC32_MONOID  ) ;
                case GB_FC64_code   : return (GxB_TIMES_FC64_MONOID  ) ;
                default: ;
            }
            break ;

        case GB_PLUS_binop_code:

            switch (zcode)
            {
                // 12 PLUS monoids: 10 real types, and 2 complex types
                case GB_INT8_code   : return (GrB_PLUS_MONOID_INT8   ) ;
                case GB_INT16_code  : return (GrB_PLUS_MONOID_INT16  ) ;
                case GB_INT32_code  : return (GrB_PLUS_MONOID_INT32  ) ;
                case GB_INT64_code  : return (GrB_PLUS_MONOID_INT64  ) ;
                case GB_UINT8_code  : return (GrB_PLUS_MONOID_UINT8  ) ;
                case GB_UINT16_code : return (GrB_PLUS_MONOID_UINT16 ) ;
                case GB_UINT32_code : return (GrB_PLUS_MONOID_UINT32 ) ;
                case GB_UINT64_code : return (GrB_PLUS_MONOID_UINT64 ) ;
                case GB_FP32_code   : return (GrB_PLUS_MONOID_FP32   ) ;
                case GB_FP64_code   : return (GrB_PLUS_MONOID_FP64   ) ;
                case GB_FC32_code   : return (GxB_PLUS_FC32_MONOID   ) ;
                case GB_FC64_code   : return (GxB_PLUS_FC64_MONOID   ) ;
                default: ;
            }
            break ;

        case GB_ANY_binop_code:

            switch (zcode)
            {
                // 13 ANY monoids: bool, 10 real types, and 2 complex types
                case GB_BOOL_code   : return (GxB_ANY_BOOL_MONOID    ) ;
                case GB_INT8_code   : return (GxB_ANY_INT8_MONOID    ) ;
                case GB_INT16_code  : return (GxB_ANY_INT16_MONOID   ) ;
                case GB_INT32_code  : return (GxB_ANY_INT32_MONOID   ) ;
                case GB_INT64_code  : return (GxB_ANY_INT64_MONOID   ) ;
                case GB_UINT8_code  : return (GxB_ANY_UINT8_MONOID   ) ;
                case GB_UINT16_code : return (GxB_ANY_UINT16_MONOID  ) ;
                case GB_UINT32_code : return (GxB_ANY_UINT32_MONOID  ) ;
                case GB_UINT64_code : return (GxB_ANY_UINT64_MONOID  ) ;
                case GB_FP32_code   : return (GxB_ANY_FP32_MONOID    ) ;
                case GB_FP64_code   : return (GxB_ANY_FP64_MONOID    ) ;
                case GB_FC32_code   : return (GxB_ANY_FC32_MONOID    ) ;
                case GB_FC64_code   : return (GxB_ANY_FC64_MONOID    ) ;
                default: ;
            }
            break ;

        // 4 boolean monoids: (see also GxB_ANY_BOOL_MONOID above)
        #define B(monoid) return ((zcode == GB_BOOL_code) ? monoid : NULL) ;
        case GB_LOR_binop_code   : B (GrB_LOR_MONOID_BOOL)   ;
        case GB_LAND_binop_code  : B (GrB_LAND_MONOID_BOOL)  ;
        case GB_LXOR_binop_code  : B (GrB_LXOR_MONOID_BOOL)  ;
        case GB_EQ_binop_code    : B (GrB_LXNOR_MONOID_BOOL) ;

        case GB_BOR_binop_code:

            switch (zcode)
            {
                // 4 BOR monoids
                case GB_UINT8_code  : return (GxB_BOR_UINT8_MONOID   ) ;
                case GB_UINT16_code : return (GxB_BOR_UINT16_MONOID  ) ;
                case GB_UINT32_code : return (GxB_BOR_UINT32_MONOID  ) ;
                case GB_UINT64_code : return (GxB_BOR_UINT64_MONOID  ) ;
                default: ;
            }
            break ;

        case GB_BAND_binop_code:

            switch (zcode)
            {
                // 4 BAND monoids
                case GB_UINT8_code  : return (GxB_BAND_UINT8_MONOID  ) ;
                case GB_UINT16_code : return (GxB_BAND_UINT16_MONOID ) ;
                case GB_UINT32_code : return (GxB_BAND_UINT32_MONOID ) ;
                case GB_UINT64_code : return (GxB_BAND_UINT64_MONOID ) ;
                default: ;
            }
            break ;

        case GB_BXOR_binop_code:

            switch (zcode)
            {
                // 4 BXOR monoids
                case GB_UINT8_code  : return (GxB_BXOR_UINT8_MONOID  ) ;
                case GB_UINT16_code : return (GxB_BXOR_UINT16_MONOID ) ;
                case GB_UINT32_code : return (GxB_BXOR_UINT32_MONOID ) ;
                case GB_UINT64_code : return (GxB_BXOR_UINT64_MONOID ) ;
                default: ;
            }
            break ;

        case GB_BXNOR_binop_code:

            switch (zcode)
            {
                // 4 BXNOR monoids
                case GB_UINT8_code  : return (GxB_BXNOR_UINT8_MONOID ) ;
                case GB_UINT16_code : return (GxB_BXNOR_UINT16_MONOID) ;
                case GB_UINT32_code : return (GxB_BXNOR_UINT32_MONOID) ;
                case GB_UINT64_code : return (GxB_BXNOR_UINT64_MONOID) ;
                default: ;
            }
            break ;

        default : ;
            // user-defined binary operatory, or other built-in binary op
    }

    //--------------------------------------------------------------------------
    // op_in binary operator does not correspond to a known monoid
    //--------------------------------------------------------------------------

    return (NULL) ;
}

