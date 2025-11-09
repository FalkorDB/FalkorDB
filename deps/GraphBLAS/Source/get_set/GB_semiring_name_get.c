//------------------------------------------------------------------------------
// GB_semiring_name_get: get the name of a built-in semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

const char *GB_semiring_name_get (GrB_Semiring semiring)
{

    if (semiring->user_name_size > 0)
    { 
        // user-defined semiring, with name defined by GrB_set
        return (semiring->user_name) ;
    }

    GB_Opcode add_opcode = semiring->add->op->opcode ;
    GB_Opcode mult_opcode = semiring->multiply->opcode ;
    GB_Type_code zcode = semiring->add->op->ztype->code ;
    GB_Type_code xcode = semiring->multiply->xtype->code ;
    bool xbool = (xcode == GB_BOOL_code) ;

    switch (mult_opcode)
    {

        //----------------------------------------------------------------------
        case GB_FIRST_binop_code   :    // z = x
        //----------------------------------------------------------------------

            // 61 semirings with FIRST:
            // 50: (min,max,plus,times,any) for 10 non-boolean real
            // 5: (or,and,xor,eq,any) for boolean
            // 6: (plus,times,any) for 2 complex

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MIN_FIRST_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MIN_FIRST_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MIN_FIRST_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MIN_FIRST_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MIN_FIRST_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MIN_FIRST_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MIN_FIRST_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MIN_FIRST_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MIN_FIRST_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MIN_FIRST_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MAX_FIRST_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MAX_FIRST_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MAX_FIRST_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MAX_FIRST_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MAX_FIRST_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MAX_FIRST_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MAX_FIRST_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MAX_FIRST_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MAX_FIRST_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MAX_FIRST_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_FIRST_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_FIRST_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_FIRST_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_FIRST_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_FIRST_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_FIRST_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_FIRST_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_FIRST_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_FIRST_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_FIRST_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_FIRST_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_FIRST_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_FIRST_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_FIRST_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_FIRST_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_FIRST_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_FIRST_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_FIRST_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_FIRST_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_FIRST_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_FIRST_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_FIRST_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_FIRST_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_FIRST_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types, plus 2 complex
                        case GB_BOOL_code   : return ("GxB_ANY_FIRST_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_FIRST_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_FIRST_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_FIRST_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_FIRST_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_FIRST_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_FIRST_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_FIRST_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_FIRST_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_FIRST_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_FIRST_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_FIRST_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_FIRST_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LOR_binop_code  : if (xbool) return ("GxB_LOR_FIRST_BOOL"  ) ;
                case GB_LAND_binop_code : if (xbool) return ("GxB_LAND_FIRST_BOOL" ) ;
                case GB_EQ_binop_code   : if (xbool) return ("GxB_EQ_FIRST_BOOL"   ) ;
                case GB_LXOR_binop_code : if (xbool) return ("GxB_LXOR_FIRST_BOOL") ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_SECOND_binop_code  :    // z = y
        //----------------------------------------------------------------------

            // 61 semirings with SECOND:
            // 50: (min,max,plus,times,any) for 10 real non-boolean
            // 5: (or,and,xor,eq,any) for boolean
            // 6: (plus,times,any) for 2 complex

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MIN_SECOND_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MIN_SECOND_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MIN_SECOND_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MIN_SECOND_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MIN_SECOND_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MIN_SECOND_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MIN_SECOND_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MIN_SECOND_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MIN_SECOND_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MIN_SECOND_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MAX_SECOND_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MAX_SECOND_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MAX_SECOND_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MAX_SECOND_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MAX_SECOND_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MAX_SECOND_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MAX_SECOND_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MAX_SECOND_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MAX_SECOND_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MAX_SECOND_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_SECOND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_SECOND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_SECOND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_SECOND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_SECOND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_SECOND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_SECOND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_SECOND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_SECOND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_SECOND_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_SECOND_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_SECOND_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_SECOND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_SECOND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_SECOND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_SECOND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_SECOND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_SECOND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_SECOND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_SECOND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_SECOND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_SECOND_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_SECOND_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_SECOND_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types, plus 2 complex
                        case GB_BOOL_code   : return ("GxB_ANY_SECOND_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_SECOND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_SECOND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_SECOND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_SECOND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_SECOND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_SECOND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_SECOND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_SECOND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_SECOND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_SECOND_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_SECOND_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_SECOND_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LOR_binop_code  : if (xbool) return ("GxB_LOR_SECOND_BOOL"  ) ;
                case GB_LAND_binop_code : if (xbool) return ("GxB_LAND_SECOND_BOOL" ) ;
                case GB_EQ_binop_code   : if (xbool) return ("GxB_EQ_SECOND_BOOL"   ) ;
                case GB_LXOR_binop_code : if (xbool) return ("GxB_LXOR_SECOND_BOOL") ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MIN_binop_code     :    // z = min(x,y)
        //----------------------------------------------------------------------

            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_MIN_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_MIN_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_MIN_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_MIN_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_MIN_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_MIN_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_MIN_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_MIN_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_MIN_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_MIN_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MAX_MIN_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MAX_MIN_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MAX_MIN_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MAX_MIN_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MAX_MIN_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MAX_MIN_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MAX_MIN_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MAX_MIN_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MAX_MIN_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MAX_MIN_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_PLUS_MIN_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_PLUS_MIN_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_PLUS_MIN_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_PLUS_MIN_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_PLUS_MIN_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_PLUS_MIN_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_PLUS_MIN_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_PLUS_MIN_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_PLUS_MIN_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_PLUS_MIN_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_MIN_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_MIN_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_MIN_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_MIN_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_MIN_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_MIN_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_MIN_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_MIN_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_MIN_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_MIN_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_MIN_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_MIN_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_MIN_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_MIN_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_MIN_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_MIN_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_MIN_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_MIN_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_MIN_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_MIN_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_binop_code     :    // z = max(x,y)
        //----------------------------------------------------------------------

            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MIN_MAX_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MIN_MAX_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MIN_MAX_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MIN_MAX_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MIN_MAX_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MIN_MAX_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MIN_MAX_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MIN_MAX_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MIN_MAX_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MIN_MAX_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_MAX_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_MAX_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_MAX_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_MAX_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_MAX_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_MAX_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_MAX_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_MAX_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_MAX_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_MAX_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_MAX_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_MAX_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_MAX_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_MAX_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_MAX_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_MAX_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_MAX_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_MAX_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_MAX_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_MAX_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_MAX_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_MAX_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_MAX_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_MAX_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_MAX_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_MAX_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_MAX_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_MAX_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_MAX_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_MAX_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_MAX_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_MAX_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_MAX_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_MAX_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_MAX_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_MAX_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_MAX_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_MAX_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_MAX_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_MAX_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_binop_code    :    // z = x + y
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MIN_PLUS_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MIN_PLUS_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MIN_PLUS_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MIN_PLUS_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MIN_PLUS_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MIN_PLUS_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MIN_PLUS_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MIN_PLUS_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MIN_PLUS_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MIN_PLUS_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MAX_PLUS_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MAX_PLUS_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MAX_PLUS_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MAX_PLUS_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MAX_PLUS_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MAX_PLUS_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MAX_PLUS_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MAX_PLUS_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MAX_PLUS_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MAX_PLUS_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_PLUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_PLUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_PLUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_PLUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_PLUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_PLUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_PLUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_PLUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_PLUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_PLUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_PLUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_PLUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_PLUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_PLUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_PLUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_PLUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_PLUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_PLUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_PLUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_PLUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_PLUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_PLUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_PLUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_PLUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_PLUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_PLUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_PLUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_PLUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_PLUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_PLUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_PLUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_PLUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_PLUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_PLUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_PLUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_PLUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MINUS_binop_code   :    // z = x - y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_MINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_MINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_MINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_MINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_MINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_MINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_MINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_MINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_MINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_MINUS_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_MINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_MINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_MINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_MINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_MINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_MINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_MINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_MINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_MINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_MINUS_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_MINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_MINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_MINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_MINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_MINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_MINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_MINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_MINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_MINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_MINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_MINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_MINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_MINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_MINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_MINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_MINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_MINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_MINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_MINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_MINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_MINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_MINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_MINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_MINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_MINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_MINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_MINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_MINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_MINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_MINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_MINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_MINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_MINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_MINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_MINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_MINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISEQ_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISEQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISEQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISEQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISEQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISEQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISEQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISEQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISEQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISEQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISEQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISEQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISEQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISEQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISEQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISEQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISEQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISEQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISEQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISEQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISEQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISEQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISEQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISEQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISEQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISEQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISEQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISEQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISEQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISEQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISEQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISEQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISEQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISEQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISEQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISEQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISEQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISEQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISEQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISEQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISEQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISEQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISEQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISEQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISEQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISEQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISEQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISEQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISEQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISEQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISEQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISNE_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISNE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISNE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISNE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISNE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISNE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISNE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISNE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISNE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISNE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISNE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISNE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISNE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISNE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISNE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISNE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISNE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISNE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISNE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISNE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISNE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISNE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISNE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISNE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISNE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISNE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISNE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISNE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISNE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISNE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISNE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISNE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISNE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISNE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISNE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISNE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISNE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISNE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISNE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISNE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISNE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISNE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISNE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISNE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISNE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISNE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISNE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISNE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISNE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISNE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISNE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISGT_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISGT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISGT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISGT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISGT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISGT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISGT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISGT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISGT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISGT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISGT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISGT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISGT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISGT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISGT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISGT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISGT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISGT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISGT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISGT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISGT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISGT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISGT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISGT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISGT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISGT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISGT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISGT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISGT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISGT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISGT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISGT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISGT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISGT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISGT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISGT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISGT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISGT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISGT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISGT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISGT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISGT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISGT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISGT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISGT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISGT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISGT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISGT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISGT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISGT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISGT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLT_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISLT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISLT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISLT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISLT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISLT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISLT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISLT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISLT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISLT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISLT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISLT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISLT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISLT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISLT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISLT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISLT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISLT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISLT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISLT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISLT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISLT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISLT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISLT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISLT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISLT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISLT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISLT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISLT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISLT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISLT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISLT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISLT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISLT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISLT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISLT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISLT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISLT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISLT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISLT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISLT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISLT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISLT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISLT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISLT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISLT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISLT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISLT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISLT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISLT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISLT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISGE_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISGE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISGE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISGE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISGE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISGE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISGE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISGE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISGE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISGE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISGE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISGE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISGE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISGE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISGE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISGE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISGE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISGE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISGE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISGE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISGE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISGE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISGE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISGE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISGE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISGE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISGE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISGE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISGE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISGE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISGE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISGE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISGE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISGE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISGE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISGE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISGE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISGE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISGE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISGE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISGE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISGE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISGE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISGE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISGE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISGE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISGE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISGE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISGE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISGE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISGE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLE_binop_code   :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_ISLE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_ISLE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_ISLE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_ISLE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_ISLE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_ISLE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_ISLE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_ISLE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_ISLE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_ISLE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_ISLE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_ISLE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_ISLE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_ISLE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_ISLE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_ISLE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_ISLE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_ISLE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_ISLE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_ISLE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_ISLE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_ISLE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_ISLE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_ISLE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_ISLE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_ISLE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_ISLE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_ISLE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_ISLE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_ISLE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_ISLE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_ISLE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_ISLE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_ISLE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_ISLE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_ISLE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_ISLE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_ISLE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_ISLE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_ISLE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_ISLE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_ISLE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_ISLE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_ISLE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_ISLE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_ISLE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_ISLE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_ISLE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_ISLE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_ISLE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RMINUS_binop_code   :    // z = y - x (reverse minus)
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_RMINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_RMINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_RMINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_RMINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_RMINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_RMINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_RMINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_RMINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_RMINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_RMINUS_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_RMINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_RMINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_RMINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_RMINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_RMINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_RMINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_RMINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_RMINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_RMINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_RMINUS_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_RMINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_RMINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_RMINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_RMINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_RMINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_RMINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_RMINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_RMINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_RMINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_RMINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_RMINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_RMINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_RMINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_RMINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_RMINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_RMINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_RMINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_RMINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_RMINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_RMINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_RMINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_RMINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_RMINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_RMINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_RMINUS_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_RMINUS_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_RMINUS_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_RMINUS_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_RMINUS_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_RMINUS_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_RMINUS_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_RMINUS_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_RMINUS_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_RMINUS_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_RMINUS_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_RMINUS_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_binop_code   :    // z = x * y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MIN_TIMES_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MIN_TIMES_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MIN_TIMES_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MIN_TIMES_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MIN_TIMES_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MIN_TIMES_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MIN_TIMES_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MIN_TIMES_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MIN_TIMES_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MIN_TIMES_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GrB_MAX_TIMES_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_MAX_TIMES_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_MAX_TIMES_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_MAX_TIMES_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_MAX_TIMES_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_MAX_TIMES_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_MAX_TIMES_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_MAX_TIMES_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_MAX_TIMES_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_MAX_TIMES_SEMIRING_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GrB_PLUS_TIMES_SEMIRING_INT8"  ) ;
                        case GB_INT16_code  : return ("GrB_PLUS_TIMES_SEMIRING_INT16" ) ;
                        case GB_INT32_code  : return ("GrB_PLUS_TIMES_SEMIRING_INT32" ) ;
                        case GB_INT64_code  : return ("GrB_PLUS_TIMES_SEMIRING_INT64" ) ;
                        case GB_UINT8_code  : return ("GrB_PLUS_TIMES_SEMIRING_UINT8" ) ;
                        case GB_UINT16_code : return ("GrB_PLUS_TIMES_SEMIRING_UINT16") ;
                        case GB_UINT32_code : return ("GrB_PLUS_TIMES_SEMIRING_UINT32") ;
                        case GB_UINT64_code : return ("GrB_PLUS_TIMES_SEMIRING_UINT64") ;
                        case GB_FP32_code   : return ("GrB_PLUS_TIMES_SEMIRING_FP32"  ) ;
                        case GB_FP64_code   : return ("GrB_PLUS_TIMES_SEMIRING_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_TIMES_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_TIMES_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_TIMES_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_TIMES_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_TIMES_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_TIMES_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_TIMES_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_TIMES_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_TIMES_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_TIMES_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_TIMES_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_TIMES_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_TIMES_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_TIMES_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_TIMES_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_TIMES_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_TIMES_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_TIMES_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_TIMES_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_TIMES_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_TIMES_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_TIMES_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_TIMES_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_TIMES_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_TIMES_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_TIMES_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_DIV_binop_code   :      // z = x / y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_DIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_DIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_DIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_DIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_DIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_DIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_DIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_DIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_DIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_DIV_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_DIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_DIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_DIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_DIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_DIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_DIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_DIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_DIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_DIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_DIV_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_DIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_DIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_DIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_DIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_DIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_DIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_DIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_DIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_DIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_DIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_DIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_DIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_DIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_DIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_DIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_DIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_DIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_DIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_DIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_DIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_DIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_DIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_DIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_DIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_DIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_DIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_DIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_DIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_DIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_DIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_DIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_DIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_DIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_DIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_DIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_DIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RDIV_binop_code   :     // z = y / x (reverse division)
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_RDIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_RDIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_RDIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_RDIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_RDIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_RDIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_RDIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_RDIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_RDIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_RDIV_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_RDIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_RDIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_RDIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_RDIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_RDIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_RDIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_RDIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_RDIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_RDIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_RDIV_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_RDIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_RDIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_RDIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_RDIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_RDIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_RDIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_RDIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_RDIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_RDIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_RDIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_RDIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_RDIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_TIMES_RDIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_RDIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_RDIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_RDIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_RDIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_RDIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_RDIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_RDIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_RDIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_RDIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_TIMES_RDIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_TIMES_RDIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_ANY_RDIV_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_RDIV_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_RDIV_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_RDIV_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_RDIV_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_RDIV_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_RDIV_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_RDIV_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_RDIV_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_RDIV_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_RDIV_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_RDIV_FC64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_EQ_binop_code      :    // z = (x == y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LOR_EQ_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LOR_EQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_EQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_EQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_EQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_EQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_EQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_EQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_EQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_EQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_EQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LAND_EQ_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LAND_EQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_EQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_EQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_EQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_EQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_EQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_EQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_EQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_EQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_EQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LXOR_EQ_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LXOR_EQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_EQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_EQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_EQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_EQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_EQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_EQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_EQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_EQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_EQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_EQ_EQ_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_EQ_EQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_EQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_EQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_EQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_EQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_EQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_EQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_EQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_EQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_EQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_ANY_EQ_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_EQ_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_EQ_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_EQ_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_EQ_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_EQ_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_EQ_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_EQ_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_EQ_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_EQ_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_EQ_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_NE_binop_code      :    // z = (x != y)
        //----------------------------------------------------------------------

            // 50 semirings: (and,or,xor,eq,any) * (10 real non-boolean types)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_LOR_NE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_NE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_NE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_NE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_NE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_NE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_NE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_NE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_NE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_NE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_LAND_NE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_NE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_NE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_NE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_NE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_NE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_NE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_NE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_NE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_NE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_LXOR_NE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_NE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_NE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_NE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_NE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_NE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_NE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_NE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_NE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_NE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_EQ_NE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_NE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_NE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_NE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_NE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_NE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_NE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_NE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_NE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_NE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_ANY_NE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_NE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_NE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_NE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_NE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_NE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_NE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_NE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_NE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_NE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_GT_binop_code      :    // z = (x >  y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LOR_GT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LOR_GT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_GT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_GT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_GT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_GT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_GT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_GT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_GT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_GT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_GT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LAND_GT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LAND_GT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_GT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_GT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_GT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_GT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_GT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_GT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_GT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_GT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_GT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LXOR_GT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LXOR_GT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_GT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_GT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_GT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_GT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_GT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_GT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_GT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_GT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_GT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_EQ_GT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_EQ_GT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_GT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_GT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_GT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_GT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_GT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_GT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_GT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_GT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_GT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_ANY_GT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_GT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_GT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_GT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_GT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_GT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_GT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_GT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_GT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_GT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_GT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LT_binop_code      :    // z = (x <  y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LOR_LT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LOR_LT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_LT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_LT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_LT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_LT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_LT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_LT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_LT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_LT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_LT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LAND_LT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LAND_LT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_LT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_LT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_LT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_LT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_LT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_LT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_LT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_LT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_LT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LXOR_LT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LXOR_LT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_LT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_LT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_LT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_LT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_LT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_LT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_LT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_LT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_LT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_EQ_LT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_EQ_LT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_LT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_LT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_LT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_LT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_LT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_LT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_LT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_LT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_LT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_ANY_LT_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_LT_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_LT_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_LT_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_LT_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_LT_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_LT_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_LT_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_LT_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_LT_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_LT_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_GE_binop_code      :    // z = (x >= y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LOR_GE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LOR_GE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_GE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_GE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_GE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_GE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_GE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_GE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_GE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_GE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_GE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LAND_GE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LAND_GE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_GE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_GE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_GE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_GE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_GE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_GE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_GE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_GE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_GE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LXOR_GE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LXOR_GE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_GE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_GE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_GE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_GE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_GE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_GE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_GE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_GE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_GE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_EQ_GE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_EQ_GE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_GE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_GE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_GE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_GE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_GE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_GE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_GE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_GE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_GE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_ANY_GE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_GE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_GE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_GE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_GE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_GE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_GE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_GE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_GE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_GE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_GE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LE_binop_code      :    // z = (x <= y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)

            switch (add_opcode)
            {

                case GB_LOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LOR_LE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LOR_LE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LOR_LE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LOR_LE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LOR_LE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LOR_LE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LOR_LE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LOR_LE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LOR_LE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LOR_LE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LOR_LE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LAND_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LAND_LE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LAND_LE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LAND_LE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LAND_LE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LAND_LE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LAND_LE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LAND_LE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LAND_LE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LAND_LE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LAND_LE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LAND_LE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_LXOR_LE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_LXOR_LE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_LXOR_LE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_LXOR_LE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_LXOR_LE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_LXOR_LE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_LXOR_LE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_LXOR_LE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_LXOR_LE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_LXOR_LE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_LXOR_LE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_EQ_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_EQ_LE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_EQ_LE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_EQ_LE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_EQ_LE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_EQ_LE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_EQ_LE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_EQ_LE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_EQ_LE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_EQ_LE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_EQ_LE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_EQ_LE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus boolean
                        case GB_BOOL_code   : return ("GxB_ANY_LE_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_LE_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_LE_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_LE_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_LE_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_LE_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_LE_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_LE_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_LE_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_LE_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_LE_FP64"  ) ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_PAIR_binop_code   :    // z = 1
        //----------------------------------------------------------------------

            // 26 semirings with PAIR:
            // 13: ANY_PAIR semirings
            // 12: (plus) for 10 real non-boolean and 2 complex
            // 1: (xor) for boolean

            switch (add_opcode)
            {

                case GB_MIN_binop_code:
                case GB_MAX_binop_code:
                case GB_TIMES_binop_code:
                case GB_LOR_binop_code: 
                case GB_LAND_binop_code: 
                case GB_EQ_binop_code: 
                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types, plus 2 complex
                        case GB_BOOL_code   : return ("GxB_ANY_PAIR_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_PAIR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_PAIR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_PAIR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_PAIR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_PAIR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_PAIR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_PAIR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_PAIR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_PAIR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_PAIR_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_ANY_PAIR_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_ANY_PAIR_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types, plus 2 complex
                        case GB_INT8_code   : return ("GxB_PLUS_PAIR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_PAIR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_PAIR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_PAIR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_PAIR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_PAIR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_PAIR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_PAIR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_PAIR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_PAIR_FP64"  ) ;
                        case GB_FC32_code   : return ("GxB_PLUS_PAIR_FC32"  ) ;
                        case GB_FC64_code   : return ("GxB_PLUS_PAIR_FC64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LXOR_binop_code : if (xbool) return ("GxB_LXOR_PAIR_BOOL") ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LOR_binop_code     :    // z = x || y
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_LOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_LOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_LOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_LOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_LOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_LOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_LOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_LOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_LOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_LOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_LOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_LOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_LOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_LOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_LOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_LOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_LOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_LOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_LOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_LOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_LOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_LOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_LOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_LOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_LOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_LOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_LOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_LOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_LOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_LOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_LOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_LOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_LOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_LOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_LOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_LOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_LOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_LOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_LOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_LOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types
                        case GB_BOOL_code   : return ("GxB_ANY_LOR_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_LOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_LOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_LOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_LOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_LOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_LOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_LOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_LOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_LOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_LOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LOR_binop_code  : if (xbool) return ("GxB_LOR_LOR_BOOL"           ) ;
                case GB_LAND_binop_code : if (xbool) return ("GrB_LAND_LOR_SEMIRING_BOOL" ) ;
                case GB_EQ_binop_code   : if (xbool) return ("GrB_LXNOR_LOR_SEMIRING_BOOL") ;
                case GB_LXOR_binop_code : if (xbool) return ("GxB_LXOR_LOR_BOOL"          ) ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_binop_code    :    // z = x && y
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_LAND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_LAND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_LAND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_LAND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_LAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_LAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_LAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_LAND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_LAND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_LAND_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_LAND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_LAND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_LAND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_LAND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_LAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_LAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_LAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_LAND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_LAND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_LAND_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_LAND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_LAND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_LAND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_LAND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_LAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_LAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_LAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_LAND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_LAND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_LAND_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_LAND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_LAND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_LAND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_LAND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_LAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_LAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_LAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_LAND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_LAND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_LAND_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types
                        case GB_BOOL_code   : return ("GxB_ANY_LAND_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_LAND_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_LAND_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_LAND_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_LAND_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_LAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_LAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_LAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_LAND_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_LAND_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_LAND_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LOR_binop_code  : if (xbool) return ("GrB_LOR_LAND_SEMIRING_BOOL" ) ;
                case GB_LAND_binop_code : if (xbool) return ("GxB_LAND_LAND_BOOL"         ) ;
                case GB_EQ_binop_code   : if (xbool) return ("GxB_EQ_LAND_BOOL"           ) ;
                case GB_LXOR_binop_code : if (xbool) return ("GrB_LXOR_LAND_SEMIRING_BOOL") ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_binop_code    :    // z = x != y
        //----------------------------------------------------------------------

            switch (add_opcode)
            {

                case GB_MIN_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MIN_LXOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MIN_LXOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MIN_LXOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MIN_LXOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MIN_LXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MIN_LXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MIN_LXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MIN_LXOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MIN_LXOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MIN_LXOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_MAX_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_MAX_LXOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_MAX_LXOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_MAX_LXOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_MAX_LXOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_MAX_LXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_MAX_LXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_MAX_LXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_MAX_LXOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_MAX_LXOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_MAX_LXOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_PLUS_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_PLUS_LXOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_PLUS_LXOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_PLUS_LXOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_PLUS_LXOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_PLUS_LXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_PLUS_LXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_PLUS_LXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_PLUS_LXOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_PLUS_LXOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_PLUS_LXOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_TIMES_binop_code:

                    switch (xcode)
                    {
                        // 10 real, non-boolean types
                        case GB_INT8_code   : return ("GxB_TIMES_LXOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_TIMES_LXOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_TIMES_LXOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_TIMES_LXOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_TIMES_LXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_TIMES_LXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_TIMES_LXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_TIMES_LXOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_TIMES_LXOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_TIMES_LXOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_ANY_binop_code:

                    switch (xcode)
                    {
                        // 11 real and boolean types
                        case GB_BOOL_code   : return ("GxB_ANY_LXOR_BOOL"  ) ;
                        case GB_INT8_code   : return ("GxB_ANY_LXOR_INT8"  ) ;
                        case GB_INT16_code  : return ("GxB_ANY_LXOR_INT16" ) ;
                        case GB_INT32_code  : return ("GxB_ANY_LXOR_INT32" ) ;
                        case GB_INT64_code  : return ("GxB_ANY_LXOR_INT64" ) ;
                        case GB_UINT8_code  : return ("GxB_ANY_LXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_ANY_LXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_ANY_LXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_ANY_LXOR_UINT64") ;
                        case GB_FP32_code   : return ("GxB_ANY_LXOR_FP32"  ) ;
                        case GB_FP64_code   : return ("GxB_ANY_LXOR_FP64"  ) ;
                        default: ;
                    }
                    break ;

                case GB_LOR_binop_code  : if (xbool) return ("GxB_LOR_LXOR_BOOL"  ) ;
                case GB_LAND_binop_code : if (xbool) return ("GxB_LAND_LXOR_BOOL" ) ;
                case GB_EQ_binop_code   : if (xbool) return ("GxB_EQ_LXOR_BOOL"   ) ;
                case GB_LXOR_binop_code : if (xbool) return ("GxB_LXOR_LXOR_BOOL" ) ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)

            switch (add_opcode)
            {

                case GB_BOR_binop_code :     // z = (x | y), bitwise or

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BOR_BOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BOR_BOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BOR_BOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BOR_BOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BAND_binop_code :    // z = (x & y), bitwise and

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BAND_BOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BAND_BOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BAND_BOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BAND_BOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXOR_BOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXOR_BOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXOR_BOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXOR_BOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXNOR_BOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXNOR_BOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXNOR_BOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXNOR_BOR_UINT64") ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)

            switch (add_opcode)
            {

                case GB_BOR_binop_code :     // z = (x | y), bitwise or

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BOR_BAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BOR_BAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BOR_BAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BOR_BAND_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BAND_binop_code :    // z = (x & y), bitwise and

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BAND_BAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BAND_BAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BAND_BAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BAND_BAND_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXOR_BAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXOR_BAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXOR_BAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXOR_BAND_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXNOR_BAND_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXNOR_BAND_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXNOR_BAND_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXNOR_BAND_UINT64") ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)

            switch (add_opcode)
            {

                case GB_BOR_binop_code :     // z = (x | y), bitwise or

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BOR_BXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BOR_BXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BOR_BXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BOR_BXOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BAND_binop_code :    // z = (x & y), bitwise and

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BAND_BXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BAND_BXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BAND_BXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BAND_BXOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXOR_BXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXOR_BXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXOR_BXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXOR_BXOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXNOR_BXOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXNOR_BXOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXNOR_BXOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXNOR_BXOR_UINT64") ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)

            switch (add_opcode)
            {

                case GB_BOR_binop_code :     // z = (x | y), bitwise or

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BOR_BXNOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BOR_BXNOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BOR_BXNOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BOR_BXNOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BAND_binop_code :    // z = (x & y), bitwise and

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BAND_BXNOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BAND_BXNOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BAND_BXNOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BAND_BXNOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXOR_BXNOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXOR_BXNOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXOR_BXNOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXOR_BXNOR_UINT64") ;
                        default: ;
                    }
                    break ;

                case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor

                    switch (zcode)
                    {
                        case GB_UINT8_code  : return ("GxB_BXNOR_BXNOR_UINT8" ) ;
                        case GB_UINT16_code : return ("GxB_BXNOR_BXNOR_UINT16") ;
                        case GB_UINT32_code : return ("GxB_BXNOR_BXNOR_UINT32") ;
                        case GB_UINT64_code : return ("GxB_BXNOR_BXNOR_UINT64") ;
                        default: ;
                    }
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_FIRSTI_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_FIRSTI_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_FIRSTI_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_FIRSTI_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_FIRSTI_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_FIRSTI_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_FIRSTI_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_FIRSTI_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_FIRSTI_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_FIRSTI_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_FIRSTI1_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_FIRSTI1_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_FIRSTI1_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_FIRSTI1_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_FIRSTI1_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_FIRSTI1_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_FIRSTI1_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_FIRSTI1_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_FIRSTI1_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_FIRSTI1_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
        case GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            // FIRSTJ and SECONDI are identical when used in a semiring

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDI_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDI_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDI_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDI_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDI_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDI_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDI_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDI_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDI_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDI_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
        case GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j)) == k+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            // FIRSTJ1 and SECONDI1 are identical when used in a semiring

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDI1_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDI1_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDI1_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDI1_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDI1_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDI1_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDI1_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDI1_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDI1_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDI1_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        //----------------------------------------------------------------------
        case GB_SECONDJ_binop_code  :   // z = second_j(x,B(i,j)) == j
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDJ_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDJ_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDJ_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDJ_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDJ_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDJ_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDJ_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDJ_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDJ_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDJ_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        //----------------------------------------------------------------------
        case GB_SECONDJ1_binop_code :   // z = second_j1(x,B(i,j)) == j+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)

            if (zcode == GB_INT32_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDJ1_INT32"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDJ1_INT32"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDJ1_INT32") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDJ1_INT32" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDJ1_INT32"  ) ;
                    default: ;
                }
            }
            else if (zcode == GB_INT64_code)
            {
                switch (add_opcode)
                {
                    case GB_MIN_binop_code   : return ("GxB_MIN_SECONDJ1_INT64"  ) ;
                    case GB_MAX_binop_code   : return ("GxB_MAX_SECONDJ1_INT64"  ) ;
                    case GB_TIMES_binop_code : return ("GxB_TIMES_SECONDJ1_INT64") ;
                    case GB_PLUS_binop_code  : return ("GxB_PLUS_SECONDJ1_INT64" ) ;
                    case GB_ANY_binop_code   : return ("GxB_ANY_SECONDJ1_INT64"  ) ;
                    default: ;
                }
            }
            break ;

        default: ;
    }

    return (NULL) ;
}

