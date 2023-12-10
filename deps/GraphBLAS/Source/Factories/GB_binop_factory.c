//------------------------------------------------------------------------------
// GB_binop_factory: switch factory for built-in methods for C=binop(A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The #include'ing file defines the GB_BINOP_WORKER macro, and opcode, xcode,
// ycode, and zcode, to call one of 388 builtin binary operators.  The binary
// operators are all named GrB_[OPNAME]_[XTYPE], according to the opcode/
// opname, and the xtype of the operator.  The type of z and y are not in the
// name.  Except for the GxB_BSHIFT_[XTYPE] operators (where y always has type
// int8), the types of x and y are the same.

{

    // this switch factory does not handle positional operators
    ASSERT (!GB_OPCODE_IS_POSITIONAL (opcode)) ;

    switch (opcode)
    {

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

        //----------------------------------------------------------------------
        case GB_MIN_binop_code     :    // z = min(x,y)
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_min, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_min, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_min, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_min, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_min, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_min, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_min, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_min, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_min, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_min, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_binop_code     :    // z = max(x,y)
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_max, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_max, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_max, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_max, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_max, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_max, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_max, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_max, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_max, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_max, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_binop_code    :    // z = x + y
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_plus, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_plus, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_plus, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_plus, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_plus, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_plus, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_plus, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_plus, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_plus, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_plus, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_plus, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_plus, _fc64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_binop_code   :    // z = x * y
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_times, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_times, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_times, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_times, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_times, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_times, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_times, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_times, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_times, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_times, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_times, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_times, _fc64  )
                default: ;
            }
            break ;
#endif

        //----------------------------------------------------------------------
        case GB_MINUS_binop_code   :    // z = x - y
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_minus, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_minus, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_minus, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_minus, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_minus, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_minus, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_minus, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_minus, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_minus, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_minus, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_minus, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_minus, _fc64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RMINUS_binop_code   :    // z = y - x (reverse minus)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_rminus, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_rminus, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_rminus, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_rminus, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_rminus, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_rminus, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_rminus, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_rminus, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_rminus, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_rminus, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_rminus, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_rminus, _fc64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_DIV_binop_code   :      // z = x / y
        //----------------------------------------------------------------------

            // FIRST == DIV for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_div, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_div, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_div, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_div, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_div, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_div, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_div, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_div, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_div, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_div, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_div, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_div, _fc64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RDIV_binop_code   :     // z = y / x (reverse division)
        //----------------------------------------------------------------------

            // SECOND == RDIV for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_rdiv, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_rdiv, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_rdiv, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_rdiv, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_rdiv, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_rdiv, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_rdiv, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_rdiv, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_rdiv, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_rdiv, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_rdiv, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_rdiv, _fc64  )
                default: ;
            }
            break ;

#ifndef GB_BINOP_SUBSET

        // These operators are not used in C+=A+B by GB_dense_eWise3_accum
        // when all 3 matrices are dense.

#ifndef GB_NO_FIRST

        //----------------------------------------------------------------------
        case GB_FIRST_binop_code   :    // z = x
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_first, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_first, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_first, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_first, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_first, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_first, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_first, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_first, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_first, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_first, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_first, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_first, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_first, _fc64  )
                default: ;
            }
            break ;
#endif

#ifndef GB_NO_SECOND

        //----------------------------------------------------------------------
        case GB_SECOND_binop_code  :    // z = y
        case GB_ANY_binop_code  :       // z = y
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_second, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_second, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_second, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_second, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_second, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_second, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_second, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_second, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_second, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_second, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_second, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_second, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_second, _fc64  )
                default: ;
            }
            break ;
#endif

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

#ifndef GB_NO_PAIR

        //----------------------------------------------------------------------
        case GB_PAIR_binop_code   :    // z = 1
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_pair, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_pair, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_pair, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_pair, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_pair, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_pair, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_pair, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_pair, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_pair, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_pair, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_pair, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_pair, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_pair, _fc64  )
                default: ;
            }
            break ;
#endif

        //----------------------------------------------------------------------
        case GB_ISEQ_binop_code    :    // z = (x == y)
        //----------------------------------------------------------------------

            // ISEQ == EQ for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_iseq, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_iseq, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_iseq, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_iseq, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_iseq, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_iseq, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_iseq, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_iseq, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_iseq, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_iseq, _fp64  )
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // ISEQ does not appear in a builtin complex semiring
                case GB_FC32_code   : GB_BINOP_WORKER (_iseq, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_iseq, _fc64  )
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISNE_binop_code    :    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_isne, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_isne, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_isne, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_isne, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_isne, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_isne, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_isne, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_isne, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_isne, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_isne, _fp64  )
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // ISNE does not appear in a builtin complex semiring
                case GB_FC32_code   : GB_BINOP_WORKER (_isne, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_isne, _fc64  )
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_EQ_binop_code      :    // z = (x == y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_eq, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_eq, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_eq, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_eq, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_eq, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_eq, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_eq, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_eq, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_eq, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_eq, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_eq, _fp64  )
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // EQ does not appear in a builtin complex semiring
                case GB_FC32_code   : GB_BINOP_WORKER (_eq, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_eq, _fc64  )
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_NE_binop_code      :    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_ne, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_ne, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_ne, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_ne, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_ne, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_ne, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_ne, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_ne, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_ne, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_ne, _fp64  )
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // NE does not appear in a builtin complex semiring
                case GB_FC32_code   : GB_BINOP_WORKER (_ne, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_ne, _fc64  )
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LOR_binop_code     :    // z = x || y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_lor, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_lor, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_lor, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_lor, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_lor, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_lor, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_lor, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_lor, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_lor, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_lor, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_lor, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_binop_code    :    // z = x && y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_land, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_land, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_land, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_land, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_land, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_land, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_land, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_land, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_land, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_land, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_land, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_binop_code    :    // z = x != y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_lxor, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_lxor, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_lxor, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_lxor, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_lxor, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_lxor, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_lxor, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_lxor, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_lxor, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_lxor, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_lxor, _fp64  )
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_ISGT_binop_code    :    // z = (x >  y)
        //----------------------------------------------------------------------

            // ISGT == GT for boolean.  no complex case
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_isgt, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_isgt, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_isgt, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_isgt, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_isgt, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_isgt, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_isgt, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_isgt, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_isgt, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_isgt, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLT_binop_code    :    // z = (x <  y)
        //----------------------------------------------------------------------

            // ISLT == LT for boolean.  no complex case
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_islt, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_islt, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_islt, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_islt, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_islt, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_islt, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_islt, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_islt, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_islt, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_islt, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISGE_binop_code    :    // z = (x >= y)
        //----------------------------------------------------------------------

            // POW == ISGE == GE for boolean. no complex case.
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_isge, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_isge, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_isge, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_isge, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_isge, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_isge, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_isge, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_isge, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_isge, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_isge, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLE_binop_code     :    // z = (x <= y)
        //----------------------------------------------------------------------

            // ISLE == LE for boolean.  no complex case
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_isle, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_isle, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_isle, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_isle, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_isle, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_isle, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_isle, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_isle, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_isle, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_isle, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_GT_binop_code      :    // z = (x >  y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_gt, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_gt, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_gt, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_gt, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_gt, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_gt, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_gt, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_gt, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_gt, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_gt, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_gt, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LT_binop_code      :    // z = (x <  y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_lt, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_lt, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_lt, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_lt, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_lt, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_lt, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_lt, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_lt, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_lt, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_lt, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_lt, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_GE_binop_code      :    // z = (x >= y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_ge, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_ge, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_ge, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_ge, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_ge, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_ge, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_ge, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_ge, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_ge, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_ge, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_ge, _fp64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LE_binop_code      :    // z = (x <= y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                case GB_BOOL_code   : GB_BINOP_WORKER (_le, _bool  )
                case GB_INT8_code   : GB_BINOP_WORKER (_le, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_le, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_le, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_le, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_le, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_le, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_le, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_le, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_le, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_le, _fp64  )
                default: ;
            }
            break ;

#ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER

        // pow, atan2, hypot, ... are not used as multiplicative operators in
        // any semiring, so they are not called by GB_rowscale or GB_colscale.

        //----------------------------------------------------------------------
        case GB_POW_binop_code    :    // z = x ^ y
        //----------------------------------------------------------------------

            // POW == ISGE == GE for boolean
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_pow, _int8  )
                case GB_INT16_code  : GB_BINOP_WORKER (_pow, _int16 )
                case GB_INT32_code  : GB_BINOP_WORKER (_pow, _int32 )
                case GB_INT64_code  : GB_BINOP_WORKER (_pow, _int64 )
                case GB_UINT8_code  : GB_BINOP_WORKER (_pow, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_pow, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_pow, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_pow, _uint64)
                case GB_FP32_code   : GB_BINOP_WORKER (_pow, _fp32  )
                case GB_FP64_code   : GB_BINOP_WORKER (_pow, _fp64  )
                case GB_FC32_code   : GB_BINOP_WORKER (_pow, _fc32  )
                case GB_FC64_code   : GB_BINOP_WORKER (_pow, _fc64  )
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ATAN2_binop_code    :    // z = atan2 (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_atan2, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_atan2, _fp64)
                default: ;
            }
            break ;

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

        //----------------------------------------------------------------------
        case GB_HYPOT_binop_code    :    // z = hypot (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_hypot, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_hypot, _fp64)
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_FMOD_binop_code    :    // z = fmod (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_fmod, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_fmod, _fp64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_REMAINDER_binop_code    :    // z = remainder (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_remainder, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_remainder, _fp64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LDEXP_binop_code    :    // z = ldexp (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_ldexp, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_ldexp, _fp64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_COPYSIGN_binop_code    :    // z = copysign (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_copysign, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_copysign, _fp64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_CMPLX_binop_code    :    // z = cmplx (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_FP32_code : GB_BINOP_WORKER (_cmplx, _fp32)
                case GB_FP64_code : GB_BINOP_WORKER (_cmplx, _fp64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BGET_binop_code :   // z = bitget (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_bget, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bget, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bget, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bget, _int64)
                case GB_UINT8_code  : GB_BINOP_WORKER (_bget, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bget, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bget, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bget, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BSET_binop_code :   // z = bitset (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_bset, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bset, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bset, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bset, _int64)
                case GB_UINT8_code  : GB_BINOP_WORKER (_bset, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bset, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bset, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bset, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BCLR_binop_code :   // z = bitclr (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_bclr, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bclr, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bclr, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bclr, _int64)
                case GB_UINT8_code  : GB_BINOP_WORKER (_bclr, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bclr, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bclr, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bclr, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BSHIFT_binop_code :   // z = bitshift (x,y)
        //----------------------------------------------------------------------

            // y is always int8; z and x have int* or uint* type
            switch (xcode)
            {
                case GB_INT8_code   : GB_BINOP_WORKER (_bshift, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bshift, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bshift, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bshift, _int64)
                case GB_UINT8_code  : GB_BINOP_WORKER (_bshift, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bshift, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bshift, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bshift, _uint64)
                default: ;
            }
            break ;

#endif

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            switch (xcode)
            {
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // BOR for signed integers is not in any builtin semiring
                case GB_INT8_code   : GB_BINOP_WORKER (_bor, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bor, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bor, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bor, _int64)
                #endif
                case GB_UINT8_code  : GB_BINOP_WORKER (_bor, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bor, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bor, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bor, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            switch (xcode)
            {
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // BAND for signed integers is not in any builtin semiring
                case GB_INT8_code   : GB_BINOP_WORKER (_band, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_band, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_band, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_band, _int64)
                #endif
                case GB_UINT8_code  : GB_BINOP_WORKER (_band, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_band, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_band, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_band, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            switch (xcode)
            {
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // BXOR for signed integers is not in any builtin semiring
                case GB_INT8_code   : GB_BINOP_WORKER (_bxor, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bxor, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bxor, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bxor, _int64)
                #endif
                case GB_UINT8_code  : GB_BINOP_WORKER (_bxor, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bxor, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bxor, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bxor, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            switch (xcode)
            {
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // BXNOR for signed integers is not in any builtin semiring
                case GB_INT8_code   : GB_BINOP_WORKER (_bxnor, _int8 )
                case GB_INT16_code  : GB_BINOP_WORKER (_bxnor, _int16)
                case GB_INT32_code  : GB_BINOP_WORKER (_bxnor, _int32)
                case GB_INT64_code  : GB_BINOP_WORKER (_bxnor, _int64)
                #endif
                case GB_UINT8_code  : GB_BINOP_WORKER (_bxnor, _uint8 )
                case GB_UINT16_code : GB_BINOP_WORKER (_bxnor, _uint16)
                case GB_UINT32_code : GB_BINOP_WORKER (_bxnor, _uint32)
                case GB_UINT64_code : GB_BINOP_WORKER (_bxnor, _uint64)
                default: ;
            }
            break ;
#endif

#endif

        default: ;
    }
}

#undef GB_NO_FIRST
#undef GB_NO_SECOND
#undef GB_NO_PAIR

