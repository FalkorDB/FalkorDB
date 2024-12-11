//------------------------------------------------------------------------------
// GB_binop_factory.c: switch factory for built-in methods for C=binop(A,B)
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

#if defined (GxB_NO_BOOL)
#define GB_CASE_BOOL(op)
#else
#define GB_CASE_BOOL(op)   case GB_BOOL_code:   GB_BINOP_WORKER (op, _bool  )
#endif

#if defined (GxB_NO_INT8)
#define GB_CASE_INT8(op)
#else
#define GB_CASE_INT8(op)   case GB_INT8_code:   GB_BINOP_WORKER (op, _int8  )
#endif

#if defined (GxB_NO_INT16)
#define GB_CASE_INT16(op)
#else
#define GB_CASE_INT16(op)  case GB_INT16_code:  GB_BINOP_WORKER (op, _int16 )
#endif

#if defined (GxB_NO_INT32)
#define GB_CASE_INT32(op)
#else
#define GB_CASE_INT32(op)  case GB_INT32_code:  GB_BINOP_WORKER (op, _int32 )
#endif

#if defined (GxB_NO_INT64)
#define GB_CASE_INT64(op)
#else
#define GB_CASE_INT64(op)  case GB_INT64_code:  GB_BINOP_WORKER (op, _int64 )
#endif

#if defined (GxB_NO_UINT8)
#define GB_CASE_UINT8(op)
#else
#define GB_CASE_UINT8(op)  case GB_UINT8_code:  GB_BINOP_WORKER (op, _uint8 )
#endif

#if defined (GxB_NO_UINT16)
#define GB_CASE_UINT16(op)
#else
#define GB_CASE_UINT16(op) case GB_UINT16_code: GB_BINOP_WORKER (op, _uint16)
#endif

#if defined (GxB_NO_UINT32)
#define GB_CASE_UINT32(op)
#else
#define GB_CASE_UINT32(op) case GB_UINT32_code: GB_BINOP_WORKER (op, _uint32)
#endif

#if defined (GxB_NO_UINT64)
#define GB_CASE_UINT64(op)
#else
#define GB_CASE_UINT64(op) case GB_UINT64_code: GB_BINOP_WORKER (op, _uint64)
#endif

#if defined (GxB_NO_FP32)
#define GB_CASE_FP32(op)
#else
#define GB_CASE_FP32(op)   case GB_FP32_code:   GB_BINOP_WORKER (op, _fp32  )
#endif

#if defined (GxB_NO_FP64)
#define GB_CASE_FP64(op)
#else
#define GB_CASE_FP64(op)   case GB_FP64_code:   GB_BINOP_WORKER (op, _fp64  )
#endif

#if defined (GxB_NO_FC32)
#define GB_CASE_FC32(op)
#else
#define GB_CASE_FC32(op)   case GB_FC32_code:   GB_BINOP_WORKER (op, _fc32  )
#endif

#if defined (GxB_NO_FC64)
#define GB_CASE_FC64(op)
#else
#define GB_CASE_FC64(op)   case GB_FC64_code:   GB_BINOP_WORKER (op, _fc64  )
#endif

{

    // this switch factory does not handle positional operators
    ASSERT (!GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode)) ;

    switch (opcode)
    {

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

        //----------------------------------------------------------------------
        case GB_MIN_binop_code     :    // z = min(x,y)
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_min)
                GB_CASE_INT16  (_min)
                GB_CASE_INT32  (_min)
                GB_CASE_INT64  (_min)
                GB_CASE_UINT8  (_min)
                GB_CASE_UINT16 (_min)
                GB_CASE_UINT32 (_min)
                GB_CASE_UINT64 (_min)
                GB_CASE_FP32   (_min)
                GB_CASE_FP64   (_min)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_binop_code     :    // z = max(x,y)
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_max)
                GB_CASE_INT16  (_max)
                GB_CASE_INT32  (_max)
                GB_CASE_INT64  (_max)
                GB_CASE_UINT8  (_max)
                GB_CASE_UINT16 (_max)
                GB_CASE_UINT32 (_max)
                GB_CASE_UINT64 (_max)
                GB_CASE_FP32   (_max)
                GB_CASE_FP64   (_max)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_binop_code    :    // z = x + y
        //----------------------------------------------------------------------

            // MAX == PLUS == OR for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_plus)
                GB_CASE_INT16  (_plus)
                GB_CASE_INT32  (_plus)
                GB_CASE_INT64  (_plus)
                GB_CASE_UINT8  (_plus)
                GB_CASE_UINT16 (_plus)
                GB_CASE_UINT32 (_plus)
                GB_CASE_UINT64 (_plus)
                GB_CASE_FP32   (_plus)
                GB_CASE_FP64   (_plus)
                GB_CASE_FC32   (_plus)
                GB_CASE_FC64   (_plus)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_binop_code   :    // z = x * y
        //----------------------------------------------------------------------

            // MIN == TIMES == AND for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_times)
                GB_CASE_INT16  (_times)
                GB_CASE_INT32  (_times)
                GB_CASE_INT64  (_times)
                GB_CASE_UINT8  (_times)
                GB_CASE_UINT16 (_times)
                GB_CASE_UINT32 (_times)
                GB_CASE_UINT64 (_times)
                GB_CASE_FP32   (_times)
                GB_CASE_FP64   (_times)
                GB_CASE_FC32   (_times)
                GB_CASE_FC64   (_times)
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
                GB_CASE_INT8   (_minus)
                GB_CASE_INT16  (_minus)
                GB_CASE_INT32  (_minus)
                GB_CASE_INT64  (_minus)
                GB_CASE_UINT8  (_minus)
                GB_CASE_UINT16 (_minus)
                GB_CASE_UINT32 (_minus)
                GB_CASE_UINT64 (_minus)
                GB_CASE_FP32   (_minus)
                GB_CASE_FP64   (_minus)
                GB_CASE_FC32   (_minus)
                GB_CASE_FC64   (_minus)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RMINUS_binop_code   :    // z = y - x (reverse minus)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_rminus)
                GB_CASE_INT16  (_rminus)
                GB_CASE_INT32  (_rminus)
                GB_CASE_INT64  (_rminus)
                GB_CASE_UINT8  (_rminus)
                GB_CASE_UINT16 (_rminus)
                GB_CASE_UINT32 (_rminus)
                GB_CASE_UINT64 (_rminus)
                GB_CASE_FP32   (_rminus)
                GB_CASE_FP64   (_rminus)
                GB_CASE_FC32   (_rminus)
                GB_CASE_FC64   (_rminus)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_DIV_binop_code   :      // z = x / y
        //----------------------------------------------------------------------

            // FIRST == DIV for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_div)
                GB_CASE_INT16  (_div)
                GB_CASE_INT32  (_div)
                GB_CASE_INT64  (_div)
                GB_CASE_UINT8  (_div)
                GB_CASE_UINT16 (_div)
                GB_CASE_UINT32 (_div)
                GB_CASE_UINT64 (_div)
                GB_CASE_FP32   (_div)
                GB_CASE_FP64   (_div)
                GB_CASE_FC32   (_div)
                GB_CASE_FC64   (_div)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_RDIV_binop_code   :     // z = y / x (reverse division)
        //----------------------------------------------------------------------

            // SECOND == RDIV for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_rdiv)
                GB_CASE_INT16  (_rdiv)
                GB_CASE_INT32  (_rdiv)
                GB_CASE_INT64  (_rdiv)
                GB_CASE_UINT8  (_rdiv)
                GB_CASE_UINT16 (_rdiv)
                GB_CASE_UINT32 (_rdiv)
                GB_CASE_UINT64 (_rdiv)
                GB_CASE_FP32   (_rdiv)
                GB_CASE_FP64   (_rdiv)
                GB_CASE_FC32   (_rdiv)
                GB_CASE_FC64   (_rdiv)
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
                GB_CASE_BOOL   (_first)
                GB_CASE_INT8   (_first)
                GB_CASE_INT16  (_first)
                GB_CASE_INT32  (_first)
                GB_CASE_INT64  (_first)
                GB_CASE_UINT8  (_first)
                GB_CASE_UINT16 (_first)
                GB_CASE_UINT32 (_first)
                GB_CASE_UINT64 (_first)
                GB_CASE_FP32   (_first)
                GB_CASE_FP64   (_first)
                GB_CASE_FC32   (_first)
                GB_CASE_FC64   (_first)
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
                GB_CASE_BOOL   (_second)
                GB_CASE_INT8   (_second)
                GB_CASE_INT16  (_second)
                GB_CASE_INT32  (_second)
                GB_CASE_INT64  (_second)
                GB_CASE_UINT8  (_second)
                GB_CASE_UINT16 (_second)
                GB_CASE_UINT32 (_second)
                GB_CASE_UINT64 (_second)
                GB_CASE_FP32   (_second)
                GB_CASE_FP64   (_second)
                GB_CASE_FC32   (_second)
                GB_CASE_FC64   (_second)
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
                GB_CASE_BOOL   (_pair)
                GB_CASE_INT8   (_pair)
                GB_CASE_INT16  (_pair)
                GB_CASE_INT32  (_pair)
                GB_CASE_INT64  (_pair)
                GB_CASE_UINT8  (_pair)
                GB_CASE_UINT16 (_pair)
                GB_CASE_UINT32 (_pair)
                GB_CASE_UINT64 (_pair)
                GB_CASE_FP32   (_pair)
                GB_CASE_FP64   (_pair)
                GB_CASE_FC32   (_pair)
                GB_CASE_FC64   (_pair)
                default: ;
            }
            break ;
#endif

#if 0

        //----------------------------------------------------------------------
        // IS* operators fully disabled
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        case GB_ISEQ_binop_code:    // z = (x == y)
        //----------------------------------------------------------------------

            // ISEQ == EQ for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_iseq)
                GB_CASE_INT16  (_iseq)
                GB_CASE_INT32  (_iseq)
                GB_CASE_INT64  (_iseq)
                GB_CASE_UINT8  (_iseq)
                GB_CASE_UINT16 (_iseq)
                GB_CASE_UINT32 (_iseq)
                GB_CASE_UINT64 (_iseq)
                GB_CASE_FP32   (_iseq)
                GB_CASE_FP64   (_iseq)
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // ISEQ does not appear in a builtin complex semiring
                GB_CASE_FC32   (_iseq)
                GB_CASE_FC64   (_iseq)
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISNE_binop_code:    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_isne)
                GB_CASE_INT16  (_isne)
                GB_CASE_INT32  (_isne)
                GB_CASE_INT64  (_isne)
                GB_CASE_UINT8  (_isne)
                GB_CASE_UINT16 (_isne)
                GB_CASE_UINT32 (_isne)
                GB_CASE_UINT64 (_isne)
                GB_CASE_FP32   (_isne)
                GB_CASE_FP64   (_isne)
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // ISNE does not appear in a builtin complex semiring
                GB_CASE_FC32   (_isne)
                GB_CASE_FC64   (_isne)
                #endif
                default: ;
            }
            break ;
#endif

        //----------------------------------------------------------------------
        case GB_EQ_binop_code      :    // z = (x == y)
        //----------------------------------------------------------------------

            // For eq, ge, gt, le, lt, ne: z is bool, while the type of
            // x and y can be non-boolean.  Some factory kernels require the
            // types of x and z to match (subassign_22 and subassign_23).

            switch (xcode)
            {
                GB_CASE_BOOL   (_eq)
                #ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH
                GB_CASE_INT8   (_eq)
                GB_CASE_INT16  (_eq)
                GB_CASE_INT32  (_eq)
                GB_CASE_INT64  (_eq)
                GB_CASE_UINT8  (_eq)
                GB_CASE_UINT16 (_eq)
                GB_CASE_UINT32 (_eq)
                GB_CASE_UINT64 (_eq)
                GB_CASE_FP32   (_eq)
                GB_CASE_FP64   (_eq)
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // EQ does not appear in a builtin complex semiring
                GB_CASE_FC32   (_eq)
                GB_CASE_FC64   (_eq)
                #endif
                #endif
                default: ;
            }
            break ;

#ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH

        //----------------------------------------------------------------------
        case GB_NE_binop_code      :    // z = (x != y)
        //----------------------------------------------------------------------

            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            switch (xcode)
            {
                GB_CASE_INT8   (_ne)
                GB_CASE_INT16  (_ne)
                GB_CASE_INT32  (_ne)
                GB_CASE_INT64  (_ne)
                GB_CASE_UINT8  (_ne)
                GB_CASE_UINT16 (_ne)
                GB_CASE_UINT32 (_ne)
                GB_CASE_UINT64 (_ne)
                GB_CASE_FP32   (_ne)
                GB_CASE_FP64   (_ne)
                #ifndef GB_BINOP_IS_SEMIRING_MULTIPLIER
                // NE does not appear in a builtin complex semiring
                GB_CASE_FC32   (_ne)
                GB_CASE_FC64   (_ne)
                #endif
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_LOR_binop_code     :    // z = x || y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_lor)
                GB_CASE_INT8   (_lor)
                GB_CASE_INT16  (_lor)
                GB_CASE_INT32  (_lor)
                GB_CASE_INT64  (_lor)
                GB_CASE_UINT8  (_lor)
                GB_CASE_UINT16 (_lor)
                GB_CASE_UINT32 (_lor)
                GB_CASE_UINT64 (_lor)
                GB_CASE_FP32   (_lor)
                GB_CASE_FP64   (_lor)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_binop_code    :    // z = x && y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_land)
                GB_CASE_INT8   (_land)
                GB_CASE_INT16  (_land)
                GB_CASE_INT32  (_land)
                GB_CASE_INT64  (_land)
                GB_CASE_UINT8  (_land)
                GB_CASE_UINT16 (_land)
                GB_CASE_UINT32 (_land)
                GB_CASE_UINT64 (_land)
                GB_CASE_FP32   (_land)
                GB_CASE_FP64   (_land)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_binop_code    :    // z = x != y
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_lxor)
                GB_CASE_INT8   (_lxor)
                GB_CASE_INT16  (_lxor)
                GB_CASE_INT32  (_lxor)
                GB_CASE_INT64  (_lxor)
                GB_CASE_UINT8  (_lxor)
                GB_CASE_UINT16 (_lxor)
                GB_CASE_UINT32 (_lxor)
                GB_CASE_UINT64 (_lxor)
                GB_CASE_FP32   (_lxor)
                GB_CASE_FP64   (_lxor)
                default: ;
            }
            break ;

#endif

#if 0
        //----------------------------------------------------------------------
        // IS* operators fully disabled
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        case GB_ISGT_binop_code:    // z = (x >  y)
        //----------------------------------------------------------------------

            // ISGT == GT for boolean.  no complex case
            switch (xcode)
            {
                GB_CASE_INT8   (_isgt)
                GB_CASE_INT16  (_isgt)
                GB_CASE_INT32  (_isgt)
                GB_CASE_INT64  (_isgt)
                GB_CASE_UINT8  (_isgt)
                GB_CASE_UINT16 (_isgt)
                GB_CASE_UINT32 (_isgt)
                GB_CASE_UINT64 (_isgt)
                GB_CASE_FP32   (_isgt)
                GB_CASE_FP64   (_isgt)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLT_binop_code:    // z = (x <  y)
        //----------------------------------------------------------------------

            // ISLT == LT for boolean.  no complex case
            switch (xcode)
            {
                GB_CASE_INT8   (_islt)
                GB_CASE_INT16  (_islt)
                GB_CASE_INT32  (_islt)
                GB_CASE_INT64  (_islt)
                GB_CASE_UINT8  (_islt)
                GB_CASE_UINT16 (_islt)
                GB_CASE_UINT32 (_islt)
                GB_CASE_UINT64 (_islt)
                GB_CASE_FP32   (_islt)
                GB_CASE_FP64   (_islt)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISGE_binop_code:    // z = (x >= y)
        //----------------------------------------------------------------------

            // POW == ISGE == GE for boolean. no complex case.
            switch (xcode)
            {
                GB_CASE_INT8   (_isge)
                GB_CASE_INT16  (_isge)
                GB_CASE_INT32  (_isge)
                GB_CASE_INT64  (_isge)
                GB_CASE_UINT8  (_isge)
                GB_CASE_UINT16 (_isge)
                GB_CASE_UINT32 (_isge)
                GB_CASE_UINT64 (_isge)
                GB_CASE_FP32   (_isge)
                GB_CASE_FP64   (_isge)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ISLE_binop_code:    // z = (x <= y)
        //----------------------------------------------------------------------

            // ISLE == LE for boolean.  no complex case
            switch (xcode)
            {
                GB_CASE_INT8   (_isle)
                GB_CASE_INT16  (_isle)
                GB_CASE_INT32  (_isle)
                GB_CASE_INT64  (_isle)
                GB_CASE_UINT8  (_isle)
                GB_CASE_UINT16 (_isle)
                GB_CASE_UINT32 (_isle)
                GB_CASE_UINT64 (_isle)
                GB_CASE_FP32   (_isle)
                GB_CASE_FP64   (_isle)
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_GT_binop_code      :    // z = (x >  y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_gt)
                #ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH
                GB_CASE_INT8   (_gt)
                GB_CASE_INT16  (_gt)
                GB_CASE_INT32  (_gt)
                GB_CASE_INT64  (_gt)
                GB_CASE_UINT8  (_gt)
                GB_CASE_UINT16 (_gt)
                GB_CASE_UINT32 (_gt)
                GB_CASE_UINT64 (_gt)
                GB_CASE_FP32   (_gt)
                GB_CASE_FP64   (_gt)
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LT_binop_code      :    // z = (x <  y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_lt)
                #ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH
                GB_CASE_INT8   (_lt)
                GB_CASE_INT16  (_lt)
                GB_CASE_INT32  (_lt)
                GB_CASE_INT64  (_lt)
                GB_CASE_UINT8  (_lt)
                GB_CASE_UINT16 (_lt)
                GB_CASE_UINT32 (_lt)
                GB_CASE_UINT64 (_lt)
                GB_CASE_FP32   (_lt)
                GB_CASE_FP64   (_lt)
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_GE_binop_code      :    // z = (x >= y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_ge)
                #ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH
                GB_CASE_INT8   (_ge)
                GB_CASE_INT16  (_ge)
                GB_CASE_INT32  (_ge)
                GB_CASE_INT64  (_ge)
                GB_CASE_UINT8  (_ge)
                GB_CASE_UINT16 (_ge)
                GB_CASE_UINT32 (_ge)
                GB_CASE_UINT64 (_ge)
                GB_CASE_FP32   (_ge)
                GB_CASE_FP64   (_ge)
                #endif
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LE_binop_code      :    // z = (x <= y)
        //----------------------------------------------------------------------

            // no complex case
            switch (xcode)
            {
                GB_CASE_BOOL   (_le)
                #ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH
                GB_CASE_INT8   (_le)
                GB_CASE_INT16  (_le)
                GB_CASE_INT32  (_le)
                GB_CASE_INT64  (_le)
                GB_CASE_UINT8  (_le)
                GB_CASE_UINT16 (_le)
                GB_CASE_UINT32 (_le)
                GB_CASE_UINT64 (_le)
                GB_CASE_FP32   (_le)
                GB_CASE_FP64   (_le)
                #endif
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
                GB_CASE_INT8   (_pow)
                GB_CASE_INT16  (_pow)
                GB_CASE_INT32  (_pow)
                GB_CASE_INT64  (_pow)
                GB_CASE_UINT8  (_pow)
                GB_CASE_UINT16 (_pow)
                GB_CASE_UINT32 (_pow)
                GB_CASE_UINT64 (_pow)
                GB_CASE_FP32   (_pow)
                GB_CASE_FP64   (_pow)
                GB_CASE_FC32   (_pow)
                GB_CASE_FC64   (_pow)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_ATAN2_binop_code    :    // z = atan2 (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_atan2)
                GB_CASE_FP64 (_atan2)
                default: ;
            }
            break ;

#ifndef GB_NO_COMMUTATIVE_BINARY_OPS

        //----------------------------------------------------------------------
        case GB_HYPOT_binop_code    :    // z = hypot (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_hypot)
                GB_CASE_FP64 (_hypot)
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_FMOD_binop_code    :    // z = fmod (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_fmod)
                GB_CASE_FP64 (_fmod)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_REMAINDER_binop_code    :    // z = remainder (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_remainder)
                GB_CASE_FP64 (_remainder)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_LDEXP_binop_code    :    // z = ldexp (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_ldexp)
                GB_CASE_FP64 (_ldexp)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_COPYSIGN_binop_code    :    // z = copysign (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_copysign)
                GB_CASE_FP64 (_copysign)
                default: ;
            }
            break ;

#ifndef GB_XTYPE_AND_ZTYPE_MUST_MATCH

        //----------------------------------------------------------------------
        case GB_CMPLX_binop_code    :    // z = cmplx (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_FP32 (_cmplx)
                GB_CASE_FP64 (_cmplx)
                default: ;
            }
            break ;

#endif

        //----------------------------------------------------------------------
        case GB_BGET_binop_code :   // z = bitget (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_INT8   (_bget)
                GB_CASE_INT16  (_bget)
                GB_CASE_INT32  (_bget)
                GB_CASE_INT64  (_bget)
                GB_CASE_UINT8  (_bget)
                GB_CASE_UINT16 (_bget)
                GB_CASE_UINT32 (_bget)
                GB_CASE_UINT64 (_bget)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BSET_binop_code :   // z = bitset (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_INT8   (_bset)
                GB_CASE_INT16  (_bset)
                GB_CASE_INT32  (_bset)
                GB_CASE_INT64  (_bset)
                GB_CASE_UINT8  (_bset)
                GB_CASE_UINT16 (_bset)
                GB_CASE_UINT32 (_bset)
                GB_CASE_UINT64 (_bset)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BCLR_binop_code :   // z = bitclr (x,y)
        //----------------------------------------------------------------------

            switch (xcode)
            {
                GB_CASE_INT8   (_bclr)
                GB_CASE_INT16  (_bclr)
                GB_CASE_INT32  (_bclr)
                GB_CASE_INT64  (_bclr)
                GB_CASE_UINT8  (_bclr)
                GB_CASE_UINT16 (_bclr)
                GB_CASE_UINT32 (_bclr)
                GB_CASE_UINT64 (_bclr)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BSHIFT_binop_code :   // z = bitshift (x,y)
        //----------------------------------------------------------------------

            // y is always int8; z and x have int* or uint* type
            switch (xcode)
            {
                GB_CASE_INT8   (_bshift)
                GB_CASE_INT16  (_bshift)
                GB_CASE_INT32  (_bshift)
                GB_CASE_INT64  (_bshift)
                GB_CASE_UINT8  (_bshift)
                GB_CASE_UINT16 (_bshift)
                GB_CASE_UINT32 (_bshift)
                GB_CASE_UINT64 (_bshift)
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
                GB_CASE_INT8   (_bor)
                GB_CASE_INT16  (_bor)
                GB_CASE_INT32  (_bor)
                GB_CASE_INT64  (_bor)
                #endif
                GB_CASE_UINT8  (_bor)
                GB_CASE_UINT16 (_bor)
                GB_CASE_UINT32 (_bor)
                GB_CASE_UINT64 (_bor)
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
                GB_CASE_INT8   (_band)
                GB_CASE_INT16  (_band)
                GB_CASE_INT32  (_band)
                GB_CASE_INT64  (_band)
                #endif
                GB_CASE_UINT8  (_band)
                GB_CASE_UINT16 (_band)
                GB_CASE_UINT32 (_band)
                GB_CASE_UINT64 (_band)
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
                GB_CASE_INT8   (_bxor)
                GB_CASE_INT16  (_bxor)
                GB_CASE_INT32  (_bxor)
                GB_CASE_INT64  (_bxor)
                #endif
                GB_CASE_UINT8  (_bxor)
                GB_CASE_UINT16 (_bxor)
                GB_CASE_UINT32 (_bxor)
                GB_CASE_UINT64 (_bxor)
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
                GB_CASE_INT8   (_bxnor)
                GB_CASE_INT16  (_bxnor)
                GB_CASE_INT32  (_bxnor)
                GB_CASE_INT64  (_bxnor)
                #endif
                GB_CASE_UINT8  (_bxnor)
                GB_CASE_UINT16 (_bxnor)
                GB_CASE_UINT32 (_bxnor)
                GB_CASE_UINT64 (_bxnor)
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
#undef GB_XTYPE_AND_ZTYPE_MUST_MATCH

#undef GB_CASE_BOOL
#undef GB_CASE_INT8
#undef GB_CASE_INT16
#undef GB_CASE_INT32
#undef GB_CASE_INT64
#undef GB_CASE_UINT8
#undef GB_CASE_UINT16
#undef GB_CASE_UINT32
#undef GB_CASE_UINT64
#undef GB_CASE_FP32
#undef GB_CASE_FP64
#undef GB_CASE_FC32
#undef GB_CASE_FC64

