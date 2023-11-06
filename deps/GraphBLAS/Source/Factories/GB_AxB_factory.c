//------------------------------------------------------------------------------
// GB_AxB_factory: switch factory for C=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is used by GB_AxB_saxpy3.c and GB_AxB_dot[234].c to create the built-in
// versions of sparse matrix-matrix multiplication.  The #include'ing file
// defines the GB_AxB_WORKER macro, and mult_binop_code, add_binop_code, xcode,
// ycode, and zcode.

// Three 2nd level switch factories are used:

//      GB_AxB_type_factory: handles all semirings where the multiply operator
//          is TxT->T (as is the monoid).

//      GB_AxB_compare_factory: handles all semirings where the multiply
//          operator is TxT -> bool (for the comparators, LT, GT,
//          etc), and where the monoid is bool x bool -> bool.

//      GB_AxB_bitwise_factory: handles all semirings for bitwise operators.

//      GxB_AxB_positional_factory: handles all semirings for positional
//          multiply operators.  Those operators are of the for XxX -> int64,
//          where "X" denotes any type.  No typecasting is needed from the
//          types of A and B.

// If the multiplicative operator is ANY, then it has already been renamed to
// SECOND, prior to using this factory, since that is faster for the
// saxpy-based methods (y is the value of B(k,j), which is loaded less
// frequently from memory than A(i,k)).

// This switch factory is not used to call the ANY_PAIR iso semiring.

ASSERT (mult_binop_code != GB_ANY_binop_code) ;

{

    switch (mult_binop_code)
    {

        //----------------------------------------------------------------------
        case GB_FIRST_binop_code   :    // z = x
        //----------------------------------------------------------------------

            // 61 semirings with FIRST:
            // 50: (min,max,plus,times,any) for 10 non-boolean real
            // 5: (or,and,xor,eq,any) for boolean
            // 6: (plus,times,any) for 2 complex
            #define GB_MNAME _first
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_SECOND_binop_code  :    // z = y
        //----------------------------------------------------------------------

            // 61 semirings with SECOND:
            // 50: (min,max,plus,times,any) for 10 real non-boolean
            // 5: (or,and,xor,eq,any) for boolean
            // 6: (plus,times,any) for 2 complex
            #define GB_MNAME _second
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_MIN_binop_code     :    // z = min(x,y)
        //----------------------------------------------------------------------

            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // MIN == TIMES == AND for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _min
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_MAX_binop_code     :    // z = max(x,y)
        //----------------------------------------------------------------------

            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // MAX == PLUS == OR for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _max
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_PLUS_binop_code    :    // z = x + y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            // MAX == PLUS == OR for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _plus
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_MINUS_binop_code   :    // z = x - y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _minus
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_RMINUS_binop_code   :    // z = y - x (reverse minus)
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            #define GB_NO_BOOLEAN
            #define GB_MNAME _rminus
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_TIMES_binop_code   :    // z = x * y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            #define GB_NO_BOOLEAN
            #define GB_MNAME _times
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_DIV_binop_code   :      // z = x / y
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            // FIRST == DIV for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _div
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_RDIV_binop_code   :     // z = y / x (reverse division)
        //----------------------------------------------------------------------

            // 56 semirings:
            // 50 semirings: (min,max,plus,times,any) for 10 real non-boolean
            // 6: (plus,times,any) for 2 complex types
            // SECOND == RDIV for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _rdiv
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_EQ_binop_code      :    // z = (x == y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)
            #define GB_MNAME _eq
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_NE_binop_code      :    // z = (x != y)
        //----------------------------------------------------------------------

            // 50 semirings: (and,or,xor,eq,any) * (10 real non-boolean types)
            // MINUS == RMINUS == NE == ISNE == XOR for boolean
            #define GB_NO_BOOLEAN
            #define GB_MNAME _ne
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_GT_binop_code      :    // z = (x >  y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)
            #define GB_MNAME _gt
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LT_binop_code      :    // z = (x <  y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)
            #define GB_MNAME _lt
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_GE_binop_code      :    // z = (x >= y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)
            #define GB_MNAME _ge
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LE_binop_code      :    // z = (x <= y)
        //----------------------------------------------------------------------

            // 55 semirings: (and,or,xor,eq,any) * 11 types (all but complex)
            #define GB_MNAME _le
            #include "GB_AxB_compare_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_PAIR_binop_code   :    // z = 1
        //----------------------------------------------------------------------

            // 13 semirings with PAIR: (not including ANY_PAIR)
            // 12: (plus) for 10 real non-boolean and 2 complex
            // 1: (xor) for boolean
            #define GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
            #define GB_MULT_IS_PAIR_OPERATOR
            #define GB_MNAME _pair
            #define GB_COMPLEX
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LOR_binop_code     :    // z = x || y
        //----------------------------------------------------------------------

            // 15 semirings:
            // 10 semirings: plus_lor for 10 real non-boolean types
            // 5 semirings: (lor,land,eq,lxor,any) for boolean
            #define GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
            #define GB_MNAME _lor
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LAND_binop_code    :    // z = x && y
        //----------------------------------------------------------------------

            // 15 semirings: same as LOR
            #define GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
            #define GB_MNAME _land
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_LXOR_binop_code    :    // z = x != y
        //----------------------------------------------------------------------

            // 15 semirings: same as LOR
            #define GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
            #define GB_MNAME _lxor
            #include "GB_AxB_type_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)
            #define GB_MNAME _bor
            #include "GB_AxB_bitwise_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)
            #define GB_MNAME _band
            #include "GB_AxB_bitwise_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)
            #define GB_MNAME _bxor
            #include "GB_AxB_bitwise_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            // 16 semirings: (bor,band,bxor,bxnor) * (uint8,16,32,64)
            #define GB_MNAME _bxnor
            #include "GB_AxB_bitwise_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            #define GB_MNAME _firsti
            #include "GB_AxB_positional_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            #define GB_MNAME _firsti1
            #include "GB_AxB_positional_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
        case GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            // FIRSTJ and SECONDI are identical when used in a semiring
            #define GB_MNAME _firstj
            #include "GB_AxB_positional_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
        case GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j)) == k+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            // FIRSTJ1 and SECONDI1 are identical when used in a semiring
            #define GB_MNAME _firstj1
            #include "GB_AxB_positional_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_SECONDJ_binop_code  :   // z = second_j(x,B(i,j)) == j
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            #define GB_MNAME _secondj
            #include "GB_AxB_positional_factory.c"
            break ;

        //----------------------------------------------------------------------
        case GB_SECONDJ1_binop_code :   // z = second_j1(x,B(i,j)) == j+1
        //----------------------------------------------------------------------

            // 10 semirings: (min,max,times,plus,any) * (int32,int64)
            #define GB_MNAME _secondj1
            #include "GB_AxB_positional_factory.c"
            break ;

        default: ;
    }
}

