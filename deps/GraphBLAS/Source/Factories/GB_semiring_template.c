//------------------------------------------------------------------------------
// GB_semiring_template.c: built-in unary and binary functions and operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd many times in GB_ops.c to define the built-in
// semirings.  That file has defined either GB_BOOLEAN, or GB_TYPE as one of
// the 10 non-boolean types.

// Using built-in types and operators, many unique semirings can be built.
// Below is the list of pre-defined semirings in SuiteSparse:GraphBLAS:

// 1000 semirings with a multiply operator TxT -> T where T is non-Boolean, from
// the complete cross product of:

//      5 monoids: MIN, MAX, PLUS, TIMES, ANY
//      20 multiply operators:
//          FIRST, SECOND, PAIR, MIN, MAX, PLUS, MINUS, RMINUS, TIMES, DIV, RDIV
//          ISEQ, ISNE, ISGT, ISLT, ISGE, ISLE,
//          LOR, LAND, LXOR
//      10 non-Boolean types, T

//      a single instance of this file creates 100 semirings of this
//      form, of one type T, when T is not BOOL

// 300 semirings with a comparator TxT -> bool, where T is
// non-Boolean, from the complete cross product of:

//      5 Boolean monoids: LAND, LOR, LXOR, EQ, ANY
//      6 multiply operators: EQ, NE, GT, LT, GE, LE
//      10 non-Boolean types, T

//      a single instance of this file creates 30 semirings of this form,
//      of one type T, when T is not BOOL

// 55 semirings with purely Boolean types, bool x bool -> bool, from the
// complete cross product of:

//      5 Boolean monoids: LAND, LOR, LXOR, EQ, ANY
//      11 multiply operators:
//          FIRST, SECOND, PAIR, LOR, LAND, LXOR, EQ, GT, LT, GE, LE

//      a single instance of this file creates all 5*11 = 55 purely Boolean
//      semirings, when T is BOOL and GB_BOOLEAN is defined

// 54 complex semirings:

//      3 complex monoids: PLUS, TIMES, ANY
//      9 multiply operators: FIRST, SECOND, PAIR, PLUS, MINUS, TIMES,
//          DIV, RDIV, RMINUS
//      2 complex types (FC32 and FC64)

// 64 bitwise semirings:

//      4 bitwise monoids: BOR, BAND, BXOR, BXNOR
//      4 bitwise multiply operators: BOR, BAND, BXOR, BXNOR
//      4 unsigned integer types: UINT8, UINT16, UINT32, UINT64

// 80 positional semirings:

//      5 monoids: MIN, MAX, PLUS, TIMES, ANY
//      8 multiply operators:
//          FIRSTI, FIRSTI1, FIRSTJ, FIRSTJ1,
//          SECONDI, SECONDI1, SECONDJ, SECONDJ1
//      2 types: INT32, INT64

#if defined ( GB_BOOLEAN )

    //--------------------------------------------------------------------------
    // 55 purely Boolean semirings
    //--------------------------------------------------------------------------

    // All types in these 44 semirings are BOOL

    // 11 semirings with LOR monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( LOR   , FIRST  )
    GXB_SEMIRING ( LOR   , SECOND )
    GXB_SEMIRING ( LOR   , PAIR   )
    GXB_SEMIRING ( LOR   , LOR    )
    GXB_SEMIRING ( LOR   , LAND   )
    GXB_SEMIRING ( LOR   , LXOR   )
    GXB_SEMIRING ( LOR   , EQ     )
    GXB_SEMIRING ( LOR   , GT     )
    GXB_SEMIRING ( LOR   , LT     )
    GXB_SEMIRING ( LOR   , GE     )
    GXB_SEMIRING ( LOR   , LE     )

    // 11 semirings with LAND monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( LAND  , FIRST  )
    GXB_SEMIRING ( LAND  , SECOND )
    GXB_SEMIRING ( LAND  , PAIR   )
    GXB_SEMIRING ( LAND  , LOR    )
    GXB_SEMIRING ( LAND  , LAND   )
    GXB_SEMIRING ( LAND  , LXOR   )
    GXB_SEMIRING ( LAND  , EQ     )
    GXB_SEMIRING ( LAND  , GT     )
    GXB_SEMIRING ( LAND  , LT     )
    GXB_SEMIRING ( LAND  , GE     )
    GXB_SEMIRING ( LAND  , LE     )

    // 11 semirings with LXOR monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( LXOR  , FIRST  )
    GXB_SEMIRING ( LXOR  , SECOND )
    GXB_SEMIRING ( LXOR  , PAIR   )
    GXB_SEMIRING ( LXOR  , LOR    )
    GXB_SEMIRING ( LXOR  , LAND   )
    GXB_SEMIRING ( LXOR  , LXOR   )
    GXB_SEMIRING ( LXOR  , EQ     )
    GXB_SEMIRING ( LXOR  , GT     )
    GXB_SEMIRING ( LXOR  , LT     )
    GXB_SEMIRING ( LXOR  , GE     )
    GXB_SEMIRING ( LXOR  , LE     )

    // 11 semirings with EQ monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( EQ    , FIRST  )
    GXB_SEMIRING ( EQ    , SECOND )
    GXB_SEMIRING ( EQ    , PAIR   )
    GXB_SEMIRING ( EQ    , LOR    )
    GXB_SEMIRING ( EQ    , LAND   )
    GXB_SEMIRING ( EQ    , LXOR   )
    GXB_SEMIRING ( EQ    , EQ     )
    GXB_SEMIRING ( EQ    , GT     )
    GXB_SEMIRING ( EQ    , LT     )
    GXB_SEMIRING ( EQ    , GE     )
    GXB_SEMIRING ( EQ    , LE     )

    // 11 semirings with ANY monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( ANY   , FIRST  )
    GXB_SEMIRING ( ANY   , SECOND )
    GXB_SEMIRING ( ANY   , PAIR   )
    GXB_SEMIRING ( ANY   , LOR    )
    GXB_SEMIRING ( ANY   , LAND   )
    GXB_SEMIRING ( ANY   , LXOR   )
    GXB_SEMIRING ( ANY   , EQ     )
    GXB_SEMIRING ( ANY   , GT     )
    GXB_SEMIRING ( ANY   , LT     )
    GXB_SEMIRING ( ANY   , GE     )
    GXB_SEMIRING ( ANY   , LE     )

#elif defined ( GB_COMPLEX )

    //--------------------------------------------------------------------------
    // 27 complex semirings of the form TxT->t
    //--------------------------------------------------------------------------

    // 9 semirings with PLUS monoid
    GXB_SEMIRING ( PLUS  , FIRST  )
    GXB_SEMIRING ( PLUS  , SECOND )
    GXB_SEMIRING ( PLUS  , PAIR   )
    GXB_SEMIRING ( PLUS  , PLUS   )
    GXB_SEMIRING ( PLUS  , MINUS  )
    GXB_SEMIRING ( PLUS  , RMINUS )
    GXB_SEMIRING ( PLUS  , TIMES  )
    GXB_SEMIRING ( PLUS  , DIV    )
    GXB_SEMIRING ( PLUS  , RDIV   )

    // 9 semirings with TIMES monoid
    GXB_SEMIRING ( TIMES , FIRST  )
    GXB_SEMIRING ( TIMES , SECOND )
    GXB_SEMIRING ( TIMES , PAIR   )
    GXB_SEMIRING ( TIMES , PLUS   )
    GXB_SEMIRING ( TIMES , MINUS  )
    GXB_SEMIRING ( TIMES , RMINUS )
    GXB_SEMIRING ( TIMES , TIMES  )
    GXB_SEMIRING ( TIMES , DIV    )
    GXB_SEMIRING ( TIMES , RDIV   )

    // 9 semirings with ANY monoid
    GXB_SEMIRING ( ANY   , FIRST  )
    GXB_SEMIRING ( ANY   , SECOND )
    GXB_SEMIRING ( ANY   , PAIR   )
    GXB_SEMIRING ( ANY   , PLUS   )
    GXB_SEMIRING ( ANY   , MINUS  )
    GXB_SEMIRING ( ANY   , RMINUS )
    GXB_SEMIRING ( ANY   , TIMES  )
    GXB_SEMIRING ( ANY   , DIV    )
    GXB_SEMIRING ( ANY   , RDIV   )

#else

    //--------------------------------------------------------------------------
    // 100 semirings of the form TxT->T
    //--------------------------------------------------------------------------

    // All types in these semirings are the same.  These are defined for
    // the 10 non-Boolean types, not when T is BOOL.

    // 20 semirings with MIN monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( MIN   , FIRST  )
    GXB_SEMIRING ( MIN   , SECOND )
    GXB_SEMIRING ( MIN   , PAIR   )
    GXB_SEMIRING ( MIN   , MIN    )
    GXB_SEMIRING ( MIN   , MAX    )
    GXB_SEMIRING ( MIN   , PLUS   )
    GXB_SEMIRING ( MIN   , MINUS  )
    GXB_SEMIRING ( MIN   , RMINUS )
    GXB_SEMIRING ( MIN   , TIMES  )
    GXB_SEMIRING ( MIN   , DIV    )
    GXB_SEMIRING ( MIN   , RDIV   )
    GXB_SEMIRING ( MIN   , ISEQ   )
    GXB_SEMIRING ( MIN   , ISNE   )
    GXB_SEMIRING ( MIN   , ISGT   )
    GXB_SEMIRING ( MIN   , ISLT   )
    GXB_SEMIRING ( MIN   , ISGE   )
    GXB_SEMIRING ( MIN   , ISLE   )
    GXB_SEMIRING ( MIN   , LOR    )
    GXB_SEMIRING ( MIN   , LAND   )
    GXB_SEMIRING ( MIN   , LXOR   )

    // 20 semirings with MAX monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( MAX   , FIRST  )
    GXB_SEMIRING ( MAX   , SECOND )
    GXB_SEMIRING ( MAX   , PAIR   )
    GXB_SEMIRING ( MAX   , MIN    )
    GXB_SEMIRING ( MAX   , MAX    )
    GXB_SEMIRING ( MAX   , PLUS   )
    GXB_SEMIRING ( MAX   , MINUS  )
    GXB_SEMIRING ( MAX   , RMINUS )
    GXB_SEMIRING ( MAX   , TIMES  )
    GXB_SEMIRING ( MAX   , DIV    )
    GXB_SEMIRING ( MAX   , RDIV   )
    GXB_SEMIRING ( MAX   , ISEQ   )
    GXB_SEMIRING ( MAX   , ISNE   )
    GXB_SEMIRING ( MAX   , ISGT   )
    GXB_SEMIRING ( MAX   , ISLT   )
    GXB_SEMIRING ( MAX   , ISGE   )
    GXB_SEMIRING ( MAX   , ISLE   )
    GXB_SEMIRING ( MAX   , LOR    )
    GXB_SEMIRING ( MAX   , LAND   )
    GXB_SEMIRING ( MAX   , LXOR   )

    // 20 semirings with PLUS monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( PLUS  , FIRST  )
    GXB_SEMIRING ( PLUS  , SECOND )
    GXB_SEMIRING ( PLUS  , PAIR   )
    GXB_SEMIRING ( PLUS  , MIN    )
    GXB_SEMIRING ( PLUS  , MAX    )
    GXB_SEMIRING ( PLUS  , PLUS   )
    GXB_SEMIRING ( PLUS  , MINUS  )
    GXB_SEMIRING ( PLUS  , RMINUS )
    GXB_SEMIRING ( PLUS  , TIMES  )
    GXB_SEMIRING ( PLUS  , DIV    )
    GXB_SEMIRING ( PLUS  , RDIV   )
    GXB_SEMIRING ( PLUS  , ISEQ   )
    GXB_SEMIRING ( PLUS  , ISNE   )
    GXB_SEMIRING ( PLUS  , ISGT   )
    GXB_SEMIRING ( PLUS  , ISLT   )
    GXB_SEMIRING ( PLUS  , ISGE   )
    GXB_SEMIRING ( PLUS  , ISLE   )
    GXB_SEMIRING ( PLUS  , LOR    )
    GXB_SEMIRING ( PLUS  , LAND   )
    GXB_SEMIRING ( PLUS  , LXOR   )

    // 20 semirings with TIMES monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( TIMES , FIRST  )
    GXB_SEMIRING ( TIMES , SECOND )
    GXB_SEMIRING ( TIMES , PAIR   )
    GXB_SEMIRING ( TIMES , MIN    )
    GXB_SEMIRING ( TIMES , MAX    )
    GXB_SEMIRING ( TIMES , PLUS   )
    GXB_SEMIRING ( TIMES , MINUS  )
    GXB_SEMIRING ( TIMES , RMINUS )
    GXB_SEMIRING ( TIMES , TIMES  )
    GXB_SEMIRING ( TIMES , DIV    )
    GXB_SEMIRING ( TIMES , RDIV   )
    GXB_SEMIRING ( TIMES , ISEQ   )
    GXB_SEMIRING ( TIMES , ISNE   )
    GXB_SEMIRING ( TIMES , ISGT   )
    GXB_SEMIRING ( TIMES , ISLT   )
    GXB_SEMIRING ( TIMES , ISGE   )
    GXB_SEMIRING ( TIMES , ISLE   )
    GXB_SEMIRING ( TIMES , LOR    )
    GXB_SEMIRING ( TIMES , LAND   )
    GXB_SEMIRING ( TIMES , LXOR   )

    // 20 semirings with ANY monoid; the 2nd argument is the multiply operator
    GXB_SEMIRING ( ANY   , FIRST  )
    GXB_SEMIRING ( ANY   , SECOND )
    GXB_SEMIRING ( ANY   , PAIR   )
    GXB_SEMIRING ( ANY   , MIN    )
    GXB_SEMIRING ( ANY   , MAX    )
    GXB_SEMIRING ( ANY   , PLUS   )
    GXB_SEMIRING ( ANY   , MINUS  )
    GXB_SEMIRING ( ANY   , RMINUS )
    GXB_SEMIRING ( ANY   , TIMES  )
    GXB_SEMIRING ( ANY   , DIV    )
    GXB_SEMIRING ( ANY   , RDIV   )
    GXB_SEMIRING ( ANY   , ISEQ   )
    GXB_SEMIRING ( ANY   , ISNE   )
    GXB_SEMIRING ( ANY   , ISGT   )
    GXB_SEMIRING ( ANY   , ISLT   )
    GXB_SEMIRING ( ANY   , ISGE   )
    GXB_SEMIRING ( ANY   , ISLE   )
    GXB_SEMIRING ( ANY   , LOR    )
    GXB_SEMIRING ( ANY   , LAND   )
    GXB_SEMIRING ( ANY   , LXOR   )

    //--------------------------------------------------------------------------
    // 30 semirings of the form TxT -> bool
    //--------------------------------------------------------------------------

    // The multiply operator has the form z=compare(x,y), where x and y are of
    // type T, and z is Boolean.  These operators are combined with the four
    // Boolean monoids.

    // These are defined when T is one of the 10 non-Boolean types, not when T
    // is BOOL

    // 6 semrings with LOR monoid; the 2nd argument is the comparator
    GXB_SEMIRING_COMPARE ( LOR  , EQ )
    GXB_SEMIRING_COMPARE ( LOR  , NE )
    GXB_SEMIRING_COMPARE ( LOR  , GT )
    GXB_SEMIRING_COMPARE ( LOR  , LT )
    GXB_SEMIRING_COMPARE ( LOR  , GE )
    GXB_SEMIRING_COMPARE ( LOR  , LE )

    // 6 semrings with LAND monoid; the 2nd argument is the comparator
    GXB_SEMIRING_COMPARE ( LAND , EQ )
    GXB_SEMIRING_COMPARE ( LAND , NE )
    GXB_SEMIRING_COMPARE ( LAND , GT )
    GXB_SEMIRING_COMPARE ( LAND , LT )
    GXB_SEMIRING_COMPARE ( LAND , GE )
    GXB_SEMIRING_COMPARE ( LAND , LE )

    // 6 semrings with LXOR monoid; the 2nd argument is the comparator
    GXB_SEMIRING_COMPARE ( LXOR , EQ )
    GXB_SEMIRING_COMPARE ( LXOR , NE )
    GXB_SEMIRING_COMPARE ( LXOR , GT )
    GXB_SEMIRING_COMPARE ( LXOR , LT )
    GXB_SEMIRING_COMPARE ( LXOR , GE )
    GXB_SEMIRING_COMPARE ( LXOR , LE )

    // 6 semrings with EQ monoid; the 2nd argument is the comparator
    GXB_SEMIRING_COMPARE ( EQ   , EQ )
    GXB_SEMIRING_COMPARE ( EQ   , NE )
    GXB_SEMIRING_COMPARE ( EQ   , GT )
    GXB_SEMIRING_COMPARE ( EQ   , LT )
    GXB_SEMIRING_COMPARE ( EQ   , GE )
    GXB_SEMIRING_COMPARE ( EQ   , LE )

    // 6 semrings with ANY monoid; the 2nd argument is the comparator
    GXB_SEMIRING_COMPARE ( ANY  , EQ )
    GXB_SEMIRING_COMPARE ( ANY  , NE )
    GXB_SEMIRING_COMPARE ( ANY  , GT )
    GXB_SEMIRING_COMPARE ( ANY  , LT )
    GXB_SEMIRING_COMPARE ( ANY  , GE )
    GXB_SEMIRING_COMPARE ( ANY  , LE )

#endif

#if defined ( GB_UNSIGNED_INT )

    //--------------------------------------------------------------------------
    // 16 bitwise semirings (for unsigned integers only)
    //--------------------------------------------------------------------------

    GXB_SEMIRING ( BOR   , BOR    )
    GXB_SEMIRING ( BOR   , BAND   )
    GXB_SEMIRING ( BOR   , BXOR   )
    GXB_SEMIRING ( BOR   , BXNOR  )

    GXB_SEMIRING ( BAND  , BOR    )
    GXB_SEMIRING ( BAND  , BAND   )
    GXB_SEMIRING ( BAND  , BXOR   )
    GXB_SEMIRING ( BAND  , BXNOR  )

    GXB_SEMIRING ( BXOR  , BOR    )
    GXB_SEMIRING ( BXOR  , BAND   )
    GXB_SEMIRING ( BXOR  , BXOR   )
    GXB_SEMIRING ( BXOR  , BXNOR  )

    GXB_SEMIRING ( BXNOR , BOR    )
    GXB_SEMIRING ( BXNOR , BAND   )
    GXB_SEMIRING ( BXNOR , BXOR   )
    GXB_SEMIRING ( BXNOR , BXNOR  )

#endif

#if defined ( GB_POSITIONAL )

    //--------------------------------------------------------------------------
    // 40 positional semirings:
    //--------------------------------------------------------------------------

    GXB_SEMIRING ( MIN   , FIRSTI   )
    GXB_SEMIRING ( MIN   , FIRSTI1  )
    GXB_SEMIRING ( MIN   , FIRSTJ   )
    GXB_SEMIRING ( MIN   , FIRSTJ1  )
    GXB_SEMIRING ( MIN   , SECONDI  )
    GXB_SEMIRING ( MIN   , SECONDI1 )
    GXB_SEMIRING ( MIN   , SECONDJ  )
    GXB_SEMIRING ( MIN   , SECONDJ1 )

    GXB_SEMIRING ( MAX   , FIRSTI   )
    GXB_SEMIRING ( MAX   , FIRSTI1  )
    GXB_SEMIRING ( MAX   , FIRSTJ   )
    GXB_SEMIRING ( MAX   , FIRSTJ1  )
    GXB_SEMIRING ( MAX   , SECONDI  )
    GXB_SEMIRING ( MAX   , SECONDI1 )
    GXB_SEMIRING ( MAX   , SECONDJ  )
    GXB_SEMIRING ( MAX   , SECONDJ1 )

    GXB_SEMIRING ( ANY   , FIRSTI   )
    GXB_SEMIRING ( ANY   , FIRSTI1  )
    GXB_SEMIRING ( ANY   , FIRSTJ   )
    GXB_SEMIRING ( ANY   , FIRSTJ1  )
    GXB_SEMIRING ( ANY   , SECONDI  )
    GXB_SEMIRING ( ANY   , SECONDI1 )
    GXB_SEMIRING ( ANY   , SECONDJ  )
    GXB_SEMIRING ( ANY   , SECONDJ1 )

    GXB_SEMIRING ( PLUS  , FIRSTI   )
    GXB_SEMIRING ( PLUS  , FIRSTI1  )
    GXB_SEMIRING ( PLUS  , FIRSTJ   )
    GXB_SEMIRING ( PLUS  , FIRSTJ1  )
    GXB_SEMIRING ( PLUS  , SECONDI  )
    GXB_SEMIRING ( PLUS  , SECONDI1 )
    GXB_SEMIRING ( PLUS  , SECONDJ  )
    GXB_SEMIRING ( PLUS  , SECONDJ1 )

    GXB_SEMIRING ( TIMES , FIRSTI   )
    GXB_SEMIRING ( TIMES , FIRSTI1  )
    GXB_SEMIRING ( TIMES , FIRSTJ   )
    GXB_SEMIRING ( TIMES , FIRSTJ1  )
    GXB_SEMIRING ( TIMES , SECONDI  )
    GXB_SEMIRING ( TIMES , SECONDI1 )
    GXB_SEMIRING ( TIMES , SECONDJ  )
    GXB_SEMIRING ( TIMES , SECONDJ1 )

#endif

#undef GB_XTYPE
#undef GB_BOOLEAN
#undef GB_COMPLEX
#undef GB_UNSIGNED_INT
#undef GB_POSITIONAL

