//------------------------------------------------------------------------------
// GB_control.h:  disable hard-coded functions to reduce code size
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The maximal way to reduce code size and compile time is to use the COMPACT
// cmake option (-DCOMPACT=1).  That setting disables all FactoryKernels.
// Performance will then rely on the JIT kernels, once they are compiled at run
// time, or from an PreJIT kernels.  See the User Guide for details.  If
// COMPACT is enabled, the settings in this file have no additional effect.

// The installer of SuiteSparse:GraphBLAS can edit this file to reduce the code
// size of the compiled library, by seletively disabling the corresponding
// hard-coded functions in Source/FactoryKernels.  For example, if
// SuiteSparse:GraphBLAS is integrated into an application that makes no use of
// the GrB_INT16 data type, or just occassional use where performance is not a
// concern, then uncomment the line "#define GxB_NO_INT16 1".  Alternatively,
// SuiteSparse:GraphBLAS can be compiled with a list of options, such as
// -DGxB_NO_INT16=1, which does the same thing.

// GraphBLAS will still work as expected.  It will simply use a JIT or PreJIT
// kernel, if the JIT is enabled.  Failing that, it relies on a generic method
// instead of the type- or operator-specific code.  The JIT and PreJIT kernels
// are fast.  The generic methods will be slower, by about 2x or 3x, depending
// on the operation, but their results will be the same.  A few operations will
// be 10x slower when using generic methods, such as GrB_reduce to scalar using
// the GrB_MAX_FP64 operator.

// Note that some semirings are renamed.  For example, C=A*B when all matrices
// are in CSC format, uses the semiring as-is.  If all matrices are in CSR
// format instead, then C'=B'*A' is computed, treating the internal matrices
// as if they are in CSC format.  To accomplish this, the semiring may be
// "flipped", if the multiply operator is not commutative.  That is,
// the GxB_PLUS_FIRST_* semiring is replaced with GxB_PLUS_SECOND_*.  Below
// is a list of all multiplicative operators and their "flipped" pair.

// As a result of the "flip", if the FIRST operator is disabled, it may disable
// some uses of the GxB_*_SECOND_* semirings, and visa versa, depending on the
// matrix formats.  I recommend that if you want to use the FIRST operator with
// fast hard-coded semirings, then do not disable FIRST or SECOND.  The
// following is a complete list of all pairs of operators that may be replaced
// with the other.  I recommend either keeping both of each pair, or disabling
// both.

        // FIRST and SECOND
        // LT and GT
        // LE and GE
        // ISLT and ISGT
        // ISLE and ISGE
        // DIV and RDIV
        // MINUS and RMINUS

// In addition, many of the Boolean operators are not unique, and are renamed
// internally.  The operators listed here are just the unique ones.  In
// particular, GrB_DIV_BOOL is identical to GrB_FIRST_BOOL, so the GrB_DIV_BOOL
// operator is replaced internally with GrB_FIRST_BOOL.

        // FIRST and DIV                        : FIRST is used for both
        // SECOND and RDIV                      : SECOND
        // MIN, TIMES, and LAND                 : LAND
        // MAX, PLUS, and LOR                   : LOR
        // ISNE, NE, MINUS, RMINUS, and LXOR    : LXOR
        // ISEQ, EQ                             : EQ
        // ISGT, GT                             : GT
        // ISLT, LT                             : LT
        // ISGE, GE                             : GE
        // ISLE, LE                             : LE

// Thus, below there is a #define GxB_NO_LAND_FIRST_BOOL, but no #define
// GxB_NO_LAND_DIV_BOOL.

// The ANY_PAIR semirings appear in Source/GB_AxB__any_pair_iso.c and
// cannot be disabled.

// In this version of SuiteSparse:GraphBLAS, some of the fast hard-coded
// semirings have been disabled below.  They still work, but are now slower
// since the work is now done by the generic semiring instead.  These semirings
// are likely not needed by any application, and disabling them here saves
// compile time and reduces the size of the compiled library.  Hard-coded
// semirings removed:

//  (1) *_IS* semirings are removed.
//  (2) semirings with DIV, RDIV, MINUS, RMINUS and ANY multiplicative
//      operators are removed.  ANY still appears as a monoid for many fast
//      hard-coded semirings, just not as a multiplicative operator.
//  (3) MIN, MAX, and TIMES monoids with the PAIR, LAND, LOR and LXOR
//      multiplicative operators are removed.
//  (4) PLUS monoids with LAND, LOR, and LXOR multiplicative ops.
//  (5) MAX_MAX, MIN_MIN, semirings are removed.
//  (5) ANY monoids with the EQ, NE, GE, LE, GT, LT, LAND, LOR, LXOR, MAX, MIN,
//      PLUS and TIMES operators.
//  (6) boolean semirings with non-boolean inputs removed
//      (EQ_LT_FP32 for example), and boolean inputs with comparators
//      (GE, GT, LE, LT)
//  (7) positional semirings with PLUS and TIMES monoid removed.
//  (8) some complex semirings removed.

// With the above semirings removed, the remaining 398 semirings are:

//  25 boolean semirings
//
//      monoid  multiply ops
//      EQ:     EQ (=LXNOR), LAND, LOR, LXOR (=NE), FIRST, SECOND
//      LAND:   EQ (=LXNOR), LAND, LOR, LXOR (=NE), FIRST, SECOND
//      LOR:    EQ (=LXNOR), LAND, LOR, LXOR (=NE), FIRST, SECOND
//      LXOR:   EQ (=LXNOR), LAND, LOR, LXOR (=NE), FIRST, SECOND, PAIR.
//
//              note: EQ_BOOL and LXNOR are the same operator, and
//                    NE_BOOL and LXOR  are the same operator.
//
//  100 semirings with MIN/MAX monoids (10 kinds, 10 real types each):
//
//      monoid  multiply ops
//      MAX:    MIN, PLUS, TIMES, FIRST, SECOND
//      MIN:    MAX, PLUS, TIMES, FIRST, SECOND
//
//  70 semirings with PLUS monoids (10 real types each):
//  60 semirings with TIMES monoids (10 real types each):
//
//      monoid  multiply ops
//      PLUS:   MIN, MAX, PLUS, TIMES, FIRST, SECOND, PAIR
//      TIMES:  MIN, MAX, PLUS, TIMES, FIRST, SECOND
//
//  10 semirings for 2 complex types (PLUS and TIMES monoids only):
//
//      monoid  multiply ops
//      PLUS:  TIMES, FIRST, SECOND
//      TIMES: FIRST, SECOND
//
//  33 semirings enabled with the ANY monoid:
//
//      ANY:   FIRST, SECOND, PAIR (with bool, 10 real types)
//
//  64 bitwise semirings: for 4 unsigned integer types:
//
//      (BOR, BAND, BXOR, BXNOR) x (BOR, BAND, BXOR, BXNOR)
//
//  36 positional semirings:
//
//      monoids: (MIN, MAX, ANY) x
//      mult:    (FIRSTI, FIRSTI1, FIRSTJ, FIRSTJ1, SECONDJ, SECONDJ1) x
//      types:   (int32, int64)
//
//  note:
//      EQ_PAIR_BOOL, XNOR_PAIR_BOOL, LAND_PAIR_BOOL, LOR_PAIR_BOOL, are all
//      the same as ANY_PAIR_BOOL.  For the other types, MAX_PAIR, MIN_PAIR,
//      and TIMES_PAIR are the same as ANY_PAIR.  These are excluded from the
//      list above, but are just as fast as ANY_PAIR.

// These changes have no effect on the performance of binary operations such
// as eWiseAdd, eWiseMult, or the unary GrB_apply to GrB_reduce.  They only
// affect GrB_mxm, GrB_mxv, and GrB_vxm.

// To renable the fast versions of these semirings, simply comment out the
// specific "#define GxB_NO..." statements below, and recompile this library.

//------------------------------------------------------------------------------
// uncomment any of these lines to disable the types
//------------------------------------------------------------------------------

// These disable all semirings with the corresponding type of x,y for the
// multiplicative operator, for GrB_mxm, GrB_vxm, and GrB_mxv.

// They also disable the hard-coded functions for GrB_eWiseAdd, GrB_eWiseMult,
// GrB_reduce, GrB_*_build, GrB_apply, and GrB_transpose for this type.

// If disabled, the types still work just fine, but operations on them will be
// slower.

// #define GxB_NO_BOOL      1
// #define GxB_NO_FP32      1
// #define GxB_NO_FP64      1
// #define GxB_NO_FC32      1
// #define GxB_NO_FC64      1
// #define GxB_NO_INT16     1
// #define GxB_NO_INT32     1
// #define GxB_NO_INT64     1
// #define GxB_NO_INT8      1
// #define GxB_NO_UINT16    1
// #define GxB_NO_UINT32    1
// #define GxB_NO_UINT64    1
// #define GxB_NO_UINT8     1

//------------------------------------------------------------------------------
// uncomment any of these lines to disable the unary operators
//------------------------------------------------------------------------------

// These disable all unary operators for GrB_apply.  These options have no
// effect on GrB_mxm, GrB_vxm, GrB_mxv, GrB_eWiseAdd, GrB_eWiseMult,
// GrB_reduce, or GrB_*_build.

// Any disabled unary operators will still work just fine, but operations using
// them will be slower.

// #define GxB_NO_ABS       1
// #define GxB_NO_AINV      1
// #define GxB_NO_IDENTITY  1
// #define GxB_NO_LNOT      1
// #define GxB_NO_MINV      1
// #define GxB_NO_ONE       1
// #define GxB_NO_BNOT      1

// #define GxB_NO_SQRT      1
// #define GxB_NO_LOG       1
// #define GxB_NO_EXP       1

// #define GxB_NO_SIN       1
// #define GxB_NO_COS       1
// #define GxB_NO_TAN       1

// #define GxB_NO_ASIN      1
// #define GxB_NO_ACOS      1
// #define GxB_NO_ATAN      1

// #define GxB_NO_SINH      1
// #define GxB_NO_COSH      1
// #define GxB_NO_TANH      1

// #define GxB_NO_ASINH     1
// #define GxB_NO_ACOSH     1
// #define GxB_NO_ATANH     1

// #define GxB_NO_SIGNUM    1
// #define GxB_NO_CEIL      1
// #define GxB_NO_FLOOR     1
// #define GxB_NO_ROUND     1
// #define GxB_NO_TRUNC     1

// #define GxB_NO_EXP2      1
// #define GxB_NO_EXPM1     1
// #define GxB_NO_LOG10     1
// #define GxB_NO_LOG1P     1
// #define GxB_NO_LOG2      1

// #define GxB_NO_LGAMMA    1
// #define GxB_NO_TGAMMA    1
// #define GxB_NO_ERF       1
// #define GxB_NO_ERFC      1
// #define GxB_NO_CBRT      1

// #define GxB_NO_FREXPX    1
// #define GxB_NO_FREXPE    1

// #define GxB_NO_CONJ      1
// #define GxB_NO_CREAL     1
// #define GxB_NO_CIMAG     1
// #define GxB_NO_CARG      1

// #define GxB_NO_ISINF     1
// #define GxB_NO_ISNAN     1
// #define GxB_NO_ISFINITE  1

//------------------------------------------------------------------------------
// uncomment any of these lines to disable the binary operators for all types
//------------------------------------------------------------------------------

// These disable all semirings with the corresponding additive or
// multiplicative operator, for GrB_mxm, GrB_vxm, and GrB_mxv.

// They also disable the hard-coded functions for GrB_eWiseAdd, GrB_eWiseMult,
// GrB_reduce, and GrB_*_build for this binary operator.

// Any disabled binary operators will still work just fine, but operations
// using them will be slower.

// #define GxB_NO_FIRST     1
// #define GxB_NO_SECOND    1
// #define GxB_NO_PAIR      1
// #define GxB_NO_ANY       1
// #define GxB_NO_MIN       1
// #define GxB_NO_MAX       1
// #define GxB_NO_PLUS      1
// #define GxB_NO_MINUS     1
// #define GxB_NO_RMINUS    1
// #define GxB_NO_TIMES     1
// #define GxB_NO_DIV       1
// #define GxB_NO_RDIV      1
// #define GxB_NO_ISEQ      1
// #define GxB_NO_ISNE      1
// #define GxB_NO_ISGT      1
// #define GxB_NO_ISGE      1
// #define GxB_NO_ISLT      1
// #define GxB_NO_ISLE      1
// #define GxB_NO_EQ        1
// #define GxB_NO_NE        1
// #define GxB_NO_GT        1
// #define GxB_NO_LT        1
// #define GxB_NO_LE        1
// #define GxB_NO_GE        1
// #define GxB_NO_LAND      1
// #define GxB_NO_LOR       1
// #define GxB_NO_LXOR      1

// #define GxB_NO_BOR       1
// #define GxB_NO_BAND      1
// #define GxB_NO_BXOR      1
// #define GxB_NO_BXNOR     1
// #define GxB_NO_BGET      1
// #define GxB_NO_BSET      1
// #define GxB_NO_BCLR      1
// #define GxB_NO_BSHIFT    1

// #define GxB_NO_ATAN2     1
// #define GxB_NO_HYPOT     1
// #define GxB_NO_FMOD      1
// #define GxB_NO_REMAINDER 1
// #define GxB_NO_COPYSIGN  1
// #define GxB_NO_LDEXP     1

// #define GxB_NO_CMPLX     1
// #define GxB_NO_POW       1

// #define GxB_NO_FIRSTI    1
// #define GxB_NO_FIRSTI1   1
// #define GxB_NO_FIRSTJ    1
// #define GxB_NO_FIRSTJ1   1
// #define GxB_NO_SECONDI   1
// #define GxB_NO_SECONDI1  1
// #define GxB_NO_SECONDJ   1
// #define GxB_NO_SECONDJ1  1

//------------------------------------------------------------------------------
// uncomment any of these lines to disable the binary operators for each type
//------------------------------------------------------------------------------

// These disable all semirings with the corresponding additive or
// multiplicative operator with the particular type, for GrB_mxm, GrB_vxm, and
// GrB_mxv.

// They also disable the hard-coded functions for GrB_eWiseAdd, GrB_eWiseMult,
// GrB_reduce, and GrB_*_build for this binary operator, for the particular
// type.

// These options are more precise than the ones above.  For example,
// GxB_NO_TIMES, above, disables the GrB_TIMES_<type> operator for all types
// (except bool, which is renamed as GrB_LAND).  GxB_NO_INT16, above, disables
// all operators of the form GrB_<op>_INT16.  But GxB_NO_TIMES_INT16, listed
// below, disables just the single GrB_TIMES_INT16 operator, and all semirings
// that use it.  It has no effect on GrB_TIMES_FP64, for example, or any other
// operator.

// Note that some boolean operators do not appear.  GrB_MIN_BOOL and
// GrB_TIMES_BOOL are renamed internally to GrB_LAND_BOOL, for example, so
// uncommenting GxB_NO_LAND_BOOL disables all three operators.

// Any disabled binary operators will still work just fine, but operations
// using them will be slower.

// also needed by GrB_reduce to vector
// #define GxB_NO_FIRST_INT8    1
// #define GxB_NO_FIRST_INT16   1
// #define GxB_NO_FIRST_INT32   1
// #define GxB_NO_FIRST_INT64   1
// #define GxB_NO_FIRST_UINT8   1
// #define GxB_NO_FIRST_UINT16  1
// #define GxB_NO_FIRST_UINT32  1
// #define GxB_NO_FIRST_UINT64  1
// #define GxB_NO_FIRST_FP32    1
// #define GxB_NO_FIRST_FP64    1
// #define GxB_NO_FIRST_FC32    1
// #define GxB_NO_FIRST_FC64    1
// #define GxB_NO_FIRST_BOOL    1

// also needed by GrB_reduce to vector
// #define GxB_NO_SECOND_INT8   1
// #define GxB_NO_SECOND_INT16  1
// #define GxB_NO_SECOND_INT32  1
// #define GxB_NO_SECOND_INT64  1
// #define GxB_NO_SECOND_UINT8  1
// #define GxB_NO_SECOND_UINT16 1
// #define GxB_NO_SECOND_UINT32 1
// #define GxB_NO_SECOND_UINT64 1
// #define GxB_NO_SECOND_FP32   1
// #define GxB_NO_SECOND_FP64   1
// #define GxB_NO_SECOND_FC32   1
// #define GxB_NO_SECOND_FC64   1
// #define GxB_NO_SECOND_BOOL   1

// #define GxB_NO_PAIR_INT8     1
// #define GxB_NO_PAIR_INT16    1
// #define GxB_NO_PAIR_INT32    1
// #define GxB_NO_PAIR_INT64    1
// #define GxB_NO_PAIR_UINT8    1
// #define GxB_NO_PAIR_UINT16   1
// #define GxB_NO_PAIR_UINT32   1
// #define GxB_NO_PAIR_UINT64   1
// #define GxB_NO_PAIR_FP32     1
// #define GxB_NO_PAIR_FP64     1
// #define GxB_NO_PAIR_FC32     1
// #define GxB_NO_PAIR_FC64     1
// #define GxB_NO_PAIR_BOOL     1

// #define GxB_NO_ANY_INT8      1
// #define GxB_NO_ANY_INT16     1
// #define GxB_NO_ANY_INT32     1
// #define GxB_NO_ANY_INT64     1
// #define GxB_NO_ANY_UINT8     1
// #define GxB_NO_ANY_UINT16    1
// #define GxB_NO_ANY_UINT32    1
// #define GxB_NO_ANY_UINT64    1
// #define GxB_NO_ANY_FP32      1
// #define GxB_NO_ANY_FP64      1
// #define GxB_NO_ANY_FC32      1
// #define GxB_NO_ANY_FC64      1
// #define GxB_NO_ANY_BOOL      1

// #define GxB_NO_MIN_INT8      1
// #define GxB_NO_MIN_INT16     1
// #define GxB_NO_MIN_INT32     1
// #define GxB_NO_MIN_INT64     1
// #define GxB_NO_MIN_UINT8     1
// #define GxB_NO_MIN_UINT16    1
// #define GxB_NO_MIN_UINT32    1
// #define GxB_NO_MIN_UINT64    1
// #define GxB_NO_MIN_FP32      1
// #define GxB_NO_MIN_FP64      1

// #define GxB_NO_MAX_INT8      1
// #define GxB_NO_MAX_INT16     1
// #define GxB_NO_MAX_INT32     1
// #define GxB_NO_MAX_INT64     1
// #define GxB_NO_MAX_UINT8     1
// #define GxB_NO_MAX_UINT16    1
// #define GxB_NO_MAX_UINT32    1
// #define GxB_NO_MAX_UINT64    1
// #define GxB_NO_MAX_FP32      1
// #define GxB_NO_MAX_FP64      1

// #define GxB_NO_PLUS_INT8     1
// #define GxB_NO_PLUS_INT16    1
// #define GxB_NO_PLUS_INT32    1
// #define GxB_NO_PLUS_INT64    1
// #define GxB_NO_PLUS_UINT8    1
// #define GxB_NO_PLUS_UINT16   1
// #define GxB_NO_PLUS_UINT32   1
// #define GxB_NO_PLUS_UINT64   1
// #define GxB_NO_PLUS_FP32     1
// #define GxB_NO_PLUS_FP64     1
// #define GxB_NO_PLUS_FC32     1
// #define GxB_NO_PLUS_FC64     1

// #define GxB_NO_MINUS_INT8    1
// #define GxB_NO_MINUS_INT16   1
// #define GxB_NO_MINUS_INT32   1
// #define GxB_NO_MINUS_INT64   1
// #define GxB_NO_MINUS_UINT8   1
// #define GxB_NO_MINUS_UINT16  1
// #define GxB_NO_MINUS_UINT32  1
// #define GxB_NO_MINUS_UINT64  1
// #define GxB_NO_MINUS_FP32    1
// #define GxB_NO_MINUS_FP64    1
// #define GxB_NO_MINUS_FC32    1
// #define GxB_NO_MINUS_FC64    1

// #define GxB_NO_RMINUS_INT8   1
// #define GxB_NO_RMINUS_INT16  1
// #define GxB_NO_RMINUS_INT32  1
// #define GxB_NO_RMINUS_INT64  1
// #define GxB_NO_RMINUS_UINT8  1
// #define GxB_NO_RMINUS_UINT16 1
// #define GxB_NO_RMINUS_UINT32 1
// #define GxB_NO_RMINUS_UINT64 1
// #define GxB_NO_RMINUS_FP32   1
// #define GxB_NO_RMINUS_FP64   1
// #define GxB_NO_RMINUS_FC32   1
// #define GxB_NO_RMINUS_FC64   1

// #define GxB_NO_TIMES_INT8    1
// #define GxB_NO_TIMES_INT16   1
// #define GxB_NO_TIMES_INT32   1
// #define GxB_NO_TIMES_INT64   1
// #define GxB_NO_TIMES_UINT8   1
// #define GxB_NO_TIMES_UINT16  1
// #define GxB_NO_TIMES_UINT32  1
// #define GxB_NO_TIMES_UINT64  1
// #define GxB_NO_TIMES_FP32    1
// #define GxB_NO_TIMES_FP64    1
// #define GxB_NO_TIMES_FC32    1
// #define GxB_NO_TIMES_FC64    1

// #define GxB_NO_DIV_INT8      1
// #define GxB_NO_DIV_INT16     1
// #define GxB_NO_DIV_INT32     1
// #define GxB_NO_DIV_INT64     1
// #define GxB_NO_DIV_UINT8     1
// #define GxB_NO_DIV_UINT16    1
// #define GxB_NO_DIV_UINT32    1
// #define GxB_NO_DIV_UINT64    1
// #define GxB_NO_DIV_FP32      1
// #define GxB_NO_DIV_FP64      1
// #define GxB_NO_DIV_FC32      1
// #define GxB_NO_DIV_FC64      1

// #define GxB_NO_RDIV_INT8     1
// #define GxB_NO_RDIV_INT16    1
// #define GxB_NO_RDIV_INT32    1
// #define GxB_NO_RDIV_INT64    1
// #define GxB_NO_RDIV_UINT8    1
// #define GxB_NO_RDIV_UINT16   1
// #define GxB_NO_RDIV_UINT32   1
// #define GxB_NO_RDIV_UINT64   1
// #define GxB_NO_RDIV_FP32     1
// #define GxB_NO_RDIV_FP64     1
// #define GxB_NO_RDIV_FC32     1
// #define GxB_NO_RDIV_FC64     1

// #define GxB_NO_ISEQ_INT8     1
// #define GxB_NO_ISEQ_INT16    1
// #define GxB_NO_ISEQ_INT32    1
// #define GxB_NO_ISEQ_INT64    1
// #define GxB_NO_ISEQ_UINT8    1
// #define GxB_NO_ISEQ_UINT16   1
// #define GxB_NO_ISEQ_UINT32   1
// #define GxB_NO_ISEQ_UINT64   1
// #define GxB_NO_ISEQ_FP32     1
// #define GxB_NO_ISEQ_FP64     1
// #define GxB_NO_ISEQ_FC32     1
// #define GxB_NO_ISEQ_FC64     1

// #define GxB_NO_ISNE_INT8     1
// #define GxB_NO_ISNE_INT16    1
// #define GxB_NO_ISNE_INT32    1
// #define GxB_NO_ISNE_INT64    1
// #define GxB_NO_ISNE_UINT8    1
// #define GxB_NO_ISNE_UINT16   1
// #define GxB_NO_ISNE_UINT32   1
// #define GxB_NO_ISNE_UINT64   1
// #define GxB_NO_ISNE_FP32     1
// #define GxB_NO_ISNE_FP64     1
// #define GxB_NO_ISNE_FC32     1
// #define GxB_NO_ISNE_FC64     1

// #define GxB_NO_ISGT_INT8     1
// #define GxB_NO_ISGT_INT16    1
// #define GxB_NO_ISGT_INT32    1
// #define GxB_NO_ISGT_INT64    1
// #define GxB_NO_ISGT_UINT8    1
// #define GxB_NO_ISGT_UINT16   1
// #define GxB_NO_ISGT_UINT32   1
// #define GxB_NO_ISGT_UINT64   1
// #define GxB_NO_ISGT_FP32     1
// #define GxB_NO_ISGT_FP64     1

// #define GxB_NO_ISLT_INT8     1
// #define GxB_NO_ISLT_INT16    1
// #define GxB_NO_ISLT_INT32    1
// #define GxB_NO_ISLT_INT64    1
// #define GxB_NO_ISLT_UINT8    1
// #define GxB_NO_ISLT_UINT16   1
// #define GxB_NO_ISLT_UINT32   1
// #define GxB_NO_ISLT_UINT64   1
// #define GxB_NO_ISLT_FP32     1
// #define GxB_NO_ISLT_FP64     1

// #define GxB_NO_ISGE_INT8     1
// #define GxB_NO_ISGE_INT16    1
// #define GxB_NO_ISGE_INT32    1
// #define GxB_NO_ISGE_INT64    1
// #define GxB_NO_ISGE_UINT8    1
// #define GxB_NO_ISGE_UINT16   1
// #define GxB_NO_ISGE_UINT32   1
// #define GxB_NO_ISGE_UINT64   1
// #define GxB_NO_ISGE_FP32     1
// #define GxB_NO_ISGE_FP64     1

// #define GxB_NO_ISLE_INT8     1
// #define GxB_NO_ISLE_INT16    1
// #define GxB_NO_ISLE_INT32    1
// #define GxB_NO_ISLE_INT64    1
// #define GxB_NO_ISLE_UINT8    1
// #define GxB_NO_ISLE_UINT16   1
// #define GxB_NO_ISLE_UINT32   1
// #define GxB_NO_ISLE_UINT64   1
// #define GxB_NO_ISLE_FP32     1
// #define GxB_NO_ISLE_FP64     1

// #define GxB_NO_EQ_INT8       1
// #define GxB_NO_EQ_INT16      1
// #define GxB_NO_EQ_INT32      1
// #define GxB_NO_EQ_INT64      1
// #define GxB_NO_EQ_UINT8      1
// #define GxB_NO_EQ_UINT16     1
// #define GxB_NO_EQ_UINT32     1
// #define GxB_NO_EQ_UINT64     1
// #define GxB_NO_EQ_FP32       1
// #define GxB_NO_EQ_FP64       1
// #define GxB_NO_EQ_BOOL       1
// #define GxB_NO_EQ_FC32       1
// #define GxB_NO_EQ_FC64       1

// #define GxB_NO_NE_INT8       1
// #define GxB_NO_NE_INT16      1
// #define GxB_NO_NE_INT32      1
// #define GxB_NO_NE_INT64      1
// #define GxB_NO_NE_UINT8      1
// #define GxB_NO_NE_UINT16     1
// #define GxB_NO_NE_UINT32     1
// #define GxB_NO_NE_UINT64     1
// #define GxB_NO_NE_FP32       1
// #define GxB_NO_NE_FP64       1
// #define GxB_NO_NE_FC32       1
// #define GxB_NO_NE_FC64       1

// #define GxB_NO_GT_INT8       1
// #define GxB_NO_GT_INT16      1
// #define GxB_NO_GT_INT32      1
// #define GxB_NO_GT_INT64      1
// #define GxB_NO_GT_UINT8      1
// #define GxB_NO_GT_UINT16     1
// #define GxB_NO_GT_UINT32     1
// #define GxB_NO_GT_UINT64     1
// #define GxB_NO_GT_FP32       1
// #define GxB_NO_GT_FP64       1
// #define GxB_NO_GT_BOOL       1

// #define GxB_NO_LT_INT8       1
// #define GxB_NO_LT_INT16      1
// #define GxB_NO_LT_INT32      1
// #define GxB_NO_LT_INT64      1
// #define GxB_NO_LT_UINT8      1
// #define GxB_NO_LT_UINT16     1
// #define GxB_NO_LT_UINT32     1
// #define GxB_NO_LT_UINT64     1
// #define GxB_NO_LT_FP32       1
// #define GxB_NO_LT_FP64       1
// #define GxB_NO_LT_BOOL       1

// #define GxB_NO_GE_INT8       1
// #define GxB_NO_GE_INT16      1
// #define GxB_NO_GE_INT32      1
// #define GxB_NO_GE_INT64      1
// #define GxB_NO_GE_UINT8      1
// #define GxB_NO_GE_UINT16     1
// #define GxB_NO_GE_UINT32     1
// #define GxB_NO_GE_UINT64     1
// #define GxB_NO_GE_FP32       1
// #define GxB_NO_GE_FP64       1
// #define GxB_NO_GE_BOOL       1

// #define GxB_NO_LE_INT8       1
// #define GxB_NO_LE_INT16      1
// #define GxB_NO_LE_INT32      1
// #define GxB_NO_LE_INT64      1
// #define GxB_NO_LE_UINT8      1
// #define GxB_NO_LE_UINT16     1
// #define GxB_NO_LE_UINT32     1
// #define GxB_NO_LE_UINT64     1
// #define GxB_NO_LE_FP32       1
// #define GxB_NO_LE_FP64       1
// #define GxB_NO_LE_BOOL

// #define GxB_NO_LOR_INT8      1
// #define GxB_NO_LOR_INT16     1
// #define GxB_NO_LOR_INT32     1
// #define GxB_NO_LOR_INT64     1
// #define GxB_NO_LOR_UINT8     1
// #define GxB_NO_LOR_UINT16    1
// #define GxB_NO_LOR_UINT32    1
// #define GxB_NO_LOR_UINT64    1
// #define GxB_NO_LOR_FP32      1
// #define GxB_NO_LOR_FP64      1
// #define GxB_NO_LOR_BOOL      1

// #define GxB_NO_LAND_INT8     1
// #define GxB_NO_LAND_INT16    1
// #define GxB_NO_LAND_INT32    1
// #define GxB_NO_LAND_INT64    1
// #define GxB_NO_LAND_UINT8    1
// #define GxB_NO_LAND_UINT16   1
// #define GxB_NO_LAND_UINT32   1
// #define GxB_NO_LAND_UINT64   1
// #define GxB_NO_LAND_FP32     1
// #define GxB_NO_LAND_FP64     1
// #define GxB_NO_LAND_BOOL     1

// #define GxB_NO_LXOR_INT8     1
// #define GxB_NO_LXOR_INT16    1
// #define GxB_NO_LXOR_INT32    1
// #define GxB_NO_LXOR_INT64    1
// #define GxB_NO_LXOR_UINT8    1
// #define GxB_NO_LXOR_UINT16   1
// #define GxB_NO_LXOR_UINT32   1
// #define GxB_NO_LXOR_UINT64   1
// #define GxB_NO_LXOR_FP32     1
// #define GxB_NO_LXOR_FP64     1
// #define GxB_NO_LXOR_BOOL     1

// #define GxB_NO_ATAN2_FP32    1
// #define GxB_NO_ATAN2_FP64    1

// #define GxB_NO_HYPOT_FP32    1
// #define GxB_NO_HYPOT_FP64    1

// #define GxB_NO_FMOD_FP32     1
// #define GxB_NO_FMOD_FP64     1

// #define GxB_NO_REMAINDER_FP32 1
// #define GxB_NO_REMAINDER_FP64 1

// #define GxB_NO_COPYSIGN_FP32 1
// #define GxB_NO_COPYSIGN_FP64 1

// #define GxB_NO_LDEXP_FP32    1
// #define GxB_NO_LDEXP_FP64    1

// #define GxB_NO_CMPLX_FP32    1
// #define GxB_NO_CMPLX_FP64    1

// #define GxB_NO_POW_INT8      1
// #define GxB_NO_POW_INT16     1
// #define GxB_NO_POW_INT32     1
// #define GxB_NO_POW_INT64     1
// #define GxB_NO_POW_UINT8     1
// #define GxB_NO_POW_UINT16    1
// #define GxB_NO_POW_UINT32    1
// #define GxB_NO_POW_UINT64    1
// #define GxB_NO_POW_FP32      1
// #define GxB_NO_POW_FP64      1
// #define GxB_NO_POW_FC32      1
// #define GxB_NO_POW_FC64      1
// #define GxB_NO_POW_BOOL      1

// #define GxB_NO_BOR_INT8      1
// #define GxB_NO_BOR_INT16     1
// #define GxB_NO_BOR_INT32     1
// #define GxB_NO_BOR_INT64     1
// #define GxB_NO_BOR_UINT8     1
// #define GxB_NO_BOR_UINT16    1
// #define GxB_NO_BOR_UINT32    1
// #define GxB_NO_BOR_UINT64    1

// #define GxB_NO_BAND_INT8      1
// #define GxB_NO_BAND_INT16     1
// #define GxB_NO_BAND_INT32     1
// #define GxB_NO_BAND_INT64     1
// #define GxB_NO_BAND_UINT8     1
// #define GxB_NO_BAND_UINT16    1
// #define GxB_NO_BAND_UINT32    1
// #define GxB_NO_BAND_UINT64    1

// #define GxB_NO_BXOR_INT8      1
// #define GxB_NO_BXOR_INT16     1
// #define GxB_NO_BXOR_INT32     1
// #define GxB_NO_BXOR_INT64     1
// #define GxB_NO_BXOR_UINT8     1
// #define GxB_NO_BXOR_UINT16    1
// #define GxB_NO_BXOR_UINT32    1
// #define GxB_NO_BXOR_UINT64    1

// #define GxB_NO_BXNOR_INT8      1
// #define GxB_NO_BXNOR_INT16     1
// #define GxB_NO_BXNOR_INT32     1
// #define GxB_NO_BXNOR_INT64     1
// #define GxB_NO_BXNOR_UINT8     1
// #define GxB_NO_BXNOR_UINT16    1
// #define GxB_NO_BXNOR_UINT32    1
// #define GxB_NO_BXNOR_UINT64    1

// #define GxB_NO_BGET_INT8      1
// #define GxB_NO_BGET_INT16     1
// #define GxB_NO_BGET_INT32     1
// #define GxB_NO_BGET_INT64     1
// #define GxB_NO_BGET_UINT8     1
// #define GxB_NO_BGET_UINT16    1
// #define GxB_NO_BGET_UINT32    1
// #define GxB_NO_BGET_UINT64    1

// #define GxB_NO_BSET_INT8      1
// #define GxB_NO_BSET_INT16     1
// #define GxB_NO_BSET_INT32     1
// #define GxB_NO_BSET_INT64     1
// #define GxB_NO_BSET_UINT8     1
// #define GxB_NO_BSET_UINT16    1
// #define GxB_NO_BSET_UINT32    1
// #define GxB_NO_BSET_UINT64    1

// #define GxB_NO_BCLR_INT8      1
// #define GxB_NO_BCLR_INT16     1
// #define GxB_NO_BCLR_INT32     1
// #define GxB_NO_BCLR_INT64     1
// #define GxB_NO_BCLR_UINT8     1
// #define GxB_NO_BCLR_UINT16    1
// #define GxB_NO_BCLR_UINT32    1
// #define GxB_NO_BCLR_UINT64    1

// #define GxB_NO_BSHIFT_INT8      1
// #define GxB_NO_BSHIFT_INT16     1
// #define GxB_NO_BSHIFT_INT32     1
// #define GxB_NO_BSHIFT_INT64     1
// #define GxB_NO_BSHIFT_UINT8     1
// #define GxB_NO_BSHIFT_UINT16    1
// #define GxB_NO_BSHIFT_UINT32    1
// #define GxB_NO_BSHIFT_UINT64    1

// #define GxB_NO_FIRSTI_INT64    1
// #define GxB_NO_FIRSTI1_INT64   1
// #define GxB_NO_FIRSTJ_INT64    1
// #define GxB_NO_FIRSTJ1_INT64   1
// #define GxB_NO_SECONDI_INT64   1
// #define GxB_NO_SECONDI1_INT64  1
// #define GxB_NO_SECONDJ_INT64   1
// #define GxB_NO_SECONDJ1_INT64  1

// #define GxB_NO_FIRSTI_INT32    1
// #define GxB_NO_FIRSTI1_INT32   1
// #define GxB_NO_FIRSTJ_INT32    1
// #define GxB_NO_FIRSTJ1_INT32   1
// #define GxB_NO_SECONDI_INT32   1
// #define GxB_NO_SECONDI1_INT32  1
// #define GxB_NO_SECONDJ_INT32   1
// #define GxB_NO_SECONDJ1_INT32  1

//------------------------------------------------------------------------------
// uncomment any of these lines to disable the corresponding semiring
//------------------------------------------------------------------------------

// These options are still more precise.  They disable all semirings with the
// corresponding semiring, for GrB_mxm, GrB_vxm, and GrB_mxv.  For example, if
// GrB_NO_FP64 is enabled above, then all semirings of the form
// GxB_<addop>_<multop>_FP64 are disabled, for any addop or multop operators.
// If GrB_NO_PLUS is enabled above, then all semirings of the form
// GxB_PLUS_<multop>_<type> and GxB_<multop>_<PLUS>_<type> are disabled.

// These options have no effect on GrB_eWiseAdd, GrB_eWiseMult,
// GrB_*_build, or GrB_apply.  They do affect GrB_reduce to vector, which
// converts the reduction of a vector into a matrix-vector multiplication.

// Any disabled semirings will still work just fine, but operations using them
// will be slower.

//------------------------------------------------------------
// semirings with the boolean EQ monoid (also called XNOR)
//------------------------------------------------------------

// The only builtin GrB_* semiring that uses the EQ (LXNOR) monoid
// is LXNOR_LOR_BOOL == EQ_LOR_BOOL.

// 6 semirings with the EQ monoid

// #define GxB_NO_EQ_EQ_BOOL            1
// #define GxB_NO_EQ_LAND_BOOL          1
// builtin: GrB_LXNOR_LOR_SEMIRING_BOOL == GxB_EQ_LOR_BOOL:
// #define GxB_NO_EQ_LOR_BOOL           1
// #define GxB_NO_EQ_LXOR_BOOL          1
// FIRST and SECOND are needed by GrB_reduce to vector:
// #define GxB_NO_EQ_FIRST_BOOL         1
// #define GxB_NO_EQ_SECOND_BOOL        1

   #define GxB_NO_EQ_EQ_FP32            1
   #define GxB_NO_EQ_EQ_FP64            1
   #define GxB_NO_EQ_EQ_INT16           1
   #define GxB_NO_EQ_EQ_INT32           1
   #define GxB_NO_EQ_EQ_INT64           1
   #define GxB_NO_EQ_EQ_INT8            1
   #define GxB_NO_EQ_EQ_UINT16          1
   #define GxB_NO_EQ_EQ_UINT32          1
   #define GxB_NO_EQ_EQ_UINT64          1
   #define GxB_NO_EQ_EQ_UINT8           1
   #define GxB_NO_EQ_EQ_FC32            1
   #define GxB_NO_EQ_EQ_FC64            1

   #define GxB_NO_EQ_ANY_BOOL           1

   #define GxB_NO_EQ_GE_BOOL            1
   #define GxB_NO_EQ_GE_FP32            1
   #define GxB_NO_EQ_GE_FP64            1
   #define GxB_NO_EQ_GE_INT16           1
   #define GxB_NO_EQ_GE_INT32           1
   #define GxB_NO_EQ_GE_INT64           1
   #define GxB_NO_EQ_GE_INT8            1
   #define GxB_NO_EQ_GE_UINT16          1
   #define GxB_NO_EQ_GE_UINT32          1
   #define GxB_NO_EQ_GE_UINT64          1
   #define GxB_NO_EQ_GE_UINT8           1

   #define GxB_NO_EQ_GT_BOOL            1
   #define GxB_NO_EQ_GT_FP32            1
   #define GxB_NO_EQ_GT_FP64            1
   #define GxB_NO_EQ_GT_INT16           1
   #define GxB_NO_EQ_GT_INT32           1
   #define GxB_NO_EQ_GT_INT64           1
   #define GxB_NO_EQ_GT_INT8            1
   #define GxB_NO_EQ_GT_UINT16          1
   #define GxB_NO_EQ_GT_UINT32          1
   #define GxB_NO_EQ_GT_UINT64          1
   #define GxB_NO_EQ_GT_UINT8           1

   #define GxB_NO_EQ_LE_BOOL            1
   #define GxB_NO_EQ_LE_FP32            1
   #define GxB_NO_EQ_LE_FP64            1
   #define GxB_NO_EQ_LE_INT16           1
   #define GxB_NO_EQ_LE_INT32           1
   #define GxB_NO_EQ_LE_INT64           1
   #define GxB_NO_EQ_LE_INT8            1
   #define GxB_NO_EQ_LE_UINT16          1
   #define GxB_NO_EQ_LE_UINT32          1
   #define GxB_NO_EQ_LE_UINT64          1
   #define GxB_NO_EQ_LE_UINT8           1

   #define GxB_NO_EQ_LT_BOOL            1
   #define GxB_NO_EQ_LT_FP32            1
   #define GxB_NO_EQ_LT_FP64            1
   #define GxB_NO_EQ_LT_INT16           1
   #define GxB_NO_EQ_LT_INT32           1
   #define GxB_NO_EQ_LT_INT64           1
   #define GxB_NO_EQ_LT_INT8            1
   #define GxB_NO_EQ_LT_UINT16          1
   #define GxB_NO_EQ_LT_UINT32          1
   #define GxB_NO_EQ_LT_UINT64          1
   #define GxB_NO_EQ_LT_UINT8           1

   #define GxB_NO_EQ_NE_FP32            1
   #define GxB_NO_EQ_NE_FP64            1
   #define GxB_NO_EQ_NE_INT16           1
   #define GxB_NO_EQ_NE_INT32           1
   #define GxB_NO_EQ_NE_INT64           1
   #define GxB_NO_EQ_NE_INT8            1
   #define GxB_NO_EQ_NE_UINT16          1
   #define GxB_NO_EQ_NE_UINT32          1
   #define GxB_NO_EQ_NE_UINT64          1
   #define GxB_NO_EQ_NE_UINT8           1
   #define GxB_NO_EQ_NE_FC32            1
   #define GxB_NO_EQ_NE_FC64            1

//------------------------------------------------------------
// semirings with the boolean LAND monoid
//------------------------------------------------------------

// The only builtin GrB_* semiring that uses the LAND monoid is LAND_LOR_BOOL

// 6 semirings with the LAND monoid

// #define GxB_NO_LAND_EQ_BOOL          1
// #define GxB_NO_LAND_LAND_BOOL        1
// builtin: GrB_LAND_LOR_SEMIRING_BOOL == GxB_LAND_LOR_BOOL:
// #define GxB_NO_LAND_LOR_BOOL         1
// #define GxB_NO_LAND_LXOR_BOOL        1
// FIRST and SECOND are needed by GrB_reduce to vector:
// #define GxB_NO_LAND_FIRST_BOOL       1
// #define GxB_NO_LAND_SECOND_BOOL      1

   #define GxB_NO_LAND_EQ_FP32          1
   #define GxB_NO_LAND_EQ_FP64          1
   #define GxB_NO_LAND_EQ_INT16         1
   #define GxB_NO_LAND_EQ_INT32         1
   #define GxB_NO_LAND_EQ_INT64         1
   #define GxB_NO_LAND_EQ_INT8          1
   #define GxB_NO_LAND_EQ_UINT16        1
   #define GxB_NO_LAND_EQ_UINT32        1
   #define GxB_NO_LAND_EQ_UINT64        1
   #define GxB_NO_LAND_EQ_UINT8         1
   #define GxB_NO_LAND_EQ_FC32          1
   #define GxB_NO_LAND_EQ_FC64          1

   #define GxB_NO_LAND_ANY_BOOL         1

   #define GxB_NO_LAND_GE_BOOL          1
   #define GxB_NO_LAND_GE_FP32          1
   #define GxB_NO_LAND_GE_FP64          1
   #define GxB_NO_LAND_GE_INT16         1
   #define GxB_NO_LAND_GE_INT32         1
   #define GxB_NO_LAND_GE_INT64         1
   #define GxB_NO_LAND_GE_INT8          1
   #define GxB_NO_LAND_GE_UINT16        1
   #define GxB_NO_LAND_GE_UINT32        1
   #define GxB_NO_LAND_GE_UINT64        1
   #define GxB_NO_LAND_GE_UINT8         1

   #define GxB_NO_LAND_GT_BOOL          1

   #define GxB_NO_LAND_GT_FP32          1
   #define GxB_NO_LAND_GT_FP64          1
   #define GxB_NO_LAND_GT_INT16         1
   #define GxB_NO_LAND_GT_INT32         1
   #define GxB_NO_LAND_GT_INT64         1
   #define GxB_NO_LAND_GT_INT8          1
   #define GxB_NO_LAND_GT_UINT16        1
   #define GxB_NO_LAND_GT_UINT32        1
   #define GxB_NO_LAND_GT_UINT64        1
   #define GxB_NO_LAND_GT_UINT8         1

   #define GxB_NO_LAND_LE_BOOL          1
   #define GxB_NO_LAND_LE_FP32          1
   #define GxB_NO_LAND_LE_FP64          1
   #define GxB_NO_LAND_LE_INT16         1
   #define GxB_NO_LAND_LE_INT32         1
   #define GxB_NO_LAND_LE_INT64         1
   #define GxB_NO_LAND_LE_INT8          1
   #define GxB_NO_LAND_LE_UINT16        1
   #define GxB_NO_LAND_LE_UINT32        1
   #define GxB_NO_LAND_LE_UINT64        1
   #define GxB_NO_LAND_LE_UINT8         1

   #define GxB_NO_LAND_LT_BOOL          1
   #define GxB_NO_LAND_LT_FP32          1
   #define GxB_NO_LAND_LT_FP64          1
   #define GxB_NO_LAND_LT_INT16         1
   #define GxB_NO_LAND_LT_INT32         1
   #define GxB_NO_LAND_LT_INT64         1
   #define GxB_NO_LAND_LT_INT8          1
   #define GxB_NO_LAND_LT_UINT16        1
   #define GxB_NO_LAND_LT_UINT32        1
   #define GxB_NO_LAND_LT_UINT64        1
   #define GxB_NO_LAND_LT_UINT8         1

   #define GxB_NO_LAND_NE_FP32          1
   #define GxB_NO_LAND_NE_FP64          1
   #define GxB_NO_LAND_NE_INT16         1
   #define GxB_NO_LAND_NE_INT32         1
   #define GxB_NO_LAND_NE_INT64         1
   #define GxB_NO_LAND_NE_INT8          1
   #define GxB_NO_LAND_NE_UINT16        1
   #define GxB_NO_LAND_NE_UINT32        1
   #define GxB_NO_LAND_NE_UINT64        1
   #define GxB_NO_LAND_NE_UINT8         1
   #define GxB_NO_LAND_NE_FC32          1
   #define GxB_NO_LAND_NE_FC64          1

//------------------------------------------------------------
// semirings with the boolean LOR monoid
//------------------------------------------------------------

// The only builtin GrB_* semiring that uses the LOR monoid is LOR_LAND_BOOL

// 6 semirings with the LOR monoid

// #define GxB_NO_LOR_EQ_BOOL           1
// builtin GrB_LOR_LAND_SEMIRING_BOOL == GxB_LOR_LAND_BOOL:
// #define GxB_NO_LOR_LAND_BOOL         1
// #define GxB_NO_LOR_LOR_BOOL          1
// #define GxB_NO_LOR_LXOR_BOOL         1
// FIRST and SECOND are needed by GrB_reduce to vector:
// #define GxB_NO_LOR_FIRST_BOOL        1
// #define GxB_NO_LOR_SECOND_BOOL       1

   #define GxB_NO_LOR_EQ_FP32           1
   #define GxB_NO_LOR_EQ_FP64           1
   #define GxB_NO_LOR_EQ_INT16          1
   #define GxB_NO_LOR_EQ_INT32          1
   #define GxB_NO_LOR_EQ_INT64          1
   #define GxB_NO_LOR_EQ_INT8           1
   #define GxB_NO_LOR_EQ_UINT16         1
   #define GxB_NO_LOR_EQ_UINT32         1
   #define GxB_NO_LOR_EQ_UINT64         1
   #define GxB_NO_LOR_EQ_UINT8          1
   #define GxB_NO_LOR_EQ_FC32           1
   #define GxB_NO_LOR_EQ_FC64           1

   #define GxB_NO_LOR_ANY_BOOL          1

   #define GxB_NO_LOR_GE_BOOL           1
   #define GxB_NO_LOR_GE_FP32           1
   #define GxB_NO_LOR_GE_FP64           1
   #define GxB_NO_LOR_GE_INT16          1
   #define GxB_NO_LOR_GE_INT32          1
   #define GxB_NO_LOR_GE_INT64          1
   #define GxB_NO_LOR_GE_INT8           1
   #define GxB_NO_LOR_GE_UINT16         1
   #define GxB_NO_LOR_GE_UINT32         1
   #define GxB_NO_LOR_GE_UINT64         1
   #define GxB_NO_LOR_GE_UINT8          1

   #define GxB_NO_LOR_GT_BOOL           1
   #define GxB_NO_LOR_GT_FP32           1
   #define GxB_NO_LOR_GT_FP64           1
   #define GxB_NO_LOR_GT_INT16          1
   #define GxB_NO_LOR_GT_INT32          1
   #define GxB_NO_LOR_GT_INT64          1
   #define GxB_NO_LOR_GT_INT8           1
   #define GxB_NO_LOR_GT_UINT16         1
   #define GxB_NO_LOR_GT_UINT32         1
   #define GxB_NO_LOR_GT_UINT64         1
   #define GxB_NO_LOR_GT_UINT8          1

   #define GxB_NO_LOR_LE_BOOL           1
   #define GxB_NO_LOR_LE_FP32           1
   #define GxB_NO_LOR_LE_FP64           1
   #define GxB_NO_LOR_LE_INT16          1
   #define GxB_NO_LOR_LE_INT32          1
   #define GxB_NO_LOR_LE_INT64          1
   #define GxB_NO_LOR_LE_INT8           1
   #define GxB_NO_LOR_LE_UINT16         1
   #define GxB_NO_LOR_LE_UINT32         1
   #define GxB_NO_LOR_LE_UINT64         1
   #define GxB_NO_LOR_LE_UINT8          1

   #define GxB_NO_LOR_LT_BOOL           1
   #define GxB_NO_LOR_LT_FP32           1
   #define GxB_NO_LOR_LT_FP64           1
   #define GxB_NO_LOR_LT_INT16          1
   #define GxB_NO_LOR_LT_INT32          1
   #define GxB_NO_LOR_LT_INT64          1
   #define GxB_NO_LOR_LT_INT8           1
   #define GxB_NO_LOR_LT_UINT16         1
   #define GxB_NO_LOR_LT_UINT32         1
   #define GxB_NO_LOR_LT_UINT64         1
   #define GxB_NO_LOR_LT_UINT8          1

   #define GxB_NO_LOR_NE_FP32           1
   #define GxB_NO_LOR_NE_FP64           1
   #define GxB_NO_LOR_NE_INT16          1
   #define GxB_NO_LOR_NE_INT32          1
   #define GxB_NO_LOR_NE_INT64          1
   #define GxB_NO_LOR_NE_INT8           1
   #define GxB_NO_LOR_NE_UINT16         1
   #define GxB_NO_LOR_NE_UINT32         1
   #define GxB_NO_LOR_NE_UINT64         1
   #define GxB_NO_LOR_NE_UINT8          1
   #define GxB_NO_LOR_NE_FC32           1
   #define GxB_NO_LOR_NE_FC64           1

//------------------------------------------------------------
// semirings with the boolean LXOR monoid (also called NE)
//------------------------------------------------------------

// The only builtin GrB_* semiring that uses the LXOR monoid is LXOR_LAND_BOOL

// 7 semirings with the LXOR monoid (PAIR is unique to this monoid)

// #define GxB_NO_LXOR_EQ_BOOL          1
// builtin: GrB_LXOR_LAND_SEMIRING_BOOL == GxB_LXOR_LAND_BOOL:
// #define GxB_NO_LXOR_LAND_BOOL        1
// #define GxB_NO_LXOR_LOR_BOOL         1
// #define GxB_NO_LXOR_LXOR_BOOL        1
// FIRST and SECOND are needed by GrB_reduce to vector:
// #define GxB_NO_LXOR_FIRST_BOOL       1
// #define GxB_NO_LXOR_SECOND_BOOL      1
// #define GxB_NO_LXOR_PAIR_BOOL        1

   #define GxB_NO_LXOR_EQ_FP32          1
   #define GxB_NO_LXOR_EQ_FP64          1
   #define GxB_NO_LXOR_EQ_INT16         1
   #define GxB_NO_LXOR_EQ_INT32         1
   #define GxB_NO_LXOR_EQ_INT64         1
   #define GxB_NO_LXOR_EQ_INT8          1
   #define GxB_NO_LXOR_EQ_UINT16        1
   #define GxB_NO_LXOR_EQ_UINT32        1
   #define GxB_NO_LXOR_EQ_UINT64        1
   #define GxB_NO_LXOR_EQ_UINT8         1
   #define GxB_NO_LXOR_EQ_FC32          1
   #define GxB_NO_LXOR_EQ_FC64          1

   #define GxB_NO_LXOR_ANY_BOOL         1

   #define GxB_NO_LXOR_GE_BOOL          1
   #define GxB_NO_LXOR_GE_FP32          1
   #define GxB_NO_LXOR_GE_FP64          1
   #define GxB_NO_LXOR_GE_INT16         1
   #define GxB_NO_LXOR_GE_INT32         1
   #define GxB_NO_LXOR_GE_INT64         1
   #define GxB_NO_LXOR_GE_INT8          1
   #define GxB_NO_LXOR_GE_UINT16        1
   #define GxB_NO_LXOR_GE_UINT32        1
   #define GxB_NO_LXOR_GE_UINT64        1
   #define GxB_NO_LXOR_GE_UINT8         1

   #define GxB_NO_LXOR_GT_BOOL          1
   #define GxB_NO_LXOR_GT_FP32          1
   #define GxB_NO_LXOR_GT_FP64          1
   #define GxB_NO_LXOR_GT_INT16         1
   #define GxB_NO_LXOR_GT_INT32         1
   #define GxB_NO_LXOR_GT_INT64         1
   #define GxB_NO_LXOR_GT_INT8          1
   #define GxB_NO_LXOR_GT_UINT16        1
   #define GxB_NO_LXOR_GT_UINT32        1
   #define GxB_NO_LXOR_GT_UINT64        1
   #define GxB_NO_LXOR_GT_UINT8         1

   #define GxB_NO_LXOR_LE_BOOL          1
   #define GxB_NO_LXOR_LE_FP32          1
   #define GxB_NO_LXOR_LE_FP64          1
   #define GxB_NO_LXOR_LE_INT16         1
   #define GxB_NO_LXOR_LE_INT32         1
   #define GxB_NO_LXOR_LE_INT64         1
   #define GxB_NO_LXOR_LE_INT8          1
   #define GxB_NO_LXOR_LE_UINT16        1
   #define GxB_NO_LXOR_LE_UINT32        1
   #define GxB_NO_LXOR_LE_UINT64        1
   #define GxB_NO_LXOR_LE_UINT8         1

   #define GxB_NO_LXOR_LT_BOOL          1
   #define GxB_NO_LXOR_LT_FP32          1
   #define GxB_NO_LXOR_LT_FP64          1
   #define GxB_NO_LXOR_LT_INT16         1
   #define GxB_NO_LXOR_LT_INT32         1
   #define GxB_NO_LXOR_LT_INT64         1
   #define GxB_NO_LXOR_LT_INT8          1
   #define GxB_NO_LXOR_LT_UINT16        1
   #define GxB_NO_LXOR_LT_UINT32        1
   #define GxB_NO_LXOR_LT_UINT64        1
   #define GxB_NO_LXOR_LT_UINT8         1

   #define GxB_NO_LXOR_NE_FP32          1
   #define GxB_NO_LXOR_NE_FP64          1
   #define GxB_NO_LXOR_NE_INT16         1
   #define GxB_NO_LXOR_NE_INT32         1
   #define GxB_NO_LXOR_NE_INT64         1
   #define GxB_NO_LXOR_NE_INT8          1
   #define GxB_NO_LXOR_NE_UINT16        1
   #define GxB_NO_LXOR_NE_UINT32        1
   #define GxB_NO_LXOR_NE_UINT64        1
   #define GxB_NO_LXOR_NE_UINT8         1
   #define GxB_NO_LXOR_NE_FC32          1
   #define GxB_NO_LXOR_NE_FC64          1

//------------------------------------------------------------
// semirings with the MAX monoid
//------------------------------------------------------------

// MAX_PLUS, MAX_TIMES, MAX_FIRST, MAX_SECOND, and MAX_MIN are GrB_* builtins.

//  50 semirings with MAX monoids (5 kinds, 10 real types each):
//
//      monoid  multiply ops
//      MAX:    MIN, PLUS, TIMES, FIRST, SECOND

// builtin GrB_*:
// #define GxB_NO_MAX_MIN_FP32          1
// #define GxB_NO_MAX_MIN_FP64          1
// #define GxB_NO_MAX_MIN_INT16         1
// #define GxB_NO_MAX_MIN_INT32         1
// #define GxB_NO_MAX_MIN_INT64         1
// #define GxB_NO_MAX_MIN_INT8          1
// #define GxB_NO_MAX_MIN_UINT16        1
// #define GxB_NO_MAX_MIN_UINT32        1
// #define GxB_NO_MAX_MIN_UINT64        1
// #define GxB_NO_MAX_MIN_UINT8         1

// builtin GrB_*:
// #define GxB_NO_MAX_PLUS_FP32         1
// #define GxB_NO_MAX_PLUS_FP64         1
// #define GxB_NO_MAX_PLUS_INT16        1
// #define GxB_NO_MAX_PLUS_INT32        1
// #define GxB_NO_MAX_PLUS_INT64        1
// #define GxB_NO_MAX_PLUS_INT8         1
// #define GxB_NO_MAX_PLUS_UINT16       1
// #define GxB_NO_MAX_PLUS_UINT32       1
// #define GxB_NO_MAX_PLUS_UINT64       1
// #define GxB_NO_MAX_PLUS_UINT8        1

// builtin GrB_*:
// #define GxB_NO_MAX_TIMES_FP32        1
// #define GxB_NO_MAX_TIMES_FP64        1
// #define GxB_NO_MAX_TIMES_INT16       1
// #define GxB_NO_MAX_TIMES_INT32       1
// #define GxB_NO_MAX_TIMES_INT64       1
// #define GxB_NO_MAX_TIMES_INT8        1
// #define GxB_NO_MAX_TIMES_UINT16      1
// #define GxB_NO_MAX_TIMES_UINT32      1
// #define GxB_NO_MAX_TIMES_UINT64      1
// #define GxB_NO_MAX_TIMES_UINT8       1

// builtin GrB_*: also needed by GrB_reduce to vector
// #define GxB_NO_MAX_FIRST_FP32        1
// #define GxB_NO_MAX_FIRST_FP64        1
// #define GxB_NO_MAX_FIRST_INT16       1
// #define GxB_NO_MAX_FIRST_INT32       1
// #define GxB_NO_MAX_FIRST_INT64       1
// #define GxB_NO_MAX_FIRST_INT8        1
// #define GxB_NO_MAX_FIRST_UINT16      1
// #define GxB_NO_MAX_FIRST_UINT32      1
// #define GxB_NO_MAX_FIRST_UINT64      1
// #define GxB_NO_MAX_FIRST_UINT8       1

// builtin GrB_*: also needed by GrB_reduce to vector
// #define GxB_NO_MAX_SECOND_FP32       1
// #define GxB_NO_MAX_SECOND_FP64       1
// #define GxB_NO_MAX_SECOND_INT16      1
// #define GxB_NO_MAX_SECOND_INT32      1
// #define GxB_NO_MAX_SECOND_INT64      1
// #define GxB_NO_MAX_SECOND_INT8       1
// #define GxB_NO_MAX_SECOND_UINT16     1
// #define GxB_NO_MAX_SECOND_UINT32     1
// #define GxB_NO_MAX_SECOND_UINT64     1
// #define GxB_NO_MAX_SECOND_UINT8      1

   #define GxB_NO_MAX_DIV_FP32          1
   #define GxB_NO_MAX_DIV_FP64          1
   #define GxB_NO_MAX_DIV_INT16         1
   #define GxB_NO_MAX_DIV_INT32         1
   #define GxB_NO_MAX_DIV_INT64         1
   #define GxB_NO_MAX_DIV_INT8          1
   #define GxB_NO_MAX_DIV_UINT16        1
   #define GxB_NO_MAX_DIV_UINT32        1
   #define GxB_NO_MAX_DIV_UINT64        1
   #define GxB_NO_MAX_DIV_UINT8         1

   #define GxB_NO_MAX_ANY_FP32          1
   #define GxB_NO_MAX_ANY_FP64          1
   #define GxB_NO_MAX_ANY_INT16         1
   #define GxB_NO_MAX_ANY_INT32         1
   #define GxB_NO_MAX_ANY_INT64         1
   #define GxB_NO_MAX_ANY_INT8          1
   #define GxB_NO_MAX_ANY_UINT16        1
   #define GxB_NO_MAX_ANY_UINT32        1
   #define GxB_NO_MAX_ANY_UINT64        1
   #define GxB_NO_MAX_ANY_UINT8         1

   #define GxB_NO_MAX_LAND_FP32         1
   #define GxB_NO_MAX_LAND_FP64         1
   #define GxB_NO_MAX_LAND_INT16        1
   #define GxB_NO_MAX_LAND_INT32        1
   #define GxB_NO_MAX_LAND_INT64        1
   #define GxB_NO_MAX_LAND_INT8         1
   #define GxB_NO_MAX_LAND_UINT16       1
   #define GxB_NO_MAX_LAND_UINT32       1
   #define GxB_NO_MAX_LAND_UINT64       1
   #define GxB_NO_MAX_LAND_UINT8        1

   #define GxB_NO_MAX_LOR_FP32          1
   #define GxB_NO_MAX_LOR_FP64          1
   #define GxB_NO_MAX_LOR_INT16         1
   #define GxB_NO_MAX_LOR_INT32         1
   #define GxB_NO_MAX_LOR_INT64         1
   #define GxB_NO_MAX_LOR_INT8          1
   #define GxB_NO_MAX_LOR_UINT16        1
   #define GxB_NO_MAX_LOR_UINT32        1
   #define GxB_NO_MAX_LOR_UINT64        1
   #define GxB_NO_MAX_LOR_UINT8         1

   #define GxB_NO_MAX_LXOR_FP32         1
   #define GxB_NO_MAX_LXOR_FP64         1
   #define GxB_NO_MAX_LXOR_INT16        1
   #define GxB_NO_MAX_LXOR_INT32        1
   #define GxB_NO_MAX_LXOR_INT64        1
   #define GxB_NO_MAX_LXOR_INT8         1
   #define GxB_NO_MAX_LXOR_UINT16       1
   #define GxB_NO_MAX_LXOR_UINT32       1
   #define GxB_NO_MAX_LXOR_UINT64       1
   #define GxB_NO_MAX_LXOR_UINT8        1

   #define GxB_NO_MAX_MAX_FP32          1
   #define GxB_NO_MAX_MAX_FP64          1
   #define GxB_NO_MAX_MAX_INT16         1
   #define GxB_NO_MAX_MAX_INT32         1
   #define GxB_NO_MAX_MAX_INT64         1
   #define GxB_NO_MAX_MAX_INT8          1
   #define GxB_NO_MAX_MAX_UINT16        1
   #define GxB_NO_MAX_MAX_UINT32        1
   #define GxB_NO_MAX_MAX_UINT64        1
   #define GxB_NO_MAX_MAX_UINT8         1

   #define GxB_NO_MAX_MINUS_FP32        1
   #define GxB_NO_MAX_MINUS_FP64        1
   #define GxB_NO_MAX_MINUS_INT16       1
   #define GxB_NO_MAX_MINUS_INT32       1
   #define GxB_NO_MAX_MINUS_INT64       1
   #define GxB_NO_MAX_MINUS_INT8        1
   #define GxB_NO_MAX_MINUS_UINT16      1
   #define GxB_NO_MAX_MINUS_UINT32      1
   #define GxB_NO_MAX_MINUS_UINT64      1
   #define GxB_NO_MAX_MINUS_UINT8       1

   #define GxB_NO_MAX_RDIV_FP32         1
   #define GxB_NO_MAX_RDIV_FP64         1
   #define GxB_NO_MAX_RDIV_INT16        1
   #define GxB_NO_MAX_RDIV_INT32        1
   #define GxB_NO_MAX_RDIV_INT64        1
   #define GxB_NO_MAX_RDIV_INT8         1
   #define GxB_NO_MAX_RDIV_UINT16       1
   #define GxB_NO_MAX_RDIV_UINT32       1
   #define GxB_NO_MAX_RDIV_UINT64       1
   #define GxB_NO_MAX_RDIV_UINT8        1

   #define GxB_NO_MAX_RMINUS_FP32       1
   #define GxB_NO_MAX_RMINUS_FP64       1
   #define GxB_NO_MAX_RMINUS_INT16      1
   #define GxB_NO_MAX_RMINUS_INT32      1
   #define GxB_NO_MAX_RMINUS_INT64      1
   #define GxB_NO_MAX_RMINUS_INT8       1
   #define GxB_NO_MAX_RMINUS_UINT16     1
   #define GxB_NO_MAX_RMINUS_UINT32     1
   #define GxB_NO_MAX_RMINUS_UINT64     1
   #define GxB_NO_MAX_RMINUS_UINT8      1

//------------------------------------------------------------
// semirings with the MIN monoid
//------------------------------------------------------------

// MIN_PLUS, MIN_TIMES, MIN_FIRST, MIN_SECOND, and MIN_MAX are GrB_* builtins.

//  50 semirings with MIN monoids (6 kinds, 10 real types each):
//
//      monoid  multiply ops
//      MIN:    MAX, PLUS, TIMES, FIRST, SECOND

// builtin GrB_*:
// #define GxB_NO_MIN_MAX_FP32          1
// #define GxB_NO_MIN_MAX_FP64          1
// #define GxB_NO_MIN_MAX_INT16         1
// #define GxB_NO_MIN_MAX_INT32         1
// #define GxB_NO_MIN_MAX_INT64         1
// #define GxB_NO_MIN_MAX_INT8          1
// #define GxB_NO_MIN_MAX_UINT16        1
// #define GxB_NO_MIN_MAX_UINT32        1
// #define GxB_NO_MIN_MAX_UINT64        1
// #define GxB_NO_MIN_MAX_UINT8         1

// builtin GrB_*:
// #define GxB_NO_MIN_PLUS_FP32         1
// #define GxB_NO_MIN_PLUS_FP64         1
// #define GxB_NO_MIN_PLUS_INT16        1
// #define GxB_NO_MIN_PLUS_INT32        1
// #define GxB_NO_MIN_PLUS_INT64        1
// #define GxB_NO_MIN_PLUS_INT8         1
// #define GxB_NO_MIN_PLUS_UINT16       1
// #define GxB_NO_MIN_PLUS_UINT32       1
// #define GxB_NO_MIN_PLUS_UINT64       1
// #define GxB_NO_MIN_PLUS_UINT8        1

// builtin GrB_*:
// #define GxB_NO_MIN_TIMES_FP32        1
// #define GxB_NO_MIN_TIMES_FP64        1
// #define GxB_NO_MIN_TIMES_INT16       1
// #define GxB_NO_MIN_TIMES_INT32       1
// #define GxB_NO_MIN_TIMES_INT64       1
// #define GxB_NO_MIN_TIMES_INT8        1
// #define GxB_NO_MIN_TIMES_UINT16      1
// #define GxB_NO_MIN_TIMES_UINT32      1
// #define GxB_NO_MIN_TIMES_UINT64      1
// #define GxB_NO_MIN_TIMES_UINT8       1

// builtin GrB_*: also needed by GrB_reduce to vector
// #define GxB_NO_MIN_FIRST_FP32        1
// #define GxB_NO_MIN_FIRST_FP64        1
// #define GxB_NO_MIN_FIRST_INT16       1
// #define GxB_NO_MIN_FIRST_INT32       1
// #define GxB_NO_MIN_FIRST_INT64       1
// #define GxB_NO_MIN_FIRST_INT8        1
// #define GxB_NO_MIN_FIRST_UINT16      1
// #define GxB_NO_MIN_FIRST_UINT32      1
// #define GxB_NO_MIN_FIRST_UINT64      1
// #define GxB_NO_MIN_FIRST_UINT8       1

// builtin GrB_*: also needed by GrB_reduce to vector
// #define GxB_NO_MIN_SECOND_FP32       1
// #define GxB_NO_MIN_SECOND_FP64       1
// #define GxB_NO_MIN_SECOND_INT16      1
// #define GxB_NO_MIN_SECOND_INT32      1
// #define GxB_NO_MIN_SECOND_INT64      1
// #define GxB_NO_MIN_SECOND_INT8       1
// #define GxB_NO_MIN_SECOND_UINT16     1
// #define GxB_NO_MIN_SECOND_UINT32     1
// #define GxB_NO_MIN_SECOND_UINT64     1
// #define GxB_NO_MIN_SECOND_UINT8      1

   #define GxB_NO_MIN_DIV_FP32          1
   #define GxB_NO_MIN_DIV_FP64          1
   #define GxB_NO_MIN_DIV_INT16         1
   #define GxB_NO_MIN_DIV_INT32         1
   #define GxB_NO_MIN_DIV_INT64         1
   #define GxB_NO_MIN_DIV_INT8          1
   #define GxB_NO_MIN_DIV_UINT16        1
   #define GxB_NO_MIN_DIV_UINT32        1
   #define GxB_NO_MIN_DIV_UINT64        1
   #define GxB_NO_MIN_DIV_UINT8         1

   #define GxB_NO_MIN_ANY_FP32          1
   #define GxB_NO_MIN_ANY_FP64          1
   #define GxB_NO_MIN_ANY_INT16         1
   #define GxB_NO_MIN_ANY_INT32         1
   #define GxB_NO_MIN_ANY_INT64         1
   #define GxB_NO_MIN_ANY_INT8          1
   #define GxB_NO_MIN_ANY_UINT16        1
   #define GxB_NO_MIN_ANY_UINT32        1
   #define GxB_NO_MIN_ANY_UINT64        1
   #define GxB_NO_MIN_ANY_UINT8         1

   #define GxB_NO_MIN_LAND_FP32         1
   #define GxB_NO_MIN_LAND_FP64         1
   #define GxB_NO_MIN_LAND_INT16        1
   #define GxB_NO_MIN_LAND_INT32        1
   #define GxB_NO_MIN_LAND_INT64        1
   #define GxB_NO_MIN_LAND_INT8         1
   #define GxB_NO_MIN_LAND_UINT16       1
   #define GxB_NO_MIN_LAND_UINT32       1
   #define GxB_NO_MIN_LAND_UINT64       1
   #define GxB_NO_MIN_LAND_UINT8        1

   #define GxB_NO_MIN_LOR_FP32          1
   #define GxB_NO_MIN_LOR_FP64          1
   #define GxB_NO_MIN_LOR_INT16         1
   #define GxB_NO_MIN_LOR_INT32         1
   #define GxB_NO_MIN_LOR_INT64         1
   #define GxB_NO_MIN_LOR_INT8          1
   #define GxB_NO_MIN_LOR_UINT16        1
   #define GxB_NO_MIN_LOR_UINT32        1
   #define GxB_NO_MIN_LOR_UINT64        1
   #define GxB_NO_MIN_LOR_UINT8         1

   #define GxB_NO_MIN_LXOR_FP32         1
   #define GxB_NO_MIN_LXOR_FP64         1
   #define GxB_NO_MIN_LXOR_INT16        1
   #define GxB_NO_MIN_LXOR_INT32        1
   #define GxB_NO_MIN_LXOR_INT64        1
   #define GxB_NO_MIN_LXOR_INT8         1
   #define GxB_NO_MIN_LXOR_UINT16       1
   #define GxB_NO_MIN_LXOR_UINT32       1
   #define GxB_NO_MIN_LXOR_UINT64       1
   #define GxB_NO_MIN_LXOR_UINT8        1

   #define GxB_NO_MIN_MINUS_FP32        1
   #define GxB_NO_MIN_MINUS_FP64        1
   #define GxB_NO_MIN_MINUS_INT16       1
   #define GxB_NO_MIN_MINUS_INT32       1
   #define GxB_NO_MIN_MINUS_INT64       1
   #define GxB_NO_MIN_MINUS_INT8        1
   #define GxB_NO_MIN_MINUS_UINT16      1
   #define GxB_NO_MIN_MINUS_UINT32      1
   #define GxB_NO_MIN_MINUS_UINT64      1
   #define GxB_NO_MIN_MINUS_UINT8       1

   #define GxB_NO_MIN_MIN_FP32          1
   #define GxB_NO_MIN_MIN_FP64          1
   #define GxB_NO_MIN_MIN_INT16         1
   #define GxB_NO_MIN_MIN_INT32         1
   #define GxB_NO_MIN_MIN_INT64         1
   #define GxB_NO_MIN_MIN_INT8          1
   #define GxB_NO_MIN_MIN_UINT16        1
   #define GxB_NO_MIN_MIN_UINT32        1
   #define GxB_NO_MIN_MIN_UINT64        1
   #define GxB_NO_MIN_MIN_UINT8         1

   #define GxB_NO_MIN_RDIV_FP32         1
   #define GxB_NO_MIN_RDIV_FP64         1
   #define GxB_NO_MIN_RDIV_INT16        1
   #define GxB_NO_MIN_RDIV_INT32        1
   #define GxB_NO_MIN_RDIV_INT64        1
   #define GxB_NO_MIN_RDIV_INT8         1
   #define GxB_NO_MIN_RDIV_UINT16       1
   #define GxB_NO_MIN_RDIV_UINT32       1
   #define GxB_NO_MIN_RDIV_UINT64       1
   #define GxB_NO_MIN_RDIV_UINT8        1

   #define GxB_NO_MIN_RMINUS_FP32       1
   #define GxB_NO_MIN_RMINUS_FP64       1
   #define GxB_NO_MIN_RMINUS_INT16      1
   #define GxB_NO_MIN_RMINUS_INT32      1
   #define GxB_NO_MIN_RMINUS_INT64      1
   #define GxB_NO_MIN_RMINUS_INT8       1
   #define GxB_NO_MIN_RMINUS_UINT16     1
   #define GxB_NO_MIN_RMINUS_UINT32     1
   #define GxB_NO_MIN_RMINUS_UINT64     1
   #define GxB_NO_MIN_RMINUS_UINT8      1

//------------------------------------------------------------
// semirings with the PLUS monoid
//------------------------------------------------------------

// PLUS_TIMES and PLUS_MIN are GrB_* builtin (not for FC23 or FC64).

//  70 semirings with PLUS monoids (10 real types each):
//
//      monoid  multiply ops
//      PLUS:  MIN, MAX, PLUS, TIMES, FIRST, SECOND, PAIR

// not GrB_*, used in LAGraph: triangle count and BFS
// #define GxB_NO_PLUS_PAIR_FP32        1
// #define GxB_NO_PLUS_PAIR_FP64        1
// #define GxB_NO_PLUS_PAIR_INT16       1
// #define GxB_NO_PLUS_PAIR_INT32       1
// #define GxB_NO_PLUS_PAIR_INT64       1
// #define GxB_NO_PLUS_PAIR_INT8        1
// #define GxB_NO_PLUS_PAIR_UINT16      1
// #define GxB_NO_PLUS_PAIR_UINT32      1
// #define GxB_NO_PLUS_PAIR_UINT64      1
// #define GxB_NO_PLUS_PAIR_UINT8       1

// builtin GrB_*:
// #define GxB_NO_PLUS_MIN_FP32         1
// #define GxB_NO_PLUS_MIN_FP64         1
// #define GxB_NO_PLUS_MIN_INT16        1
// #define GxB_NO_PLUS_MIN_INT32        1
// #define GxB_NO_PLUS_MIN_INT64        1
// #define GxB_NO_PLUS_MIN_INT8         1
// #define GxB_NO_PLUS_MIN_UINT16       1
// #define GxB_NO_PLUS_MIN_UINT32       1
// #define GxB_NO_PLUS_MIN_UINT64       1
// #define GxB_NO_PLUS_MIN_UINT8        1

// #define GxB_NO_PLUS_MAX_FP32         1
// #define GxB_NO_PLUS_MAX_FP64         1
// #define GxB_NO_PLUS_MAX_INT16        1
// #define GxB_NO_PLUS_MAX_INT32        1
// #define GxB_NO_PLUS_MAX_INT64        1
// #define GxB_NO_PLUS_MAX_INT8         1
// #define GxB_NO_PLUS_MAX_UINT16       1
// #define GxB_NO_PLUS_MAX_UINT32       1
// #define GxB_NO_PLUS_MAX_UINT64       1
// #define GxB_NO_PLUS_MAX_UINT8        1

// not GrB_*, used in LAGraph: sparse deep neural network
// #define GxB_NO_PLUS_PLUS_FP32        1
// #define GxB_NO_PLUS_PLUS_FP64        1
// #define GxB_NO_PLUS_PLUS_INT16       1
// #define GxB_NO_PLUS_PLUS_INT32       1
// #define GxB_NO_PLUS_PLUS_INT64       1
// #define GxB_NO_PLUS_PLUS_INT8        1
// #define GxB_NO_PLUS_PLUS_UINT16      1
// #define GxB_NO_PLUS_PLUS_UINT32      1
// #define GxB_NO_PLUS_PLUS_UINT64      1
// #define GxB_NO_PLUS_PLUS_UINT8       1

// builtin GrB_*: the classical semiring of linear algebra
// #define GxB_NO_PLUS_TIMES_FP32       1
// #define GxB_NO_PLUS_TIMES_FP64       1
// #define GxB_NO_PLUS_TIMES_INT16      1
// #define GxB_NO_PLUS_TIMES_INT32      1
// #define GxB_NO_PLUS_TIMES_INT64      1
// #define GxB_NO_PLUS_TIMES_INT8       1
// #define GxB_NO_PLUS_TIMES_UINT16     1
// #define GxB_NO_PLUS_TIMES_UINT32     1
// #define GxB_NO_PLUS_TIMES_UINT64     1
// #define GxB_NO_PLUS_TIMES_UINT8      1

// not GrB_*, used in LAGraph: pagerank and Betweeness-Centrality
// also needed by GrB_reduce to vector
// #define GxB_NO_PLUS_FIRST_FP32       1
// #define GxB_NO_PLUS_FIRST_FP64       1
// #define GxB_NO_PLUS_FIRST_INT16      1
// #define GxB_NO_PLUS_FIRST_INT32      1
// #define GxB_NO_PLUS_FIRST_INT64      1
// #define GxB_NO_PLUS_FIRST_INT8       1
// #define GxB_NO_PLUS_FIRST_UINT16     1
// #define GxB_NO_PLUS_FIRST_UINT32     1
// #define GxB_NO_PLUS_FIRST_UINT64     1
// #define GxB_NO_PLUS_FIRST_UINT8      1

// not GrB_*, used in LAGraph: Betweeness-Centrality and PageRank
// also needed by GrB_reduce to vector
// #define GxB_NO_PLUS_SECOND_FP32      1
// #define GxB_NO_PLUS_SECOND_FP64      1
// #define GxB_NO_PLUS_SECOND_INT16     1
// #define GxB_NO_PLUS_SECOND_INT32     1
// #define GxB_NO_PLUS_SECOND_INT64     1
// #define GxB_NO_PLUS_SECOND_INT8      1
// #define GxB_NO_PLUS_SECOND_UINT16    1
// #define GxB_NO_PLUS_SECOND_UINT32    1
// #define GxB_NO_PLUS_SECOND_UINT64    1
// #define GxB_NO_PLUS_SECOND_UINT8     1

   #define GxB_NO_PLUS_DIV_FP32         1
   #define GxB_NO_PLUS_DIV_FP64         1
   #define GxB_NO_PLUS_DIV_INT16        1
   #define GxB_NO_PLUS_DIV_INT32        1
   #define GxB_NO_PLUS_DIV_INT64        1
   #define GxB_NO_PLUS_DIV_INT8         1
   #define GxB_NO_PLUS_DIV_UINT16       1
   #define GxB_NO_PLUS_DIV_UINT32       1
   #define GxB_NO_PLUS_DIV_UINT64       1
   #define GxB_NO_PLUS_DIV_UINT8        1

   #define GxB_NO_PLUS_ANY_FP32         1
   #define GxB_NO_PLUS_ANY_FP64         1
   #define GxB_NO_PLUS_ANY_INT16        1
   #define GxB_NO_PLUS_ANY_INT32        1
   #define GxB_NO_PLUS_ANY_INT64        1
   #define GxB_NO_PLUS_ANY_INT8         1
   #define GxB_NO_PLUS_ANY_UINT16       1
   #define GxB_NO_PLUS_ANY_UINT32       1
   #define GxB_NO_PLUS_ANY_UINT64       1
   #define GxB_NO_PLUS_ANY_UINT8        1

   #define GxB_NO_PLUS_LAND_FP32        1
   #define GxB_NO_PLUS_LAND_FP64        1
   #define GxB_NO_PLUS_LAND_INT16       1
   #define GxB_NO_PLUS_LAND_INT32       1
   #define GxB_NO_PLUS_LAND_INT64       1
   #define GxB_NO_PLUS_LAND_INT8        1
   #define GxB_NO_PLUS_LAND_UINT16      1
   #define GxB_NO_PLUS_LAND_UINT32      1
   #define GxB_NO_PLUS_LAND_UINT64      1
   #define GxB_NO_PLUS_LAND_UINT8       1

   #define GxB_NO_PLUS_LOR_FP32         1
   #define GxB_NO_PLUS_LOR_FP64         1
   #define GxB_NO_PLUS_LOR_INT16        1
   #define GxB_NO_PLUS_LOR_INT32        1
   #define GxB_NO_PLUS_LOR_INT64        1
   #define GxB_NO_PLUS_LOR_INT8         1
   #define GxB_NO_PLUS_LOR_UINT16       1
   #define GxB_NO_PLUS_LOR_UINT32       1
   #define GxB_NO_PLUS_LOR_UINT64       1
   #define GxB_NO_PLUS_LOR_UINT8        1

   #define GxB_NO_PLUS_LXOR_FP32        1
   #define GxB_NO_PLUS_LXOR_FP64        1
   #define GxB_NO_PLUS_LXOR_INT16       1
   #define GxB_NO_PLUS_LXOR_INT32       1
   #define GxB_NO_PLUS_LXOR_INT64       1
   #define GxB_NO_PLUS_LXOR_INT8        1
   #define GxB_NO_PLUS_LXOR_UINT16      1
   #define GxB_NO_PLUS_LXOR_UINT32      1
   #define GxB_NO_PLUS_LXOR_UINT64      1
   #define GxB_NO_PLUS_LXOR_UINT8       1

   #define GxB_NO_PLUS_MINUS_FP32       1
   #define GxB_NO_PLUS_MINUS_FP64       1
   #define GxB_NO_PLUS_MINUS_INT16      1
   #define GxB_NO_PLUS_MINUS_INT32      1
   #define GxB_NO_PLUS_MINUS_INT64      1
   #define GxB_NO_PLUS_MINUS_INT8       1
   #define GxB_NO_PLUS_MINUS_UINT16     1
   #define GxB_NO_PLUS_MINUS_UINT32     1
   #define GxB_NO_PLUS_MINUS_UINT64     1
   #define GxB_NO_PLUS_MINUS_UINT8      1

   #define GxB_NO_PLUS_RDIV_FP32        1
   #define GxB_NO_PLUS_RDIV_FP64        1
   #define GxB_NO_PLUS_RDIV_INT16       1
   #define GxB_NO_PLUS_RDIV_INT32       1
   #define GxB_NO_PLUS_RDIV_INT64       1
   #define GxB_NO_PLUS_RDIV_INT8        1
   #define GxB_NO_PLUS_RDIV_UINT16      1
   #define GxB_NO_PLUS_RDIV_UINT32      1
   #define GxB_NO_PLUS_RDIV_UINT64      1
   #define GxB_NO_PLUS_RDIV_UINT8       1

   #define GxB_NO_PLUS_RMINUS_FP32      1
   #define GxB_NO_PLUS_RMINUS_FP64      1
   #define GxB_NO_PLUS_RMINUS_INT16     1
   #define GxB_NO_PLUS_RMINUS_INT32     1
   #define GxB_NO_PLUS_RMINUS_INT64     1
   #define GxB_NO_PLUS_RMINUS_INT8      1
   #define GxB_NO_PLUS_RMINUS_UINT16    1
   #define GxB_NO_PLUS_RMINUS_UINT32    1
   #define GxB_NO_PLUS_RMINUS_UINT64    1
   #define GxB_NO_PLUS_RMINUS_UINT8     1

//------------------------------------------------------------
// semirings with the TIMES monoid
//------------------------------------------------------------

// No builtin GrB_* semirings use the TIMES monoid, and none are used in
// LAGraph yet.  TIMES_FIRST and TIMES_SECOND are needed by GrB_reduce to
// vector.

//  60 semirings with TIMES monoids (10 real types each):
//
//      monoid  multiply ops
//      TIMES:  MIN, MAX, PLUS, TIMES, FIRST, SECOND

// #define GxB_NO_TIMES_MIN_FP32        1
// #define GxB_NO_TIMES_MIN_FP64        1
// #define GxB_NO_TIMES_MIN_INT16       1
// #define GxB_NO_TIMES_MIN_INT32       1
// #define GxB_NO_TIMES_MIN_INT64       1
// #define GxB_NO_TIMES_MIN_INT8        1
// #define GxB_NO_TIMES_MIN_UINT16      1
// #define GxB_NO_TIMES_MIN_UINT32      1
// #define GxB_NO_TIMES_MIN_UINT64      1
// #define GxB_NO_TIMES_MIN_UINT8       1

// #define GxB_NO_TIMES_MAX_FP32        1
// #define GxB_NO_TIMES_MAX_FP64        1
// #define GxB_NO_TIMES_MAX_INT16       1
// #define GxB_NO_TIMES_MAX_INT32       1
// #define GxB_NO_TIMES_MAX_INT64       1
// #define GxB_NO_TIMES_MAX_INT8        1
// #define GxB_NO_TIMES_MAX_UINT16      1
// #define GxB_NO_TIMES_MAX_UINT32      1
// #define GxB_NO_TIMES_MAX_UINT64      1
// #define GxB_NO_TIMES_MAX_UINT8       1

// #define GxB_NO_TIMES_PLUS_FP32       1
// #define GxB_NO_TIMES_PLUS_FP64       1
// #define GxB_NO_TIMES_PLUS_INT16      1
// #define GxB_NO_TIMES_PLUS_INT32      1
// #define GxB_NO_TIMES_PLUS_INT64      1
// #define GxB_NO_TIMES_PLUS_INT8       1
// #define GxB_NO_TIMES_PLUS_UINT16     1
// #define GxB_NO_TIMES_PLUS_UINT32     1
// #define GxB_NO_TIMES_PLUS_UINT64     1
// #define GxB_NO_TIMES_PLUS_UINT8      1

// #define GxB_NO_TIMES_TIMES_FP32      1
// #define GxB_NO_TIMES_TIMES_FP64      1
// #define GxB_NO_TIMES_TIMES_INT16     1
// #define GxB_NO_TIMES_TIMES_INT32     1
// #define GxB_NO_TIMES_TIMES_INT64     1
// #define GxB_NO_TIMES_TIMES_INT8      1
// #define GxB_NO_TIMES_TIMES_UINT16    1
// #define GxB_NO_TIMES_TIMES_UINT32    1
// #define GxB_NO_TIMES_TIMES_UINT64    1
// #define GxB_NO_TIMES_TIMES_UINT8     1

// needed by GrB_reduce to vector
// #define GxB_NO_TIMES_FIRST_FP32      1
// #define GxB_NO_TIMES_FIRST_FP64      1
// #define GxB_NO_TIMES_FIRST_INT16     1
// #define GxB_NO_TIMES_FIRST_INT32     1
// #define GxB_NO_TIMES_FIRST_INT64     1
// #define GxB_NO_TIMES_FIRST_INT8      1
// #define GxB_NO_TIMES_FIRST_UINT16    1
// #define GxB_NO_TIMES_FIRST_UINT32    1
// #define GxB_NO_TIMES_FIRST_UINT64    1
// #define GxB_NO_TIMES_FIRST_UINT8     1

// needed by GrB_reduce to vector
// #define GxB_NO_TIMES_SECOND_FP32     1
// #define GxB_NO_TIMES_SECOND_FP64     1
// #define GxB_NO_TIMES_SECOND_INT16    1
// #define GxB_NO_TIMES_SECOND_INT32    1
// #define GxB_NO_TIMES_SECOND_INT64    1
// #define GxB_NO_TIMES_SECOND_INT8     1
// #define GxB_NO_TIMES_SECOND_UINT16   1
// #define GxB_NO_TIMES_SECOND_UINT32   1
// #define GxB_NO_TIMES_SECOND_UINT64   1
// #define GxB_NO_TIMES_SECOND_UINT8    1

   #define GxB_NO_TIMES_DIV_FP32        1
   #define GxB_NO_TIMES_DIV_FP64        1
   #define GxB_NO_TIMES_DIV_INT16       1
   #define GxB_NO_TIMES_DIV_INT32       1
   #define GxB_NO_TIMES_DIV_INT64       1
   #define GxB_NO_TIMES_DIV_INT8        1
   #define GxB_NO_TIMES_DIV_UINT16      1
   #define GxB_NO_TIMES_DIV_UINT32      1
   #define GxB_NO_TIMES_DIV_UINT64      1
   #define GxB_NO_TIMES_DIV_UINT8       1

   #define GxB_NO_TIMES_ANY_FP32        1
   #define GxB_NO_TIMES_ANY_FP64        1
   #define GxB_NO_TIMES_ANY_INT16       1
   #define GxB_NO_TIMES_ANY_INT32       1
   #define GxB_NO_TIMES_ANY_INT64       1
   #define GxB_NO_TIMES_ANY_INT8        1
   #define GxB_NO_TIMES_ANY_UINT16      1
   #define GxB_NO_TIMES_ANY_UINT32      1
   #define GxB_NO_TIMES_ANY_UINT64      1
   #define GxB_NO_TIMES_ANY_UINT8       1

   #define GxB_NO_TIMES_LAND_FP32       1
   #define GxB_NO_TIMES_LAND_FP64       1
   #define GxB_NO_TIMES_LAND_INT16      1
   #define GxB_NO_TIMES_LAND_INT32      1
   #define GxB_NO_TIMES_LAND_INT64      1
   #define GxB_NO_TIMES_LAND_INT8       1
   #define GxB_NO_TIMES_LAND_UINT16     1
   #define GxB_NO_TIMES_LAND_UINT32     1
   #define GxB_NO_TIMES_LAND_UINT64     1
   #define GxB_NO_TIMES_LAND_UINT8      1

   #define GxB_NO_TIMES_LOR_FP32        1
   #define GxB_NO_TIMES_LOR_FP64        1
   #define GxB_NO_TIMES_LOR_INT16       1
   #define GxB_NO_TIMES_LOR_INT32       1
   #define GxB_NO_TIMES_LOR_INT64       1
   #define GxB_NO_TIMES_LOR_INT8        1
   #define GxB_NO_TIMES_LOR_UINT16      1
   #define GxB_NO_TIMES_LOR_UINT32      1
   #define GxB_NO_TIMES_LOR_UINT64      1
   #define GxB_NO_TIMES_LOR_UINT8       1

   #define GxB_NO_TIMES_LXOR_FP32       1
   #define GxB_NO_TIMES_LXOR_FP64       1
   #define GxB_NO_TIMES_LXOR_INT16      1
   #define GxB_NO_TIMES_LXOR_INT32      1
   #define GxB_NO_TIMES_LXOR_INT64      1
   #define GxB_NO_TIMES_LXOR_INT8       1
   #define GxB_NO_TIMES_LXOR_UINT16     1
   #define GxB_NO_TIMES_LXOR_UINT32     1
   #define GxB_NO_TIMES_LXOR_UINT64     1
   #define GxB_NO_TIMES_LXOR_UINT8      1

   #define GxB_NO_TIMES_MINUS_FP32      1
   #define GxB_NO_TIMES_MINUS_FP64      1
   #define GxB_NO_TIMES_MINUS_INT16     1
   #define GxB_NO_TIMES_MINUS_INT32     1
   #define GxB_NO_TIMES_MINUS_INT64     1
   #define GxB_NO_TIMES_MINUS_INT8      1
   #define GxB_NO_TIMES_MINUS_UINT16    1
   #define GxB_NO_TIMES_MINUS_UINT32    1
   #define GxB_NO_TIMES_MINUS_UINT64    1
   #define GxB_NO_TIMES_MINUS_UINT8     1

   #define GxB_NO_TIMES_RDIV_FP32       1
   #define GxB_NO_TIMES_RDIV_FP64       1
   #define GxB_NO_TIMES_RDIV_INT16      1
   #define GxB_NO_TIMES_RDIV_INT32      1
   #define GxB_NO_TIMES_RDIV_INT64      1
   #define GxB_NO_TIMES_RDIV_INT8       1
   #define GxB_NO_TIMES_RDIV_UINT16     1
   #define GxB_NO_TIMES_RDIV_UINT32     1
   #define GxB_NO_TIMES_RDIV_UINT64     1
   #define GxB_NO_TIMES_RDIV_UINT8      1

   #define GxB_NO_TIMES_RMINUS_FP32     1
   #define GxB_NO_TIMES_RMINUS_FP64     1
   #define GxB_NO_TIMES_RMINUS_INT16    1
   #define GxB_NO_TIMES_RMINUS_INT32    1
   #define GxB_NO_TIMES_RMINUS_INT64    1
   #define GxB_NO_TIMES_RMINUS_INT8     1
   #define GxB_NO_TIMES_RMINUS_UINT16   1
   #define GxB_NO_TIMES_RMINUS_UINT32   1
   #define GxB_NO_TIMES_RMINUS_UINT64   1
   #define GxB_NO_TIMES_RMINUS_UINT8    1

//----------------------------------------
// complex semirings:
//----------------------------------------

// _FIRST and _SECOND are needed by GrB_reduce to vector

//  10 semirings for 2 complex types (PLUS and TIMES monoids only):
//
//      monoid  multiply ops
//      PLUS:  TIMES, FIRST, SECOND
//      TIMES: FIRST, SECOND

   #define GxB_NO_PLUS_PLUS_FC32        1
   #define GxB_NO_PLUS_PLUS_FC64        1

// the conventional semiring, used by C=A*B for built-in matrices
// #define GxB_NO_PLUS_TIMES_FC32       1
// #define GxB_NO_PLUS_TIMES_FC64       1

// needed by GrB_reduce to vector, or s = sum (A) in @GrB interface
// #define GxB_NO_PLUS_FIRST_FC32       1
// #define GxB_NO_PLUS_FIRST_FC64       1

// needed by GrB_reduce to vector, or s = sum (A) in @GrB interface
// #define GxB_NO_PLUS_SECOND_FC32      1
// #define GxB_NO_PLUS_SECOND_FC64      1

   #define GxB_NO_PLUS_PAIR_FC32        1
   #define GxB_NO_PLUS_PAIR_FC64        1

   #define GxB_NO_PLUS_MINUS_FC32       1
   #define GxB_NO_PLUS_MINUS_FC64       1

   #define GxB_NO_PLUS_DIV_FC32         1
   #define GxB_NO_PLUS_DIV_FC64         1

   #define GxB_NO_PLUS_RDIV_FC32        1
   #define GxB_NO_PLUS_RDIV_FC64        1

   #define GxB_NO_PLUS_RMINUS_FC32      1
   #define GxB_NO_PLUS_RMINUS_FC64      1

   #define GxB_NO_TIMES_PLUS_FC32       1
   #define GxB_NO_TIMES_PLUS_FC64       1

// GxB_TIMES_TIMES_FC32 is required for testing (GraphBLAS/Test and Tcov),
// so it is not disabled when GraphBLAS is compiled for MATLAB.
   #ifndef GBMATLAB
   #define GxB_NO_TIMES_TIMES_FC32      1
   #endif
   #define GxB_NO_TIMES_TIMES_FC64      1

// needed by GrB_reduce to vector, or s = prod (A) in @GrB interface
// #define GxB_NO_TIMES_FIRST_FC32      1
// #define GxB_NO_TIMES_FIRST_FC64      1

// needed by GrB_reduce to vector, or s = prod (A) in @GrB interface
// #define GxB_NO_TIMES_SECOND_FC32     1
// #define GxB_NO_TIMES_SECOND_FC64     1

   #define GxB_NO_TIMES_MINUS_FC32      1
   #define GxB_NO_TIMES_MINUS_FC64      1

   #define GxB_NO_TIMES_DIV_FC32        1
   #define GxB_NO_TIMES_DIV_FC64        1

   #define GxB_NO_TIMES_RDIV_FC32       1
   #define GxB_NO_TIMES_RDIV_FC64       1

   #define GxB_NO_TIMES_RMINUS_FC32     1
   #define GxB_NO_TIMES_RMINUS_FC64     1

//----------------------------------------
// semirings with the ANY monoid
//----------------------------------------

// None of these are GrB_*, since the ANY monoid is a GxB* extension.
// However, semirings based on the ANY monoid are common: BFS in particular
// uses ANY_FIRST, ANY_SECOND, and ANY_PAIR.

// The ANY_PAIR_ISO semiring cannot be disabled.

//  33 semirings enabled with the ANY monoid:
//
//      ANY:   FIRST, SECOND, PAIR (with bool, 10 real types)

// used in LAGraph: BFS, and GrB_reduce to vector
// #define GxB_NO_ANY_FIRST_BOOL        1
// #define GxB_NO_ANY_FIRST_FP32        1
// #define GxB_NO_ANY_FIRST_FP64        1
// #define GxB_NO_ANY_FIRST_INT16       1
// #define GxB_NO_ANY_FIRST_INT32       1
// #define GxB_NO_ANY_FIRST_INT64       1
// #define GxB_NO_ANY_FIRST_INT8        1
// #define GxB_NO_ANY_FIRST_UINT16      1
// #define GxB_NO_ANY_FIRST_UINT32      1
// #define GxB_NO_ANY_FIRST_UINT64      1
// #define GxB_NO_ANY_FIRST_UINT8       1
   #define GxB_NO_ANY_FIRST_FC32        1
   #define GxB_NO_ANY_FIRST_FC64        1

// used in LAGraph: BFS, and GrB_reduce to vector
// #define GxB_NO_ANY_SECOND_BOOL       1
// #define GxB_NO_ANY_SECOND_FP32       1
// #define GxB_NO_ANY_SECOND_FP64       1
// #define GxB_NO_ANY_SECOND_INT16      1
// #define GxB_NO_ANY_SECOND_INT32      1
// #define GxB_NO_ANY_SECOND_INT64      1
// #define GxB_NO_ANY_SECOND_INT8       1
// #define GxB_NO_ANY_SECOND_UINT16     1
// #define GxB_NO_ANY_SECOND_UINT32     1
// #define GxB_NO_ANY_SECOND_UINT64     1
// #define GxB_NO_ANY_SECOND_UINT8      1
   #define GxB_NO_ANY_SECOND_FC32       1
   #define GxB_NO_ANY_SECOND_FC64       1

   #define GxB_NO_ANY_DIV_FP32          1
   #define GxB_NO_ANY_DIV_FP64          1
   #define GxB_NO_ANY_DIV_INT16         1
   #define GxB_NO_ANY_DIV_INT32         1
   #define GxB_NO_ANY_DIV_INT64         1
   #define GxB_NO_ANY_DIV_INT8          1
   #define GxB_NO_ANY_DIV_UINT16        1
   #define GxB_NO_ANY_DIV_UINT32        1
   #define GxB_NO_ANY_DIV_UINT64        1
   #define GxB_NO_ANY_DIV_UINT8         1
   #define GxB_NO_ANY_DIV_FC32          1
   #define GxB_NO_ANY_DIV_FC64          1

   // argmin and argmax use these semirings, but only for row/col scale,
   // which is controlled by the same as a binop, not a semiring.
   #define GxB_NO_ANY_EQ_BOOL           1
   #define GxB_NO_ANY_EQ_FP32           1
   #define GxB_NO_ANY_EQ_FP64           1
   #define GxB_NO_ANY_EQ_INT16          1
   #define GxB_NO_ANY_EQ_INT32          1
   #define GxB_NO_ANY_EQ_INT64          1
   #define GxB_NO_ANY_EQ_INT8           1
   #define GxB_NO_ANY_EQ_UINT16         1
   #define GxB_NO_ANY_EQ_UINT32         1
   #define GxB_NO_ANY_EQ_UINT64         1
   #define GxB_NO_ANY_EQ_UINT8          1
   #define GxB_NO_ANY_EQ_FC32           1
   #define GxB_NO_ANY_EQ_FC64           1

   #define GxB_NO_ANY_GE_BOOL           1
   #define GxB_NO_ANY_GE_FP32           1
   #define GxB_NO_ANY_GE_FP64           1
   #define GxB_NO_ANY_GE_INT16          1
   #define GxB_NO_ANY_GE_INT32          1
   #define GxB_NO_ANY_GE_INT64          1
   #define GxB_NO_ANY_GE_INT8           1
   #define GxB_NO_ANY_GE_UINT16         1
   #define GxB_NO_ANY_GE_UINT32         1
   #define GxB_NO_ANY_GE_UINT64         1
   #define GxB_NO_ANY_GE_UINT8          1

   #define GxB_NO_ANY_GT_BOOL           1
   #define GxB_NO_ANY_GT_FP32           1
   #define GxB_NO_ANY_GT_FP64           1
   #define GxB_NO_ANY_GT_INT16          1
   #define GxB_NO_ANY_GT_INT32          1
   #define GxB_NO_ANY_GT_INT64          1
   #define GxB_NO_ANY_GT_INT8           1
   #define GxB_NO_ANY_GT_UINT16         1
   #define GxB_NO_ANY_GT_UINT32         1
   #define GxB_NO_ANY_GT_UINT64         1
   #define GxB_NO_ANY_GT_UINT8          1

   #define GxB_NO_ANY_LAND_BOOL         1
   #define GxB_NO_ANY_LAND_FP32         1
   #define GxB_NO_ANY_LAND_FP64         1
   #define GxB_NO_ANY_LAND_INT16        1
   #define GxB_NO_ANY_LAND_INT32        1
   #define GxB_NO_ANY_LAND_INT64        1
   #define GxB_NO_ANY_LAND_INT8         1
   #define GxB_NO_ANY_LAND_UINT16       1
   #define GxB_NO_ANY_LAND_UINT32       1
   #define GxB_NO_ANY_LAND_UINT64       1
   #define GxB_NO_ANY_LAND_UINT8        1

   #define GxB_NO_ANY_LE_BOOL           1
   #define GxB_NO_ANY_LE_FP32           1
   #define GxB_NO_ANY_LE_FP64           1
   #define GxB_NO_ANY_LE_INT16          1
   #define GxB_NO_ANY_LE_INT32          1
   #define GxB_NO_ANY_LE_INT64          1
   #define GxB_NO_ANY_LE_INT8           1
   #define GxB_NO_ANY_LE_UINT16         1
   #define GxB_NO_ANY_LE_UINT32         1
   #define GxB_NO_ANY_LE_UINT64         1
   #define GxB_NO_ANY_LE_UINT8          1

   #define GxB_NO_ANY_LOR_BOOL          1
   #define GxB_NO_ANY_LOR_FP32          1
   #define GxB_NO_ANY_LOR_FP64          1
   #define GxB_NO_ANY_LOR_INT16         1
   #define GxB_NO_ANY_LOR_INT32         1
   #define GxB_NO_ANY_LOR_INT64         1
   #define GxB_NO_ANY_LOR_INT8          1
   #define GxB_NO_ANY_LOR_UINT16        1
   #define GxB_NO_ANY_LOR_UINT32        1
   #define GxB_NO_ANY_LOR_UINT64        1
   #define GxB_NO_ANY_LOR_UINT8         1

   #define GxB_NO_ANY_LT_BOOL           1
   #define GxB_NO_ANY_LT_FP32           1
   #define GxB_NO_ANY_LT_FP64           1
   #define GxB_NO_ANY_LT_INT16          1
   #define GxB_NO_ANY_LT_INT32          1
   #define GxB_NO_ANY_LT_INT64          1
   #define GxB_NO_ANY_LT_INT8           1
   #define GxB_NO_ANY_LT_UINT16         1
   #define GxB_NO_ANY_LT_UINT32         1
   #define GxB_NO_ANY_LT_UINT64         1
   #define GxB_NO_ANY_LT_UINT8          1

   #define GxB_NO_ANY_LXOR_BOOL         1
   #define GxB_NO_ANY_LXOR_FP32         1
   #define GxB_NO_ANY_LXOR_FP64         1
   #define GxB_NO_ANY_LXOR_INT16        1
   #define GxB_NO_ANY_LXOR_INT32        1
   #define GxB_NO_ANY_LXOR_INT64        1
   #define GxB_NO_ANY_LXOR_INT8         1
   #define GxB_NO_ANY_LXOR_UINT16       1
   #define GxB_NO_ANY_LXOR_UINT32       1
   #define GxB_NO_ANY_LXOR_UINT64       1
   #define GxB_NO_ANY_LXOR_UINT8        1

   #define GxB_NO_ANY_MAX_FP32          1
   #define GxB_NO_ANY_MAX_FP64          1
   #define GxB_NO_ANY_MAX_INT16         1
   #define GxB_NO_ANY_MAX_INT32         1
   #define GxB_NO_ANY_MAX_INT64         1
   #define GxB_NO_ANY_MAX_INT8          1
   #define GxB_NO_ANY_MAX_UINT16        1
   #define GxB_NO_ANY_MAX_UINT32        1
   #define GxB_NO_ANY_MAX_UINT64        1
   #define GxB_NO_ANY_MAX_UINT8         1

   #define GxB_NO_ANY_MIN_FP32          1
   #define GxB_NO_ANY_MIN_FP64          1
   #define GxB_NO_ANY_MIN_INT16         1
   #define GxB_NO_ANY_MIN_INT32         1
   #define GxB_NO_ANY_MIN_INT64         1
   #define GxB_NO_ANY_MIN_INT8          1
   #define GxB_NO_ANY_MIN_UINT16        1
   #define GxB_NO_ANY_MIN_UINT32        1
   #define GxB_NO_ANY_MIN_UINT64        1
   #define GxB_NO_ANY_MIN_UINT8         1

   #define GxB_NO_ANY_MINUS_FP32        1
   #define GxB_NO_ANY_MINUS_FP64        1
   #define GxB_NO_ANY_MINUS_INT16       1
   #define GxB_NO_ANY_MINUS_INT32       1
   #define GxB_NO_ANY_MINUS_INT64       1
   #define GxB_NO_ANY_MINUS_INT8        1
   #define GxB_NO_ANY_MINUS_UINT16      1
   #define GxB_NO_ANY_MINUS_UINT32      1
   #define GxB_NO_ANY_MINUS_UINT64      1
   #define GxB_NO_ANY_MINUS_UINT8       1
   #define GxB_NO_ANY_MINUS_FC32        1
   #define GxB_NO_ANY_MINUS_FC64        1

   #define GxB_NO_ANY_NE_FP32           1
   #define GxB_NO_ANY_NE_FP64           1
   #define GxB_NO_ANY_NE_INT16          1
   #define GxB_NO_ANY_NE_INT32          1
   #define GxB_NO_ANY_NE_INT64          1
   #define GxB_NO_ANY_NE_INT8           1
   #define GxB_NO_ANY_NE_UINT16         1
   #define GxB_NO_ANY_NE_UINT32         1
   #define GxB_NO_ANY_NE_UINT64         1
   #define GxB_NO_ANY_NE_UINT8          1
   #define GxB_NO_ANY_NE_FC32           1
   #define GxB_NO_ANY_NE_FC64           1

   #define GxB_NO_ANY_PLUS_FP32         1
   #define GxB_NO_ANY_PLUS_FP64         1
   #define GxB_NO_ANY_PLUS_INT16        1
   #define GxB_NO_ANY_PLUS_INT32        1
   #define GxB_NO_ANY_PLUS_INT64        1
   #define GxB_NO_ANY_PLUS_INT8         1
   #define GxB_NO_ANY_PLUS_UINT16       1
   #define GxB_NO_ANY_PLUS_UINT32       1
   #define GxB_NO_ANY_PLUS_UINT64       1
   #define GxB_NO_ANY_PLUS_UINT8        1
   #define GxB_NO_ANY_PLUS_FC32         1
   #define GxB_NO_ANY_PLUS_FC64         1

   #define GxB_NO_ANY_RDIV_FP32         1
   #define GxB_NO_ANY_RDIV_FP64         1
   #define GxB_NO_ANY_RDIV_INT16        1
   #define GxB_NO_ANY_RDIV_INT32        1
   #define GxB_NO_ANY_RDIV_INT64        1
   #define GxB_NO_ANY_RDIV_INT8         1
   #define GxB_NO_ANY_RDIV_UINT16       1
   #define GxB_NO_ANY_RDIV_UINT32       1
   #define GxB_NO_ANY_RDIV_UINT64       1
   #define GxB_NO_ANY_RDIV_UINT8        1
   #define GxB_NO_ANY_RDIV_FC32         1
   #define GxB_NO_ANY_RDIV_FC64         1

   #define GxB_NO_ANY_RMINUS_FP32       1
   #define GxB_NO_ANY_RMINUS_FP64       1
   #define GxB_NO_ANY_RMINUS_INT16      1
   #define GxB_NO_ANY_RMINUS_INT32      1
   #define GxB_NO_ANY_RMINUS_INT64      1
   #define GxB_NO_ANY_RMINUS_INT8       1
   #define GxB_NO_ANY_RMINUS_UINT16     1
   #define GxB_NO_ANY_RMINUS_UINT32     1
   #define GxB_NO_ANY_RMINUS_UINT64     1
   #define GxB_NO_ANY_RMINUS_UINT8      1
   #define GxB_NO_ANY_RMINUS_FC32       1
   #define GxB_NO_ANY_RMINUS_FC64       1

   #define GxB_NO_ANY_TIMES_FP32        1
   #define GxB_NO_ANY_TIMES_FP64        1
   #define GxB_NO_ANY_TIMES_INT16       1
   #define GxB_NO_ANY_TIMES_INT32       1
   #define GxB_NO_ANY_TIMES_INT64       1
   #define GxB_NO_ANY_TIMES_INT8        1
   #define GxB_NO_ANY_TIMES_UINT16      1
   #define GxB_NO_ANY_TIMES_UINT32      1
   #define GxB_NO_ANY_TIMES_UINT64      1
   #define GxB_NO_ANY_TIMES_UINT8       1
   #define GxB_NO_ANY_TIMES_FC32        1
   #define GxB_NO_ANY_TIMES_FC64        1

//----------------------------------------
// bitwise semirings:
//----------------------------------------

//  64 bitwise semirings: for 4 unsigned integer types, all enabled below:
//
//      (BOR, BAND, BXOR, BXNOR) x (BOR, BAND, BXOR, BXNOR)

// #define GxB_NO_BOR_BOR_UINT8         1
// #define GxB_NO_BOR_BOR_UINT16        1
// #define GxB_NO_BOR_BOR_UINT32        1
// #define GxB_NO_BOR_BOR_UINT64        1

// #define GxB_NO_BOR_BAND_UINT8        1
// #define GxB_NO_BOR_BAND_UINT16       1
// #define GxB_NO_BOR_BAND_UINT32       1
// #define GxB_NO_BOR_BAND_UINT64       1

// #define GxB_NO_BOR_BXOR_UINT8        1
// #define GxB_NO_BOR_BXOR_UINT16       1
// #define GxB_NO_BOR_BXOR_UINT32       1
// #define GxB_NO_BOR_BXOR_UINT64       1

// #define GxB_NO_BOR_BXNOR_UINT8       1
// #define GxB_NO_BOR_BXNOR_UINT16      1
// #define GxB_NO_BOR_BXNOR_UINT32      1
// #define GxB_NO_BOR_BXNOR_UINT64      1

// #define GxB_NO_BAND_BOR_UINT8        1
// #define GxB_NO_BAND_BOR_UINT16       1
// #define GxB_NO_BAND_BOR_UINT32       1
// #define GxB_NO_BAND_BOR_UINT64       1

// #define GxB_NO_BAND_BAND_UINT8       1
// #define GxB_NO_BAND_BAND_UINT16      1
// #define GxB_NO_BAND_BAND_UINT32      1
// #define GxB_NO_BAND_BAND_UINT64      1

// #define GxB_NO_BAND_BXOR_UINT8       1
// #define GxB_NO_BAND_BXOR_UINT16      1
// #define GxB_NO_BAND_BXOR_UINT32      1
// #define GxB_NO_BAND_BXOR_UINT64      1

// #define GxB_NO_BAND_BXNOR_UINT8      1
// #define GxB_NO_BAND_BXNOR_UINT16     1
// #define GxB_NO_BAND_BXNOR_UINT32     1
// #define GxB_NO_BAND_BXNOR_UINT64     1

// #define GxB_NO_BXOR_BOR_UINT8        1
// #define GxB_NO_BXOR_BOR_UINT16       1
// #define GxB_NO_BXOR_BOR_UINT32       1
// #define GxB_NO_BXOR_BOR_UINT64       1

// #define GxB_NO_BXOR_BAND_UINT8       1
// #define GxB_NO_BXOR_BAND_UINT16      1
// #define GxB_NO_BXOR_BAND_UINT32      1
// #define GxB_NO_BXOR_BAND_UINT64      1

// #define GxB_NO_BXOR_BXOR_UINT8       1
// #define GxB_NO_BXOR_BXOR_UINT16      1
// #define GxB_NO_BXOR_BXOR_UINT32      1
// #define GxB_NO_BXOR_BXOR_UINT64      1

// #define GxB_NO_BXOR_BXNOR_UINT8      1
// #define GxB_NO_BXOR_BXNOR_UINT16     1
// #define GxB_NO_BXOR_BXNOR_UINT32     1
// #define GxB_NO_BXOR_BXNOR_UINT64     1

// #define GxB_NO_BXNOR_BOR_UINT8       1
// #define GxB_NO_BXNOR_BOR_UINT16      1
// #define GxB_NO_BXNOR_BOR_UINT32      1
// #define GxB_NO_BXNOR_BOR_UINT64      1

// #define GxB_NO_BXNOR_BAND_UINT8      1
// #define GxB_NO_BXNOR_BAND_UINT16     1
// #define GxB_NO_BXNOR_BAND_UINT32     1
// #define GxB_NO_BXNOR_BAND_UINT64     1

// #define GxB_NO_BXNOR_BXOR_UINT8      1
// #define GxB_NO_BXNOR_BXOR_UINT16     1
// #define GxB_NO_BXNOR_BXOR_UINT32     1
// #define GxB_NO_BXNOR_BXOR_UINT64     1

// #define GxB_NO_BXNOR_BXNOR_UINT8     1
// #define GxB_NO_BXNOR_BXNOR_UINT16    1
// #define GxB_NO_BXNOR_BXNOR_UINT32    1
// #define GxB_NO_BXNOR_BXNOR_UINT64    1

//----------------------------------------
// semirings with positional multiplicative operators:
//----------------------------------------

// No builtin GrB_* semirings use positional multiplicative operators.
// BFS_parent uses ANY_SECONDI.  1-based semirings are important for 1-based
// framewarks such as Julia and Octave.  In a semiring, the multiplicative
// operator SECONDI is the same as FIRSTJ.

//  36 positional semirings:
//
//      monoids: (MIN, MAX, ANY) x
//      mult:    (FIRSTI, FIRSTI1, FIRSTJ, FIRSTJ1, SECONDJ, SECONDJ1) x
//      types:   (int32, int64)

// enable GxB_MIN_FIRSTI_INT32 for test coverage:
// #define GxB_NO_MIN_FIRSTI_INT32      1
   #define GxB_NO_MIN_FIRSTI_INT64      1
   #define GxB_NO_MIN_FIRSTI1_INT32     1
   #define GxB_NO_MIN_FIRSTI1_INT64     1
// #define GxB_NO_MIN_FIRSTJ_INT32      1
// #define GxB_NO_MIN_FIRSTJ_INT64      1
// #define GxB_NO_MIN_FIRSTJ1_INT32     1
// #define GxB_NO_MIN_FIRSTJ1_INT64     1
   #define GxB_NO_MIN_SECONDJ_INT32     1
   #define GxB_NO_MIN_SECONDJ_INT64     1
   #define GxB_NO_MIN_SECONDJ1_INT32    1
   #define GxB_NO_MIN_SECONDJ1_INT64    1

   #define GxB_NO_MAX_FIRSTI_INT32      1
   #define GxB_NO_MAX_FIRSTI_INT64      1
   #define GxB_NO_MAX_FIRSTI1_INT32     1
   #define GxB_NO_MAX_FIRSTI1_INT64     1
// #define GxB_NO_MAX_FIRSTJ_INT32      1
// #define GxB_NO_MAX_FIRSTJ_INT64      1
// #define GxB_NO_MAX_FIRSTJ1_INT32     1
// #define GxB_NO_MAX_FIRSTJ1_INT64     1
   #define GxB_NO_MAX_SECONDJ_INT32     1
   #define GxB_NO_MAX_SECONDJ_INT64     1
   #define GxB_NO_MAX_SECONDJ1_INT32    1
   #define GxB_NO_MAX_SECONDJ1_INT64    1

   #define GxB_NO_ANY_FIRSTI_INT32      1
   #define GxB_NO_ANY_FIRSTI_INT64      1
   #define GxB_NO_ANY_FIRSTI1_INT32     1
   #define GxB_NO_ANY_FIRSTI1_INT64     1
// #define GxB_NO_ANY_FIRSTJ_INT32      1
// #define GxB_NO_ANY_FIRSTJ_INT64      1
// #define GxB_NO_ANY_FIRSTJ1_INT32     1
// #define GxB_NO_ANY_FIRSTJ1_INT64     1
   #define GxB_NO_ANY_SECONDJ_INT32     1
   #define GxB_NO_ANY_SECONDJ_INT64     1
   #define GxB_NO_ANY_SECONDJ1_INT32    1
   #define GxB_NO_ANY_SECONDJ1_INT64    1

   #define GxB_NO_PLUS_FIRSTI_INT32     1
   #define GxB_NO_PLUS_FIRSTI_INT64     1
   #define GxB_NO_PLUS_FIRSTI1_INT32    1
   #define GxB_NO_PLUS_FIRSTI1_INT64    1
   #define GxB_NO_PLUS_FIRSTJ_INT32     1
   #define GxB_NO_PLUS_FIRSTJ_INT64     1
   #define GxB_NO_PLUS_FIRSTJ1_INT32    1
   #define GxB_NO_PLUS_FIRSTJ1_INT64    1
   #define GxB_NO_PLUS_SECONDJ_INT32    1
   #define GxB_NO_PLUS_SECONDJ_INT64    1
   #define GxB_NO_PLUS_SECONDJ1_INT32   1
   #define GxB_NO_PLUS_SECONDJ1_INT64   1

   #define GxB_NO_TIMES_FIRSTI_INT32    1
   #define GxB_NO_TIMES_FIRSTI_INT64    1
   #define GxB_NO_TIMES_FIRSTI1_INT32   1
   #define GxB_NO_TIMES_FIRSTI1_INT64   1
   #define GxB_NO_TIMES_FIRSTJ_INT32    1
   #define GxB_NO_TIMES_FIRSTJ_INT64    1
   #define GxB_NO_TIMES_FIRSTJ1_INT32   1
   #define GxB_NO_TIMES_FIRSTJ1_INT64   1
   #define GxB_NO_TIMES_SECONDJ_INT32   1
   #define GxB_NO_TIMES_SECONDJ_INT64   1
   #define GxB_NO_TIMES_SECONDJ1_INT32  1
   #define GxB_NO_TIMES_SECONDJ1_INT64  1

