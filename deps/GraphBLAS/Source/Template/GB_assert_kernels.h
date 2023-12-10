//------------------------------------------------------------------------------
// GB_assert_kernels.h: assertions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// debugging definitions
//------------------------------------------------------------------------------

// the JIT run time kernels use abort directly from libc:
#undef  GB_ABORT
#define GB_ABORT /* abort ( ) */ ;

#undef ASSERT

#ifdef GB_DEBUG

    // assert X is true
    #define ASSERT(X)                                                       \
    {                                                                       \
        if (!(X))                                                           \
        {                                                                   \
            GBDUMP ("JIT assertion failed: " __FILE__ " line %d\n", __LINE__) ;\
            GB_ABORT ;                                                      \
        }                                                                   \
    }

#else

    // debugging disabled
    #define ASSERT(X)

#endif

// ASSERT_OK* debugging disabled in the JIT kernels
#undef  ASSERT_OK
#define ASSERT_OK(X)

#undef  ASSERT_OK_OR_NULL
#define ASSERT_OK_OR_NULL(X)

#undef  GB_IMPLIES
#define GB_IMPLIES(p,q) (!(p) || (q))

// The JIT kernels do not trigger the 'gotcha'.
#undef  GB_GOTCHA
#define GB_GOTCHA

#undef  GB_HERE
#define GB_HERE GBDUMP ("%2d: Here: " __FILE__ "\n", __LINE__) ;

// ASSERT (GB_DEAD_CODE) marks code that is intentionally dead, leftover from
// prior versions of SuiteSparse:GraphBLAS but no longer used in the current
// version.  The code is kept in case it is required for future versions (in
// which case, the ASSERT (GB_DEAD_CODE) statement would be removed).
#undef  GB_DEAD_CODE
#define GB_DEAD_CODE 0

//------------------------------------------------------------------------------
// assertions for checking specific objects
//------------------------------------------------------------------------------

// these assertions are disabled in the JIT runtime kernels, since the
// functions do not appear in GB_callback.

#undef  ASSERT_TYPE_OK
#undef  ASSERT_TYPE_OK_OR_NULL
#undef  ASSERT_BINARYOP_OK
#undef  ASSERT_INDEXUNARYOP_OK
#undef  ASSERT_BINARYOP_OK_OR_NULL
#undef  ASSERT_UNARYOP_OK
#undef  ASSERT_UNARYOP_OK_OR_NULL
#undef  ASSERT_SELECTOP_OK
#undef  ASSERT_SELECTOP_OK_OR_NULL
#undef  ASSERT_OP_OK
#undef  ASSERT_OP_OK_OR_NULL
#undef  ASSERT_MONOID_OK
#undef  ASSERT_SEMIRING_OK
#undef  ASSERT_MATRIX_OK
#undef  ASSERT_MATRIX_OK_OR_NULL
#undef  ASSERT_VECTOR_OK
#undef  ASSERT_VECTOR_OK_OR_NULL
#undef  ASSERT_SCALAR_OK
#undef  ASSERT_SCALAR_OK_OR_NULL
#undef  ASSERT_DESCRIPTOR_OK
#undef  ASSERT_DESCRIPTOR_OK_OR_NULL

#define ASSERT_TYPE_OK(t,name,pr)
#define ASSERT_TYPE_OK_OR_NULL(t,name,pr)
#define ASSERT_BINARYOP_OK(op,name,pr)
#define ASSERT_INDEXUNARYOP_OK(op,name,pr)
#define ASSERT_BINARYOP_OK_OR_NULL(op,name,pr)
#define ASSERT_UNARYOP_OK(op,name,pr)
#define ASSERT_UNARYOP_OK_OR_NULL(op,name,pr)
#define ASSERT_SELECTOP_OK(op,name,pr)
#define ASSERT_SELECTOP_OK_OR_NULL(op,name,pr)
#define ASSERT_OP_OK(op,name,pr)
#define ASSERT_OP_OK_OR_NULL(op,name,pr)
#define ASSERT_MONOID_OK(mon,name,pr)
#define ASSERT_SEMIRING_OK(s,name,pr)
#define ASSERT_MATRIX_OK(A,name,pr)
#define ASSERT_MATRIX_OK_OR_NULL(A,name,pr)
#define ASSERT_VECTOR_OK(v,name,pr)
#define ASSERT_VECTOR_OK_OR_NULL(v,name,pr)
#define ASSERT_SCALAR_OK(s,name,pr)
#define ASSERT_SCALAR_OK_OR_NULL(s,name,pr)
#define ASSERT_DESCRIPTOR_OK(d,name,pr)
#define ASSERT_DESCRIPTOR_OK_OR_NULL(d,name,pr)

