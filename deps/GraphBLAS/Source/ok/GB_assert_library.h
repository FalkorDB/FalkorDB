//------------------------------------------------------------------------------
// GB_assert_library.h: assertions for all of GraphBLAS except JIT kernels.
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods are used in all of GraphBLAS except for JIT kernels.
// For JIT kernels, see ok/GB_assert_kernels.h.

//------------------------------------------------------------------------------
// debugging definitions
//------------------------------------------------------------------------------

#undef ASSERT_OK
#undef ASSERT_OK_OR_NULL

#ifdef GB_DEBUG

    // call a GraphBLAS method and assert that it returns GrB_SUCCESS
    #define ASSERT_OK(X)                                                    \
    {                                                                       \
        GrB_Info Info = (X) ;                                               \
        ASSERT (Info == GrB_SUCCESS) ;                                      \
    }

    // call a GraphBLAS method and assert that it returns GrB_SUCCESS
    // or GrB_NULL_POINTER.
    #define ASSERT_OK_OR_NULL(X)                                            \
    {                                                                       \
        GrB_Info Info = (X) ;                                               \
        ASSERT (Info == GrB_SUCCESS || Info == GrB_NULL_POINTER) ;          \
    }

#else

    // debugging disabled
    #define ASSERT_OK(X)
    #define ASSERT_OK_OR_NULL(X)

#endif

// for finding tests that trigger statement coverage.  If running a test
// in GraphBLAS/Tcov, the test does not terminate.
#undef GB_GOTCHA
#ifdef GBCOVER
#define GB_GOTCHA                                                       \
{                                                                       \
    fprintf (stderr, "\nGotcha: " __FILE__ " line: %d\n", __LINE__) ;   \
    GBDUMP ("\nGotcha: " __FILE__ " line: %d\n", __LINE__) ;            \
}
#else
#define GB_GOTCHA                                                       \
{                                                                       \
    fprintf (stderr, "\ngotcha: " __FILE__ " line: %d\n", __LINE__) ;   \
    GBDUMP ("\ngotcha: " __FILE__ " line: %d\n", __LINE__) ;            \
    GB_Global_abort ( ) ;                                               \
}
#endif

#undef  GB_HERE
#define GB_HERE GBDUMP ("%2d: Here: " __FILE__ "\n", __LINE__) ;

//------------------------------------------------------------------------------
// assertions for checking specific objects
//------------------------------------------------------------------------------

#undef  ASSERT_TYPE_OK
#undef  ASSERT_TYPE_OK_OR_NULL
#undef  ASSERT_BINARYOP_OK
#undef  ASSERT_INDEXUNARYOP_OK
#undef  ASSERT_INDEXBINARYOP_OK
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

#define ASSERT_TYPE_OK(t,name,pr)  \
    ASSERT_OK (GB_Type_check (t, name, pr, NULL))

#define ASSERT_TYPE_OK_OR_NULL(t,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Type_check (t, name, pr, NULL))

#define ASSERT_BINARYOP_OK(op,name,pr)  \
    ASSERT_OK (GB_BinaryOp_check (op, name, pr, NULL))

#define ASSERT_INDEXUNARYOP_OK(op,name,pr)  \
    ASSERT_OK (GB_IndexUnaryOp_check (op, name, pr, NULL))

#define ASSERT_INDEXBINARYOP_OK(op,name,pr)  \
    ASSERT_OK (GB_IndexBinaryOp_check (op, name, pr, NULL))

#define ASSERT_BINARYOP_OK_OR_NULL(op,name,pr)  \
    ASSERT_OK_OR_NULL (GB_BinaryOp_check (op, name, pr, NULL))

#define ASSERT_UNARYOP_OK(op,name,pr)  \
    ASSERT_OK (GB_UnaryOp_check (op, name, pr, NULL))

#define ASSERT_UNARYOP_OK_OR_NULL(op,name,pr)  \
    ASSERT_OK_OR_NULL (GB_UnaryOp_check (op, name, pr, NULL))

#define ASSERT_SELECTOP_OK(op,name,pr)  \
    ASSERT_OK (GB_SelectOp_check (op, name, pr, NULL))

#define ASSERT_SELECTOP_OK_OR_NULL(op,name,pr)  \
    ASSERT_OK_OR_NULL (GB_SelectOp_check (op, name, pr, NULL))

#define ASSERT_OP_OK(op,name,pr) \
    ASSERT_OK (GB_Operator_check (op, name, pr, NULL))

#define ASSERT_OP_OK_OR_NULL(op,name,pr) \
    ASSERT_OK_OR_NULL (GB_Operator_check (op, name, pr, NULL))

#define ASSERT_MONOID_OK(mon,name,pr)  \
    ASSERT_OK (GB_Monoid_check (mon, name, pr, NULL, false))

#define ASSERT_SEMIRING_OK(s,name,pr)  \
    ASSERT_OK (GB_Semiring_check (s, name, pr, NULL))

#define ASSERT_MATRIX_OK(A,name,pr)  \
    ASSERT_OK (GB_Matrix_check (A, name, pr, NULL))

#define ASSERT_MATRIX_OK_OR_NULL(A,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Matrix_check (A, name, pr, NULL))

#define ASSERT_VECTOR_OK(v,name,pr)  \
    ASSERT_OK (GB_Vector_check (v, name, pr, NULL))

#define ASSERT_VECTOR_OK_OR_NULL(v,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Vector_check (v, name, pr, NULL))

#define ASSERT_SCALAR_OK(s,name,pr)  \
    ASSERT_OK (GB_Scalar_check (s, name, pr, NULL))

#define ASSERT_SCALAR_OK_OR_NULL(s,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Scalar_check (s, name, pr, NULL))

#define ASSERT_DESCRIPTOR_OK(d,name,pr)  \
    ASSERT_OK (GB_Descriptor_check (d, name, pr, NULL))

#define ASSERT_DESCRIPTOR_OK_OR_NULL(d,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Descriptor_check (d, name, pr, NULL))

#define ASSERT_CONTEXT_OK(c,name,pr)  \
    ASSERT_OK (GB_Context_check (c, name, pr, NULL))

#define ASSERT_CONTEXT_OK_OR_NULL(c,name,pr)  \
    ASSERT_OK_OR_NULL (GB_Context_check (c, name, pr, NULL))

#if 0
// For tracking down 64-bit integers when not expected;
// this is meant for development only.
#undef  ASSERT_MATRIX_OK
#undef  ASSERT_MATRIX_OK_OR_NULL
#undef  ASSERT_VECTOR_OK
#undef  ASSERT_VECTOR_OK_OR_NULL

#define ASSERT_MATRIX_OK(A,name,pr) \
{ \
    if ((A) != NULL && GB_IS_SPARSE((A))) \
    { \
        if (!((A)->p_is_32) || !((A)->i_is_32))  \
        { \
            printf ("Hey: %s %d: %s (%d,%d)\n", \
            __FILE__, __LINE__, name, (A)->p_is_32, (A)->i_is_32) ; \
        } \
    } \
    else if ((A) != NULL && GB_IS_HYPERSPARSE((A))) \
    { \
        if (!((A)->p_is_32) || !((A)->i_is_32) || !((A)->j_is_32))  \
        { \
            printf ("Hey: %s %d: %s (%d,%d,%d)\n", \
            __FILE__, __LINE__, name, \
            (A)->p_is_32, (A)->j_is_32, (A)->i_is_32) ; \
        } \
    } \
}

#define ASSERT_MATRIX_OK_OR_NULL(A,name,pr) ASSERT_MATRIX_OK (A,name,pr)
#define ASSERT_VECTOR_OK_OR_NULL(A,name,pr) ASSERT_MATRIX_OK (A,name,pr)
#define ASSERT_VECTOR_OK(A,name,pr) ASSERT_MATRIX_OK (A,name,pr)
#endif

