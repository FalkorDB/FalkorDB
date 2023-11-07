//------------------------------------------------------------------------------
// GB_mex_errors.h: error handling macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define FAIL(s)                                                             \
{                                                                           \
    mexPrintf ("\nTest failure: %s line %d\n", __FILE__, __LINE__) ;        \
    mexPrintf ( "%s\n", GB_STR(s)) ;                                        \
    GB_mx_at_exit ( ) ;                                                     \
    mexErrMsgTxt (GB_STR(s) " line: " GB_XSTR(__LINE__)) ;                  \
}

#undef  CHECK
#define CHECK(x)    if (!(x)) FAIL(x) ;
#define CHECK2(x,s) if (!(x)) FAIL(s) ;

// assert that a method should return a particular error code
#define ERR(method)                                                         \
{                                                                           \
    info = method ;                                                         \
    if (info != expected)                                                   \
    {                                                                       \
        mexPrintf ("got %d expected %d\n", info, expected) ;                \
    }                                                                       \
    CHECK2 (info == expected, method) ;                                     \
}

// assert that a method should return a particular error code: with logger,
// for a GrB_Matrix, GrB_Vector, or GrB_Scalar
#define ERR1(C,method)                                                      \
{                                                                           \
    info = method ;                                                         \
    if (info != expected)                                                   \
    {                                                                       \
        const char *error_logger = NULL ;                                   \
        GrB_Matrix_error_(&error_logger, ((GrB_Matrix) C)) ;                \
        if (error_logger != NULL) mexPrintf ("[%s]\n", error_logger) ;      \
        mexPrintf ("got %d expected %d\n", info, expected) ;                \
    }                                                                       \
    CHECK2 (info == expected, method) ;                                     \
}

// assert that a method should return a particular error code: with logger,
// for a GrB_Descriptor
#define ERRD(descriptor,method)                                             \
{                                                                           \
    info = method ;                                                         \
    if (info != expected)                                                   \
    {                                                                       \
        const char *error_logger = NULL ;                                   \
        GrB_Descriptor_error_(&error_logger, descriptor) ;                  \
        if (error_logger != NULL) mexPrintf ("[%s]\n", error_logger) ;      \
        mexPrintf ("got %d expected %d\n", info, expected) ;                \
    }                                                                       \
    CHECK2 (info == expected, method) ;                                     \
}

// assert that a method should succeed
#define OK(method)                                                          \
{                                                                           \
    info = method ;                                                         \
    if (info < 0)                                                           \
    {                                                                       \
        mexPrintf ("[%d] Test failed\n", info) ;                            \
        FAIL (method) ;                                                     \
    }                                                                       \
}

