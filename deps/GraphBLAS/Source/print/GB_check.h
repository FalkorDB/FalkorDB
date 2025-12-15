//------------------------------------------------------------------------------
// GB_check.h: check and optionally print an object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CHECK_H
#define GB_CHECK_H

// pr values for *_check functions
#define GB0 GxB_SILENT
#define GB1 GxB_SUMMARY
#define GB2 GxB_SHORT
#define GB3 GxB_COMPLETE
#define GB4 GxB_SHORT_VERBOSE
#define GB5 GxB_COMPLETE_VERBOSE

// as above, but do not check zombie counts
#define GB0_Z 6
#define GB1_Z 7
#define GB2_Z 8
#define GB3_Z 9
#define GB4_Z 10
#define GB5_Z 11

GrB_Info GB_entry_check     // print a single value
(
    const GrB_Type type,    // type of value to print
    const void *x,          // value to print
    int pr,                 // print level
    FILE *f,                // file to print to
    // for user-defined types only:
    char **string_handle,   // string buffer for printing
    size_t *string_size     // size of the string buffer
) ;

GrB_Info GB_Type_check      // check a GraphBLAS Type
(
    const GrB_Type type,    // GraphBLAS type to print and check
    const char *name,       // name of the type from the caller; optional
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_UnaryOp_check   // check a GraphBLAS unary operator
(
    const GrB_UnaryOp op,   // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_BinaryOp_check  // check a GraphBLAS binary operator
(
    const GrB_BinaryOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_IndexUnaryOp_check  // check a GraphBLAS index_unary operator
(
    const GrB_IndexUnaryOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_IndexBinaryOp_check  // check a GraphBLAS index_binary operator
(
    const GxB_IndexBinaryOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;

#ifndef GB_CUDA_FOLDER
GrB_Info GB_SelectOp_check  // check a GraphBLAS select operator
(
    const GxB_SelectOp op,  // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;
#endif

GrB_Info GB_Operator_check  // check a GraphBLAS operator
(
    const GB_Operator op,   // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_Monoid_check        // check a GraphBLAS monoid
(
    const GrB_Monoid monoid,    // GraphBLAS monoid to print and check
    const char *name,           // name of the monoid, optional
    int pr,                     // print level
    FILE *f,                    // file for output
    bool in_semiring            // if true, then called by GB_Semiring_check
) ;

GrB_Info GB_Semiring_check          // check a GraphBLAS semiring
(
    const GrB_Semiring semiring,    // GraphBLAS semiring to print and check
    const char *name,               // name of the semiring, optional
    int pr,                         // print level
    FILE *f                         // file for output
) ;

GrB_Info GB_Descriptor_check    // check a GraphBLAS descriptor
(
    const GrB_Descriptor D,     // GraphBLAS descriptor to print and check
    const char *name,           // name of the descriptor, optional
    int pr,                     // print level
    FILE *f                     // file for output
) ;

GrB_Info GB_Context_check       // check a GraphBLAS Context
(
    const GxB_Context Context,  // GraphBLAS Context to print and check
    const char *name,           // name of the Context, optional
    int pr,                     // print level
    FILE *f                     // file for output
) ;

GrB_Info GB_matvec_check    // check a GraphBLAS matrix or vector
(
    const GrB_Matrix A,     // GraphBLAS matrix to print and check
    const char *name,       // name of the matrix, optional
    int pr,                 // print level
    FILE *f,                // file for output
    const char *kind        // "matrix" or "vector"
) ;

GrB_Info GB_Matrix_check    // check a GraphBLAS matrix
(
    const GrB_Matrix A,     // GraphBLAS matrix to print and check
    const char *name,       // name of the matrix
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_Vector_check    // check a GraphBLAS vector
(
    const GrB_Vector v,     // GraphBLAS vector to print and check
    const char *name,       // name of the vector
    int pr,                 // print level
    FILE *f                 // file for output
) ;

GrB_Info GB_Scalar_check    // check a GraphBLAS GrB_Scalar
(
    const GrB_Scalar v,     // GraphBLAS GrB_Scalar to print and check
    const char *name,       // name of the GrB_Scalar
    int pr,                 // print level
    FILE *f                 // file for output
) ;

#endif

