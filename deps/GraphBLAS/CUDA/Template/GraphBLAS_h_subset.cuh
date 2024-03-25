//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GraphBLAS_h_subset.cuh: subset of GraphBLAS.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Note the header gaurd is the same as GraphBLAS.h:
#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

typedef enum
{

    GrB_SUCCESS = 0,            // all is well

    //--------------------------------------------------------------------------
    // informational codes, not an error:
    //--------------------------------------------------------------------------

    GrB_NO_VALUE = 1,           // A(i,j) requested but not there
    GxB_EXHAUSTED = 7089,       // iterator is exhausted

    //--------------------------------------------------------------------------
    // errors:
    //--------------------------------------------------------------------------

    GrB_UNINITIALIZED_OBJECT = -1,  // object has not been initialized
    GrB_NULL_POINTER = -2,          // input pointer is NULL
    GrB_INVALID_VALUE = -3,         // generic error; some value is bad
    GrB_INVALID_INDEX = -4,         // row or column index is out of bounds
    GrB_DOMAIN_MISMATCH = -5,       // object domains are not compatible
    GrB_DIMENSION_MISMATCH = -6,    // matrix dimensions do not match
    GrB_OUTPUT_NOT_EMPTY = -7,      // output matrix already has values
    GrB_NOT_IMPLEMENTED = -8,       // method not implemented
    GrB_ALREADY_SET = -9,           // field already written to
    GrB_PANIC = -101,               // unknown error
    GrB_OUT_OF_MEMORY = -102,       // out of memory
    GrB_INSUFFICIENT_SPACE = -103,  // output array not large enough
    GrB_INVALID_OBJECT = -104,      // object is corrupted
    GrB_INDEX_OUT_OF_BOUNDS = -105, // row or col index out of bounds
    GrB_EMPTY_OBJECT = -106         // an object does not contain a value

}
GrB_Info ;

#undef restrict
#undef GB_restrict
#define GB_restrict __restrict__
#define restrict GB_restrict

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#undef  GB_GLOBAL
#define GB_GLOBAL extern

// GB_STR: convert the content of x into a string "x"
#define GB_XSTR(x) GB_STR(x)
#define GB_STR(x) #x

#undef  GxB_MAX_NAME_LEN
#define GxB_MAX_NAME_LEN 128

typedef uint64_t GrB_Index ;
typedef struct GB_Descriptor_opaque *GrB_Descriptor ;
typedef struct GB_Type_opaque *GrB_Type ;
typedef struct GB_UnaryOp_opaque *GrB_UnaryOp ;
typedef struct GB_BinaryOp_opaque *GrB_BinaryOp ;
typedef struct GB_IndexUnaryOp_opaque *GrB_IndexUnaryOp ;
typedef struct GB_Monoid_opaque *GrB_Monoid ;
typedef struct GB_Semiring_opaque *GrB_Semiring ;
typedef struct GB_Scalar_opaque *GrB_Scalar ;
typedef struct GB_Vector_opaque *GrB_Vector ;
typedef struct GB_Matrix_opaque *GrB_Matrix ;
typedef struct GB_Context_opaque *GxB_Context ;
typedef struct GB_Global_opaque *GrB_Global ;
typedef struct GB_Iterator_opaque *GxB_Iterator ;

#define GxB_HYPERSPARSE 1   // store matrix in hypersparse form
#define GxB_SPARSE      2   // store matrix as sparse form (compressed vector)
#define GxB_BITMAP      4   // store matrix as a bitmap
#define GxB_FULL        8   // store matrix as full; all entries must be present

typedef void (*GxB_unary_function)  (void *, const void *) ;
typedef void (*GxB_binary_function) (void *, const void *, const void *) ;

typedef bool (*GxB_select_function)      // return true if A(i,j) is kept
(
    GrB_Index i,                // row index of A(i,j)
    GrB_Index j,                // column index of A(i,j)
    const void *x,              // value of A(i,j)
    const void *thunk           // optional input for select function
) ;

typedef void (*GxB_index_unary_function)
(
    void *z,            // output value z, of type ztype
    const void *x,      // input value x of type xtype; value of v(i) or A(i,j)
    GrB_Index i,        // row index of A(i,j)
    GrB_Index j,        // column index of A(i,j), or zero for v(i)
    const void *y       // input scalar y
) ;

#define GxB_GLOBAL_GPU_ID 26

typedef enum
{
    // for all GrB_Descriptor fields:
    GxB_DEFAULT = 0,    // default behavior of the method

    // for GrB_OUTP only:
    GrB_REPLACE = 1,    // clear the output before assigning new values to it

    // for GrB_MASK only:
    GrB_COMP = 2,       // use the structural complement of the input
    GrB_SCMP = 2,       // same as GrB_COMP (historical; use GrB_COMP instead)
    GrB_STRUCTURE = 4,  // use the only pattern of the mask, not its values

    // for GrB_INP0 and GrB_INP1 only:
    GrB_TRAN = 3,       // use the transpose of the input

    // for GxB_AxB_METHOD only:
    GxB_AxB_GUSTAVSON = 1001,   // gather-scatter saxpy method
    GxB_AxB_DOT       = 1003,   // dot product
    GxB_AxB_HASH      = 1004,   // hash-based saxpy method
    GxB_AxB_SAXPY     = 1005    // saxpy method (any kind)
}
GrB_Desc_Value ;

#endif

