//------------------------------------------------------------------------------
// GxB_Matrix_subassign_[SCALAR]: assign to submatrix, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix:

// C(Rows,Cols)<M> = accum(C(Rows,Cols),x)

// The scalar x is implicitly expanded into a matrix A of size nRows-by-nCols,
// with each entry in A equal to x.

// Compare with GrB_Matrix_assign_scalar,
// which uses M and C_Replace differently.

#define GB_FREE_ALL ;
#include "assign/GB_subassign.h"

#define GB_ASSIGN_SCALAR(type,T,ampersand)                                  \
GrB_Info GB_EVAL2 (GXB (Matrix_subassign_), T) /* C(Rows,Cols)<M> += x   */ \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results   */ \
    const GrB_Matrix M,             /* optional mask for C(Rows,Cols)    */ \
    const GrB_BinaryOp accum,       /* accum for Z=accum(C(Rows,Cols),x) */ \
    type x,                         /* scalar to assign to C(Rows,Cols)  */ \
    const uint64_t *Rows,           /* row indices                       */ \
    uint64_t nRows,                 /* number of row indices             */ \
    const uint64_t *Cols,           /* column indices                    */ \
    uint64_t nCols,                 /* number of column indices          */ \
    const GrB_Descriptor desc       /* descriptor for C and M */            \
)                                                                           \
{                                                                           \
    GB_WHERE2 (C, M, "GxB_Matrix_subassign_" GB_STR(T)                      \
        " (C, M, accum, x, Rows, nRows, Cols, nCols, desc)") ;              \
    GB_RETURN_IF_NULL (C) ;                                                 \
    GB_BURBLE_START ("GxB_Matrix_subassign " GB_STR(T)) ;                   \
    info = GB_subassign_scalar (C, M, accum, ampersand x, GB_## T ## _code, \
        Rows, false, nRows, Cols, false, nCols, desc, Werk) ;               \
    GB_BURBLE_END ;                                                         \
    return (info) ;                                                         \
}

GB_ASSIGN_SCALAR (bool      , BOOL   , &)
GB_ASSIGN_SCALAR (int8_t    , INT8   , &)
GB_ASSIGN_SCALAR (uint8_t   , UINT8  , &)
GB_ASSIGN_SCALAR (int16_t   , INT16  , &)
GB_ASSIGN_SCALAR (uint16_t  , UINT16 , &)
GB_ASSIGN_SCALAR (int32_t   , INT32  , &)
GB_ASSIGN_SCALAR (uint32_t  , UINT32 , &)
GB_ASSIGN_SCALAR (int64_t   , INT64  , &)
GB_ASSIGN_SCALAR (uint64_t  , UINT64 , &)
GB_ASSIGN_SCALAR (float     , FP32   , &)
GB_ASSIGN_SCALAR (double    , FP64   , &)
GB_ASSIGN_SCALAR (GxB_FC32_t, FC32   , &)
GB_ASSIGN_SCALAR (GxB_FC64_t, FC64   , &)
GB_ASSIGN_SCALAR (void *    , UDT    ,  )

//------------------------------------------------------------------------------
// GxB_Matrix_subassign_Scalar: subassign a GrB_Scalar to a matrix
//------------------------------------------------------------------------------

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar subassignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Matrix_new (&A, stype, nRows, nCols) ;
//  GxB_Matrix_subassign (C, M, accum, A, Rows, nRows, Cols, nCols, desc) ;
//  GrB_Matrix_free (&A) ;

GrB_Info GxB_Matrix_subassign_Scalar   // C(I,J)<M> = accum (C(I,J),s)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    const GrB_Scalar scalar,        // scalar to assign to C(I,J)
    const uint64_t *I,              // row indices
    uint64_t ni,                    // number of row indices
    const uint64_t *J,              // column indices
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc       // descriptor for C and Mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (C, Mask, scalar,
        "GxB_Matrix_subassign_Scalar (C, M, accum, s, Rows, nRows, Cols, nCols,"
        " desc)") ;
    GB_BURBLE_START ("GxB_subassign") ;

    //--------------------------------------------------------------------------
    // C(I,J)<M> = accum (C(I,J), scalar)
    //--------------------------------------------------------------------------

    GB_OK (GB_Matrix_subassign_scalar (C, Mask, accum, scalar,
        I, false, ni, J, false, nj, desc, Werk)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

