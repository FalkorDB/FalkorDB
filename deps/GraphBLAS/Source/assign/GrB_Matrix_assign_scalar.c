//------------------------------------------------------------------------------
// GrB_Matrix_assign_[SCALAR]: assign a scalar to matrix, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a matrix:

// C<M>(Rows,Cols) = accum(C(Rows,Cols),x)

// The scalar x is implicitly expanded into a matrix A of size nRows-by-nCols,
// with each entry in A equal to x.

// Compare with GxB_Matrix_subassign_scalar,
// which uses M and C_Replace differently.

// The actual work is done in GB_assign_scalar.c.

#define GB_FREE_ALL ;
#include "assign/GB_assign.h"

#define GB_ASSIGN_SCALAR(prefix,type,T,ampersand)                           \
GrB_Info GB_EVAL3 (prefix, _Matrix_assign_, T) /* C<M>(Rows,Cols) += x */   \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results   */ \
    const GrB_Matrix M,             /* optional mask for C               */ \
    const GrB_BinaryOp accum,       /* accum for Z=accum(C(Rows,Cols),x) */ \
    type x,                         /* scalar to assign to C(Rows,Cols)  */ \
    const uint64_t *Rows,           /* row indices                       */ \
    uint64_t nRows,                 /* number of row indices             */ \
    const uint64_t *Cols,           /* column indices                    */ \
    uint64_t nCols,                 /* number of column indices          */ \
    const GrB_Descriptor desc       /* descriptor for C and M            */ \
)                                                                           \
{                                                                           \
    GB_WHERE2 (C, M, "GrB_Matrix_assign_" GB_STR(T)                         \
        " (C, M, accum, x, Rows, nRows, Cols, nCols, desc)") ;              \
    GB_RETURN_IF_NULL (C) ;                                                 \
    GB_BURBLE_START ("GrB_assign") ;                                        \
    info = GB_assign_scalar (C, M, accum, ampersand x, GB_## T ## _code,    \
        Rows, false, nRows, Cols, false, nCols, desc, Werk) ;               \
    GB_BURBLE_END ;                                                         \
    return (info) ;                                                         \
}

GB_ASSIGN_SCALAR (GrB, bool      , BOOL   , &)
GB_ASSIGN_SCALAR (GrB, int8_t    , INT8   , &)
GB_ASSIGN_SCALAR (GrB, uint8_t   , UINT8  , &)
GB_ASSIGN_SCALAR (GrB, int16_t   , INT16  , &)
GB_ASSIGN_SCALAR (GrB, uint16_t  , UINT16 , &)
GB_ASSIGN_SCALAR (GrB, int32_t   , INT32  , &)
GB_ASSIGN_SCALAR (GrB, uint32_t  , UINT32 , &)
GB_ASSIGN_SCALAR (GrB, int64_t   , INT64  , &)
GB_ASSIGN_SCALAR (GrB, uint64_t  , UINT64 , &)
GB_ASSIGN_SCALAR (GrB, float     , FP32   , &)
GB_ASSIGN_SCALAR (GrB, double    , FP64   , &)
GB_ASSIGN_SCALAR (GxB, GxB_FC32_t, FC32   , &)
GB_ASSIGN_SCALAR (GxB, GxB_FC64_t, FC64   , &)
GB_ASSIGN_SCALAR (GrB, void *    , UDT    ,  )

//------------------------------------------------------------------------------
// GrB_Matrix_assign_Scalar: assign a GrB_Scalar to a matrix
//------------------------------------------------------------------------------

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar assignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Matrix_new (&A, stype, nRows, nCols) ;
//  GrB_Matrix_assign (C, M, accum, A, Rows, nRows, Cols, nCols, desc) ;
//  GrB_Matrix_free (&A) ;

GrB_Info GrB_Matrix_assign_Scalar   // C<Mask>(I,J) = accum (C(I,J),s)
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
        "GrB_Matrix_assign_Scalar (C, M, accum, s, I, ni, J, nj, desc)") ;
    GB_BURBLE_START ("GrB_assign") ;

    //--------------------------------------------------------------------------
    // C<M>(I,J) = accum (C(I,J), scalar)
    //--------------------------------------------------------------------------

    GB_OK (GB_Matrix_assign_scalar (C, Mask, accum, scalar,
        I, false, ni, J, false, nj, desc, Werk)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

