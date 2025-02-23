//------------------------------------------------------------------------------
// GrB_Vector_assign_[SCALAR]: assign scalar to vector, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a vector, w<M>(I) = accum(w(I),x)
// The scalar x is implicitly expanded into a vector u of size ni-by-1,
// with each entry in u equal to x.

#define GB_FREE_ALL ;
#include "assign/GB_assign.h"

#define GB_ASSIGN_SCALAR(prefix,type,T,ampersand)                           \
GrB_Info GB_EVAL3 (prefix, _Vector_assign_, T) /* w<M>(I)=accum(w(I),x)*/   \
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results   */ \
    const GrB_Vector M,             /* optional mask for w               */ \
    const GrB_BinaryOp accum,       /* opt. accum for Z=accum(w(I),x)    */ \
    type x,                         /* scalar to assign to w(I)          */ \
    const uint64_t *I,              /* row indices                       */ \
    uint64_t ni,                    /* number of row indices             */ \
    const GrB_Descriptor desc       /* descriptor for w and mask         */ \
)                                                                           \
{                                                                           \
    GB_WHERE2 (w, M, "GrB_Vector_assign_" GB_STR(T)                         \
        " (w, M, accum, x, I, ni, desc)") ;                                 \
    GB_RETURN_IF_NULL (w) ;                                                 \
    GB_BURBLE_START ("GrB_assign") ;                                        \
    ASSERT (GB_VECTOR_OK (w)) ;                                             \
    ASSERT (GB_IMPLIES (M != NULL, GB_VECTOR_OK (M))) ;                     \
    info = GB_assign_scalar ((GrB_Matrix) w, (GrB_Matrix) M, accum,         \
        ampersand x, GB_## T ## _code,                                      \
        I, false, ni, GrB_ALL, false, 1, desc, Werk) ;                      \
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
// GrB_Vector_assign_Scalar: assign a GrB_Scalar to a matrix
//------------------------------------------------------------------------------

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar subassignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Vector_new (&A, stype, ni) ;
//  GrB_Vector_assign (w, M, accum, A, I, ni, desc) ;
//  GrB_Vector_free (&A) ;

GrB_Info GrB_Vector_assign_Scalar   // w(I)<mask> = accum (w(I),s)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar scalar,        // scalar to assign to w(I)
    const uint64_t *I,              // row indices
    uint64_t ni,                    // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (w, mask, scalar,
        "GrB_Vector_assign_Scalar (w, M, accum, s, I, ni, desc)") ;
    GB_BURBLE_START ("GrB_assign") ;

    //--------------------------------------------------------------------------
    // w<M>(I) = accum (w(I), scalar)
    //--------------------------------------------------------------------------

    GB_OK (GB_Vector_assign_scalar (w, mask, accum, scalar,
        I, false, ni, desc, Werk)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

