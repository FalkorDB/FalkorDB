//------------------------------------------------------------------------------
// GxB_Vector_subassign_[SCALAR]: assign scalar to vector, via scalar expansion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a subvector, w(I)<M> = accum(w(I),x)
// The scalar x is implicitly expanded into a vector u of size ni-by-1,
// with each entry in u equal to x.

// The actual work is done in GB_subassign_scalar.c.

#define GB_FREE_ALL ;
#include "assign/GB_subassign.h"

#define GB_ASSIGN_SCALAR(type,T,ampersand)                                  \
GrB_Info GB_EVAL2 (GXB (Vector_subassign_), T) /* w(I)<M> = accum (w(I),x)*/\
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results   */ \
    const GrB_Vector M,             /* optional mask for w(I)            */ \
    const GrB_BinaryOp accum,       /* opt. accum for Z=accum(w(I),x)    */ \
    type x,                         /* scalar to assign to w(I)          */ \
    const uint64_t *I,              /* row indices                       */ \
    uint64_t ni,                    /* number of row indices             */ \
    const GrB_Descriptor desc       /* descriptor for w(I) and M         */ \
)                                                                           \
{                                                                           \
    GB_WHERE2 (w, M, "GxB_Vector_subassign_" GB_STR(T)                      \
        " (w, M, accum, x, I, ni, desc)") ;                                 \
    GB_RETURN_IF_NULL (w) ;                                                 \
    GB_BURBLE_START ("GxB_subassign") ;                                     \
    ASSERT (GB_VECTOR_OK (w)) ;                                             \
    ASSERT (GB_IMPLIES (M != NULL, GB_VECTOR_OK (M))) ;                     \
    info = GB_subassign_scalar ((GrB_Matrix) w, (GrB_Matrix) M, accum,      \
        ampersand x, GB_## T ## _code,                                      \
        I, false, ni, GrB_ALL, false, 1, desc, Werk) ;                      \
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
// GxB_Vector_subassign_Scalar: subassign a GrB_Scalar to a vector
//------------------------------------------------------------------------------

// If the GrB_Scalar s is non-empty, then this is the same as the non-opapue
// scalar assignment above.

// If the GrB_Scalar s is empty of type stype, then this is identical to:
//  GrB_Vector_new (&A, stype, ni) ;
//  GxB_Vector_subassign (w, M, accum, A, I, ni, desc) ;
//  GrB_Vector_free (&A) ;

GrB_Info GxB_Vector_subassign_Scalar   // w(I)><mask> = accum (w(I),s)
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
        "GxB_Vector_subassign_Scalar (w, M, accum, s, I, desc)") ;
    GB_BURBLE_START ("GxB_subassign") ;

    //--------------------------------------------------------------------------
    // w(I)<M> = accum (w(I), scalar)
    //--------------------------------------------------------------------------

    GB_OK (GB_Vector_subassign_scalar (w, mask, accum, scalar,
        I, false, ni, desc, Werk)) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    GB_BURBLE_END ;
    return (info) ;
}

