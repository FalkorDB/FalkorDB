//------------------------------------------------------------------------------
// GrB_Matrix_reduce: reduce a matrix to a vector or scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "reduce/GB_reduce.h"
#include "binaryop/GB_binop.h"

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_TYPE: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// Reduce entries in a matrix to a scalar, c = accum (c, reduce_to_scalar(A)))

// All entries in the matrix are "summed" to a single scalar t using the reduce
// monoid, which must be associative (otherwise the results are undefined).
// The result is either assigned to the output scalar c (if accum is NULL), or
// it accumulated in the result c via c = accum(c,t).  If A has no entries, the
// result t is the identity value of the monoid.  Unlike most other GraphBLAS
// operations, this operation uses an accum operator but no mask.

#define GB_REDUCE_TO_CSCALAR(prefix,type,T)                                    \
GrB_Info GB_EVAL3 (prefix, _Matrix_reduce_, T) /* c = accum (c, reduce (A)) */ \
(                                                                              \
    type *c,                        /* result scalar                        */ \
    const GrB_BinaryOp accum,       /* optional accum for c=accum(c,t)      */ \
    const GrB_Monoid monoid,        /* monoid to do the reduction           */ \
    const GrB_Matrix A,             /* matrix to reduce                     */ \
    const GrB_Descriptor desc                                                  \
)                                                                              \
{                                                                              \
    GB_RETURN_IF_NULL (A) ;                                                    \
    GB_WHERE_1 (A, "GrB_Matrix_reduce_" GB_STR(T)                              \
        " (&c, accum, monoid, A, desc)") ;                                     \
    GB_BURBLE_START ("GrB_reduce") ;                                           \
    info = GB_reduce_to_scalar (c, GB_EVAL3 (prefix, _, T), accum, monoid, A,  \
        Werk) ;                                                                \
    GB_BURBLE_END ;                                                            \
    return (info) ;                                                            \
}

GB_REDUCE_TO_CSCALAR (GrB, bool      , BOOL   )
GB_REDUCE_TO_CSCALAR (GrB, int8_t    , INT8   )
GB_REDUCE_TO_CSCALAR (GrB, int16_t   , INT16  )
GB_REDUCE_TO_CSCALAR (GrB, int32_t   , INT32  )
GB_REDUCE_TO_CSCALAR (GrB, int64_t   , INT64  )
GB_REDUCE_TO_CSCALAR (GrB, uint8_t   , UINT8  )
GB_REDUCE_TO_CSCALAR (GrB, uint16_t  , UINT16 )
GB_REDUCE_TO_CSCALAR (GrB, uint32_t  , UINT32 )
GB_REDUCE_TO_CSCALAR (GrB, uint64_t  , UINT64 )
GB_REDUCE_TO_CSCALAR (GrB, float     , FP32   )
GB_REDUCE_TO_CSCALAR (GrB, double    , FP64   )
GB_REDUCE_TO_CSCALAR (GxB, GxB_FC32_t, FC32   )
GB_REDUCE_TO_CSCALAR (GxB, GxB_FC64_t, FC64   )

GrB_Info GrB_Matrix_reduce_UDT      // c = accum (c, reduce_to_scalar (A))
(
    void *c,                        // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
)
{ 
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_WHERE_1 (A, "GrB_Matrix_reduce_UDT (&c, accum, monoid, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;

    info = GB_reduce_to_scalar (c, monoid->op->ztype, accum,
        monoid, A, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_Monoid: reduce a matrix to a vector via a monoid
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_reduce_Monoid   // w<M> = accum (w,reduce(A))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid monoid,        // reduce monoid for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{ 
    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE3 (w, M, A,
        "GrB_Matrix_reduce_Monoid (w, M, accum, monoid, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;

    info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M, accum, monoid,
        A, desc, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_Monoid_Scalar: reduce a matrix to a GrB_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_reduce_Monoid_Scalar
(
    GrB_Scalar S,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
)
{ 
    GB_RETURN_IF_NULL (S) ;
    GB_WHERE2 (S, A,
        "GrB_Matrix_reduce_Monoid_Scalar (s, accum, monoid, A, desc)") ;
    GB_BURBLE_START ("GrB_reduce") ;

    info = GB_Scalar_reduce (S, accum, monoid, A, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_BinaryOp: reduce a matrix to a vector via a binary op
//------------------------------------------------------------------------------

// Only binary ops that correspond to a known monoid are supported.
// This method is not recommended.

GrB_Info GrB_Matrix_reduce_BinaryOp
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // reduce operator for t=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{
    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE3 (w, M, A, "GrB_Matrix_reduce_BinaryOp : DEPRECATED!") ;
    GB_BURBLE_START ("GrB_reduce with binary op : DEPRECATED!") ;

    // convert the binary op to its corresponding monoid
    if (op->ztype != op->xtype || op->ztype != op->ytype)
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }
    GrB_Monoid monoid = GB_binop_to_monoid (op) ;
    if (monoid == NULL)
    { 
        return (GrB_NOT_IMPLEMENTED) ;
    }

    // w<M> = reduce (A) via the monoid
    info = GB_reduce_to_vector ((GrB_Matrix) w, (GrB_Matrix) M, accum, monoid,
        A, desc, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_reduce_BinaryOp_Scalar: reduce matrix to GrB_Scalar via binary op
//------------------------------------------------------------------------------

// Only binary ops that correspond to a known monoid are supported.
// This method is not recommended.

GrB_Info GrB_Matrix_reduce_BinaryOp_Scalar
(
    GrB_Scalar S,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_BinaryOp op,          // binary op to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    const GrB_Descriptor desc
)
{
    GB_RETURN_IF_NULL (S) ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_WHERE2 (S, A, "GrB_Matrix_reduce_BinaryOp_Scalar : DEPRECATED!") ;
    GB_BURBLE_START ("GrB_reduce with binary op : DEPRECATED!") ;

    // convert the binary op to its corresponding monoid
    if (op->ztype != op->xtype || op->ztype != op->ytype)
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }
    GrB_Monoid monoid = GB_binop_to_monoid (op) ;
    if (monoid == NULL)
    { 
        return (GrB_NOT_IMPLEMENTED) ;
    }

    // S = reduce (A) via the monoid
    info = GB_Scalar_reduce (S, accum, monoid, A, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

