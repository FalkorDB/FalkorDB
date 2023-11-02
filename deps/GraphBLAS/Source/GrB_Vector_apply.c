//------------------------------------------------------------------------------
// GrB_Vector_apply: apply a unary or binary operator to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_apply.h"
#include "GB_scalar_wrap.h"
#include "GB_get_mask.h"

//------------------------------------------------------------------------------
// GrB_Vector_apply: apply a unary operator to a vector
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply           // w<M> = accum (w, op(u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_UnaryOp op,           // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Vector_apply (w, M, accum, op, u, desc)") ;
    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator; do not transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        (GrB_Matrix) w, C_replace,  // w and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(w,T)
        (GB_Operator) op, NULL, false, // operator op(.) to apply to the entries
        (GrB_Matrix) u, false,      // u, not transposed
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_1st: apply a binary operator: op(x,u)
//------------------------------------------------------------------------------

static inline GrB_Info GB_1st       // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc,      // descriptor for w and M
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (x) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator; do not transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        (GrB_Matrix) w, C_replace,  // w and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(w,T)
        (GB_Operator) op, x, true,  // operator op(x,.) to apply to the entries
        (GrB_Matrix) u,  false,     // u, not transposed
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_2nd: apply a binary operator: op(u,y) or index_unop
//------------------------------------------------------------------------------

static inline GrB_Info GB_2nd       // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GB_Operator op,           // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc,      // descriptor for w and M
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    GB_RETURN_IF_NULL_OR_FAULTY (y) ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator; do not transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        (GrB_Matrix) w, C_replace,  // w and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(w,T)
        op, y, false,               // operator op(.,y) to apply to the entries
        (GrB_Matrix) u, false,      // u, not transposed
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp1st_Scalar: apply a binary operator: op(x,u)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_BinaryOp1st_Scalar    // w<mask> = accum (w, op(x,u))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_BinaryOp1st_Scalar"
        " (w, M, accum, op, x, u, desc)") ;
    return (GB_1st (w, M, accum, op, x, u, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_apply_BinaryOp1st: historical
//------------------------------------------------------------------------------

// identical to GrB_Vector_apply_BinaryOp1st_Scalar
GrB_Info GxB_Vector_apply_BinaryOp1st
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and M
)
{
    return (GrB_Vector_apply_BinaryOp1st_Scalar (w, M, accum, op, x, u, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp2nd_Scalar: apply a binary operator: op(u,y)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_BinaryOp2nd_Scalar    // w<mask> = accum (w, op(u,y))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_BinaryOp2nd_Scalar"
        " (w, M, accum, op, u, y, desc)") ;
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, y, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_apply_BinaryOp2nd: historical
//------------------------------------------------------------------------------

// identical to GrB_Vector_apply_BinaryOp2nd_Scalar
GrB_Info GxB_Vector_apply_BinaryOp2nd
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and M
)
{
    return (GrB_Vector_apply_BinaryOp2nd_Scalar (w, M, accum, op, u, y, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp1st_TYPE: apply a binary operator: op(x,u)
//------------------------------------------------------------------------------

#define GB_BIND1ST(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Vector_apply_BinaryOp1st_, T)                   \
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results */   \
    const GrB_Vector M,             /* optional mask for w */               \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(w,T) */   \
    const GrB_BinaryOp op,          /* operator to apply to the entries */  \
    const type x,                   /* first input:  scalar x */            \
    const GrB_Vector u,             /* second input: vector u */            \
    const GrB_Descriptor desc       /* descriptor for w and M */            \
)                                                                           \
{                                                                           \
    GB_WHERE (w, GB_STR(prefix) "_Vector_apply_BinaryOp1st_" GB_STR(T)      \
        " (w, M, accum, op, x, u, desc)") ;                                 \
    GB_SCALAR_WRAP (scalar, x, GB_EVAL3 (prefix, _, T)) ;                   \
    return (GB_1st (w, M, accum, op, scalar, u, desc, Werk)) ;              \
}

GB_BIND1ST (GrB, bool      , BOOL  )
GB_BIND1ST (GrB, int8_t    , INT8  )
GB_BIND1ST (GrB, int16_t   , INT16 )
GB_BIND1ST (GrB, int32_t   , INT32 )
GB_BIND1ST (GrB, int64_t   , INT64 )
GB_BIND1ST (GrB, uint8_t   , UINT8 )
GB_BIND1ST (GrB, uint16_t  , UINT16)
GB_BIND1ST (GrB, uint32_t  , UINT32)
GB_BIND1ST (GrB, uint64_t  , UINT64)
GB_BIND1ST (GrB, float     , FP32  )
GB_BIND1ST (GrB, double    , FP64  )
GB_BIND1ST (GxB, GxB_FC32_t, FC32  )
GB_BIND1ST (GxB, GxB_FC64_t, FC64  )

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp1st_UDT: apply a binary operator: op(x,u)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_BinaryOp1st_UDT
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const void *x,                  // first input:  scalar x
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_BinaryOp1st_UDT"
        " (w, M, accum, op, x, u, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, x, (op == NULL) ? NULL : op->xtype) ;
    return (GB_1st (w, M, accum, op, scalar, u, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp2nd_TYPE: apply a binary operator: op(u,y)
//------------------------------------------------------------------------------

#define GB_BIND2ND(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Vector_apply_BinaryOp2nd_, T)                   \
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results */   \
    const GrB_Vector M,             /* optional mask for w*/                \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(w,T) */   \
    const GrB_BinaryOp op,          /* operator to apply to the entries */  \
    const GrB_Vector u,             /* first input:  vector u */            \
    const type y,                   /* second input: scalar y */            \
    const GrB_Descriptor desc       /* descriptor for w and M */            \
)                                                                           \
{                                                                           \
    GB_WHERE (w, GB_STR(prefix) "_Vector_apply_BinaryOp2nd_" GB_STR(T)      \
        " (w, M, accum, op, u, y, desc)") ;                                 \
    GB_SCALAR_WRAP (scalar, y, GB_EVAL3 (prefix, _, T)) ;                   \
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, scalar, desc, Werk)) ;\
}

GB_BIND2ND (GrB, bool      , BOOL  )
GB_BIND2ND (GrB, int8_t    , INT8  )
GB_BIND2ND (GrB, int16_t   , INT16 )
GB_BIND2ND (GrB, int32_t   , INT32 )
GB_BIND2ND (GrB, int64_t   , INT64 )
GB_BIND2ND (GrB, uint8_t   , UINT8 )
GB_BIND2ND (GrB, uint16_t  , UINT16)
GB_BIND2ND (GrB, uint32_t  , UINT32)
GB_BIND2ND (GrB, uint64_t  , UINT64)
GB_BIND2ND (GrB, float     , FP32  )
GB_BIND2ND (GrB, double    , FP64  )
GB_BIND2ND (GxB, GxB_FC32_t, FC32  )
GB_BIND2ND (GxB, GxB_FC64_t, FC64  )

//------------------------------------------------------------------------------
// GrB_Vector_apply_BinaryOp2nd_UDT: apply a binary operator: op(u,y)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_BinaryOp2nd_UDT
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_BinaryOp2nd_UDT"
        " (w, M, accum, op, u, y, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, y, (op == NULL) ? NULL : op->ytype) ;
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, scalar, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_IndexOp_TYPE: apply an idxunop operator: op(u,i,j,thunk)
//------------------------------------------------------------------------------

#define GB_IDXUNOP(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Vector_apply_IndexOp_, T)                       \
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results */   \
    const GrB_Vector M,             /* optional mask for w*/                \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(w,T) */   \
    const GrB_IndexUnaryOp op,      /* operator to apply to the entries */  \
    const GrB_Vector u,             /* first input:  vector u */            \
    const type thunk,               /* second input: scalar thunk */        \
    const GrB_Descriptor desc       /* descriptor for w and M */            \
)                                                                           \
{                                                                           \
    GB_WHERE (w, GB_STR(prefix) "_Vector_apply_IndexOp_" GB_STR(T)          \
        " (w, M, accum, op, u, thunk, desc)") ;                             \
    GB_SCALAR_WRAP (scalar, thunk, GB_EVAL3 (prefix, _, T)) ;               \
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, scalar, desc, Werk)) ;\
}

GB_IDXUNOP (GrB, bool      , BOOL  )
GB_IDXUNOP (GrB, int8_t    , INT8  )
GB_IDXUNOP (GrB, int16_t   , INT16 )
GB_IDXUNOP (GrB, int32_t   , INT32 )
GB_IDXUNOP (GrB, int64_t   , INT64 )
GB_IDXUNOP (GrB, uint8_t   , UINT8 )
GB_IDXUNOP (GrB, uint16_t  , UINT16)
GB_IDXUNOP (GrB, uint32_t  , UINT32)
GB_IDXUNOP (GrB, uint64_t  , UINT64)
GB_IDXUNOP (GrB, float     , FP32  )
GB_IDXUNOP (GrB, double    , FP64  )
GB_IDXUNOP (GxB, GxB_FC32_t, FC32  )
GB_IDXUNOP (GxB, GxB_FC64_t, FC64  )

//------------------------------------------------------------------------------
// GrB_Vector_apply_IndexOp_UDT: apply an idxunop operator: op(u,i,j,thunk)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_IndexOp_UDT
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const void *thunk,              // second input: scalar thunk
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_IndexOp_UDT"
        " (w, M, accum, op, u, thunk, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, thunk, (op == NULL) ? NULL : op->ytype) ;
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, scalar, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_apply_IndexOp_Scalar: apply an idxunop operator: op(u,i,j,thunk)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_apply_IndexOp_Scalar
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar thunk,         // second input: scalar thunk
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE (w, "GrB_Vector_apply_IndexOp_Scalar"
        " (w, M, accum, op, u, thunk, desc)") ;
    return (GB_2nd (w, M, accum, (GB_Operator) op, u, thunk, desc, Werk)) ;
}

