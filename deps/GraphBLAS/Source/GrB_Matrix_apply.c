//------------------------------------------------------------------------------
// GrB_Matrix_apply: apply a unary or binary operator to a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_apply.h"
#include "GB_scalar_wrap.h"
#include "GB_get_mask.h"

//------------------------------------------------------------------------------
// GrB_Matrix_apply: apply a unary operator to a matrix
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply           // C<M> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_UnaryOp op,           // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_apply (C, M, accum, op, A, desc)") ;
    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator and optionally transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        C, C_replace,               // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        (GB_Operator) op, NULL, false, // operator op(.) to apply to the entries
        A, A_transpose,             // A and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_1st: apply a binary operator: op(x,A)
//------------------------------------------------------------------------------

static inline GrB_Info GB_1st       // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (x) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // get the descriptor, using GrB_INP1 to transpose the matrix
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, A_transpose, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator and optionally transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        C, C_replace,               // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        (GB_Operator) op, x, true,  // operator op(x,.) to apply to the entries
        A, A_transpose,             // A and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_2nd: apply a binary operator or idxunop: op(A,y)
//------------------------------------------------------------------------------

static inline GrB_Info GB_2nd       // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GB_Operator op,           // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_BURBLE_START ("GrB_apply") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (y) ;

    // get the descriptor, using GrB_INP0 to transpose the matrix
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // apply the operator and optionally transpose
    //--------------------------------------------------------------------------

    info = GB_apply (
        C, C_replace,               // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        op, y, false,               // operator op(.,y) to apply to the entries
        A, A_transpose,             // A and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_BinaryOp1st_Scalar: apply a binary operator: op(x,A)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_BinaryOp1st_Scalar    // C<M>=accum(C,op(x,A))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_BinaryOp1st_Scalar"
        " (C, M, accum, op, x, A, desc)") ;
    return (GB_1st (C, M, accum, op, x, A, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_apply_BinaryOp1st: historical
//------------------------------------------------------------------------------

// identical to GrB_Matrix_apply_BinaryOp1st_Scalar
GrB_Info GxB_Matrix_apply_BinaryOp1st
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Scalar x,             // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{
    return (GrB_Matrix_apply_BinaryOp1st_Scalar (C, M, accum, op, x, A, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_BinaryOp2nd_Scalar: apply a binary operator: op(A,y)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_BinaryOp2nd_Scalar    // C<M>=accum(C,op(A,y))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_BinaryOp2nd_Scalar"
        " (C, M, accum, op, A, y, desc)") ;
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, y, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_Matrix_apply_BinaryOp2nd: historical
//------------------------------------------------------------------------------

// identical to GrB_Matrix_apply_BinaryOp2nd_Scalar
GrB_Info GxB_Matrix_apply_BinaryOp2nd
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{
    return (GrB_Matrix_apply_BinaryOp2nd_Scalar (C, M, accum, op, A, y, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_BinaryOp1st_TYPE: apply a binary operator: op(x,A)
//------------------------------------------------------------------------------

#define GB_BIND1ST(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Matrix_apply_BinaryOp1st_, T)                   \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results */   \
    const GrB_Matrix M,             /* optional mask for C*/                \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(C,T) */   \
    const GrB_BinaryOp op,          /* operator to apply to the entries */  \
    const type x,                   /* first input:  scalar x */            \
    const GrB_Matrix A,             /* second input: matrix A */            \
    const GrB_Descriptor desc       /* descriptor for C, M, and A */        \
)                                                                           \
{                                                                           \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_apply_BinaryOp1st_" GB_STR(T)      \
        " (C, M, accum, op, x, A, desc)") ;                                 \
    GB_SCALAR_WRAP (scalar, x, GB_EVAL3 (prefix, _, T)) ;                   \
    return (GB_1st (C, M, accum, op, scalar, A, desc, Werk)) ;              \
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
// GrB_Matrix_apply_BinaryOp1st_UDT: apply a binary operator: op(x,A)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_BinaryOp1st_UDT
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const void *x,                  // first input:  scalar x
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_BinaryOp1st_UDT "
        " (C, M, accum, op, x, A, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, x, (op == NULL) ? NULL : op->xtype) ;
    return (GB_1st (C, M, accum, op, scalar, A, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_BinaryOp2nd_TYPE: apply a binary operator: op(A,y)
//------------------------------------------------------------------------------

#define GB_BIND2ND(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Matrix_apply_BinaryOp2nd_, T)                   \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results */   \
    const GrB_Matrix M,             /* optional mask for C*/                \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(C,T) */   \
    const GrB_BinaryOp op,          /* operator to apply to the entries */  \
    const GrB_Matrix A,             /* first input:  matrix A */            \
    const type y,                   /* second input: scalar y */            \
    const GrB_Descriptor desc       /* descriptor for C, M, and A */        \
)                                                                           \
{                                                                           \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_apply_BinaryOp2nd_" GB_STR(T)      \
        " (C, M, accum, op, A, y, desc)") ;                                 \
    GB_SCALAR_WRAP (scalar, y, GB_EVAL3 (prefix, _, T)) ;                   \
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, scalar, desc, Werk)) ;\
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
// GrB_Matrix_apply_BinaryOp2nd_UDT: apply a binary operator: op(A,y)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_BinaryOp2nd_UDT
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_BinaryOp2nd_UDT"
        " (C, M, accum, op, A, y, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, y, (op == NULL) ? NULL : op->ytype) ;
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, scalar, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_IndexOp_TYPE: apply a binary operator: op(A,i,j,thunk)
//------------------------------------------------------------------------------

#define GB_IDXUNOP(prefix,type,T)                                           \
GrB_Info GB_EVAL3 (prefix, _Matrix_apply_IndexOp_, T)                       \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results */   \
    const GrB_Matrix M,             /* optional mask for C*/                \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(C,T) */   \
    const GrB_IndexUnaryOp op,      /* operator to apply to the entries */  \
    const GrB_Matrix A,             /* first input:  matrix A */            \
    const type thunk,               /* second input: scalar thunk */        \
    const GrB_Descriptor desc       /* descriptor for C, M, and A */        \
)                                                                           \
{                                                                           \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_apply_IndexOp_" GB_STR(T)          \
        " (C, M, accum, op, A, thunk, desc)") ;                             \
    GB_SCALAR_WRAP (scalar, thunk, GB_EVAL3 (prefix, _, T)) ;               \
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, scalar, desc, Werk)) ;\
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
// GrB_Matrix_apply_IndexOp_UDT: apply a binary operator: op(A,i,j,thunk)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_IndexOp_UDT
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *thunk,              // second input: scalar thunk
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_IndexOp_UDT"
        " (C, M, accum, op, A, thunk, desc)") ;
    GB_SCALAR_WRAP_UDT (scalar, thunk, (op == NULL) ? NULL : op->ytype) ;
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, scalar, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_apply_IndexOp_Scalar: apply a binary operator: op(A,i,j,thunk)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_apply_IndexOp_Scalar    // C<M>=accum(C,op(A,i,j,thunk))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to apply to the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar thunk,         // second input: scalar thunk
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_apply_IndexOp_Scalar"
        " (C, M, accum, op, A, thunk, desc)") ;
    return (GB_2nd (C, M, accum, (GB_Operator) op, A, thunk, desc, Werk)) ;
}

