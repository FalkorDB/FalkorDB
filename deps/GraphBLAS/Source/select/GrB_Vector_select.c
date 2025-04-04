//------------------------------------------------------------------------------
// GrB_Vector_select: select entries from a vector using a GrB_IndexUnaryOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "select/GB_select.h"
#include "mask/GB_get_mask.h"
#include "scalar/GB_Scalar_wrap.h"

//------------------------------------------------------------------------------
// GB_sel: select using a GrB_IndexUnaryOp
//------------------------------------------------------------------------------

static inline GrB_Info GB_sel   // w<M> = accum (w, select(w,k))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc,      // descriptor for w and M
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_BURBLE_START ("GrB_select") ;

    // get the descriptor
    GrB_Info info ;
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx0, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // select the entries and optionally transpose; assemble pending tuples
    //--------------------------------------------------------------------------

    info = GB_select (
        (GrB_Matrix) w, C_replace,  // w and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(w,T)
        op,                         // operator to select the entries
        (GrB_Matrix) u,             // first input: u
        y,                          // second input: scalar y
        false,                      // vector u is never transposed
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_select_TYPE: select entries from a vector (built-in types)
//------------------------------------------------------------------------------

#define GB_SEL(prefix,type,T)                                               \
GrB_Info GB_EVAL3 (prefix, _Vector_select_, T)                              \
(                                                                           \
    GrB_Vector w,                   /* input/output vector for results */   \
    const GrB_Vector M,             /* optional mask for w, or NULL */      \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(w,T) */   \
    const GrB_IndexUnaryOp op,      /* operator to select the entries */    \
    const GrB_Vector u,             /* first input:  vector u */            \
    const type y,                   /* second input: scalary y */           \
    const GrB_Descriptor desc       /* descriptor for w and M */            \
)                                                                           \
{                                                                           \
    GB_WHERE3 (w, M, u, GB_STR(prefix) "_Vector_select_" GB_STR(T)          \
        " (w, M, accum, op, u, y, desc)") ;                                 \
    GB_SCALAR_WRAP (yscalar, y, GB_EVAL3 (prefix, _, T)) ;                  \
    return (GB_sel (w, M, accum, op, u, yscalar, desc, Werk)) ;             \
}

GB_SEL (GrB, bool      , BOOL  )
GB_SEL (GrB, int8_t    , INT8  )
GB_SEL (GrB, int16_t   , INT16 )
GB_SEL (GrB, int32_t   , INT32 )
GB_SEL (GrB, int64_t   , INT64 )
GB_SEL (GrB, uint8_t   , UINT8 )
GB_SEL (GrB, uint16_t  , UINT16)
GB_SEL (GrB, uint32_t  , UINT32)
GB_SEL (GrB, uint64_t  , UINT64)
GB_SEL (GrB, float     , FP32  )
GB_SEL (GrB, double    , FP64  )
GB_SEL (GxB, GxB_FC32_t, FC32  )
GB_SEL (GxB, GxB_FC64_t, FC64  )

//------------------------------------------------------------------------------
// GrB_Vector_select_UDT: select entries from vector (y: user-defined type)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_select_UDT
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, or NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Vector u,             // first input:  vector u
    const void *y,                  // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE3 (w, M, u,
        "GrB_Vector_select_UDT (w, M, accum, op, u, y, desc)") ;
    GB_SCALAR_WRAP_UDT (yscalar, y, (op == NULL) ? NULL : op->ytype) ;
    return (GB_sel (w, M, accum, op, u, yscalar, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_select_Scalar: select entries from a vector (y is GrB_Scalar)
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_select_Scalar
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, or NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar y,             // second input: scalar y
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_WHERE4 (w, M, u, y,
        "GrB_Vector_select_Scalar (w, M, accum, op, u, y, desc)") ;
    return (GB_sel (w, M, accum, op, u, y, desc, Werk)) ;
}

