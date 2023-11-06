//------------------------------------------------------------------------------
// GrB_Matrix_select: select entries from a matrix using a GrB_IndexUnaryOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum(C,select(A,k)) or accum(C,select(A',))

#include "GB_select.h"
#include "GB_get_mask.h"
#include "GB_scalar_wrap.h"

//------------------------------------------------------------------------------
// GB_sel: select using a GrB_IndexUnaryOp
//------------------------------------------------------------------------------

static inline GrB_Info GB_sel   // C<M> = accum (C, select(A,k)) or select(A',k)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar Thunk,         // optional input for select operator
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_BURBLE_START ("GrB_select") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // select the entries and optionally transpose; assemble pending tuples
    //--------------------------------------------------------------------------

    info = GB_select (
        C, C_replace,               // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        op,                         // operator to select the entries
        A,                          // first input: A
        Thunk,                      // optional input for select operator
        A_transpose,                // descriptor for A
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_select_TYPE: select entries from a matrix (built-in types)
//------------------------------------------------------------------------------

#define GB_SEL(prefix,type,T)                                               \
GrB_Info GB_EVAL3 (prefix, _Matrix_select_, T)                              \
(                                                                           \
    GrB_Matrix C,                   /* input/output matrix for results */   \
    const GrB_Matrix M,             /* optional mask for C, or NULL */      \
    const GrB_BinaryOp accum,       /* optional accum for Z=accum(C,T) */   \
    const GrB_IndexUnaryOp op,      /* operator to select the entries */    \
    const GrB_Matrix A,             /* first input:  matrix A */            \
    const type thunk,               /* optional input for select operator */\
    const GrB_Descriptor desc       /* descriptor for C, M, and A */        \
)                                                                           \
{                                                                           \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_select_" GB_STR(T)                 \
        " (C, M, accum, op, A, thunk, desc)") ;                             \
    GB_SCALAR_WRAP (Thunk, thunk, GB_EVAL3 (prefix, _, T)) ;                \
    return (GB_sel (C, M, accum, op, A, Thunk, desc, Werk)) ;               \
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
// GrB_Matrix_select_UDT: select entries from matrix (thunk: user-defined type)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_select_UDT
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, or NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Matrix A,             // first input:  matrix A
    const void *thunk,              // optional input for select operator
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_select_UDT (C, M, accum, op, A, thunk, desc)") ;
    GB_SCALAR_WRAP_UDT (Thunk, thunk, (op == NULL) ? NULL : op->ytype) ;
    return (GB_sel (C, M, accum, op, A, Thunk, desc, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_select_Scalar: select entries from a matrix (thunk is GrB_Scalar)
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_select_Scalar
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, or NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op,      // operator to select the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar Thunk,         // optional input for select operator
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 
    GB_WHERE (C, "GrB_Matrix_select_Scalar (C, M, accum, op, A, thunk, desc)") ;
    return (GB_sel (C, M, accum, op, A, Thunk, desc, Werk)) ;
}

