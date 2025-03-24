//------------------------------------------------------------------------------
// GrB_Vector_eWiseAdd: vector element-wise operations, set union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w<M> = accum (w,u+v)

#include "ewise/GB_ewise.h"
#include "mask/GB_get_mask.h"

//------------------------------------------------------------------------------
// GrB_Vector_eWiseAdd_BinaryOp: vector addition
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_eWiseAdd_BinaryOp       // w<M> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector Mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp op,          // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE4 (w, Mask, u, v, "GrB_Vector_eWiseAdd (w, M, accum, op, u, v, "
        "desc)") ;
    GB_BURBLE_START ("GrB_Vector_eWiseAdd") ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (GB_VECTOR_OK (u)) ;
    ASSERT (GB_VECTOR_OK (v)) ;
    ASSERT (Mask == NULL || GB_VECTOR_OK (Mask)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // w<M> = accum (w,t) where t = u+v, u'+v, u+v', or u'+v'
    //--------------------------------------------------------------------------

    info = GB_ewise (
        (GrB_Matrix) w, C_replace,  // w and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // accumulate operator
        op,                         // operator that defines '+'
        (GrB_Matrix) u, false,      // u, never transposed
        (GrB_Matrix) v, false,      // v, never transposed
        true,                       // eWiseAdd
        false, NULL, NULL,          // not eWiseUnion
        Werk) ;
    GB_BURBLE_END ;

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_eWiseAdd_Monoid: vector addition
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_eWiseAdd_Monoid         // w<M> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Monoid monoid,        // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GrB_BinaryOp op = monoid->op ;
    return (GrB_Vector_eWiseAdd_BinaryOp (w, M, accum, op, u, v, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_eWiseAdd_Semiring: vector addition
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_eWiseAdd_Semiring       // w<M> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M,             // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector v,             // second input: vector v
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GrB_BinaryOp op = semiring->add->op ;
    return (GrB_Vector_eWiseAdd_BinaryOp (w, M, accum, op, u, v, desc)) ;
}

