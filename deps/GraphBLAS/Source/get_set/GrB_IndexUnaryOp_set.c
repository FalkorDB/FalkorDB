//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_*: set a field in a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_Scalar
(
    GrB_IndexUnaryOp op,
    GrB_Scalar scalar,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_String
(
    GrB_IndexUnaryOp op,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_INT32
(
    GrB_IndexUnaryOp op,
    int32_t value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_VOID
(
    GrB_IndexUnaryOp op,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

