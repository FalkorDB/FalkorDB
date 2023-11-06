//------------------------------------------------------------------------------
// GrB_UnaryOp_set_*: set a field in a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_UnaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_set_Scalar
(
    GrB_UnaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_set_String
(
    GrB_UnaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_UnaryOp_set_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_UNARYOP_OK (op, "unaryop for set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_set_INT32
(
    GrB_UnaryOp op,
    int32_t value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_set_VOID
(
    GrB_UnaryOp op,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

