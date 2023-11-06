//------------------------------------------------------------------------------
// GrB_UnaryOp_get_*: get a field in a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_UnaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_get_Scalar
(
    GrB_UnaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_UnaryOp_get_Scalar (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_UNARYOP_OK (op, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_get_String
(
    GrB_UnaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_UnaryOp_get_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_UNARYOP_OK (op, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_string_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_get_INT32
(
    GrB_UnaryOp op,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_UnaryOp_get_INT32 (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_UNARYOP_OK (op, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_get_SIZE
(
    GrB_UnaryOp op,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_UnaryOp_get_SIZE (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_UNARYOP_OK (op, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_size_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_UnaryOp_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_UnaryOp_get_VOID
(
    GrB_UnaryOp op,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

