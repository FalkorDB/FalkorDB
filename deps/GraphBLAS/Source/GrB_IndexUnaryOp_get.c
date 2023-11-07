//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_*: get a field in a idxunop
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_Scalar
(
    GrB_IndexUnaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_IndexUnaryOp_get_Scalar (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_String
(
    GrB_IndexUnaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_IndexUnaryOp_get_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_string_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_INT32
(
    GrB_IndexUnaryOp op,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_IndexUnaryOp_get_INT32 (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_SIZE
(
    GrB_IndexUnaryOp op,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_IndexUnaryOp_get_SIZE (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_size_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_VOID
(
    GrB_IndexUnaryOp op,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

