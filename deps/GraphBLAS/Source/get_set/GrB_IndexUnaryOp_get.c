//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_*: get a field in a idxunop
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_Scalar
(
    GrB_IndexUnaryOp op,
    GrB_Scalar scalar,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_1 (scalar, "GrB_IndexUnaryOp_get_Scalar (op, scalar, field)") ;

    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, scalar, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_get_String
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
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
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
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
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
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

