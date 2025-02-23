//------------------------------------------------------------------------------
// GrB_BinaryOp_get_*: get a field in a binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_Scalar
(
    GrB_BinaryOp op,
    GrB_Scalar scalar,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (scalar) ;
    GB_WHERE_1 (scalar, "GrB_BinaryOp_get_Scalar (op, scalar, field)") ;

    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for get", GB0) ;
    }

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, scalar, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_String
(
    GrB_BinaryOp op,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for get", GB0) ;
    }
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_string_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_INT32
(
    GrB_BinaryOp op,
    int32_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for get", GB0) ;
    }
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_SIZE
(
    GrB_BinaryOp op,
    size_t * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for get", GB0) ;
    }
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_size_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_VOID
(
    GrB_BinaryOp op,
    void * value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

