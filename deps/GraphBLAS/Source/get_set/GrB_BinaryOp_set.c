//------------------------------------------------------------------------------
// GrB_BinaryOp_set_*: set a field in a binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_BinaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_set_Scalar
(
    GrB_BinaryOp op,
    GrB_Scalar scalar,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_set_String
(
    GrB_BinaryOp op,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT
    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for set", GB0) ;
    }
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_set_INT32
(
    GrB_BinaryOp op,
    int32_t value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_set_VOID
(
    GrB_BinaryOp op,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

