//------------------------------------------------------------------------------
// GrB_BinaryOp_get_*: get a field in a binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_Scalar
(
    GrB_BinaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_BinaryOp_get_Scalar (op, value, field)") ;
    if (op != GxB_IGNORE_DUP) 
    { 
        GB_RETURN_IF_NULL_OR_FAULTY (op) ;
        ASSERT_BINARYOP_OK (op, "binaryop for get", GB0) ;
    }
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_BinaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_BinaryOp_get_String
(
    GrB_BinaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_BinaryOp_get_String (op, value, field)") ;
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
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_BinaryOp_get_INT32 (op, value, field)") ;
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
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_BinaryOp_get_SIZE (op, value, field)") ;
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
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

