//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_*: get a field in a index binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_get_Scalar
(
    GxB_IndexBinaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexBinaryOp_get_Scalar (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) op, value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_String
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_get_String
(
    GxB_IndexBinaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexBinaryOp_get_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_string_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_INT32
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_get_INT32
(
    GxB_IndexBinaryOp op,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexBinaryOp_get_INT32 (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_get_SIZE
(
    GxB_IndexBinaryOp op,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_IndexBinaryOp_get_SIZE (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_size_get ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_get_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_get_VOID
(
    GxB_IndexBinaryOp op,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

