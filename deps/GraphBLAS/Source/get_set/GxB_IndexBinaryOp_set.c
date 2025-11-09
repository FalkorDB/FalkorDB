//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_set_*: set a field in a index binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_set_Scalar
(
    GxB_IndexBinaryOp op,
    GrB_Scalar scalar,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_set_String
(
    GxB_IndexBinaryOp op,
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
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_set_INT32
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_set_INT32
(
    GxB_IndexBinaryOp op,
    int32_t value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GxB_IndexBinaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_IndexBinaryOp_set_VOID
(
    GxB_IndexBinaryOp op,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

