//------------------------------------------------------------------------------
// gb_first_binop: return the GrB_FIRST operator for a given type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_first_binop     // construct GrB_FIRST_[type] operator
(
    // output
    GrB_BinaryOp *op,       // return GrB_FIRST_[type] operator
    // input
    const GrB_Type type,
    char err [ERRLEN]
)
{ 

    if      (type == GrB_BOOL)   (*op) = GrB_FIRST_BOOL ;
    else if (type == GrB_INT8)   (*op) = GrB_FIRST_INT8 ;
    else if (type == GrB_INT16)  (*op) = GrB_FIRST_INT16 ;
    else if (type == GrB_INT32)  (*op) = GrB_FIRST_INT32 ;
    else if (type == GrB_INT64)  (*op) = GrB_FIRST_INT64 ;
    else if (type == GrB_UINT8)  (*op) = GrB_FIRST_UINT8 ;
    else if (type == GrB_UINT16) (*op) = GrB_FIRST_UINT16 ;
    else if (type == GrB_UINT32) (*op) = GrB_FIRST_UINT32 ;
    else if (type == GrB_UINT64) (*op) = GrB_FIRST_UINT64 ;
    else if (type == GrB_FP32)   (*op) = GrB_FIRST_FP32 ;
    else if (type == GrB_FP64)   (*op) = GrB_FIRST_FP64 ;
    else if (type == GxB_FC32)   (*op) = GxB_FIRST_FC32 ;
    else if (type == GxB_FC64)   (*op) = GxB_FIRST_FC64 ;
    else
    {
        (*op) = NULL ;
        ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
    }

    return (GrB_SUCCESS) ;
}

