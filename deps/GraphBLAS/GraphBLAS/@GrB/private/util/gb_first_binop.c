//------------------------------------------------------------------------------
// gb_first_binop: return the GrB_FIRST operator for a given type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_BinaryOp gb_first_binop         // return GrB_FIRST_[type] operator
(
    const GrB_Type type
)
{ 

    if      (type == GrB_BOOL)   return (GrB_FIRST_BOOL) ;
    else if (type == GrB_INT8)   return (GrB_FIRST_INT8) ;
    else if (type == GrB_INT16)  return (GrB_FIRST_INT16) ;
    else if (type == GrB_INT32)  return (GrB_FIRST_INT32) ;
    else if (type == GrB_INT64)  return (GrB_FIRST_INT64) ;
    else if (type == GrB_UINT8)  return (GrB_FIRST_UINT8) ;
    else if (type == GrB_UINT16) return (GrB_FIRST_UINT16) ;
    else if (type == GrB_UINT32) return (GrB_FIRST_UINT32) ;
    else if (type == GrB_UINT64) return (GrB_FIRST_UINT64) ;
    else if (type == GrB_FP32)   return (GrB_FIRST_FP32) ;
    else if (type == GrB_FP64)   return (GrB_FIRST_FP64) ;
    else if (type == GxB_FC32)   return (GxB_FIRST_FC32) ;
    else if (type == GxB_FC64)   return (GxB_FIRST_FC64) ;
    else
    {
        ERROR ("unsupported type") ;
    }

    return (NULL) ;
}

