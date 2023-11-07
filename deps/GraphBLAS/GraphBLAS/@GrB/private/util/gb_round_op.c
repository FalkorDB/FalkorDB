//------------------------------------------------------------------------------
// gb_round_op: get a rounding operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_UnaryOp gb_round_op (GrB_Type type)
{ 
    if (type == GrB_FP32) return (GxB_ROUND_FP32) ;
    if (type == GrB_FP64) return (GxB_ROUND_FP64) ;
    if (type == GxB_FC32) return (GxB_ROUND_FC32) ;
    if (type == GxB_FC64) return (GxB_ROUND_FC64) ;
    return (NULL) ;
}

