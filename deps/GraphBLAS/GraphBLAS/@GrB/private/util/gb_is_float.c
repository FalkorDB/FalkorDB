//------------------------------------------------------------------------------
// gb_is_float: check if a GrB_Type is floating-point
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

bool gb_is_float (const GrB_Type type)
{ 
    return ((type == GrB_FP32 ) ||
            (type == GrB_FP64 ) ||
            (type == GxB_FC32 ) ||
            (type == GxB_FC64 )) ;
}

