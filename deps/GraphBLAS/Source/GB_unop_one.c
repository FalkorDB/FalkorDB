//------------------------------------------------------------------------------
// GB_unop_one: return the ONE operator for a given type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_unop.h"

GrB_UnaryOp GB_unop_one (GB_Type_code xcode)
{
    switch (xcode)
    {
        case GB_BOOL_code    : return (GxB_ONE_BOOL  ) ;
        case GB_INT8_code    : return (GxB_ONE_INT8  ) ;
        case GB_INT16_code   : return (GxB_ONE_INT16 ) ;
        case GB_INT32_code   : return (GxB_ONE_INT32 ) ;
        case GB_INT64_code   : return (GxB_ONE_INT64 ) ;
        case GB_UINT8_code   : return (GxB_ONE_UINT8 ) ;
        case GB_UINT16_code  : return (GxB_ONE_UINT16) ;
        case GB_UINT32_code  : return (GxB_ONE_UINT32) ;
        case GB_UINT64_code  : return (GxB_ONE_UINT64) ;
        case GB_FP32_code    : return (GxB_ONE_FP32  ) ;
        case GB_FP64_code    : return (GxB_ONE_FP64  ) ;
        case GB_FC32_code    : return (GxB_ONE_FC32  ) ;
        case GB_FC64_code    : return (GxB_ONE_FC64  ) ;
        default              : return (NULL) ;
    }
}

