//------------------------------------------------------------------------------
// gb_code_to_type: get the GrB_Type from the GrB_Type_Code
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns NULL if the type is user-defined.

#include "gb_interface.h"

GrB_Type gb_code_to_type
(
    GrB_Type_Code code
)
{ 
    switch (code)
    {
        case GrB_BOOL_CODE   : return (GrB_BOOL) ;
        case GrB_INT8_CODE   : return (GrB_INT8) ;
        case GrB_INT16_CODE  : return (GrB_INT16) ;
        case GrB_INT32_CODE  : return (GrB_INT32) ;
        case GrB_INT64_CODE  : return (GrB_INT64) ;
        case GrB_UINT8_CODE  : return (GrB_UINT8) ;
        case GrB_UINT16_CODE : return (GrB_UINT16) ;
        case GrB_UINT32_CODE : return (GrB_UINT32) ;
        case GrB_UINT64_CODE : return (GrB_UINT64) ;
        case GrB_FP32_CODE   : return (GrB_FP32) ;
        case GrB_FP64_CODE   : return (GrB_FP64) ;
        case GxB_FC32_CODE   : return (GxB_FC32) ;
        case GxB_FC64_CODE   : return (GxB_FC64) ;
        default              : return (NULL) ;
    }
}

