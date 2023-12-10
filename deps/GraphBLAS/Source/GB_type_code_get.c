//------------------------------------------------------------------------------
// GB_type_code_get: convert a GB_Type_code to a GrB_Type_Code
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Type_Code GB_type_code_get  // return the GrB_Type_Code for the code
(
    const GB_Type_code code     // type code to convert
)
{

    ASSERT (code >= 0 && code <= GB_UDT_code) ;
    switch (code)
    {
        case GB_BOOL_code   : return (GrB_BOOL_CODE)   ;
        case GB_INT8_code   : return (GrB_INT8_CODE)   ;
        case GB_INT16_code  : return (GrB_INT16_CODE)  ;
        case GB_INT32_code  : return (GrB_INT32_CODE)  ;
        case GB_INT64_code  : return (GrB_INT64_CODE)  ;
        case GB_UINT8_code  : return (GrB_UINT8_CODE)  ;
        case GB_UINT16_code : return (GrB_UINT16_CODE) ;
        case GB_UINT32_code : return (GrB_UINT32_CODE) ;
        case GB_UINT64_code : return (GrB_UINT64_CODE) ;
        case GB_FP32_code   : return (GrB_FP32_CODE)   ;
        case GB_FP64_code   : return (GrB_FP64_CODE)   ;
        case GB_FC32_code   : return (GxB_FC32_CODE)   ;
        case GB_FC64_code   : return (GxB_FC64_CODE)   ;
        default:
        case GB_UDT_code    : return (GrB_UDT_CODE)    ;
    }
}

