//------------------------------------------------------------------------------
// GB_code_name_get: get the user_name of a type, from a code
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

const char *GB_code_name_get (GB_Type_code code, char *user_name)
{
    switch (code)
    {
        case GB_BOOL_code   : return ("GrB_BOOL")   ;
        case GB_INT8_code   : return ("GrB_INT8")   ;
        case GB_INT16_code  : return ("GrB_INT16")  ;
        case GB_INT32_code  : return ("GrB_INT32")  ;
        case GB_INT64_code  : return ("GrB_INT64")  ;
        case GB_UINT8_code  : return ("GrB_UINT8")  ;
        case GB_UINT16_code : return ("GrB_UINT16") ;
        case GB_UINT32_code : return ("GrB_UINT32") ;
        case GB_UINT64_code : return ("GrB_UINT64") ;
        case GB_FP32_code   : return ("GrB_FP32")   ;
        case GB_FP64_code   : return ("GrB_FP64")   ;
        case GB_FC32_code   : return ("GxB_FC32")   ;
        case GB_FC64_code   : return ("GxB_FC64")   ;
        default:
        case GB_UDT_code    : return (user_name) ;
    }
}

