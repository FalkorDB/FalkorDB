//------------------------------------------------------------------------------
// GB_code_size: given a type code, return sizeof (type)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The user-defined type has no known size, so this must be provided on input.

#include "GB.h"

size_t GB_code_size             // return the size of a type, given its code
(
    const GB_Type_code code,    // input code of the type to find the size of
    const size_t usize          // known size of user-defined type
)
{

    switch (code)
    {
        case GB_BOOL_code   : return (sizeof (bool))     ;
        case GB_INT8_code   : return (sizeof (int8_t))   ;
        case GB_UINT8_code  : return (sizeof (uint8_t))  ;
        case GB_INT16_code  : return (sizeof (int16_t))  ;
        case GB_UINT16_code : return (sizeof (uint16_t)) ;
        case GB_INT32_code  : return (sizeof (int32_t))  ;
        case GB_UINT32_code : return (sizeof (uint32_t)) ;
        case GB_INT64_code  : return (sizeof (int64_t))  ;
        case GB_UINT64_code : return (sizeof (uint64_t)) ;
        case GB_FP32_code   : return (sizeof (float))    ;
        case GB_FP64_code   : return (sizeof (double))   ;
        case GB_FC32_code   : return (sizeof (GxB_FC32_t)) ;
        case GB_FC64_code   : return (sizeof (GxB_FC64_t)) ;
        case GB_UDT_code    : return (usize) ;
        default             : return (0) ;
    }
}

