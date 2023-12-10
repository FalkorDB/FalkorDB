//------------------------------------------------------------------------------
// GB_code_string: convert a type code into a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Given GB_Type_code, return a string with the name of the type

#include "GB.h"

char *GB_code_string            // return a static string for a type name
(
    const GB_Type_code code     // code to convert to string
)
{

    switch (code)
    {
        case GB_BOOL_code   : return ("bool"          ) ;
        case GB_INT8_code   : return ("int8_t"        ) ;
        case GB_INT16_code  : return ("int16_t"       ) ;
        case GB_INT32_code  : return ("int32_t"       ) ;
        case GB_INT64_code  : return ("int64_t"       ) ;
        case GB_UINT8_code  : return ("uint8_t"       ) ;
        case GB_UINT16_code : return ("uint16_t"      ) ;
        case GB_UINT32_code : return ("uint32_t"      ) ;
        case GB_UINT64_code : return ("uint64_t"      ) ;
        case GB_FP32_code   : return ("float"         ) ;
        case GB_FP64_code   : return ("double"        ) ;
        case GB_FC32_code   : return ("float complex" ) ;
        case GB_FC64_code   : return ("double complex") ;
        case GB_UDT_code    : return ("user-defined"  ) ;
        default             : return ("unknown type!" ) ;
    }
}

