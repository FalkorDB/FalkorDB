//------------------------------------------------------------------------------
// GB_mx_Type_to_string: return a built-in string for a GrB_Type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Given a GrB_Type, constructs a built-in string with the type name

#include "GB_mex.h"

mxArray *GB_mx_Type_to_mxstring        // returns a built-in string
(
    const GrB_Type type
)
{

    switch (type->code)
    {
        case GB_BOOL_code    : return (mxCreateString ("logical")) ;
        case GB_INT8_code    : return (mxCreateString ("int8")) ;
        case GB_INT16_code   : return (mxCreateString ("int16")) ;
        case GB_INT32_code   : return (mxCreateString ("int32")) ;
        case GB_INT64_code   : return (mxCreateString ("int64")) ;
        case GB_UINT8_code   : return (mxCreateString ("uint8")) ;
        case GB_UINT16_code  : return (mxCreateString ("uint16")) ;
        case GB_UINT32_code  : return (mxCreateString ("uint32")) ;
        case GB_UINT64_code  : return (mxCreateString ("uint64")) ;
        case GB_FP32_code    : return (mxCreateString ("single")) ;
        case GB_FP64_code    : return (mxCreateString ("double")) ;
        case GB_FC32_code    : return (mxCreateString ("single complex")) ;
        case GB_FC64_code    : return (mxCreateString ("double complex")) ;
        case GB_UDT_code     : return (mxCreateString ("double complex")) ;
        default              : return (mxCreateString ("other")) ;
    }
}

