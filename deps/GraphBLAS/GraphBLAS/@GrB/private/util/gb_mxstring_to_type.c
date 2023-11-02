//------------------------------------------------------------------------------
// gb_mxstring_to_type: return the GraphBLAS type from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Type gb_mxstring_to_type    // return the GrB_Type from a built-in string
(
    const mxArray *mxstring
)
{ 

    #define LEN 256
    char s [LEN+2] ;
    gb_mxstring_to_string (s, LEN, mxstring, "type") ;
    return (gb_string_to_type (s)) ;
}

