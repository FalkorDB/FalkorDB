//------------------------------------------------------------------------------
// gb_string_to_type: return the GraphBLAS type from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Type gb_string_to_type      // return the GrB_Type from a string
(
    const char *typename
)
{ 

    if (MATCH (typename, "logical" )) return (GrB_BOOL) ;
    if (MATCH (typename, "int8"    )) return (GrB_INT8) ;
    if (MATCH (typename, "int16"   )) return (GrB_INT16) ;
    if (MATCH (typename, "int32"   )) return (GrB_INT32) ;
    if (MATCH (typename, "int64"   )) return (GrB_INT64) ;
    if (MATCH (typename, "uint8"   )) return (GrB_UINT8) ;
    if (MATCH (typename, "uint16"  )) return (GrB_UINT16) ;
    if (MATCH (typename, "uint32"  )) return (GrB_UINT32) ;
    if (MATCH (typename, "uint64"  )) return (GrB_UINT64) ;
    if (MATCH (typename, "single"  )) return (GrB_FP32) ;
    if (MATCH (typename, "double"  )) return (GrB_FP64) ;

    if (MATCH (typename, "single complex") ||
        MATCH (typename, "float complex"))
    { 
        return (GxB_FC32) ;
    }

    if (MATCH (typename, "double complex") ||
        MATCH (typename, "complex"))
    { 
        return (GxB_FC64) ;
    }

    // The string is not a type, but this is not an error here.  For example,
    // G = GrB (m,n,'double','by row') queries both its string input arguments
    // with gb_mxstring_to_type and gb_mxstring_to_format, to parse its inputs.
    return (NULL) ;
}

