//------------------------------------------------------------------------------
// GB_mx_string_to_Type.c: return the GrB_type from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Also returns the GrB_Type from the string.

#include "GB_mex.h"

GrB_Type GB_mx_string_to_Type       // GrB_Type from the string
(
    const mxArray *type_mx,         // string with type name
    const GrB_Type default_type     // default type if string empty
)
{

    // get the string
    #define LEN 256
    char type [LEN+2] ;
    int len = GB_mx_mxArray_to_string (type, LEN, type_mx) ;

    if (len <  0) mexErrMsgTxt ("unknown class") ;
    if (len == 0) return (default_type) ;
    if (MATCH (type, "logical")) return (GrB_BOOL) ;
    if (MATCH (type, "bool"   )) return (GrB_BOOL) ;
    if (MATCH (type, "int8"   )) return (GrB_INT8) ;
    if (MATCH (type, "int16"  )) return (GrB_INT16) ;
    if (MATCH (type, "int32"  )) return (GrB_INT32) ;
    if (MATCH (type, "int64"  )) return (GrB_INT64) ;
    if (MATCH (type, "uint8"  )) return (GrB_UINT8) ;
    if (MATCH (type, "uint16" )) return (GrB_UINT16) ;
    if (MATCH (type, "uint32" )) return (GrB_UINT32) ;
    if (MATCH (type, "uint64" )) return (GrB_UINT64) ;
    if (MATCH (type, "single" )) return (GrB_FP32) ;
    if (MATCH (type, "double" )) return (GrB_FP64) ;
    if (MATCH (type, "single complex" )) return (GxB_FC32) ;
    if (MATCH (type, "double complex" )) return (Complex) ;
    if (MATCH (type, "GxB_FC32_t" )) return (GxB_FC32) ;
    if (MATCH (type, "GxB_FC64_t" )) return (GxB_FC64) ;

    mexErrMsgTxt ("unknown class") ;
    return (NULL) ;
}

