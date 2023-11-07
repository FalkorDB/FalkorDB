//------------------------------------------------------------------------------
// gb_type_to_mxstring: create a built-in string from a GraphBLAS type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

mxArray * gb_type_to_mxstring    // return the built-in string from a GrB_Type
(
    const GrB_Type type
)
{ 

    if      (type == GrB_BOOL)   return (mxCreateString ("logical")) ;
    else if (type == GrB_INT8)   return (mxCreateString ("int8")) ;
    else if (type == GrB_INT16)  return (mxCreateString ("int16")) ;
    else if (type == GrB_INT32)  return (mxCreateString ("int32")) ;
    else if (type == GrB_INT64)  return (mxCreateString ("int64")) ;
    else if (type == GrB_UINT8)  return (mxCreateString ("uint8")) ;
    else if (type == GrB_UINT16) return (mxCreateString ("uint16")) ;
    else if (type == GrB_UINT32) return (mxCreateString ("uint32")) ;
    else if (type == GrB_UINT64) return (mxCreateString ("uint64")) ;
    else if (type == GrB_FP32)   return (mxCreateString ("single")) ;
    else if (type == GrB_FP64)   return (mxCreateString ("double")) ;
    else if (type == GxB_FC32)   return (mxCreateString ("single complex")) ;
    else if (type == GxB_FC64)   return (mxCreateString ("double complex")) ;
    else
    {
        ERROR ("unsupported type") ;
    }

    return (NULL) ;
}

