//------------------------------------------------------------------------------
// gb_error_string: return a string from a GraphBLAS GrB_info
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

const char *gb_error_string // return an error message from a GrB_Info value
(
    GrB_Info info
)
{
    switch (info)
    {

        case GrB_SUCCESS :              return ("success") ;

        //----------------------------------------------------------------------
        // informational codes, not an error:
        //----------------------------------------------------------------------

        case GrB_NO_VALUE :             return ("no entry present") ; 
        case GxB_EXHAUSTED :            return ("iterator is exhausted") ;

        //----------------------------------------------------------------------
        // errors:
        //----------------------------------------------------------------------

        case GrB_UNINITIALIZED_OBJECT : return ("uninitialized object") ;
        case GrB_NULL_POINTER :         return ("input pointer is NULL") ;
        case GrB_INVALID_VALUE :        return ("invalid value") ;
        case GrB_INVALID_INDEX :        return ("row or column index out of bounds") ;
        case GrB_DOMAIN_MISMATCH :      return ("object domains are not compatible") ;
        case GrB_DIMENSION_MISMATCH :   return ("matrix dimensions are invalid") ;
        case GrB_OUTPUT_NOT_EMPTY :     return ("output matrix already has values") ;
        case GrB_NOT_IMPLEMENTED :      return ("method not implemented") ;
        case GrB_ALREADY_SET :          return ("name already set") ;
        case GrB_OUT_OF_MEMORY :        return ("out of memory") ;
        case GrB_INSUFFICIENT_SPACE :   return ("output array not large enough") ;
        case GrB_INVALID_OBJECT :       return ("object is corrupted") ;
        case GrB_INDEX_OUT_OF_BOUNDS :  return ("row or column index out of bounds") ;
        case GrB_EMPTY_OBJECT :         return ("an object does not contain a value") ;
        case GxB_JIT_ERROR :            return ("jit error") ;
//      case GxB_GPU_ERROR:             return ("GPU error") ; // in progress
        case GxB_OUTPUT_IS_READONLY :   return ("output is readonly") ;
        default :
        case GrB_PANIC :                break ;
    }

    return ("unknown error") ;
}

