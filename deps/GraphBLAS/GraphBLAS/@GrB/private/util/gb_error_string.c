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

        case GrB_SUCCESS :              return ("GraphBLAS: success") ;

        //----------------------------------------------------------------------
        // informational codes, not an error:
        //----------------------------------------------------------------------

        case GrB_NO_VALUE :             return ("GraphBLAS: no entry present") ; 
        case GxB_EXHAUSTED :            return ("GraphBLAS: iterator is exhausted") ;

        //----------------------------------------------------------------------
        // errors:
        //----------------------------------------------------------------------

        case GrB_UNINITIALIZED_OBJECT : return ("GraphBLAS: uninitialized object") ;
        case GrB_NULL_POINTER :         return ("GraphBLAS: input pointer is NULL") ;
        case GrB_INVALID_VALUE :        return ("GraphBLAS: invalid value") ;
        case GrB_INVALID_INDEX :        return ("GraphBLAS: row or column index out of bounds") ;
        case GrB_DOMAIN_MISMATCH :      return ("GraphBLAS: object domains are not compatible") ;
        case GrB_DIMENSION_MISMATCH :   return ("GraphBLAS: matrix dimensions are invalid") ;
        case GrB_OUTPUT_NOT_EMPTY :     return ("GraphBLAS: output matrix already has values") ;
        case GrB_NOT_IMPLEMENTED :      return ("GraphBLAS: method not implemented") ;
        case GrB_ALREADY_SET :          return ("GraphBLAS: name already set") ;
        case GrB_OUT_OF_MEMORY :        return ("GraphBLAS: out of memory") ;
        case GrB_INSUFFICIENT_SPACE :   return ("GraphBLAS: output array not large enough") ;
        case GrB_INVALID_OBJECT :       return ("GraphBLAS: object is corrupted") ;
        case GrB_INDEX_OUT_OF_BOUNDS :  return ("GraphBLAS: row or column index out of bounds") ;
        case GrB_EMPTY_OBJECT :         return ("GraphBLAS: an object does not contain a value") ;
        case GxB_JIT_ERROR :            return ("GraphBLAS: JIT failure.  JIT now disabled; see GrB.jit to re-enable") ;
//      case GxB_GPU_ERROR:             return ("GPU error") ; // in progress
        case GxB_OUTPUT_IS_READONLY :   return ("GraphBLAS: output is readonly") ;
        default :
        case GrB_PANIC :                break ;
    }

    return ("GraphBLAS: unknown error") ;
}

