//------------------------------------------------------------------------------
// gbmex_get: get a property of any matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The matrix can be any matrix: GhB, GrB, or builtin

#include "gb_interface.h"
#include "gbmx_interface.h"
#include "GB_memory.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&C_to_free) ;

#define USAGE "usage: value = gbmex_get (C, state)"

//------------------------------------------------------------------------------
// safe_strlen:  compute the length of a string
//------------------------------------------------------------------------------

// This is the same as the POSIX strnlen, but just using my own method.
// Non-POSIX systems do not have strnlen.

// This method returns the # of bytes in the string s, excluding the null
// terminating byte.  If s has no null terminating byte, it returns maxlen.

size_t safe_strlen (const char *s, size_t maxlen) ;

size_t safe_strlen (const char *s, size_t maxlen)
{
    if (s == NULL) return (0) ;
    for (size_t k = 0 ; k < maxlen ; k++)
    {
        if (s [k] == '\0') return (k) ;
    }
    return (maxlen) ;
}

//------------------------------------------------------------------------------
// gbmex_get mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for all GhB matrices

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    char state [LEN+2], str [LEN+2] ;
    int value = 0 ;
    bool iso = false ;
    gbmx_mxstring_to_string (state, LEN, pargin [1], "state") ;
    bool value_is_string = false, value_is_bool = false ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&C, &C_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // get the state
    //--------------------------------------------------------------------------

    if (MATCH (state, "format"))
    { 

        //----------------------------------------------------------------------
        // get the format and sparsity control
        //----------------------------------------------------------------------

        int sparsity, fmt ;
        OK (GrB_Matrix_get_INT32 (C, &sparsity, GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Matrix_get_INT32 (C, &fmt, GxB_FORMAT)) ;
        value_is_string = true ;

        switch (sparsity)
        {
            case GxB_HYPERSPARSE :                              // 1
                GB_string_copy (str, "hypersparse by ", LEN) ;
                break ;
            case GxB_SPARSE :                                   // 2
                GB_string_copy (str, "sparse by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_SPARSE :                 // 3
                GB_string_copy (str, "sparse/hypersparse by ", LEN) ;
                break ;
            case GxB_BITMAP :                                   // 4
                GB_string_copy (str, "bitmap by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_BITMAP :                 // 5
                GB_string_copy (str, "hypersparse/bitmap by ", LEN) ;
                break ;
            case GxB_SPARSE + GxB_BITMAP :                      // 6
                GB_string_copy (str, "sparse/bitmap by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP :    // 7
                GB_string_copy (str, "sparse/hypersparse/bitmap by ", LEN) ;
                break ;
            case GxB_FULL :                                     // 8
                GB_string_copy (str, "full by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_FULL :                   // 9
                GB_string_copy (str, "hypersparse/full by ", LEN) ;
                break ;
            case GxB_SPARSE + GxB_FULL :                        // 10
                GB_string_copy (str, "sparse/full by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL :      // 11
                GB_string_copy (str, "sparse/hypersparse/full by ", LEN) ;
                break ;
            case GxB_BITMAP + GxB_FULL :                        // 12
                GB_string_copy (str, "bitmap/full by ", LEN) ;
                break ;
            case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL :      // 13
                GB_string_copy (str, "hypersparse/bitmap/full by ", LEN) ;
                break ;
            case GxB_SPARSE + GxB_BITMAP + GxB_FULL :           // 14
                GB_string_copy (str, "sparse/bitmap/full by ", LEN) ;
                break ;
            default :
            case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP + GxB_FULL : // 15
                GB_string_copy (str, "sparse/hypersparse/bitmap/full by ",
                    LEN) ;
                break ;
        }

        // append the format ('by row' or 'by col')
        int len = safe_strlen (str, LEN) ;
        char *str2 = str + len ;
        len = LEN - len ;
        switch (fmt)
        {
            case GxB_BY_ROW    : GB_string_copy (str2, "row", len) ; break ;
            case GxB_BY_COL    : GB_string_copy (str2, "col", len) ; break ;
            case GxB_NO_FORMAT :
            default            : GB_string_copy (str2, "default", len) ; break ;
        }

    }
    else if (MATCH (state, "iso"))
    { 
        // get the iso state
        OK (GrB_Matrix_get_INT32 (C, &iso, GxB_ISO)) ;
        value_is_bool = true ;
    }
    else if (MATCH (state, "offset"))
    { 
        // get the integer sizes for offsets
        OK (GrB_Matrix_get_INT32 (C, &value, GxB_OFFSET_INTEGER_BITS)) ;
    }
    else if (MATCH (state, "column") || MATCH (state, "col"))
    { 
        // get the integer sizes for column indices
        OK (GrB_Matrix_get_INT32 (C, &value, GxB_COLINDEX_INTEGER_BITS)) ;
    }
    else if (MATCH (state, "row"))
    { 
        // get the integer sizes for row indices
        OK (GrB_Matrix_get_INT32 (C, &value, GxB_ROWINDEX_INTEGER_BITS)) ;
    }
    else
    { 
        ERROR ("invalid state (must be 'format', 'iso', 'offset', "
            "'row', or 'col')", GrB_INVALID_VALUE) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;

    if (value_is_string)
    { 
        pargout [0] = mxCreateString (str) ;
    }
    else if (value_is_bool)
    { 
        pargout [0] = mxCreateLogicalScalar (iso) ;
    }
    else
    { 
        pargout [0] = mxCreateDoubleScalar ((double) value) ;
    }
    gb_wrapup ( ) ;
}

