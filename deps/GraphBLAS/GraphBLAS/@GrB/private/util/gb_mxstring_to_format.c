//------------------------------------------------------------------------------
// gb_mxstring_to_format: get the format from a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Valid format strings:

//  'by row'            auto sparsity for these 2 strings
//  'by col'

//  'sparse by row'
//  'hypersparse by row'
//  'bitmap by row'
//  'full by row'

//  'sparse by col'
//  'hypersparse by col'
//  'bitmap by col'
//  'full by col'

//  'sparse'            fmt is GxB_BY_COL for these four strings
//  'hypersparse'
//  'bitmap'
//  'full'

// The sparsity formats can be combined as well, such as:
//  'sparse/hyper by row'

// hypersparse can be abbreviated as 'hyper'

#include "gb_interface.h"

bool gb_mxstring_to_format      // true if a valid format is found
(
    // input
    const mxArray *mxformat,    // built-in string, 'by row' or 'by col'
    // output
    GxB_Format_Value *fmt,
    int *sparsity
)
{

    bool valid = false ;
    (*fmt) = GxB_BY_COL ;
    (*sparsity) = 0 ;
    #define LEN 256
    char format_string [LEN+2] ;
    gb_mxstring_to_string (format_string, LEN, mxformat, "format") ;

    //--------------------------------------------------------------------------
    // look for trailing "by row" or "by col", and set format if found
    //--------------------------------------------------------------------------

    int len = strlen (format_string) ;
    if (len >= 6)
    {
        if (MATCH (format_string + len - 6, "by row"))
        { 
            valid = true ;
            (*fmt) = GxB_BY_ROW ;
            len = len - 6 ;
            format_string [GB_IMAX (0, len-1)] = '\0' ;
        }
        else if (MATCH (format_string + len - 6, "by col"))
        { 
            valid = true ;
            (*fmt) = GxB_BY_COL ;
            len = len - 6 ;
            format_string [GB_IMAX (0, len-1)] = '\0' ;
        }
    }

    //--------------------------------------------------------------------------
    // parse the format for hypersparse/sparse/bitmap/full sparsity tokens
    //--------------------------------------------------------------------------

    int s = 0 ;
    int kstart = 0 ;
    for (int k = 0 ; k <= len ; k++)
    {
        if (format_string [k] == '/' || format_string [k] == '\0')
        {
            // mark the end of prior token
            format_string [k] = '\0' ;

            // null-terminated token is contained in format_string [kstart:k]
            if (MATCH (format_string + kstart, "sparse"))
            { 
                s += GxB_SPARSE ;
            }
            else if (MATCH (format_string + kstart, "hypersparse") ||
                     MATCH (format_string + kstart, "hyper"))
            { 
                s += GxB_HYPERSPARSE ;
            }
            else if (MATCH (format_string + kstart, "bitmap"))
            { 
                s += GxB_BITMAP ;
            }
            else if (MATCH (format_string + kstart, "full"))
            { 
                s += GxB_FULL ;
            }

            // advance to the next token
            kstart = k+1 ;
        }
    }

    if (s > 0)
    {
        valid = true ;
        (*sparsity) = s ;
    }

    return (valid) ;
}

