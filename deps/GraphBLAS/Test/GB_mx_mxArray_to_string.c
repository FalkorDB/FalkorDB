//------------------------------------------------------------------------------
// GB_mx_mxArray_to_string.c: get a built-in string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

int GB_mx_mxArray_to_string // returns length of string, or -1 if S not a string
(
    char *string,           // size maxlen
    const size_t maxlen,    // length of string
    const mxArray *S        // built-in mxArray containing a string
)
{

    size_t len = 0 ;
    string [0] = '\0' ;
    if (S != NULL && mxGetNumberOfElements (S) > 0)
    {
        if (!mxIsChar (S))
        {
            mexWarnMsgIdAndTxt ("GB:warn", "argument must be a string") ;
            return (-1) ;
        }
        len = mxGetNumberOfElements (S) ;
        if (len > 0)
        {
            mxGetString (S, string, maxlen) ;
            string [maxlen] = '\0' ;
        }
    }
    return (len) ;
}

