//------------------------------------------------------------------------------
// gbversion: string with SuiteSparse:GraphBLAS version
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// v = gbversion

#include "gb_interface.h"

#define USAGE "usage: v = gbversion"

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

    gb_usage (nargin == 0 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the version and date information and return it as a built-in string
    //--------------------------------------------------------------------------

    int version [3] ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION, version)) ;

    char *date = NULL ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_DATE, &date)) ;

    #define LEN 256
    char s [LEN+1] ;
    snprintf (s, LEN, "%d.%d.%d (%s)",
        version [0], version [1], version [2], date) ;

    pargout [0] = mxCreateString (s) ;
    GB_WRAPUP ;
}

