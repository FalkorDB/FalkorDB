//------------------------------------------------------------------------------
// gbversion: string with SuiteSparse:GraphBLAS version
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

    int major, minor, patch ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &major, GrB_LIBRARY_VER_MAJOR)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &minor, GrB_LIBRARY_VER_MINOR)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &patch, GrB_LIBRARY_VER_PATCH)) ;

    // get the date
    size_t len = 0 ;
    OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_LIBRARY_DATE)) ;
    char *date = mxMalloc (len+1) ;
    OK (GrB_Global_get_String (GrB_GLOBAL, date, GxB_LIBRARY_DATE)) ;

    #define LEN 256
    char s [LEN+1] ;
    snprintf (s, LEN, "%d.%d.%d (%s)", major, minor, patch, date) ;
    mxFree (date) ;

    pargout [0] = mxCreateString (s) ;
    gb_wrapup ( ) ;
}

