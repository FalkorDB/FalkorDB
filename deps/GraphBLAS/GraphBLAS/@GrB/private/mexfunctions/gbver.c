//------------------------------------------------------------------------------
// gbver: struct with SuiteSparse:GraphBLAS version
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// v = gbver

#include "gb_interface.h"

#define USAGE "usage: v = gbver"

static const char *vfields [3] = { "Name", "Version", "Date" } ;

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
    // get the version and date information and return it as a struct
    //--------------------------------------------------------------------------

    int version [3] ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION, version)) ;

    char *date ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_DATE, &date)) ;

    if (nargout == 0)
    {
        char *license, *about, *spec, *url ;
        printf ("----------------------------------------"
                "-----------------------------------\n") ;
        OK (GxB_Global_Option_get (GxB_LIBRARY_ABOUT, &about)) ;
        printf ("%s\n", about) ;
        printf ("Version: %d.%d.%d (%s)\n",
                version [0], version [1], version [2], date) ;
        char *compiler ;
        int cver [3] ;
        bool have_openmp ;
        OK (GxB_get (GxB_COMPILER_NAME, &compiler)) ;
        OK (GxB_get (GxB_COMPILER_VERSION, cver)) ;
        OK (GxB_get (GxB_LIBRARY_OPENMP, &have_openmp)) ;
        printf ("GraphBLAS compiled with %s (v%d.%d.%d), %s OpenMP\n", compiler,
            cver [0], cver [1], cver [2],
            have_openmp ? "with" : "without") ;
        printf ("@GrB License: Apache-2.0\n\n") ;
        OK (GxB_Global_Option_get (GxB_API_ABOUT, &spec)) ;
        printf ("Spec:\n%s\n", spec) ;
        OK (GxB_Global_Option_get (GxB_API_URL, &url)) ;
        printf ("URL: %s\n", url) ;
        printf ("----------------------------------------"
                "-----------------------------------\n") ;
    }
    else
    {
        #define LEN 256
        char s [LEN+1] ;
        snprintf (s, LEN, "%d.%d.%d", version [0], version [1], version [2]) ;
        pargout [0] = mxCreateStructMatrix (1, 1, 3, vfields) ;
        mxSetFieldByNumber (pargout [0], 0, 0,
                mxCreateString ("SuiteSparse:GraphBLAS")) ;
        mxSetFieldByNumber (pargout [0], 0, 1, mxCreateString (s)) ;
        mxSetFieldByNumber (pargout [0], 0, 2, mxCreateString (date)) ;
    }

    GB_WRAPUP ;
}

