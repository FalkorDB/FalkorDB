//------------------------------------------------------------------------------
// gbmex_ver: struct with SuiteSparse:GraphBLAS version
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// v = gbmex_ver

// Calls to GrB_* and mx* methods are intermingled since none of the GrB
// methods allocate any memory.

#include "gb_interface.h"
#include "gbmx_interface.h"

#define USAGE "usage: v = gbmex_ver"

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

    GBMX_USAGE (nargin == 0 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the version and date information and return it as a struct
    //--------------------------------------------------------------------------

    // get the GraphBLAS version
    int major, minor, patch ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &major, GrB_LIBRARY_VER_MAJOR)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &minor, GrB_LIBRARY_VER_MINOR)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &patch, GrB_LIBRARY_VER_PATCH)) ;

    // get the date
    size_t len = 0 ;
    OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_LIBRARY_DATE)) ;
    char *date = mxMalloc (len+1) ;
    OK (GrB_Global_get_String (GrB_GLOBAL, date, GxB_LIBRARY_DATE)) ;

    if (nargout == 0)
    { 
        printf ("----------------------------------------"
                "-----------------------------------\n") ;

        // about:
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_LIBRARY_ABOUT)) ;
        char *about = mxMalloc (len+1) ;
        OK (GrB_Global_get_String (GrB_GLOBAL, about, GxB_LIBRARY_ABOUT)) ;
        printf ("%s\n", about) ;
        gbmx_free ((void **) &about) ;

        // version and date:
        printf ("Version: %d.%d.%d (%s)\n", major, minor, patch, date) ;

        // compiler:
        int32_t cver [3] ;
        int32_t have_openmp ;
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_COMPILER_NAME)) ;
        char *compiler = mxMalloc (len+1) ;
        OK (GrB_Global_get_String (GrB_GLOBAL, compiler, GxB_COMPILER_NAME)) ;
        OK (GrB_Global_get_VOID (GrB_GLOBAL, (void *) cver,
            GxB_COMPILER_VERSION)) ;
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &have_openmp,
            GxB_LIBRARY_OPENMP)) ;
        printf ("GraphBLAS compiled with %s (v%d.%d.%d), %s OpenMP\n", compiler,
            cver [0], cver [1], cver [2],
            have_openmp ? "with" : "without") ;
        gbmx_free ((void **) &compiler) ;

        // license:
        printf ("GraphBLAS License: Apache-2.0\n\n") ;

        // spec:
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_API_ABOUT)) ;
        char *spec = mxMalloc (len+1) ;
        OK (GrB_Global_get_String (GrB_GLOBAL, spec, GxB_API_ABOUT)) ;
        printf ("Spec:\n%s\n", spec) ;
        gbmx_free ((void **) &spec) ;

        // url:
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &len, GxB_API_URL)) ;
        char *url = mxMalloc (len+1) ;
        OK (GrB_Global_get_String (GrB_GLOBAL, url, GxB_API_URL)) ;
        printf ("URL: %s\n", url) ;
        gbmx_free ((void **) &url) ;

        printf ("----------------------------------------"
                "-----------------------------------\n") ;
    }
    else
    { 
        char s [LEN+2] ;
        snprintf (s, LEN, "%d.%d.%d", major, minor, patch) ;
        pargout [0] = mxCreateStructMatrix (1, 1, 3, vfields) ;
        mxSetFieldByNumber (pargout [0], 0, 0,
                mxCreateString ("SuiteSparse:GraphBLAS")) ;
        mxSetFieldByNumber (pargout [0], 0, 1, mxCreateString (s)) ;
        mxSetFieldByNumber (pargout [0], 0, 2, mxCreateString (date)) ;
    }

    gbmx_free ((void **) &date) ;
    gb_wrapup ( ) ;
}

