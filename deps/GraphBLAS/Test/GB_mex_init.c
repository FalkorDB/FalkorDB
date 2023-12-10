//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns the status of all global settings.

#include "GB_mex.h"

#define USAGE "[nthreads format hyper_switch " \
"name version date about license compiledate compiletime api api_about" \
" chunk bitmap_switch] = GB_mex_init"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    mexPrintf ("usage:\n%s\n", USAGE) ;

    // finalize GraphBLAS but tell it that it can be called again
    GB_mx_at_exit ( ) ;

    // initialize GraphBLAS
    GxB_init (GrB_NONBLOCKING, mxMalloc, NULL, NULL, mxFree) ;

    // mxMalloc, mxCalloc, mxRealloc, and mxFree are not thread safe
    GB_Global_malloc_is_thread_safe_set (false) ;
    GB_Global_abort_set (GB_mx_abort) ;
    GB_Global_malloc_tracking_set (true) ;

    // built-in default is by column
    GxB_Global_Option_set_(GxB_FORMAT, GxB_BY_COL) ;

    int nthreads ;
    GxB_Global_Option_get_(GxB_NTHREADS, &nthreads) ;
    pargout [0] = mxCreateDoubleScalar (nthreads) ;

    GxB_Format_Value format ;
    GxB_Global_Option_get_(GxB_FORMAT, &format) ;
    pargout [1] = mxCreateDoubleScalar (format) ;

    double hyper_switch ;
    GxB_Global_Option_get_(GxB_HYPER_SWITCH, &hyper_switch) ;
    pargout [2] = mxCreateDoubleScalar (hyper_switch) ;

    char *name ;
    GxB_Global_Option_get_(GxB_LIBRARY_NAME, &name) ;
    pargout [3] = mxCreateString (name) ;

    int version [3] ;
    GxB_Global_Option_get_(GxB_LIBRARY_VERSION, version) ;
    pargout [4] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *p = mxGetPr (pargout [4]) ;
    p [0] = version [0] ;
    p [1] = version [1] ;
    p [2] = version [2] ;

    char *date ;
    GxB_Global_Option_get_(GxB_LIBRARY_DATE, &date) ;
    pargout [5] = mxCreateString (date) ;

    char *about ;
    GxB_Global_Option_get_(GxB_LIBRARY_ABOUT, &about) ;
    pargout [6] = mxCreateString (about) ;

    char *license ;
    GxB_Global_Option_get_(GxB_LIBRARY_LICENSE, &license) ;
    pargout [7] = mxCreateString (license) ;

    char *compile_date ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_DATE, &compile_date) ;
    pargout [8] = mxCreateString (compile_date) ;

    char *compile_time ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_TIME, &compile_time) ;
    pargout [9] = mxCreateString (compile_time) ;

    int api [3] ;
    GxB_Global_Option_get_(GxB_API_VERSION, api) ;
    pargout [10] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *a = mxGetPr (pargout [10]) ;
    a [0] = api [0] ;
    a [1] = api [1] ;
    a [2] = api [2] ;

    char *api_about ;
    GxB_Global_Option_get_(GxB_API_ABOUT, &api_about) ;
    pargout [11] = mxCreateString (api_about) ;

    double chunk ;
    GxB_Global_Option_get_(GxB_CHUNK, &chunk) ;
    pargout [12] = mxCreateDoubleScalar (chunk) ;

    double bitmap_switch [GxB_NBITMAP_SWITCH] ;
    GxB_Global_Option_get_(GxB_BITMAP_SWITCH, bitmap_switch) ;
    pargout [13] = mxCreateDoubleMatrix (1, GxB_NBITMAP_SWITCH, mxREAL) ;
    double *bswitch = mxGetPr (pargout [13]) ;
    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        bswitch [k] = bitmap_switch [k] ;
    }

    for (int k = 0 ; k < GxB_NBITMAP_SWITCH ; k++)
    {
        printf ("bitmap_switch [%d] = %g ", k, bswitch [k]) ;
        if (k == 0)
        {
            printf ("for vectors and matrices with 1 row or column\n") ;
        }
        else if (k == GxB_NBITMAP_SWITCH - 1) 
        {
            printf ("for matrices with min dimension > %d\n", 1 << (k-1)) ;
        }
        else
        {
            printf ("for matrices with min dimension %d to %d\n",
                (1 << (k-1)) + 1, 1 << k) ;
        }
    }

    // finalize GraphBLAS but tell it that it can be called again
    GB_mx_at_exit ( ) ;
}

