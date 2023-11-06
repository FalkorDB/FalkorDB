//------------------------------------------------------------------------------
// gbjit: control the GraphBLAS JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [s,path] = gbjit
// [s,path] = gbjit (jit)

#include "gb_interface.h"

#define USAGE "usage: [s,path] = GrB.jit (s,path) ;"

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

    gb_usage (nargin <= 2 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // set the JIT control, if requested
    //--------------------------------------------------------------------------

    #define LEN 2048
    char s [LEN] ;

    if (nargin > 0)
    { 
        // set the JIT control
        #define JIT(c) { OK (GxB_Global_Option_set (GxB_JIT_C_CONTROL, c)) ; }
        gb_mxstring_to_string (s, LEN, pargin [0], "s") ; 
        if      (MATCH (s, ""     )) { /* do nothing */ ; }
        else if (MATCH (s, "off"  )) JIT (GxB_JIT_OFF)
        else if (MATCH (s, "pause")) JIT (GxB_JIT_PAUSE)
        else if (MATCH (s, "run"  )) JIT (GxB_JIT_RUN)
        else if (MATCH (s, "load" )) JIT (GxB_JIT_LOAD)
        else if (MATCH (s, "on"   )) JIT (GxB_JIT_ON)
        else if (MATCH (s, "flush")) { JIT (GxB_JIT_OFF) ; JIT (GxB_JIT_ON) ; }
        else ERROR2 ("unknown option: %s", s) ;
    }

    //--------------------------------------------------------------------------
    // set the cache path, if requested
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        // set the JIT cache path
        gb_mxstring_to_string (s, LEN, pargin [1], "path") ; 
        OK (GxB_Global_Option_set (GxB_JIT_CACHE_PATH, s)) ;
    }

    //--------------------------------------------------------------------------
    // get the JIT control, if requested
    //--------------------------------------------------------------------------

    if (nargout > 0)
    { 
        GxB_JIT_Control c ;
        OK (GxB_Global_Option_get (GxB_JIT_C_CONTROL, &c)) ;
        switch (c)
        {
            case GxB_JIT_OFF  : pargout [0] = mxCreateString ("off"  ) ; break ;
            case GxB_JIT_PAUSE: pargout [0] = mxCreateString ("pause") ; break ;
            case GxB_JIT_RUN  : pargout [0] = mxCreateString ("run"  ) ; break ;
            case GxB_JIT_LOAD : pargout [0] = mxCreateString ("load" ) ; break ;
            case GxB_JIT_ON   : pargout [0] = mxCreateString ("on"   ) ; break ;
            default           : pargout [0] = mxCreateString ("unknown") ;
                                break ;
        }
    }

    //--------------------------------------------------------------------------
    // get the JIT cache path, if requested
    //--------------------------------------------------------------------------

    if (nargout > 1)
    { 
        char *path = NULL ;
        OK (GxB_Global_Option_get (GxB_JIT_CACHE_PATH, &path)) ;
        pargout [1] = mxCreateString (path) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_WRAPUP ;
}

