//------------------------------------------------------------------------------
// gbjit: control the GraphBLAS JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [status] = gbjit
// [status, path] = gbjit (status, path)

#include "gb_interface.h"

#define USAGE "usage: [status, path] = GrB.jit (status, path) ;"

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

    if (nargin > 0)
    { 
        // set the JIT control
        #define JIT(c) \
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, c, GxB_JIT_C_CONTROL)) ;
        #define LEN 256
        char status [LEN+2]  ;
        gb_mxstring_to_string (status, LEN, pargin [0], "status") ;
        if      (MATCH (status, ""     ))
        { 
            /* do nothing */ ;
        }
        else if (MATCH (status, "off"  ))
        { 
            JIT (GxB_JIT_OFF) ;
        }
        else if (MATCH (status, "pause"))
        { 
            JIT (GxB_JIT_PAUSE) ;
        }
        else if (MATCH (status, "run"  ))
        { 
            JIT (GxB_JIT_RUN) ;
        }
        else if (MATCH (status, "load" ))
        { 
            JIT (GxB_JIT_LOAD) ;
        }
        else if (MATCH (status, "on"   ))
        { 
            JIT (GxB_JIT_ON) ;
        }
        else if (MATCH (status, "flush"))
        { 
            JIT (GxB_JIT_OFF) ;
            JIT (GxB_JIT_ON) ;
        }
        else
        { 
            ERROR2 ("unknown option: %s", status) ;
        }
    }

    //--------------------------------------------------------------------------
    // get the JIT control, if requested
    //--------------------------------------------------------------------------

    if (nargout > 0)
    { 
        int c ;
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &c, GxB_JIT_C_CONTROL)) ;
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
    // set the JIT cache path, if requested
    //--------------------------------------------------------------------------

    if (nargin > 1)
    {
        if (!mxIsChar (pargin[1]))
        {
            ERROR ("path must be a string") ;
        }
        size_t pathlen = mxGetNumberOfElements (pargin [1]) + 2 ;
        char *path = mxMalloc (pathlen + 2) ;
        path [0] = '\0' ;
        mxGetString (pargin [1], path, pathlen) ;
        OK (GrB_Global_set_String (GrB_GLOBAL, path, GxB_JIT_CACHE_PATH)) ;
        mxFree (path) ;
    }

    //--------------------------------------------------------------------------
    // get the JIT cache path, if requested
    //--------------------------------------------------------------------------

    if (nargout > 1)
    {
        size_t pathlen = 0 ;
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &pathlen, GxB_JIT_CACHE_PATH)) ;
        char *path = mxMalloc (pathlen + 2) ;
        path [0] = '\0' ;
        OK (GrB_Global_get_String (GrB_GLOBAL, path, GxB_JIT_CACHE_PATH)) ;
        pargout [1] = mxCreateString (path) ;
        mxFree (path) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

