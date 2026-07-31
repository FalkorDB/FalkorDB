//------------------------------------------------------------------------------
// gbmex_jit: control the GraphBLAS JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [status] = gbmex_jit
// [status, path] = gbmex_jit (status, path)

#include "gb_interface.h"
#include "gbmx_interface.h"

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

    GBMX_USAGE (nargin <= 2 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // set the JIT control, if requested
    //--------------------------------------------------------------------------

    if (nargin > 0)
    { 
        // set the JIT control
        #define JIT(c) \
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, c, GxB_JIT_C_CONTROL)) ;
        char status [LEN+2]  ;
        gbmx_mxstring_to_string (status, LEN, pargin [0], "status") ;
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
            ERROR2 ("unknown option: %s", status, GrB_INVALID_VALUE) ;
        }
    }

    //--------------------------------------------------------------------------
    // get the JIT control
    //--------------------------------------------------------------------------

    int c ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &c, GxB_JIT_C_CONTROL)) ;
    char *current_status = NULL ;
    switch (c)
    {
        case GxB_JIT_OFF  : current_status = "off"      ; break ;
        case GxB_JIT_PAUSE: current_status = "pause"    ; break ;
        case GxB_JIT_RUN  : current_status = "run"      ; break ;
        case GxB_JIT_LOAD : current_status = "load"     ; break ;
        case GxB_JIT_ON   : current_status = "on"       ; break ;
        default           : current_status = "unknown"  ; break ;
    }

    if (nargout > 0)
    { 
        pargout [0] = mxCreateString (current_status) ;
    }

    //--------------------------------------------------------------------------
    // set the JIT cache path, if requested
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        if (!mxIsChar (pargin[1]))
        { 
            ERROR ("path must be a string", GrB_INVALID_VALUE) ;
        }
        size_t pathlen = mxGetNumberOfElements (pargin [1]) + 2 ;
        char *path = mxMalloc (pathlen + 2) ;
        path [0] = '\0' ;
        mxGetString (pargin [1], path, pathlen) ;
        OK (GrB_Global_set_String (GrB_GLOBAL, path, GxB_JIT_CACHE_PATH)) ;
        gbmx_free ((void **) &path) ;
    }

    //--------------------------------------------------------------------------
    // get the JIT cache path, if requested
    //--------------------------------------------------------------------------

    char *path = NULL ;
    if (nargout > 1 || (nargin == 0 && nargout == 0))
    { 
        size_t pathlen = 0 ;
        OK (GrB_Global_get_SIZE (GrB_GLOBAL, &pathlen, GxB_JIT_CACHE_PATH)) ;
        path = mxMalloc (pathlen + 2) ;
        path [0] = '\0' ;
        OK (GrB_Global_get_String (GrB_GLOBAL, path, GxB_JIT_CACHE_PATH)) ;
    }

    if (nargout > 1)
    {  
        pargout [1] = mxCreateString (path) ;
    }

    //--------------------------------------------------------------------------
    // report the status, if requested
    //--------------------------------------------------------------------------

    if (nargin == 0 && nargout == 0)
    { 
        printf ("GraphBLAS jit status: %s\npath: %s\n", current_status, path) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    gbmx_free ((void **) &path) ;
    gb_wrapup ( ) ;
}

