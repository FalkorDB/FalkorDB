//------------------------------------------------------------------------------
// gbjit: control the GraphBLAS JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [status] = gbjit
// [status] = gbjit (status)

#include "gb_interface.h"

#define USAGE "usage: [status] = GrB.jit (status) ;"

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

    gb_usage (nargin <= 1 && nargout <= 1, USAGE) ;

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
    // return result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

