//------------------------------------------------------------------------------
// GB_mex_jit_control: set/get the JIT control
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    int c = -999909 ;
    if (nargin > 0)
    {
        // set the JIT control
        c = (int) mxGetScalar (pargin [0]) ;
        if (c < 0)
        {
            // reset the JIT; turn it off, then set to abs(c)
            OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
            c = GB_IABS (c) ;
        }
        OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, c)) ;
    }

    // get the JIT control
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_C_CONTROL, &c)) ;
    pargout [0] = mxCreateDoubleScalar ((double) c) ;
    GB_mx_put_global (true) ;
}

