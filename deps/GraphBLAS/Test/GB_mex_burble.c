//------------------------------------------------------------------------------
// GB_mex_burble: set/get the burble
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
    GB_mx_get_global (true) ;
    int burble ;
    if (nargin > 0)
    {
        // set the burble
        burble = (int) mxGetScalar (pargin [0]) ;
        OK (GxB_Global_Option_set_INT32 (GxB_BURBLE, burble)) ;
    }

    // get the burble
    OK (GxB_Global_Option_get_INT32 (GxB_BURBLE, &burble)) ;
    pargout [0] = mxCreateDoubleScalar ((double) burble) ;
    GB_mx_put_global (true) ;
}

