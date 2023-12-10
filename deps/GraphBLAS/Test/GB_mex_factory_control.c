//------------------------------------------------------------------------------
// GB_mex_factory_control: enable/disable the factory kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    if (nargin > 0)
    {
        // set the factory control
        GB_factory_kernels_enabled = (bool) mxGetScalar (pargin [0]) ;
    }

    // get the factory control
    pargout [0] = mxCreateDoubleScalar ((double) GB_factory_kernels_enabled) ;
}

