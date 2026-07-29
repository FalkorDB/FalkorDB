//------------------------------------------------------------------------------
// gbmex_nmalloc: # of mallocs in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This mexFunction is for testing and development only.  It returns the number
// of malloc'd spaces in GraphBLAS that have not been freed.

// Usage

// nmalloc = gbmex_nmalloc ;

#include "gb_interface.h"
#include "gbmx_interface.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    pargout [0] = mxCreateDoubleScalar (0) ;
    double *nmalloc_output = (double *) mxGetData (pargout [0]) ;
    (*nmalloc_output) = (double) GB_Global_nmalloc_get ( ) ;
    GB_Global_memtable_dump ( ) ;
}

