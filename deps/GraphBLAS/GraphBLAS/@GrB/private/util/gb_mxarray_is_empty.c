//------------------------------------------------------------------------------
// gb_mxarray_is_empty: check if a built-in mxArray is empty
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

bool gb_mxarray_is_empty    // true if built-in array is NULL, or 2D and 0-by-0
(
    const mxArray *S
)
{ 

    return ((S == NULL)
        || ((mxGetNumberOfDimensions (S) == 2) &&
            (mxGetM (S) == 0) && (mxGetN (S) == 0))) ;
}

