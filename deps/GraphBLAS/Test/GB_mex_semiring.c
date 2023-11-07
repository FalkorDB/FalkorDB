//------------------------------------------------------------------------------
// GB_mex_semiring: parse a semiring, for testing; returns nothing
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "GB_mex_semiring (semiring_struct))"

#define FREE_ALL                \
{                               \
    GB_mx_put_global (true) ;   \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Semiring semiring = NULL ;

    // check inputs
    if (nargin < 1 || nargin > 2 || nargout > 0)
    {
        FREE_ALL ;
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    bool user_complex = (Complex != GxB_FC64) ;

    GB_mx_mxArray_to_Semiring (&semiring, pargin [0], "semiring", GrB_FP64,
        user_complex) ;

    int GET_SCALAR (1, int, pr, GxB_COMPLETE) ;

    GrB_Info info = GB_Semiring_check (semiring, "semiring", pr, NULL) ;
    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexErrMsgTxt ("semiring fail") ;
    }
    FREE_ALL ;
}

