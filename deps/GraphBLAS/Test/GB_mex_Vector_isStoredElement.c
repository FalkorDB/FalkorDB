//------------------------------------------------------------------------------
// GB_mex_Vector_isStoredElement: interface for x = v(i)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "y = GB_mex_Vector_isStoredElement (v, I)"

#define FREE_ALL                                        \
{                                                       \
    GrB_Vector_free_(&v) ;                              \
    GB_mx_put_global (true) ;                           \
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
    GrB_Vector v = NULL ;
    bool *X = NULL ;
    GrB_Index *I = NULL, ni = 0, I_range [3] ;
    bool is_list ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get v (shallow copy)
    v = GB_mx_mxArray_to_Vector (pargin [0], "v input", false, true) ;
    if (v == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("v failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [1], &ni, I_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("I must be a list") ;
    }

    // create output X
    pargout [0] = GB_mx_create_full (ni, 1, GrB_BOOL) ;
    X = (bool *) mxGetData (pargout [0]) ;

    // x = v (i)
    for (int64_t k = 0 ; k < ni ; k++)
    {
        GrB_Info info = GxB_Vector_isStoredElement(v, I [k]) ;
        X [k] = (info == GrB_SUCCESS) ;
    }

    FREE_ALL ;
}

