//------------------------------------------------------------------------------
// GB_mx_mxArray_to_array: get a dense numerical built-in array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

void GB_mx_mxArray_to_array     // convert mxArray to array
(
    const mxArray *Xbuiltin,    // input built-in array
    // output:
    GB_void **X,                // pointer to numerical values (shallow)
    int64_t *nrows,             // number of rows of X
    int64_t *ncols,             // number of columns of X
    GrB_Type *xtype             // GraphBLAS type of X, NULL if error
)
{

    if (!(mxIsNumeric (Xbuiltin) || mxIsLogical (Xbuiltin)))
    {
        mexWarnMsgIdAndTxt ("GB:warn","input must be numeric or logical array");
    }
    if (mxIsSparse (Xbuiltin))
    {
        mexWarnMsgIdAndTxt ("GB:warn","input cannot be sparse") ;
    }

    (*X) = mxGetData (Xbuiltin) ;
    (*nrows) = mxGetM (Xbuiltin) ;
    (*ncols) = mxGetN (Xbuiltin) ;
    (*xtype) = GB_mx_Type (Xbuiltin) ;
}

