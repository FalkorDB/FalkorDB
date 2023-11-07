//------------------------------------------------------------------------------
// GB_mx_abort: terminate
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

void GB_mx_abort ( )                    // assertion failure
{
    mexErrMsgIdAndTxt ("GraphBLAS:assert", "assertion failed") ;
}

