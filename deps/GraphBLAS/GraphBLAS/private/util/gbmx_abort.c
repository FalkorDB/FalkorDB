//------------------------------------------------------------------------------
// gbmx_abort: terminate a GraphBLAS function immediately (debug assert only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method does not return; it immediately terminates the mexFunction.
// This is called by GB_Global_abort, which uses the C abort() method by
// default.

void gbmx_abort ( void )    // terminate immediately (debug assertions only)
{
    mexErrMsgIdAndTxt ("GraphBLAS:abort", "GraphBLAS failed") ;
}

