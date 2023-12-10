//------------------------------------------------------------------------------
// gbcov_util.c: utilities for test coverage
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These functions are compiled along with the GraphBLAS mexFunctions, to
// allow them to copy the statement coverage counts to and from the built-in
// global workspace.

#include "gb_interface.h"

//------------------------------------------------------------------------------
// gbcov_get: copy coverage counts from the built-in workspace
//------------------------------------------------------------------------------

// This function is called when a GraphBLAS mexFunction starts.
// gbcov_global is an int64 built-in array in the built-in global workspace.
// Its size is controlled by gbcov_max, defined in gbcovfinish.c.  If the
// array is empty in the workspace, or if it doesn't exist, it is created with
// the correct size.  Then the internal gbcov array is copied into it.

// mxGetData is used instead of the MATLAB-recommended mxGetDoubles, etc,
// because mxGetData works best for Octave, and it works fine for MATLAB
// since GraphBLAS requires R2018a with the interleaved complex data type.

void gbcov_get ( )
{

    // get gbcov_global from built-in global workspace
    mxArray *GB_cov_global = NULL ;
    GB_cov_global = (mxArray *) mexGetVariablePtr ("global", "gbcov_global") ;

    if (GB_cov_global == NULL || mxIsEmpty (GB_cov_global))
    {
        // doesn't exist; create it and set it to zero
        GB_cov_global = mxCreateNumericMatrix (1, gbcov_max,
            mxINT64_CLASS, mxREAL) ;
        // copy it back to the built-in global workspace
        mexPutVariable ("global", "gbcov_global", GB_cov_global) ;
    }

    // it should exist now, but double-check
    if (GB_cov_global == NULL || mxIsEmpty (GB_cov_global))
    {
        mexErrMsgIdAndTxt ("GrB:panic", "GB_cov_global still null!") ;
    }

    // get a pointer to the content of the gbcov_global array in the
    // global workspace
    int64_t *g = (int64_t *) mxGetData (GB_cov_global) ;

    // getting paranoid here; this should never happen
    if (g == NULL) mexErrMsgIdAndTxt ("GrB:panic", "g null!") ;

    // copy the count from the built-in gbcov_global into gbcov
    memcpy (gbcov, g, gbcov_max * sizeof (int64_t)) ;
}

//------------------------------------------------------------------------------
// gbcov_put: copy coverage counts back to the built-in workspace
//------------------------------------------------------------------------------

// This function is called when a GraphBLAS mexFunction finishes.  It copies
// the updated statement coverage counters in the gbcov array back to the
// gbcov_global array in the built-in global workspace where it can be
// analyzed.

void gbcov_put ( )
{
    // create a built-in array with the right size
    mxArray * GB_cov_global = mxCreateNumericMatrix (1, gbcov_max,
            mxINT64_CLASS, mxREAL) ;

    // copy the updated gbcov counter array into the built-in array
    int64_t *g = (int64_t *) mxGetData (GB_cov_global) ;
    memcpy (g, gbcov, gbcov_max * sizeof (int64_t)) ;

    // put the built-in array into the global workspace, overwriting the
    // version that was already there
    mexPutVariable ("global", "gbcov_global", GB_cov_global) ;
}

