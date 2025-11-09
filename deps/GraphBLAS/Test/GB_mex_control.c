//------------------------------------------------------------------------------
// GB_mex_control: get/set global is_csc and [pji]_control control
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "ctrl = GB_mex_control (ctrl) or ctrl = GB_mex_control"

static const char *ControlFields [ ] = {
    "is_csc",
    "p_control",
    "j_control",
    "i_control" } ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    mxArray *X ;
    bool malloc_debug = GB_mx_get_global (false) ;

    // check inputs
    if (nargout > 1 || nargin > 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    if (nargin > 0 && !mxIsStruct (pargin [0]))
    {
        mexErrMsgTxt ("input must be struct") ;
    }

    // get the current controls
    int csc, p_control, r_control, c_control ;
    GrB_get (GrB_GLOBAL, &csc,       GxB_FORMAT) ;
    GrB_get (GrB_GLOBAL, &p_control, GxB_OFFSET_INTEGER_HINT) ;
    GrB_get (GrB_GLOBAL, &r_control, GxB_ROWINDEX_INTEGER_HINT) ;
    GrB_get (GrB_GLOBAL, &c_control, GxB_COLINDEX_INTEGER_HINT) ;
    int j_control = csc ? c_control : r_control ;
    int i_control = csc ? r_control : c_control ;

    // get the input struct
    if (nargin > 0)
    {
        int fieldnumber = mxGetFieldNumber (pargin [0], "is_csc") ;
        if (fieldnumber >= 0)
        {
            X = mxGetFieldByNumber (pargin [0], 0, fieldnumber) ;
            csc = (int) mxGetScalar (X) ;
        }

        fieldnumber = mxGetFieldNumber (pargin [0], "p_control") ;
        if (fieldnumber >= 0)
        {
            X = mxGetFieldByNumber (pargin [0], 0, fieldnumber) ;
            p_control = (int) mxGetScalar (X) ;
        }

        fieldnumber = mxGetFieldNumber (pargin [0], "j_control") ;
        if (fieldnumber >= 0)
        {
            X = mxGetFieldByNumber (pargin [0], 0, fieldnumber) ;
            j_control = (int) mxGetScalar (X) ;
        }

        fieldnumber = mxGetFieldNumber (pargin [0], "i_control") ;
        if (fieldnumber >= 0)
        {
            X = mxGetFieldByNumber (pargin [0], 0, fieldnumber) ;
            i_control = (int) mxGetScalar (X) ;
        }
    }

    // set the new controls
    c_control = csc ? j_control : i_control ;
    r_control = csc ? i_control : j_control ;
    GrB_set (GrB_GLOBAL, csc,       GxB_FORMAT) ;
    GrB_set (GrB_GLOBAL, p_control, GxB_OFFSET_INTEGER_HINT) ;
    GrB_set (GrB_GLOBAL, r_control, GxB_ROWINDEX_INTEGER_HINT) ;
    GrB_set (GrB_GLOBAL, c_control, GxB_COLINDEX_INTEGER_HINT) ;

    // return new controls as a struct
    pargout [0] = mxCreateStructMatrix (1, 1, 4, ControlFields) ;

    mxArray *is_csc = mxCreateLogicalScalar (csc) ;
    mxArray *p_32 = mxCreateDoubleScalar (p_control) ;
    mxArray *j_32 = mxCreateDoubleScalar (j_control) ;
    mxArray *i_32 = mxCreateDoubleScalar (i_control) ;

    mxSetFieldByNumber (pargout [0], 0, 0, is_csc) ;
    mxSetFieldByNumber (pargout [0], 0, 1, p_32) ;
    mxSetFieldByNumber (pargout [0], 0, 2, j_32) ;
    mxSetFieldByNumber (pargout [0], 0, 3, i_32) ;

    // log the test coverage
    GB_mx_put_global (true) ;
}

