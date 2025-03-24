//------------------------------------------------------------------------------
// gbidxunopinfo : print a GraphBLAS GrB_IndexUnaryOp (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbidxunopinfo (idxunop)
// gbidxunopinfo (idxunop, type)
// ok = gbidxunopinfo (idxunop)

#include "gb_interface.h"

#define USAGE "usage: GrB.selectopinfo (selectop) or GrB.selectopinfo (op,type)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS GrB_IndexUnaryOp and print it
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, pargin [0], "select operator") ;

    GrB_Type type = GrB_FP64 ;
    if (nargin > 1)
    { 
        type = gb_mxstring_to_type (pargin [1]) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    GrB_IndexUnaryOp idxunop = NULL ;
    bool ignore1, ignore2 ;
    int64_t ignore3 = 0 ;

    gb_mxstring_to_idxunop (&idxunop, &ignore1, &ignore2, &ignore3,
        pargin [0], type) ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_IndexUnaryOp_fprint (idxunop, opstring, pr, NULL)) ;
    if (nargout == 1)
    {
        pargout [0] = mxCreateLogicalScalar (true) ;
    }

    gb_wrapup ( ) ;
}

