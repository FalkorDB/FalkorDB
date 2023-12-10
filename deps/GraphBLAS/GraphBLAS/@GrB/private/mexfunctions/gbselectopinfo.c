//------------------------------------------------------------------------------
// gbidxunopinfo : print a GraphBLAS GrB_IndexUnaryOp (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbidxunopinfo (idxunop)

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

    gb_usage (nargin >= 1 && nargin <= 2 && nargout == 0, USAGE) ;

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

    OK (GxB_IndexUnaryOp_fprint (idxunop, opstring, GxB_COMPLETE, NULL)) ;

    GB_WRAPUP ;
}

