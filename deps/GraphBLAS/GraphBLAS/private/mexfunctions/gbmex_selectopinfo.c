//------------------------------------------------------------------------------
// gbmex_selectopinfo : print a GraphBLAS GrB_IndexUnaryOp (for illustration)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_selectopinfo (idxunop)
// gbmex_selectopinfo (idxunop, type)
// ok = gbmex_selectopinfo (idxunop)

#include "gb_interface.h"
#include "gb_string_to_idxunop.c"

#include "gbmx_interface.h"

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
    // check inputs and construct output
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp idxunop = NULL ;
    GrB_Type type = NULL ;

    GBMX_USAGE (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    if (nargout == 1)
    { 
        pargout [0] = mxCreateLogicalScalar (true) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char op_string [LEN+2] ;
    char type_string [LEN+2] ;
    gbmx_mxstring_to_string (op_string, LEN, pargin [0], "select operator") ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS GrB_IndexUnaryOp and print it
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
    }

    bool ignore1, ignore2 ;
    int64_t ignore3 = 0 ;

    OK (gb_string_to_idxunop (&idxunop, &ignore1, &ignore2, &ignore3,
        op_string, type, err)) ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_IndexUnaryOp_fprint (idxunop, op_string, pr, NULL)) ;
    gb_wrapup ( ) ;
}

