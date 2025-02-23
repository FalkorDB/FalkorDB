//------------------------------------------------------------------------------
// gbmonoidinfo : print a GraphBLAS monoid (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmonoidinfo (monoid)
// gbmonoidinfo (monoid, type)
// ok = gbmonoidinfo (monoid)

#include "gb_interface.h"

#define USAGE "usage: GrB.monoidinfo (monoid) or GrB.monoidinfo (monoid,type)"

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
    // construct the GraphBLAS monoid and print it
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, pargin [0], "binary operator") ;

    GrB_Type type = NULL ;
    if (nargin > 1)
    { 
        type = gb_mxstring_to_type (pargin [1]) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    GrB_Monoid op = gb_mxstring_to_monoid (pargin [0], type) ;
    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_Monoid_fprint (op, opstring, pr, NULL)) ;
    if (nargout == 1)
    {
        pargout [0] = mxCreateLogicalScalar (true) ;
    }
    gb_wrapup ( ) ;
}

