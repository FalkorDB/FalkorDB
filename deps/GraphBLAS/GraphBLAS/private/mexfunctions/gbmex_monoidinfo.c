//------------------------------------------------------------------------------
// gbmex_monoidinfo : print a GraphBLAS monoid (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_monoidinfo (monoid)
// gbmex_monoidinfo (monoid, type)
// ok = gbmex_monoidinfo (monoid)

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"
#include "gb_binop_to_monoid.c"
#include "gb_string_to_monoid.c"

#include "gbmx_interface.h"

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

    GrB_Monoid monoid = NULL ;
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
    gbmx_mxstring_to_string (op_string, LEN, pargin [0], "binary operator") ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS monoid and print it
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    OK (gb_string_to_monoid (&monoid, op_string, type, err)) ;

    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_Monoid_fprint (monoid, op_string, pr, NULL)) ;
    gb_wrapup ( ) ;
}

