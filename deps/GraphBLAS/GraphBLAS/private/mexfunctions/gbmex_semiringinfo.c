//------------------------------------------------------------------------------
// gbmex_semiringinfo: print a GraphBLAS semiring (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_semiringinfo (semiring_string)
// gbmex_semiringinfo (semiring_string, type)
// ok = gbmex_semiringinfo (semiring_string)

#include "gb_interface.h"
#include "gb_semiring.c"
#include "gb_string_to_semiring.c"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#define USAGE "usage: GrB.semiringinfo (s) or GrB.semiringinfo (s,type)"

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

    GrB_Semiring semiring = NULL ;
    GrB_Type type = NULL ;

    GBMX_USAGE (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    if (nargout == 1)
    { 
        pargout [0] = mxCreateLogicalScalar (true) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char semiring_string [LEN+2] ;
    char type_string [LEN+2] ;
    gbmx_mxstring_to_string (semiring_string, LEN, pargin [0], "semiring") ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS semiring and print it
    //--------------------------------------------------------------------------

    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    OK (gb_string_to_semiring (&semiring, semiring_string, type, type, err)) ;
    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_Semiring_fprint (semiring, semiring_string, pr, NULL)) ;
    gb_wrapup ( ) ;
}

