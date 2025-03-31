//------------------------------------------------------------------------------
// gbsemiringinfo: print a GraphBLAS semiring (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbsemiringinfo (semiring_string)
// gbsemiringinfo (semiring_string, type)
// ok = gbsemiringinfo (semiring_string)

#include "gb_interface.h"

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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS semiring and print it
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

    GrB_Semiring semiring = gb_mxstring_to_semiring (pargin [0], type, type) ;
    int pr = (nargout < 1) ? GxB_COMPLETE : GxB_SILENT ;
    OK (GxB_Semiring_fprint (semiring, opstring, pr, NULL)) ;
    if (nargout == 1)
    {
        pargout [0] = mxCreateLogicalScalar (true) ;
    }
    gb_wrapup ( ) ;
}

