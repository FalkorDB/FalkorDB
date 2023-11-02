//------------------------------------------------------------------------------
// GxB_SelectOp_fprint: print and check a GxB_SelectOp object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_SelectOp_fprint        // print and check a GxB_SelectOp
(
    GxB_SelectOp selectop,          // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_SelectOp_fprint (selectop, name, pr, f)") ;

    //--------------------------------------------------------------------------
    // print and check the object
    //--------------------------------------------------------------------------

    return (GB_SelectOp_check (selectop, name, pr, f)) ;
}

