//------------------------------------------------------------------------------
// GxB_SelectOp_fprint: print and check a GxB_SelectOp object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_SelectOp_fprint        // print and check a GxB_SelectOp
(
    GxB_SelectOp selectop,          // object to print and check
    const char *name,               // name of the object
    int pr,                         // print level
    FILE *f                         // file for output
)
{ 
    GB_CHECK_INIT ;
    return (GB_SelectOp_check (selectop, name, pr, f)) ;
}

