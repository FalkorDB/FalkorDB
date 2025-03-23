//------------------------------------------------------------------------------
// GxB_IndexUnaryOp_fprint: print and check a GrB_IndexUnaryOp object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_IndexUnaryOp_fprint    // print and check a GrB_IndexUnaryOp
(
    GrB_IndexUnaryOp op,            // object to print and check
    const char *name,               // name of the object
    int pr,                         // print level
    FILE *f                         // file for output
)
{ 
    GB_CHECK_INIT ;
    return (GB_IndexUnaryOp_check (op, name, pr, f)) ;
}

