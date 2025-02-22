//------------------------------------------------------------------------------
// GxB_Type_fprint: print and check a GrB_Type object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Type_fprint            // print and check a GrB_Type
(
    GrB_Type type,                  // object to print and check
    const char *name,               // name of the object
    int pr,                         // print level
    FILE *f                         // file for output
)
{ 
    GB_CHECK_INIT ;
    return (GB_Type_check (type, name, pr, f)) ;
}

