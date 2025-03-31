//------------------------------------------------------------------------------
// GxB_Scalar_fprint: print and check a GrB_Scalar object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Scalar_fprint          // print and check a GrB_Scalar
(
    GrB_Scalar s,                   // object to print and check
    const char *name,               // name of the object
    int pr,                         // print level
    FILE *f                         // file for output
)
{ 
    GB_CHECK_INIT ;
    return (GB_Scalar_check (s, name, pr, f)) ;
}

