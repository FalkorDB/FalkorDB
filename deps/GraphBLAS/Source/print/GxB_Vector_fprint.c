//------------------------------------------------------------------------------
// GxB_Vector_fprint: print and check a GrB_Vector object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_fprint          // print and check a GrB_Vector
(
    GrB_Vector v,                   // object to print and check
    const char *name,               // name of the object
    int pr,                         // print level
    FILE *f                         // file for output
)
{
    GB_CHECK_INIT ;
    GrB_Info info = GB_Vector_check (v, name, pr, f) ;
    if (info == GrB_INDEX_OUT_OF_BOUNDS)
    { 
        // indices out of order
        return (GrB_INVALID_OBJECT) ;
    }
    else
    { 
        return (info) ;
    }
}

