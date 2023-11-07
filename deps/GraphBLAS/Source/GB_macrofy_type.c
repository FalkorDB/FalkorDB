//------------------------------------------------------------------------------
// GB_macrofy_type: construct macros for a type name
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_type
(
    FILE *fp,
    // input:
    const char *what,       // typically X, Y, Z, A, B, or C
    const char *what2,      // typically "_" or "2"
    const char *name        // name of the type
)
{

    if ((name == NULL) ||
        (strcmp (name, "GB_void") == 0) || (strcmp (name, "void") == 0))
    { 
        // values of the matrix are not used
        fprintf (fp, "#define GB_%s%sTYPE void\n", what, what2) ;
    }
    else
    { 
        fprintf (fp, "#define GB_%s%sTYPE %s\n", what, what2, name) ;
    }
}

