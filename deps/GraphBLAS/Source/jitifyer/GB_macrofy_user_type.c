//------------------------------------------------------------------------------
// GB_macrofy_user_type: construct a user defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_user_type       // construct a user-defined type
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    const GrB_Type type         // type to construct in a JIT kernel
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (type->hash == 0 || type->hash == UINT64_MAX)
    { 
        // skip if type is builtin or cannot be used in the JIT
        return ;
    }

    //--------------------------------------------------------------------------
    // construct the name
    //--------------------------------------------------------------------------

    fprintf (fp, "#define GB_USER_TYPE %s\n", type->name) ;

    //--------------------------------------------------------------------------
    // construct the typedef
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, NULL, NULL, NULL, type, NULL, NULL, NULL) ;
    fprintf (fp, "#define GB_USER_TYPE_DEFN GB_%s_USER_DEFN\n", type->name) ;
}

