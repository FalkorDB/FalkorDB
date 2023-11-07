//------------------------------------------------------------------------------
// GB_macrofy_user_op: construct a user defined operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_user_op         // construct a user-defined operator
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    const GB_Operator op        // op to construct in a JIT kernel
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (op->hash == 0 || op->hash == UINT64_MAX)
    { 
        // skip if op is builtin or cannot be used in the JIT
        return ;
    }

    //--------------------------------------------------------------------------
    // construct the name
    //--------------------------------------------------------------------------

    fprintf (fp, "#define GB_USER_OP_FUNCTION %s\n", op->name) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, NULL, NULL, NULL,
        op->xtype, op->ytype, op->ztype) ;

    //--------------------------------------------------------------------------
    // construct the function prototype
    //--------------------------------------------------------------------------

    for (char *p = op->defn ; *p ; p++)
    {
        int c = (int) (*p) ;
        if (c == '{')
        { 
            fprintf (fp, ";\n") ;
            break ;
        }
        fputc (c, fp) ;
    }

    //--------------------------------------------------------------------------
    // construct the user function itself
    //--------------------------------------------------------------------------

    fprintf (fp, "\n%s\n", op->defn) ;
    GB_macrofy_string (fp, op->name, op->defn) ;
    fprintf (fp, "#define GB_USER_OP_DEFN GB_%s_USER_DEFN\n", op->name) ;
}

