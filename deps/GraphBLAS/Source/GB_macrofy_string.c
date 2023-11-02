//------------------------------------------------------------------------------
// GB_macrofy_string: construct the string for a user-defined type or operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_string
(
    FILE *fp,
    const char *name,
    const char *defn
)
{

    fprintf (fp, "#define GB_%s_USER_DEFN \\\n\"", name) ;
    for (const char *p = defn ; (*p) != '\0' ; p++)
    {
        int c = (*p) ;
        if (c == '\n')
        { 
            // handle the new-line character
            fprintf (fp, "\\n\" \\\n\"") ;
        }
        else if (c == '\\')
        { 
            // handle the backslash character
            fprintf (fp, "\\\\") ;
        }
        else if (c == '"')
        { 
            // handle the quote character
            fprintf (fp, "\\\"") ;
        }
        else
        { 
            // all other characters
            fprintf (fp, "%c", c) ;
        }
    }
    fprintf (fp, "\"\n") ;
}

