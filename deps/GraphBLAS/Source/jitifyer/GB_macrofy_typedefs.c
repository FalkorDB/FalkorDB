//------------------------------------------------------------------------------
// GB_macrofy_typedefs: construct typedefs for up to 6 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_typedefs
(
    FILE *fp,
    // input:
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype,
    GrB_Type xtype,
    GrB_Type ytype,
    GrB_Type ztype,
    GrB_Type theta_type
)
{

    //--------------------------------------------------------------------------
    // create typedefs, checking for duplicates
    //--------------------------------------------------------------------------

    const char *defn [7] ;
    defn [0] = (ctype == NULL) ? NULL : ctype->defn ;
    defn [1] = (atype == NULL) ? NULL : atype->defn ;
    defn [2] = (btype == NULL) ? NULL : btype->defn ;
    defn [3] = (xtype == NULL) ? NULL : xtype->defn ;
    defn [4] = (ytype == NULL) ? NULL : ytype->defn ;
    defn [5] = (ztype == NULL) ? NULL : ztype->defn ;
    defn [6] = (theta_type == NULL) ? NULL : theta_type->defn ;

    const char *name [7] ;
    name [0] = (ctype == NULL) ? NULL : ctype->name ;
    name [1] = (atype == NULL) ? NULL : atype->name ;
    name [2] = (btype == NULL) ? NULL : btype->name ;
    name [3] = (xtype == NULL) ? NULL : xtype->name ;
    name [4] = (ytype == NULL) ? NULL : ytype->name ;
    name [5] = (ztype == NULL) ? NULL : ztype->name ;
    name [6] = (theta_type == NULL) ? NULL : theta_type->name ;

    for (int k = 0 ; k <= 6 ; k++)
    { 
        if (defn [k] != NULL && name [k] != NULL)
        { 
            // only print this typedef it is unique
            bool is_unique = true ;
            for (int j = 0 ; j < k && is_unique ; j++)
            {
                if (defn [j] != NULL && name [j] != NULL &&
                    strcmp (name [j], name [k]) == 0)
                { 
                    is_unique = false ;
                }
            }
            if (is_unique)
            { 
                // the typedef is unique: include it in the .h file
                fprintf (fp, "%s\n", defn [k]) ;

                // also include its definition as a string
                GB_macrofy_string (fp, name [k], defn [k]) ;

                fprintf (fp, "\n") ;
            }
        }
    }
}

