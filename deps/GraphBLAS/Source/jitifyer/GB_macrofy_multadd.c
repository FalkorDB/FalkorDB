//------------------------------------------------------------------------------
// GB_macrofy_multadd: create a fused multiply-add operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_multadd
(
    FILE *fp,
    const char *update_expression,      // has the form "z = f(z,y)"
    const char *multiply_expression,    // has the form "z = mult(x,y)"
    bool flipxy
)
{

    // CPU kernels can use the fused multiply-add
    if (flipxy)
    { 
        fprintf (fp, "#define GB_MULTADD(z,y,x,j,k,i) ") ;
    }
    else
    { 
        fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j) ") ;
    }
    for (const char *p = update_expression ; (*p) != '\0' ; p++)
    {
        // all update operators have a single 'y'
        if ((*p) == 'y')
        { 
            // inject the multiply operator; all have the form "z = ..."
            fprintf (fp, "%s", multiply_expression + 4) ;
        }
        else
        { 
            // otherwise, print the update operator character
            fprintf (fp, "%c", (*p)) ;
        }
    }
    fprintf (fp, "\n") ;
}

