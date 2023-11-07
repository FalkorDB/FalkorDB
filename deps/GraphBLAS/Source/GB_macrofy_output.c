//------------------------------------------------------------------------------
// GB_macrofy_output: construct a macro to store values to an output matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The macro, typically called GB_PUTC, also does typecasting from the Z
// type of the monoid or operator, into the type of the C matrix.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_output
(
    FILE *fp,
    // input:
    const char *cname,      // name of the scalar ... = cij to write
    const char *Cmacro,     // name of the macro is GB_PUT*(Cmacro)
    const char *Cname,      // name of the output matrix
    GrB_Type ctype,         // type of C, ignored if C is iso
    GrB_Type ztype,         // type of cij scalar to cast to ctype write to C
    int csparsity,          // sparsity format of the output matrix
    bool C_iso,             // true if C is iso on output
    bool C_in_iso           // true if C is iso on input
)
{

    //--------------------------------------------------------------------------
    // construct the matrix status macros: iso, type name, type size
    //--------------------------------------------------------------------------

    GB_macrofy_sparsity (fp, Cname, csparsity) ;
    GB_macrofy_nvals (fp, Cname, csparsity, C_iso && C_in_iso) ;
    fprintf (fp, "#define GB_%s_ISO %d\n", Cname, C_iso ? 1 : 0) ;
    fprintf (fp, "#define GB_%s_IN_ISO %d\n", Cname, C_in_iso ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // construct the macros to declare scalars and put values into the matrix
    //--------------------------------------------------------------------------

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // no need to access the values of C
        //----------------------------------------------------------------------

        GB_macrofy_type (fp, Cname, "_", "void") ;
        fprintf (fp, "#define GB_PUT%s(%s,%sx,p)\n", Cmacro, cname, Cname) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // construct the GB_PUTC macro
        //----------------------------------------------------------------------

        // #define GB_PUTC(c,Cx,p) Cx [p] = (ctype) c
        // to store a value into the C matrix, typecasting it from ztype to
        // ctype.  If C is iso, the GB_PUTC macro is empty.

        // For example, to store the scalar cij (of type float, in the example
        // above, into the matrix C of type double:

        //      GB_PUTC (cij,Cx,p) ;

        // becomes:

        //      Cx [p] = (double) cij ;

        // or, if C is iso: nothing happens; the macro is empty.

        GB_macrofy_type (fp, Cname, "_", ctype->name) ;

        #define SLEN 256
        char macro_name [SLEN+1], xargs [SLEN+1], xexpr [SLEN+1] ;
        snprintf (macro_name, SLEN, "GB_PUT%s", Cmacro) ;
        snprintf (xargs, SLEN, "%sx,p", Cname) ;
        snprintf (xexpr, SLEN, "%sx [p]", Cname) ;
        GB_macrofy_cast_output (fp, macro_name, cname, xargs, xexpr, ztype,
            ctype) ;
    }
}

