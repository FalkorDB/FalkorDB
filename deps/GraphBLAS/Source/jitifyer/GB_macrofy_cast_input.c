//------------------------------------------------------------------------------
// GB_macrofy_cast_input: construct a macro and defn for typecasting from input
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
    // the 14 scalar types: 13 built-in types, and one user-defined type code
    GB_ignore_code  = 0,
    GB_BOOL_code    = 1,        // 'logical' in @GrB interface
    GB_INT8_code    = 2,
    GB_UINT8_code   = 3,
    GB_INT16_code   = 4,
    GB_UINT16_code  = 5,
    GB_INT32_code   = 6,
    GB_UINT32_code  = 7,
    GB_INT64_code   = 8,
    GB_UINT64_code  = 9,
    GB_FP32_code    = 10,       // float ('single' in @GrB interface)
    GB_FP64_code    = 11,       // double
    GB_FC32_code    = 12,       // float complex ('single complex' in @GrB)
    GB_FC64_code    = 13,       // double complex
    GB_UDT_code     = 14        // void *, user-defined type
*/

// constructs a macro of the form:

//      #define macro(z,x...) z = ...

// The name of the macro is given by macro_name.
// The name of the variable z is given by zarg.
// The inputs x,... are given by xargs (which may contain a comma).
// xexpr is an expression using the xargs that produce a value of type xtype.
// z has type ztype.

// This method is very similar to GB_macrofy_cast_output.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_cast_input
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z output
    const GrB_Type xtype        // the type of the x input
)
{

    if (ztype == NULL || xtype == NULL)
    { 
        // empty macro if xtype or ztype are NULL (value not needed)
        fprintf (fp, "#define %s(%s,%s)\n", macro_name, zarg, xargs) ;
        return ;
    }

    int nargs ;
    const char *f = GB_macrofy_cast_expression (fp, ztype, xtype, &nargs) ;

    if (f == NULL)
    { 
        // C11 typecasting
        ASSERT (ztype != xtype) ;
        fprintf (fp, "#define %s(%s,%s) %s = (%s) (%s)\n",
            macro_name, zarg, xargs, zarg, ztype->name, xexpr) ;
    }
    else
    { 
        // GraphBLAS typecasting, or no typecasting
        fprintf (fp, "#define %s(%s,%s) ", macro_name, zarg, xargs) ;
        if (nargs == 3)
        { 
            fprintf (fp, f, zarg, xexpr, xexpr) ;
        }
        else
        { 
            fprintf (fp, f, zarg, xexpr) ;
        }
        fprintf (fp, "\n") ;
    }
}

