//------------------------------------------------------------------------------
// GB_macrofy_preface: print the kernel preface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_preface
(
    FILE *fp,               // target file to write, already open
    char *kernel_name,      // name of the kernel
    char *preface           // user-provided preface
)
{ 

    const char *date = GxB_IMPLEMENTATION_DATE ;
    int len = (int) strlen (date) ;
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n"
        "// %s.c\n"
        "//--------------------------------------"
        "----------------------------------------\n"
        "// SuiteSparse:GraphBLAS v%d.%d.%d, Timothy A. Davis, (c) 2017-%s,\n"
        "// All Rights Reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// The above copyright and license do not apply to any\n"
        "// user-defined types and operators defined below.\n"
        "//--------------------------------------"
        "----------------------------------------\n"
        "%s\n"
        "#include \"GB_jit_kernel.h\"\n\n",
        kernel_name,
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB,
        date + GB_IMAX (0, len - 4),
        preface) ;
}

