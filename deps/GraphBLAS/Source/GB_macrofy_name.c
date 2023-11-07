//------------------------------------------------------------------------------
// GB_macrofy_name: construct the name for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The kernel name has the following form, if the suffix is non-NULL:
//
//      namespace__kname__012345__suffix
//
// or, when suffix is NULL:
//
//      namespace__kname__012345
//
// where "012345" is a hexadecimal printing of the scode.  Note the double
// underscores (2 or 3 of them).  These are used by GB_demacrofy_name for
// parsing the kernel_name of a PreJIT kernel.
//
// The suffix is used only for user-defined types and operators.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_name
(
    // output:
    char *kernel_name,      // string of length GB_KLEN
    // input
    const char *name_space, // namespace for the kernel_name
    const char *kname,      // kname for the kernel_name
    int scode_digits,       // # of hexadecimal digits printed
    uint64_t scode,         // enumify'd code of the kernel
    const char *suffix      // suffix for the kernel_name (NULL if none)
)
{
    if (suffix == NULL)
    { 
        // kernel uses only built-in types and operators
        snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64,
            name_space, kname, scode_digits, scode) ;
    }
    else
    { 
        // kernel uses at least one built-in types and/or operator
        snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64 "__%s",
            name_space, kname, scode_digits, scode, suffix) ;
    }
}

