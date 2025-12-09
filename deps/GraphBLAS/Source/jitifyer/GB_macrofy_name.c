//------------------------------------------------------------------------------
// GB_macrofy_name: construct the name for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
// where "012345" is a hexadecimal printing of the encoding->code.  Note the
// double underscores (2 or 3 of them).  These are used by GB_demacrofy_name
// for parsing the kernel_name of a PreJIT kernel.
//
// The suffix is used only for user-defined types and operators.
//
// For CUDA kernels, the major/minor compute capability is also encoded in the
// name.  For example, if the target is sm_72, then the name will be one of:
//
//      namespace__kname__012345_72__suffix
//      namespace__kname__012345_72
//
// where kname also always prefaced with the string "cuda_".  The filename
// suffix (.c for CPU kernels, or .cu for CUDA kernels) is part of the
// kernel_name.

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_name
(
    // output:
    char *kernel_name,      // string of length GB_KLEN
    // input
    const char *name_space, // namespace for the kernel_name
    const char *kname,      // kname for the kernel_name
    int method_code_digits, // # of hexadecimal digits printed
    GB_jit_encoding *encoding,  // encoding of the kernel
    const char *suffix      // suffix for the kernel_name (NULL if none)
)
{
    if (suffix == NULL)
    { 
        // kernel uses only built-in types and operators
        #if defined ( GRAPHBLAS_HAS_CUDA )
        if (encoding->kcode >= GB_JIT_CUDA_KERNEL)
        {
            snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64 "_%d%d",
                name_space, kname, method_code_digits, encoding->code,
                (int) encoding->major, (int) encoding->minor) ;
        }
        else
        #endif
        {
            snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64,
                name_space, kname, method_code_digits, encoding->code) ;
        }
    }
    else
    { 
        // kernel uses at least one user-defined type and/or operator
        #if defined ( GRAPHBLAS_HAS_CUDA )
        if (encoding->kcode >= GB_JIT_CUDA_KERNEL)
        {
            snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64 "_%d%d__%s",
                name_space, kname, method_code_digits, encoding->code,
                (int) encoding->major, (int) encoding->minor, suffix) ;
        }
        else
        #endif
        {
            snprintf (kernel_name, GB_KLEN-1, "%s__%s__%0*" PRIx64 "__%s",
                name_space, kname, method_code_digits, encoding->code, suffix) ;
        }
    }
}

