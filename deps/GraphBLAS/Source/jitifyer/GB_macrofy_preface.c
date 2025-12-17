//------------------------------------------------------------------------------
// GB_macrofy_preface: print the kernel preface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_preface
(
    FILE *fp,               // target file to write, already open
    char *kernel_name,      // name of the kernel
    char *C_preface,        // user-provided preface for CPU JIT kernels
    char *CUDA_preface,     // user-provided preface for CUDA JIT kernels
    GB_jit_kcode kcode,
    uint8_t major,          // CUDA compute capability
    uint8_t minor
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
        "----------------------------------------\n",
        kernel_name,
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB,
        date + GB_IMAX (0, len - 4)) ;

    if (kcode >= GB_JIT_CUDA_KERNEL)
    {
        // for CUDA JIT kernels
        fprintf (fp, "#define GB_CUDA_COMPUTE_MAJOR %d\n", major) ;
        fprintf (fp, "#define GB_CUDA_COMPUTE_MINOR %d\n", minor) ;
        fprintf (fp, "#define GB_CUDA_KERNEL\n%s\n", CUDA_preface) ;
    }
    else
    {
        // CPU JIT kernels
        fprintf (fp, "%s\n", C_preface) ;
    }

    // for all kernels: CPU and CUDA
    fprintf (fp, "#include \"include/GB_jit_kernel.h\"\n\n") ;
}

