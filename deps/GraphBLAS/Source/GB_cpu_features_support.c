//------------------------------------------------------------------------------
// GB_cpu_features_support.c: Google's cpu_features package for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_compiler.h"
#include "GB_cpu_features.h"

#if !defined ( GBNCPUFEAT )

    // include the support files from cpu_features/src/*.c
    #include "filesystem.c"
    #include "hwcaps.c"
    #include "stack_line_reader.c"
    #include "string_view.c"

#endif

