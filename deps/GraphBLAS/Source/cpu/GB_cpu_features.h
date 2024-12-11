//------------------------------------------------------------------------------
// GB_cpu_features.h: GraphBLAS interface to Google's cpu_features package
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The following can be optionally #define'd at compile-time:
//
//  GBX86:  1 if the target architecture is x86_64,
//          0 if the target architecture is not x86_64.
//          default: #define'd by GB_compiler.h.
//
//  GBAVX2: 1 if the target architecture is x86_64 and supports AVX2,
//          0 otherwise.
//          default: left undefined and cpu_features/GetX86Info is used
//          to determine this feature at run-time.
//
//  GBAVX512F: 1 if the target architecture is x86_64 and supports AVX512F
//          0 otherwise.
//          default: left undefined and cpu_features/GetX86Info is used
//          to determine this feature at run-time.
//
//  GBNCPUFEAT: if #define'd then the Google cpu_features package is not used.
//          The run-time tests for AVX2 and AVX512F are replaced with
//          compile-time tests, using GBAVX2, and GBAVX512F.  If GBAVX2 or
//          GBAVX512F macros are not #define'd externally by the build system,
//          then no AVX acceleration is used.  default: in general not
//          #define'd (using Google's cpu_features), but it is #define'd
//          for some cases by GB_compiler.h.

#ifndef GB_CPU_FEATURES_H
#define GB_CPU_FEATURES_H

#if !defined ( GBNCPUFEAT )
    #include "cpu_features_macros.h"
    #define STACK_LINE_READER_BUFFER_SIZE 1024
    #if GBX86
    // Intel x86 (also AMD): other architectures are not exploited
    #include "cpuinfo_x86.h"
    #endif
#endif

#endif

