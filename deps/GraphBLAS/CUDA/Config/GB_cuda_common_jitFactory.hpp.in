//------------------------------------------------------------------------------
// GB_cuda_common_jitFactory.hpp: common defines for all jitFactory classes
//------------------------------------------------------------------------------

// (c) Nvidia Corp. 2020 All rights reserved
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Common defines for all jitFactory classes:
// iostream callback to deliver the buffer to jitify as if read from a file
// compiler flags
// Include this file along with any jitFactory you need.

// NOTE: do not edit the GB_cuda_common_jitFactory.hpp directly.  It is
// configured by cmake from the following file:
// GraphBLAS/CUDA/Config/GB_cuda_common_jitFactory.hpp.in

#ifndef GB_COMMON_JITFACTORY_H
#define GB_COMMON_JITFACTORY_H

#pragma once

extern "C"
{
    #include "GB.h"
    #include "GraphBLAS.h"
    #include "GB_stringify.h"
}

#include <iostream>
#include <cstdint>
#include "GB_jit_cache.h"
#include "GB_jit_launcher.h"
#include "GB_cuda_mxm_factory.hpp"
#include "GB_cuda_buckets.h"
#include "GB_cuda_type_wrap.hpp"
#include "GB_cuda_error.h"
#include "../rmm_wrap/rmm_wrap.h"
#include "GB_iceil.h"

// amount of shared memory to use in CUDA kernel launches
constexpr unsigned int SMEM = 0 ;

static const std::vector<std::string> GB_jit_cuda_compiler_flags{
   "-std=c++17",
   //"-G",
   "-remove-unused-globals",
   "-w",
   "-D__CUDACC_RTC__",
//   "-I.",
//   "-I..",
//   "-I../templates",
//   "-I../CUDA",
//   "-I../Source/Shared",

   // Add includes relative to GRAPHBLAS_SOURCE_PATH variable
   "-I" + jit::get_user_graphblas_source_path() + "/CUDA",
   "-I" + jit::get_user_graphblas_source_path() + "/Source/Shared",
   "-I" + jit::get_user_graphblas_source_path() + "/CUDA/templates",
   "-I/usr/local/cuda/include",

   // FIXME: add SUITESPARSE_CUDA_ARCHITECTURES here, via config
};

// FIXME: rename GB_jit_cuda_header_names or something
static const std::vector<std::string> header_names ={};

// FIXME: rename GB_jit_cuda_file_callback
inline std::istream* (*file_callback)(std::string, std::iostream&);

#endif
