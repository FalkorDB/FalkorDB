/*
 * Copyright (c) 2019,2020 NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GB_JIT_LAUNCHER_H
#define GB_JIT_LAUNCHER_H

#include <GB_jit_cache.h>
#include <unordered_map>
#include <memory>
#include <string>
#include <fstream>

#undef  JITIFY_PRINT_INSTANTIATION
#define JITIFY_PRINT_INSTANTIATION 0
#undef  JITIFY_PRINT_SOURCE
#define JITIFY_PRINT_SOURCE 1
#undef  JITIFY_PRINT_LOG
#define JITIFY_PRINT_LOG 1
#undef  JITIFY_PRINT_PTX
#define JITIFY_PRINT_PTX 1
#undef  JITIFY_PRINT_LINKER_LOG
#define JITIFY_PRINT_LINKER_LOG 0
#undef  JITIFY_PRINT_LAUNCH
#define JITIFY_PRINT_LAUNCH 1
#include "jitify.hpp"


namespace jit {

#if 0
const std::vector<std::string> GB_jit_cuda_compiler_flags{
    "-std=c++17",
            "--use_fast_math",
            "-remove-unused-globals",
            "-w",
            "-lcudart",
            "-D__CUDACC_RTC__",
            "-I.",
            "-I..",
            "-I../../Include",
            "-I../../Source",
            "-I../../Source/Template",
//          "-Ilocal_cub/block",
            "-Itemplates",
            "-I/usr/local/cuda/include" // FIXME: remove this?
};
#endif


/**
 * @brief Class used to handle compilation and execution of JIT kernels
 * 
 */
class launcher {
 public:
  launcher() = delete;
   
  /**
   * @brief C'tor of the launcher class
   * 
   * Method to generate vector containing all template types for a JIT kernel.
   *  This vector is used to get the compiled kernel for one set of types and set
   *  it as the kernel to launch using this launcher.
   * 
   * @param hash The hash to be used as the key for caching
   * @param cuda_code The CUDA code that contains the kernel to be launched
   * @param header_names Strings of header_names or strings that contain content
   * of the header files
   * @param compiler_flags Strings of compiler flags
   * @param file_callback a function that returns header file contents given header
   * file names.
   * @param stream The non-owned stream to use for execution
   */
  launcher(
    const std::string& hash,
    const std::string& cuda_source,
    const std::vector<std::string>& header_names,
    const std::vector<std::string>& compiler_flags,
    jitify::experimental::file_callback_type file_callback,
    cudaStream_t stream = 0
  );       
  launcher(launcher&&);
  launcher(const launcher&) = delete;
  launcher& operator=(launcher&&) = delete;
  launcher& operator=(const launcher&) = delete;

  /**
   * @brief Sets the kernel to launch using this launcher
   * 
   * Method to generate vector containing all template types for a JIT kernel.
   *  This vector is used to get the compiled kernel for one set of types and set
   *  it as the kernel to launch using this launcher.
   * 
   * @param kernel_name The kernel to be launched
   * @param arguments   The template arguments to be used to instantiate the kernel
   * @return launcher& ref to this launcehr object
   */
  launcher& set_kernel_inst(
    const std::string& kernel_name,
    const std::vector<std::string>& arguments
  )
  { // program is a member variable of the launcher
    kernel_inst = cache_instance.getKernelInstantiation(kernel_name, program, arguments);
    return *this;
  }

  /**
   * @brief Handle the Jitify API to launch using information 
   *  contained in the members of `this`
   * 
   * @tparam grid and block sizes 
   * @return Return launcher reference if successful
   */
  jitify::experimental::KernelLauncher configure( dim3 grid, dim3 block, unsigned int smem = 0, cudaStream_t stream = 0){
    return get_kernel().configure( grid, block, smem, stream);
    //return get_kernel().configure_1d_max_occupancy( max_block_size=block.x); 
  }


  /**
   * @brief Handle the Jitify API to launch using information 
   *  contained in the members of `this`
   * 
   * @tparam All parameters to launch the kernel
   * @return Return GDF_SUCCESS if successful
   */
  template <typename ... Args>
  void launch(Args ... args) {
    get_kernel().configure_1d_max_occupancy(32, 0, 0, stream).launch(args...);
  }

 private:
  jit::GBJitCache& cache_instance;
  jit::named_prog<jitify::experimental::Program> program;
  jit::named_prog<jitify::experimental::KernelInstantiation> kernel_inst;
  cudaStream_t stream;

  jitify::experimental::KernelInstantiation& get_kernel() { return *std::get<1>(kernel_inst); }
};

} // namespace jit

#endif
