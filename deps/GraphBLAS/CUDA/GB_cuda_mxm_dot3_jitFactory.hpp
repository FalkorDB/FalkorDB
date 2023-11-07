//------------------------------------------------------------------------------
// GB_cuda_mxm_dot3_factory
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//------------------------------------------------------------------------------

#ifndef GB_MXM_DOT3_JITFACTORY_H
#define GB_MXM_DOT3_JITFACTORY_H

#pragma once

/**
 * This file is responsible for picking all the parameters and what kernel variaiton we will use for a given instance
 * - data types
 * - semiring types
 * - binary ops
 * - monoids
 *
 * Kernel factory says "Here's the actual instance I want you to build with the given parameters"
 */


//AxB_dot3_phase1 kernel launchers
template<int threads_per_block, int chunk_size> class phase1launchFactory ;
template<int threads_per_block, int chunk_size> class dense_phase1launchFactory ;

//AxB_dot3_phase3 kernel launchers

//------------------------------------------------------------------------------
// dot3: dense_phase1launchFactory 
//------------------------------------------------------------------------------

// Handles full/bitmap cases, which means we don't need buckets and zombies.
// This is a much simpler kernel as a result, it only does the i,j lookup 
// and stores the values in Mi and Ci. 


template<int threads_per_block=32, int chunk_size = 128>
class dense_phase1launchFactory 
{
  // FIXME: this is the full name.  Why?  See below for partial name.
  // Need to be consistent in naming schemes.
  std::string kernel_name = "GB_jit_AxB_dot3_dense_phase1";

  GB_cuda_mxm_factory &mxm_factory_;

public:

  int get_number_of_blocks(GrB_Matrix M) {
      int number_of_sms = GB_Global_gpu_sm_get (0);
      int nblks = ( GB_nnz (M) + chunk_size - 1)/chunk_size;
      return GB_IMIN( nblks,  chunk_size * number_of_sms);
  }

  int get_threads_per_block() {
      return threads_per_block;
  }

  // This assumes the needed state on the GB_cuda_mxm_factory
  // has already been populated
  dense_phase1launchFactory(GB_cuda_mxm_factory &mxm_factory): mxm_factory_(mxm_factory){}

  bool jitGridBlockLaunch( GrB_Matrix C, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, cudaStream_t stream = 0) {

    // Idea is to have each task work on a continguous block of columns of C
    // Note: for small tests, mnz is small so ntasks is be governed by
    // chunksize, not chunk_size*number_of_sms.  For large problems in
    // production, chunksize is less important since ntasks will likely be
    // bounded by chunk_size*number_of_sms (say 128*80 = 10,240 on a V100, for
    // the default chunk_size of 128).

    // Defining dummy instance only so we can introspect type
//    // (1) create the mxm code and name

//    // (2) ensure the jitifier has "GB_mxm_[mymxm.sr_code].h"
    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mxm_factory_) ;

    uint64_t sr_code = mxm_factory_.sr_code  ;
    int mask_ecode = GB_RSHIFT (sr_code, 20, 4) ;
    bool mask_no_type = (mask_ecode < 4) ;
    auto sr_code_str = std::to_string(sr_code) ;
    std::vector<std::string> template_types = {
        (mask_no_type) ? "bool" : M->type->name, sr_code_str };

    std::stringstream string_to_be_jitted ;

    // FIXME: why does "templates/" appear here but not elsewhere?
    string_to_be_jitted << kernel_name << std::endl <<
    R"(#include "GB_cuda_kernel.h")" << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << mxm_factory_.filename << R"(")" << std::endl <<
    R"(#include "templates/)" << kernel_name << R"(.cuh")" << std::endl;

    bool result = false;

    dim3 grid(get_number_of_blocks(M));
    dim3 block(get_threads_per_block());

    jit::launcher( kernel_name + "_" + sr_code_str + ".jtfy",
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback /* FIXME: make NULL */)
                 .set_kernel_inst(  kernel_name, template_types)
                 .configure(grid, block, SMEM, stream)
                 .launch( C, M);

      result = true;

      return result;
     }
};

//------------------------------------------------------------------------------
// dot3: phase1launchFactory 
//------------------------------------------------------------------------------

// FIXME: We probably want to remove this type template altogether and provide
// a macro/function that can convert from a GrB_Type instance to the name of a
// type that the jitifier will accept.

template<int threads_per_block=32, int chunk_size = 128>
class phase1launchFactory 
{
  std::string kernel_name = "GB_jit_AxB_dot3_phase1";

  GB_cuda_mxm_factory &mxm_factory_;

public:

  int get_number_of_blocks(GrB_Matrix M) {
      int number_of_sms = GB_Global_gpu_sm_get (0);
      int nblks = ( GB_nnz (M) + chunk_size - 1)/chunk_size;
      return GB_IMIN( nblks,  chunk_size * number_of_sms);
  }

  int get_threads_per_block() {
      return threads_per_block;
  }

  // This assumes the needed state on the GB_cuda_mxm_factory
  // has already been populated
  phase1launchFactory(GB_cuda_mxm_factory &mxm_factory): mxm_factory_(mxm_factory){}

  bool jitGridBlockLaunch(int64_t *nanobuckets, int64_t *blockBucket,
                          GrB_Matrix C, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, cudaStream_t stream = 0) {

    // Idea is to have each task work on a continguous block of columns of C
    // Note: for small tests, mnz is small so ntasks is be governed by
    // chunksize, not chunk_size*number_of_sms.  For large problems in
    // production, chunksize is less important since ntasks will likely be
    // bounded by chunk_size*number_of_sms (say 128*80 = 10,240 on a V100, for
    // the default chunk_size of 128).

    // Defining dummy instance only so we can introspect type
//    // (1) create the mxm code and name

//    // (2) ensure the jitifier has "GB_mxm_[mymxm.sr_code].h"
    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mxm_factory_) ;

    uint64_t sr_code = mxm_factory_.sr_code ;
    int mask_ecode = GB_RSHIFT (sr_code, 20, 4) ;
    bool mask_no_type = (mask_ecode < 4) ;
    auto sr_code_str = std::to_string(sr_code) ;
    std::vector<std::string> template_types = {
        (mask_no_type) ? "bool" : M->type->name, sr_code_str };

    std::stringstream string_to_be_jitted ;

    // FIXME: why does "templates/" appear here but not elsewhere?
    string_to_be_jitted << kernel_name << std::endl <<
    R"(#include "GB_cuda_kernel.h")" << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << mxm_factory_.filename << R"(")" << std::endl <<
    R"(#include "templates/)" << kernel_name << R"(.cuh")" << std::endl;

    bool result = false;
    
 // std::cout<< "GB_jit_cuda_compiler_flags ";
 // for ( std::string s : GB_jit_cuda_compiler_flags)
 //     std::cout<< s<< std::endl; 

    dim3 grid(get_number_of_blocks(M));
    dim3 block(get_threads_per_block());

    jit::launcher( kernel_name + "_" + sr_code_str + ".jtfy",
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback)
                 .set_kernel_inst(  kernel_name, template_types)
                 .configure(grid, block, SMEM, stream)
                 .launch( nanobuckets, blockBucket, C, M, A, B);

      result = true;

      return result;
     }
};

//------------------------------------------------------------------------------
// dot3: phase2launchFactory
//------------------------------------------------------------------------------

template<int threads_per_block = 32, int chunk_size = 128>
class phase2launchFactory
{

  std::string base_name = "GB_jit";
  // FIXME: this is the partial name.  Why?  See above.
  std::string kernel_name = "AxB_phase2";

public:

  int get_threads_per_block() {
        return threads_per_block;
  }

  int get_number_of_blocks(GrB_Matrix M) {
    const int64_t mnz = GB_nnz (M) ;
    int ntasks = ( mnz +chunk_size -1)/chunk_size;
    // Idea is to have each task work on a continguous block of columns of C
    ntasks = GB_IMIN( ntasks,  chunk_size*GB_Global_gpu_sm_get (0)) ;    // ntasks will be grid.x
    return (ntasks + threads_per_block - 1) / threads_per_block ;
  }

  int get_number_of_phase1_blocks( GrB_Matrix M){
    const int64_t mnz = GB_nnz (M) ;
    int number_of_sms = GB_Global_gpu_sm_get (0);
    int nblks = ( GB_nnz (M) + chunk_size - 1)/chunk_size;
    return GB_IMIN( nblks,  chunk_size * number_of_sms);
  }

  bool jitGridBlockLaunch(// parameters to AxB_phase2:
                          int64_t *blockBucket, int64_t *offset, GrB_Matrix M, cudaStream_t stream = 0) {

    bool result = false;

      dim3 grid(get_number_of_blocks(M));
      dim3 block(get_threads_per_block());

      std::string hashable_name = base_name + "_" + kernel_name;
      std::stringstream string_to_be_jitted ;
      string_to_be_jitted << hashable_name << std::endl <<
        R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

      const int64_t mnz = GB_nnz (M) ;
      jit::launcher( hashable_name,
                     string_to_be_jitted.str(),
                     header_names,
                     GB_jit_cuda_compiler_flags,
                     file_callback)
                   .set_kernel_inst( kernel_name, {})
                   .configure(grid, block, SMEM, stream)
                   // parameters to AxB_phase2:
                   .launch( blockBucket, offset, get_number_of_phase1_blocks(M));

      result= true;

      return result;
     }

};

//------------------------------------------------------------------------------
// dot3: phase2endlaunchFactory
//------------------------------------------------------------------------------

template< int threads_per_block = 32, int chunk_size = 128>
class phase2endlaunchFactory
{

  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_phase2end";

public:

  int get_threads_per_block() {
        return threads_per_block;
  }

  int get_number_of_blocks(GrB_Matrix M) {
    const int64_t mnz = GB_nnz (M) ;
    int ntasks = ( mnz +chunk_size -1)/chunk_size;
    int number_of_sms = GB_Global_gpu_sm_get (0);

    // Idea is to have each task work on a continguous block of columns of C
    return GB_IMIN( ntasks,  chunk_size*number_of_sms) ;    // ntasks will be grid.x
  }

  bool jitGridBlockLaunch(int64_t *nanobuckets, int64_t *blockBucket,
                          int64_t *bucketp, int64_t *bucket, int64_t *offset,
                          GrB_Matrix C, GrB_Matrix M, cudaStream_t stream = 0)
     {

      bool result = false;

      dim3 grid(get_number_of_blocks(M));
      dim3 block(get_threads_per_block());

      std::string hashable_name = base_name + "_" + kernel_name;
      std::stringstream string_to_be_jitted ;
      string_to_be_jitted << hashable_name << std::endl <<
        R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

      jit::launcher( hashable_name,
                     string_to_be_jitted.str(),
                     header_names,
                     GB_jit_cuda_compiler_flags,
                     file_callback)
                   .set_kernel_inst(  kernel_name , {})
                   .configure(grid, block, SMEM, stream)
                   .launch( nanobuckets, blockBucket, bucketp, bucket, offset, C, GB_nnz (M));

      result= true;

      return result;
     }

};


//------------------------------------------------------------------------------
// dot3: mxm_dense_launchFactory
//------------------------------------------------------------------------------

class mxm_dense_launchFactory
{
  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_dot3_phase3_dndn";

  GB_cuda_mxm_factory &mxm_factory_;

public:

  /**
   * This assumes the needed state on the GB_cuda_mxm_factory has already been populated.
   * The `bucket_code` determines which kernel is launched
   */
  mxm_dense_launchFactory(GB_cuda_mxm_factory &mymxmfactory):
      mxm_factory_(mymxmfactory) {}

  bool jitGridBlockLaunch( GrB_Matrix C,  GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
                          cudaStream_t stream = 0) {

      bool result = false;

    //----------------------------------------------------------------------
    // do the numerical work
    //----------------------------------------------------------------------

    const int64_t nz = GB_nnz(M); // number of dots in the mask
    const int64_t mnvec = M->nvec ;

    int gridsz, blocksz;

    std::stringstream final_kernel_name_ss;
    final_kernel_name_ss << kernel_name;

    /**
     * Configure geometry and kernel function name based on sparsity of C and number of vectors in M
     */
    configure( nz, mnvec, final_kernel_name_ss, blocksz, gridsz);

    auto sr_code = std::to_string(mxm_factory_.sr_code);    // FIXME: make hexadecimal

    GrB_BinaryOp mult = mxm_factory_.semiring->multiply ;

    std::string hashable_name = base_name + "_" + final_kernel_name_ss.str();
    std::stringstream string_to_be_jitted ;
    std::vector<std::string> template_types =
    {
        C->type->name, A->type->name, B->type->name,
        mult->ztype->name, mult->xtype->name, mult->ytype->name,
        sr_code
    };

    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mxm_factory_) ;

    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include "GB_cuda_kernel.h")" << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << mxm_factory_.filename << R"(")" << std::endl <<
    R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

    dim3 grid(gridsz);
    dim3 block(blocksz);

    GBURBLE ("(GPU dot3 mxm dense launch nblocks,blocksize= %d,%d )\n", gridsz,blocksz) ;
    jit::launcher( hashable_name + "_" + sr_code,
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback)
               .set_kernel_inst(final_kernel_name_ss.str(), template_types )
                               // { C->type->name,
                               //   A->type->name,
                               //   B->type->name })
               .configure(grid, block, SMEM, stream) //if commented, use implicit 1D configure in launch
               .launch(
                        C,                 // final output matrix
                                           // inputs, not modified:
                        M,                 // Mi used for column index
                        A,                 // A matrix
                        B                  // B matrix
                    );

    result= true;

    return result;
  }

private:
    void configure(std::int64_t Cnz, std::int64_t mnvec, std::stringstream &opname,
                   int &blocksz, int &gridsz) {
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    int work_per_thread;

     blocksz = 64;
     work_per_thread = 8;
     
     if( Cnz > 1024){
       blocksz = 512;
       work_per_thread = 64;
     }

     // gridsz = ceiling (Cnz / work_per_thread*blocksz)
     gridsz = GB_ICEIL (Cnz, work_per_thread*blocksz) ;

  }
};

//------------------------------------------------------------------------------
// FIXME: rename GB_cuda_mxm_dot3_jitFactory_sparse_dense_launchFactory
//------------------------------------------------------------------------------

class mxm_sparse_dense_launchFactory
{
  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_dot3";

  GB_cuda_mxm_factory &mxm_factory_;

public:

  /**
   * This assumes the needed state on the GB_cuda_mxm_factory has already been populated.
   * The `bucket_code` determines which kernel is launched
   */
  mxm_sparse_dense_launchFactory(GB_cuda_mxm_factory &mymxmfactory):
      mxm_factory_(mymxmfactory) {}

  bool jitGridBlockLaunch( GrB_Matrix C,  GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
                          cudaStream_t stream = 0) {

      bool result = false;

    //----------------------------------------------------------------------
    // do the numerical work
    //----------------------------------------------------------------------

    const int64_t nz = GB_nnz(M); // number of dots in the mask
    const int64_t mnvec = M->nvec ;

    int gridsz, blocksz;

    std::stringstream final_kernel_name_ss;
    final_kernel_name_ss << kernel_name;

    /**
     * Configure geometry and kernel function name based on sparsity of C and number of vectors in M
     */
    configure( nz, mnvec, final_kernel_name_ss, blocksz, gridsz);

    auto sr_code = std::to_string(mxm_factory_.sr_code);

    GrB_BinaryOp mult = mxm_factory_.semiring->multiply ;

    std::string hashable_name = base_name + "_" + final_kernel_name_ss.str();
    std::stringstream string_to_be_jitted ;
    std::vector<std::string> template_types =
    {
        C->type->name, A->type->name, B->type->name,
        mult->ztype->name, mult->xtype->name, mult->ytype->name,
        sr_code
    };

    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mxm_factory_) ;

    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include "GB_cuda_kernel.h")" << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << mxm_factory_.filename << R"(")" << std::endl <<
    R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

    dim3 grid(gridsz);
    dim3 block(blocksz);

    GBURBLE ("(GPU dot3 mxm sparse_dense launch nblocks,blocksize= %d,%d )\n", gridsz,blocksz) ;
    jit::launcher( hashable_name + "_" + sr_code,
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback)
               .set_kernel_inst(final_kernel_name_ss.str(), template_types )
                               // { C->type->name,
                               //   A->type->name,
                               //   B->type->name })
               .configure(grid, block, SMEM, stream) //if commented, use implicit 1D configure in launch
               .launch(
                        C,                 // final output matrix
                                           // inputs, not modified:
                        M,                 // Mi used for column index
                        A,                 // A matrix
                        B                  // B matrix
                    );

    result= true;

    return result;
  }

private:
    void configure(std::int64_t Cnz, std::int64_t mnvec, std::stringstream &opname,
                   int &blocksz, int &gridsz) {
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    int work_per_thread;

     blocksz = 64;
     work_per_thread = 8;
     
     if( Cnz > 1024){
       blocksz = 512;
       work_per_thread = 64;
     }

     // gridsz = ceiling (Cnz / work_per_thread*blocksz)
     gridsz = GB_ICEIL (Cnz, work_per_thread*blocksz) ;

  }
};

//------------------------------------------------------------------------------
// dot3: phase3launchFactory
//------------------------------------------------------------------------------

class phase3launchFactory
{
  std::string base_name = "GB_jit";
  std::string kernel_name = "AxB_dot3";

  GB_cuda_mxm_factory &mxm_factory_;

  GB_bucket_code bucket_code_;

public:

   std::string Opname;

  /**
   * This assumes the needed state on the GB_cuda_mxm_factory has already been populated.
   * The `bucket_code` determines which kernel is launched
   */
  phase3launchFactory(GB_cuda_mxm_factory &mymxmfactory, GB_bucket_code bucket_code):
      mxm_factory_(mymxmfactory), bucket_code_(bucket_code) {}

  bool jitGridBlockLaunch(int64_t start, int64_t end, int64_t *bucketp, int64_t *bucket,
                          GrB_Matrix C,  GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
                          cudaStream_t stream = 0) {

      bool result = false;

    //----------------------------------------------------------------------
    // phase3: do the numerical work
    //----------------------------------------------------------------------

    const int64_t nz = end - start; // number of dots in this bucket  
    const int64_t mnvec = M->nvec ;

    int gridsz, blocksz, sz = 4;

    std::stringstream final_kernel_name_ss;
    final_kernel_name_ss << kernel_name << "_";

    /**
     * Configure geometry and kernel function name based on sparsity of C and number of vectors in M
     */
    auto sr_code = std::to_string(mxm_factory_.sr_code);

    configure2( nz, mnvec, final_kernel_name_ss, blocksz, gridsz, sz, mxm_factory_.sr_code);


    GrB_BinaryOp mult = mxm_factory_.semiring->multiply ;

    std::string hashable_name = base_name + "_" + final_kernel_name_ss.str();
    std::stringstream string_to_be_jitted ;
    std::vector<std::string> template_types =
    {
        C->type->name, A->type->name, B->type->name,
        mult->ztype->name, mult->xtype->name, mult->ytype->name,
        sr_code
    };

    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (mxm_factory_) ;

    // FIXME: why is "hashable_name" used sometimes, and sometimes "kernel_name"?
    string_to_be_jitted << hashable_name << std::endl <<
    R"(#include "GB_cuda_kernel.h")" << std::endl <<
    R"(#include ")" << jit::get_user_home_cache_dir() << "/" << mxm_factory_.filename << R"(")" << std::endl <<
    R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

    dim3 grid(gridsz);
    dim3 block(blocksz);

    GBURBLE ("(GPU phase3 launch %s st,end=%ld,%ld nblocks,blocksize= %d,%d )\n", this->Opname.c_str(),
              start,end,gridsz,blocksz) ;
    jit::launcher( hashable_name + "_" + sr_code,
                   string_to_be_jitted.str(),
                   header_names,
                   GB_jit_cuda_compiler_flags,
                   file_callback)
               .set_kernel_inst(final_kernel_name_ss.str(), template_types )
                               // { C->type->name,
                               //   A->type->name,
                               //   B->type->name })
               .configure(grid, block, SMEM, stream) //if commented, use implicit 1D configure in launch
               .launch(
                        start,             // input/output:
                        end,               // global bucket cumsum, of size NBUCKETS+1
                        bucket,            // global buckets, of size cnz (== mnz)
                        C,                 // final output matrix
                                           // inputs, not modified:
                        M,                 // Mi used for column index
                        A,                 // A matrix
                        B,                 // B matrix
                        sz                 // only used for sparse-sparse cases
                    );

    result= true;

    return result;
  }

private:
    void configure2(std::int64_t Cnz, std::int64_t mnvec, std::stringstream &opname,
                   int &blocksz, int &gridsz, int &sz, uint64_t sr_code) {
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    int work_per_thread;

    // 0:hyper, 1:sparse, 2:bitmap, 3:full
    int asparsity   = GB_RSHIFT (sr_code,  2, 2) ;
    int bsparsity   = GB_RSHIFT (sr_code,  0, 2) ;

    if (asparsity <= 1 && bsparsity <= 1)
    {
        // both A and B are sparse/hyper
        switch (bucket_code_)
        {

            //--------------------------------------------------------------
            // not a bucket ... bring out your dead:
            //--------------------------------------------------------------

            case GB_BUCKET_ZOMBIE : // C(i,j) is a zombie (not a bucket)
                break ;

            //--------------------------------------------------------------
            // CUDA kernel: vsvs bucket:
            //--------------------------------------------------------------

            case GB_BUCKET_VSVS :
                Opname = "phase3_vsvs" ;
                blocksz = 256;
                work_per_thread = 4;
                
                if( Cnz > (2<<12)){
                  blocksz = 512;
                  work_per_thread = 4;
                }

                // gridsz = ceiling (Cnz / work_per_thread*blocksz)
                gridsz = GB_ICEIL (Cnz, work_per_thread*blocksz) ;
                if (gridsz > 256*number_of_sms)  gridsz = 256*number_of_sms;
                break ;

            //--------------------------------------------------------------
            // CUDA kernel: mp, use the merge-path method:
            //--------------------------------------------------------------

            case GB_BUCKET_MERGEPATH :
                Opname = "phase3_mp" ;
                blocksz = 32;
                work_per_thread = 256 ;

                if( Cnz > (2<<20)){
                  work_per_thread = 1024;
                }
                gridsz = GB_ICEIL (Cnz, work_per_thread) ;
                if ((gridsz < number_of_sms) && (Cnz > (2<<20)))
                {
                   gridsz = number_of_sms; 
                }
                if (gridsz > 256*number_of_sms)  gridsz = 256*number_of_sms;
                break ;

            default:
                break ;
        }

    }
    else
    {
        // either A or B are bitmap/full
        switch (bucket_code_)
        {

            //--------------------------------------------------------------
            // not a bucket ... bring out your dead:
            //--------------------------------------------------------------

            case GB_BUCKET_ZOMBIE : // C(i,j) is a zombie (not a bucket)
                break ;

            //--------------------------------------------------------------
            // CUDA kernel: vsdn bucket:  one thread per C(i,j) dot product
            //--------------------------------------------------------------

            case GB_BUCKET_VSDN :
                Opname = "phase3_vsdn" ;

                // FIXME:
                blocksz = 256;
                work_per_thread = 4;
                
                if( Cnz > (2<<12)){
                  blocksz = 512;
                  work_per_thread = 4;
                }

                // gridsz = ceiling (Cnz / work_per_thread*blocksz)
                gridsz = GB_ICEIL (Cnz, work_per_thread*blocksz) ;
                if (gridsz > 256*number_of_sms)  gridsz = 256*number_of_sms;
                break ;

            //--------------------------------------------------------------
            // CUDA kernel: spdn bucket: one warp per C(i,j) dot product
            //--------------------------------------------------------------

            case GB_BUCKET_SPDN :
                Opname = "phase3_spdn" ;

                // FIXME:
                blocksz = 32;
                work_per_thread = 256 ;

                if( Cnz > (2<<20)){
                  work_per_thread = 1024;
                }
                gridsz = GB_ICEIL (Cnz, work_per_thread) ;
                if ((gridsz < number_of_sms) && (Cnz > (2<<20)))
                {
                   gridsz = number_of_sms; 
                }
                if (gridsz > 256*number_of_sms)  gridsz = 256*number_of_sms;
                break ;

            default:
                break ;
        }

    }

    opname << Opname;
  }
};

#endif
