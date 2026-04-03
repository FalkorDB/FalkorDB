//------------------------------------------------------------------------------
// GraphBLAS/CUDA/include/GB_cuda_geometry.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

//------------------------------------------------------------------------------

// CUDA kernel geometry: typically just the block size (# of threads in each
// threadblock) is #define'd here.  This file is used both in host and CUDA
// JIT kernels.

#ifndef GB_CUDA_GEOMETRY_H
#define GB_CUDA_GEOMETRY_H

// select sparse CUDA kernel
#define GB_CUDA_SELECT_SPARSE_BLOCKDIM1 512
#define GB_CUDA_SELECT_SPARSE_BLOCKDIM1_LOG2 9
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE1 4096
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE1_LOG2 12

#define GB_CUDA_SELECT_SPARSE_BLOCKDIM2 256
#define GB_CUDA_SELECT_SPARSE_BLOCKDIM2_LOG2 9
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE2 1024
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE2_LOG2 10

// select bitmap CUDA kernel
#define GB_CUDA_SELECT_BITMAP_BLOCKDIM 512
#define GB_CUDA_SELECT_BITMAP_BLOCKDIM_LOG2 9

#endif

