//------------------------------------------------------------------------------
// GraphBLAS/CUDA/include/GB_cuda_geometry.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

//------------------------------------------------------------------------------

// CUDA kernel geometry for all kernels, available for both the CUDA device
// kernels and host code.

// The GB_CUDA_TILE_SIZE definition is used for all kernels to define the # of
// threads in a single tile of a cooperative group.

// *BLOCKDIM* definitions are used with "dim3 block (BLOCKDIM)" declarations,
// which defines the # of threads in each threadblock.  These definitions vary
// per kernel.  The BLOCKDIM must be a multiple of the GB_CUDA_TILE_SIZE.

// *CHUNKSIZE* definitions give the amount of work in a single "chunk".  Each
// threadblock typically does many chunks using a grid-stride loop, where all
// threads in a threadblock to do cooperate one chunk at a time.  The chunksize
// controls the amount of shared memory each threadblock requires.  Some
// methods also require this size to be available in the host code, so that
// global workspace can be allocated of the right size.  These definitions vary
// per kernel.

// Most of these are exact powers of 2 to simplify the partitioning of work.
// For those cases, the log2 of the TILE_SIZE, BLOCKDIM, or CHUNKSIZE
// definitions are provided.

// FUTURE: tune this per GPU type.

#ifndef GB_CUDA_GEOMETRY_H
#define GB_CUDA_GEOMETRY_H

//------------------------------------------------------------------------------
// tile geometry for all kernels
//------------------------------------------------------------------------------

// The tile geometry defines the # of threads in a single tile of a cooperative
// group, and is used in all kernels.  All BLOCKDIM definitions below must be
// a multiple of the tile size.

#define GB_CUDA_TILE_SIZE 32        /* # of threads in a tile */
#define GB_CUDA_LOG2_TILE_SIZE 5    /* log2 of the tile size */

//------------------------------------------------------------------------------
// per-kernel geometry
//------------------------------------------------------------------------------

// select sparse CUDA kernel
#define GB_CUDA_SELECT_SPARSE_BLOCKDIM1 512
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE1 4096
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE1_LOG2 12
#define GB_CUDA_SELECT_SPARSE_BLOCKDIM2 256
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE2 1024
#define GB_CUDA_SELECT_SPARSE_CHUNKSIZE2_LOG2 10

// select bitmap CUDA kernel
#define GB_CUDA_SELECT_BITMAP_BLOCKDIM 512
#define GB_CUDA_SELECT_BITMAP_CHUNKSIZE 512

// builder CUDA kernel
#define GB_CUDA_BUILDER_BLOCKDIM 128
#define GB_CUDA_BUILDER_CHUNKSIZE 256
#define GB_CUDA_BUILDER_CHUNKSIZE_LOG2 8

// transpose CUDA prep kernel
#define GB_CUDA_TRANSPOSE_PREP_BLOCKDIM 256
#define GB_CUDA_TRANSPOSE_PREP_CHUNKSIZE 1024
#define GB_CUDA_TRANSPOSE_PREP_CHUNKSIZE_LOG2 10

// dot3 CUDA kernel
#define GB_CUDA_DOT3_CHUNKSIZE 128
#define GB_CUDA_DOT3_CHUNKSIZE_LOG2 7

// rowscale, colscale CUDA kernels
#define GB_CUDA_SCALE_BLOCKDIM 128
#define GB_CUDA_SCALE_CHUNKSIZE 1024
#define GB_CUDA_SCALE_CHUNKSIZE_LOG2 10

// reduce CUDA kernel
#define GB_CUDA_REDUCE_BLOCKDIM 320
#define GB_CUDA_REDUCE_CHUNKSIZE (256*320)
#define GB_CUDA_REDUCE_CHECK 64

// apply CUDA kernels (unop, bind1st, and bind2nd)
#define GB_CUDA_APPLY_BLOCKDIM 512
#define GB_CUDA_APPLY_CHUNKSIZE 1024
#define GB_CUDA_APPLY_CHUNKSIZE_LOG2 10

// add CUDA kernel
#define GB_CUDA_ADD_CHUNKSIZE0 512

#endif

