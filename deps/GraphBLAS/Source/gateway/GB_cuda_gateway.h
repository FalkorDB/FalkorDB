//------------------------------------------------------------------------------
// GB_cuda_gateway.h: definitions for interface to GB_cuda_* functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CUDA gateway functions (DRAFT: in progress)

// This file can be #include'd into any GraphBLAS/Source file that needs to
// call a CUDA gateway function, or use the typedef defined below.  It is also
// #include'd in GraphBLAS/CUDA/GB_cuda.h, for use by the CUDA/GB_cuda_*.cu
// gateway functions.

// If GRAPHBLAS_HAS_CUDA is defined in GraphBLAS/CMakeLists.txt, then GraphBLAS
// can call the C-callable gateway functions defined in GraphBLAS/CUDA/*.cu
// source files.  If GRAPHBLAS_HAS_CUDA is not defined, then these functions
// are not called.  The typedef always appears, since it is part of the
// GB_Global struct, whether or not CUDA is used.

#ifndef GB_CUDA_GATEWAY_H
#define GB_CUDA_GATEWAY_H

#define GB_CUDA_MAX_GPUS 32

// The GPU is only used if the work is larger than the GxB_GPU_CHUNK.
// The default value of this parameter is GB_GPU_CHUNK_DEFAULT:
#define GB_GPU_CHUNK_DEFAULT (1024*1024)

//------------------------------------------------------------------------------
// GB_cuda_device: properties of each GPU in the system
//------------------------------------------------------------------------------

typedef struct
{
    char    name [256] ;
    size_t  total_global_memory ;
    int  number_of_sms ;
    int  compute_capability_major ;
    int  compute_capability_minor ;
    bool use_memory_pool ;
    size_t  pool_size ;
    size_t  max_pool_size ;
    void *memory_resource ;
    // TODO: add something about the streams for this device
}
GB_cuda_device ;

//------------------------------------------------------------------------------
// GB_ngpus_to_use: determine # of GPUs to use for the next computation
//------------------------------------------------------------------------------

static inline int GB_ngpus_to_use
(
    double work                 // total work to do
)
{

    // gpu_hack: for testing only
    //  2: never use GPU
    //  1: always use GPU
    //  0: default
    int gpu_hack = (int) GB_Global_hack_get (2) ;

    // get # of GPUs avaiable
    int gpu_count = GB_Global_gpu_count_get ( ) ;

    if (gpu_hack == 2 || gpu_count == 0 || work == 0)
    {
        // never use the GPU(s)
        return (0) ;
    }
    else if (gpu_hack == 1)
    {
        // always use all available GPU(s)
        // FIXME for CUDA: allow 1 to gpu_count to be requested
        return (gpu_count) ;
    }
    else
    {
        // default: use no more than max_gpus_to_use
        double gpu_chunk = 2e6 ;
        double max_gpus_to_use = floor (work / gpu_chunk) ;
        // but use no more than the # of GPUs available
        if (max_gpus_to_use > gpu_count) return (gpu_count) ;
        return ((int) max_gpus_to_use) ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_* gateway functions
//------------------------------------------------------------------------------

GrB_Info GB_cuda_init (void) ;
GrB_Info GB_cuda_finalize (void) ;

bool GB_cuda_get_device_count   // true if OK, false if failure
(
    int *gpu_count              // return # of GPUs in the system
) ;

GrB_Info GB_cuda_stream_pool_init (void) ;
GrB_Info GB_cuda_stream_pool_finalize (void) ;

bool GB_cuda_warmup (int device) ;

bool GB_cuda_get_device( int *device) ;

bool GB_cuda_set_device( int device) ;

bool GB_cuda_get_device_properties
(
    int device,
    GB_cuda_device *prop
) ;

bool GB_cuda_type_branch            // return true if the type is OK on GPU
(
    const GrB_Type type             // type to query
) ;

bool GB_cuda_reduce_to_scalar_branch    // return true to use the GPU
(
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A              // input matrix
) ;

GrB_Info GB_cuda_reduce_to_scalar
(
    // output:
    GB_void *s,                 // note: statically allocated on CPU stack; if
                                // the result is in s then V is NULL.
    GrB_Matrix *V_handle,       // partial result if unable to reduce to scalar;
                                // NULL if result is in s.
    // input:
    const GrB_Monoid monoid,
    const GrB_Matrix A
) ;

bool GB_cuda_rowscale_branch
(
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
) ;

GrB_Info GB_cuda_rowscale
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
) ;

bool GB_cuda_colscale_branch
(
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
) ;

GrB_Info GB_cuda_colscale
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
) ;

bool GB_cuda_apply_binop_branch
(
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A
) ;

bool GB_cuda_apply_unop_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op
) ;

GrB_Info GB_cuda_apply_unop
(
    GB_void *Cx,
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk
) ;

GrB_Info GB_cuda_apply_binop
(
    GB_void *Cx,
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A, 
    const GB_void *scalarx,
    const bool bind1st
) ;

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
) ;

GrB_Info GB_cuda_select_bitmap
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op
) ;

GrB_Info GB_cuda_select_sparse
(
    GrB_Matrix C,
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *athunk,
    const GB_void *ythunk,
    GB_Werk Werk
) ;

bool GB_cuda_type_branch            // return true if the type is OK on GPU
(
    const GrB_Type type             // type to query
) ;

GrB_Info GB_cuda_AxB_dot3           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix, static header
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy               // if true, do z=fmult(b,a) vs fmult(a,b)
) ;

bool GB_cuda_AxB_dot3_branch
(
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy               // if true, do z=fmult(b,a) vs fmult(a,b)
);

#endif

