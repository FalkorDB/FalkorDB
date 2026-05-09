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
    size_t  pool_memsize ;
    size_t  max_pool_memsize ;
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
// CUDA init/finalize and device methods
//------------------------------------------------------------------------------

GrB_Info GB_cuda_init (void) ;
GrB_Info GB_cuda_finalize (void) ;
int GB_cuda_get_device_count (void) ;   // return # of GPUs in the system
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

//------------------------------------------------------------------------------
// CUDA type branch
//------------------------------------------------------------------------------

bool GB_cuda_type_branch            // return true if the type is OK on GPU
(
    const GrB_Type type             // type to query
) ;

//------------------------------------------------------------------------------
// CUDA reduce to scalar
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// CUDA apply
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// CUDA select
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// CUDA matrix-matrix multiply
//------------------------------------------------------------------------------

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

GrB_Info GB_cuda_AxB_dot3           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix, existing header
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
) ;

//------------------------------------------------------------------------------
// determine if the GPU can access the memory
//------------------------------------------------------------------------------

bool GB_cuda_pointer_ok
(
    const void *p,
    const char *name
) ;

//------------------------------------------------------------------------------
// builder
//------------------------------------------------------------------------------

bool GB_cuda_builder_branch
(
    const GrB_Matrix C,
    const GrB_BinaryOp dup,
    const GrB_Type xtype,
    const void *I,
    const void *J,
    const void *X,
    const uint64_t nvals
) ;

GrB_Info GB_cuda_builder            // build a matrix from tuples
(
    // output, not defined on input:
    GrB_Matrix *Thandle,    // matrix to build, dynamic header
    // inputs, not modified:
    const GrB_Type ttype,   // type of output matrix T
    const int64_t vlen,     // length of each vector of T
    const int64_t vdim,     // number of vectors in T
    const bool is_csc,      // true if T is CSC, false if CSR
    const bool is_matrix,   // true if T a GrB_Matrix, false if vector
    const GB_void *Key_input,  // if Key is preloaded, NULL otherwise
    const GB_void *I,       // original indices, size nvals
    const GB_void *J,       // original indices, size nvals
    const GB_void *X,       // array of values of tuples, size nvals,
                            // or size 1 if X is iso
    const bool X_iso,       // true if X is iso
    const int64_t nvals,    // number of tuples
    GrB_BinaryOp dup,       // binary function to assemble duplicates,
                            // if NULL use the SECOND operator to
                            // keep the most recent duplicate.
    const GrB_Type xtype,   // the type of X
    bool do_burble,         // if true, then burble is allowed
    bool I_is_32,       // true if I is 32 bit, false if 64
    bool J_is_32,       // true if J is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32,      // true if T->i is built as 32 bit, false if 64
    bool known_no_duplicates,   // true if tuples known to have no duplicates
    bool known_sorted           // true if tuples known to be sorted on input
) ;

static inline bool GB_cuda_builder_key_is_32
(
    const int64_t vlen,
    const int64_t vdim
)
{
    // returns true if Key_in will contain 32-bit integers (uint32_t);
    // if false, then Key_in will contain uint64_t integers
    return (vlen <= UINT32_MAX && vdim <= UINT32_MAX) ;
}

//------------------------------------------------------------------------------
// CUDA transpose
//------------------------------------------------------------------------------

bool GB_cuda_transpose_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op,           // any type of operator
    const GrB_Scalar scalar
) ;

GrB_Info GB_cuda_transpose      // T=A', T=(ctype)A' or T=op(A')
(
    GrB_Matrix *Thandle,        // output matrix T, header allocated on input
    GrB_Type ctype,             // desired type of T
    const bool C_is_csc,        // desired CSR/CSC format of C and T
    const bool C_iso,           // true if C (and T) is iso
    const GB_iso_code C_code_iso,   // iso code for C and T
    const GrB_Matrix A,         // input matrix; C == A if done in place
    const bool in_place,        // true if C and A are the same matrix
        // no operator is applied if op is NULL
        const GB_Operator op,       // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    GB_Werk Werk
) ;

#endif

