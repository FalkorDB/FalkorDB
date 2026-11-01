//------------------------------------------------------------------------------
// CUDA/apply/template/GB_jit_kernel_cuda_apply_bind2nd.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_cuda_apply_bind2nd_kernel: device kernel for binary apply (bind2nd case)
//------------------------------------------------------------------------------

__global__ void GB_cuda_apply_bind2nd_kernel
(
    GB_void *Cx_out,
    GrB_Matrix A,
    const GB_void *scalarx
)
{
    const GB_Y_TYPE x = * ((GB_Y_TYPE *) scalarx) ; // gets scalarx [0]
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) Cx_out ;

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = A->b ;
    #endif
    
    GB_A_NHELD (nvals) ;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t p = tid ; p < nvals ; p += nthreads)
    {
        if (!GBb_A (Ab, p)) { continue ; }
        GB_DECLAREA (aij) ;
        GB_GETA (aij, Ax, p, false) ;
        GB_EWISEOP (Cx, p, aij, x, /* i */, /* j */) ;
    }
}

//------------------------------------------------------------------------------
// host CUDA JIT kernel for binary op apply (bind2nd case)
//------------------------------------------------------------------------------

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block (GB_CUDA_APPLY_BLOCKDIM) ;
    GB_A_NHELD (nvals) ;
    if (nvals == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_bind2nd_kernel <<<grid, block, 0, stream>>> (Cx, A, scalarx) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}

