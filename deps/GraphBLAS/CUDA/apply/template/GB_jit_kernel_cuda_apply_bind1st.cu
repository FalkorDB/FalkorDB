//------------------------------------------------------------------------------
// CUDA/apply/template/GB_jit_kernel_cuda_apply_bind1st.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_cuda_apply_bind1st_kernel: device kernel for binary apply (bind1st case)
//------------------------------------------------------------------------------

__global__ void GB_cuda_apply_bind1st_kernel
(
    GB_void *Cx_out,
    const GB_void *scalarx,
    GrB_Matrix B
)
{
    const GB_X_TYPE x = * ((GB_X_TYPE *) scalarx) ; // gets scalarx [0]
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) Cx_out ;

    #if ( GB_B_IS_BITMAP )
    const int8_t *__restrict__ Bb = B->b ;
    #endif
    
    GB_B_NHELD (nvals) ;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t p = tid ; p < nvals ; p += nthreads)
    {
        if (!GBb_B (Bb, p)) { continue ; }
        GB_DECLAREB (bij) ;
        GB_GETB (bij, Bx, p, false) ;
        GB_EWISEOP (Cx, p, x, bij, /* i */, /* j */) ;
    }
}

//------------------------------------------------------------------------------
// host CUDA JIT kernel for binary op apply (bind1st case)
//------------------------------------------------------------------------------

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    // ASSERT (Cx != NULL) ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block (GB_CUDA_APPLY_BLOCKDIM) ;
    GB_B_NHELD (nvals) ;
    if (nvals == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_bind1st_kernel <<<grid, block, 0, stream>>> (Cx, scalarx, B) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}

