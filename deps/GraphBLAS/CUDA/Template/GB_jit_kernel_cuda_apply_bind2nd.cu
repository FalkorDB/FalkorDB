#define GB_FREE_ALL ;

using namespace cooperative_groups ;

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

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    ASSERT (Cx != NULL) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    GB_A_NHELD (nvals) ;
    if (nvals == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_bind2nd_kernel <<<grid, block, 0, stream>>> (Cx, A, scalarx) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}
