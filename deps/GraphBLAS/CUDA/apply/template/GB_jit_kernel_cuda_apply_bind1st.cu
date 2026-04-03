#define GB_FREE_ALL ;

using namespace cooperative_groups ;

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

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    ASSERT (Cx != NULL) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    GB_B_NHELD (nvals) ;
    if (nvals == 0) return (GrB_SUCCESS) ;
    
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_bind1st_kernel <<<grid, block, 0, stream>>> (Cx, scalarx, B) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}
