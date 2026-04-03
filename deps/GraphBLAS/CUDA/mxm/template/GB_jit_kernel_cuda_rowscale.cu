#define GB_FREE_ALL ;

using namespace cooperative_groups ;

__global__ void GB_cuda_rowscale_kernel
(
    GrB_Matrix C,
    GrB_Matrix D,
    GrB_Matrix B
)
{
    const GB_A_TYPE *__restrict__ Dx = (GB_A_TYPE *) D->x ;
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *) B->x ;

    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) C->x ;

    #define D_iso GB_A_ISO
    #define B_iso GB_B_ISO

    #if ( GB_B_IS_SPARSE || GB_B_IS_HYPER )
    const GB_Bi_TYPE *__restrict__ Bi = (GB_Bi_TYPE *) B->i ;
    #endif

    #if ( GB_B_IS_BITMAP )
    const int8_t *__restrict__ Bb = B->b ;
    #endif

    GB_B_NHELD (bnz) ;

    #if ( GB_B_IS_BITMAP || GB_B_IS_FULL )
    const int64_t bvlen = B->vlen ;
    #endif

    int ntasks = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t p = tid ; p < bnz ; p += ntasks)
    {
        if (!GBb_B (Bb, p)) { continue ; }

        int64_t i = GBi_B (Bi, p, bvlen) ;      // get row index of B(i,j)
        GB_DECLAREA (dii) ;
        GB_GETA (dii, Dx, i, D_iso) ;           // dii = D(i,i)
        GB_DECLAREB (bij) ;
        GB_GETB (bij, Bx, p, B_iso) ;           // bij = B(i,j)
        GB_EWISEOP (Cx, p, dii, bij, 0, 0) ;    // C(i,j) = dii*bij
    }
}

extern "C" {
    GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (!GB_IS_BITMAP (D)) ;
    ASSERT (!GB_IS_FULL (D)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!C->iso) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_rowscale_kernel <<<grid, block, 0, stream>>> (C, D, B) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}
