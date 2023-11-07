//------------------------------------------------------------------------------
// GB_jit_AxB_dot3_phase3_dndn.cuh:
//------------------------------------------------------------------------------

// This CUDA kernel produces the semiring product of two
// dense matrices of types T_A and T_B and common index space size n, to a  
// output matrix of type T_C. The matrices are dense, with uniform
// non-zeros and sparsity patterns. 
// ie. we want to produce C = A'*B in the sense of the given semi-ring.

// This version uses a simple warp-based dense dot product algorithm, when the
// vectors coming from both A and B are dense, for any size of N.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number
// of active threads = min( min(nzA, nzB), 32) 

// Thus, threadblock b owns a semi-ring dot product on a pair of vectors. 
// The work is to load the data, do the multiply and add work and finally 
// reduce this data to a scalar, and write it to Cx[pair].

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  GrB_Matrix C           <- result matrix 
//  GrB_Matrix M           <- mask matrix
//  GrB_Matrix A           <- input matrix A
//  GrB_Matrix B           <- input matrix B
//  int sz                 <- size parameter (not used) 

/* FIXME: This kernel needs to be split into 4 methods:

        (A bitmap) * (B bitmap)
        (A full ) * (B bitmap)
        (A bitmap) * (B full)
        (A full) * (B full)

    The buckets are not needed at all.  A single pass can be done.
    C and M would still be sparse or hypersparse.

    See also denseDotProduct.cu.
*/

#pragma once
#include <limits>
#include <cstdint>
#include "GB_cuda_kernel.h"
#include "GB_mxm_shared_definitions.h"
#include <cooperative_groups.h>

// Using tile size fixed at compile time, we don't need shared memory
#define tile_sz 32 

using namespace cooperative_groups;

//------------------------------------------------------------------------------
// warp_ReduceSum
//------------------------------------------------------------------------------

template< typename T_Z, int warp_sz>
__inline__ __device__ T_Z warp_ReduceSum(thread_block_tile<warp_sz> g, T_Z val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // FIXME: only works if sizeof(T_Z) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        T_Z next = g.shfl_down( val, i) ;
        GB_ADD( val, val, next ); 
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// AxB_dot3_phase3_dndn
//------------------------------------------------------------------------------

template<
    typename T_C, typename T_A, typename T_B,
    typename T_Z, typename T_X, typename T_Y,
    uint64_t srcode>
__global__ void AxB_dot3_phase3_dndn
(
    GrB_Matrix C,
    GrB_Matrix M,
    GrB_Matrix A,
    GrB_Matrix B
)
{
    // TODO: Figure out how to use graphblas-specific INFINITY macro
    #ifndef INFINITY
    #define INFINITY std::numeric_limits<T_C>::max()
    #endif

    const T_A *__restrict__ Ax = (T_A *)A->x  ;
    const T_B *__restrict__ Bx = (T_B *)B->x  ;
          T_C *__restrict__ Cx = (T_C *)C->x  ;
          int64_t *__restrict__ Ci = C->i ;
    const int64_t *__restrict__ Mi = M->i ;
    #if GB_M_IS_HYPER
    const int64_t *__restrict__ Mh = M->h ;
    #endif
    // A and B are either bitmap or full
    #if GB_A_IS_BITMAP
    const int8_t  *__restrict__ Ab = A->b ;
    #endif
    #if GB_B_IS_BITMAP
    const int8_t  *__restrict__ Bb = B->b ;
    #endif

    // zombie count
    int64_t zc = 0;

    int64_t start = 0;
    int64_t end   = M->p[M->nvec];

    // total items to be inspected
    int64_t nnzA = A->vlen;
    int64_t nnzB = B->vlen;
    int s = blockDim.x;

    // Main loop over pairs 
    for ( int64_t pair_id  = start + blockIdx.x; //warp per pair 
                  pair_id  < end;  
                  pair_id += gridDim.x )
    {

        // get M(i,j) and C(i,j)
        int64_t i = Mi[pair_id];
        int64_t kk = Ci[pair_id] >> 4;      // FIXME: can remove ">> 4"
        bool cij_exists = false ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity

        // skip if C(i,j) is a prezombie
        if (kk >= 0)
        {

            // j = kk or j = Mh [kk] if C and M are hypersparse
            int64_t j = GBH_M (Mh, kk) ;

            int64_t pA     = (A->vlen)*i;
            int64_t pA_end = pA +(A->vlen);

            int64_t pB     = (B->vlen)*j;
            int64_t pB_end = pB +(B->vlen);

            //      if (threadIdx.x == 0 ){
            //          printf("tid=%d, i,j = %d,%d  nnzA= %d, nnzB=%d\n",
            //                 threadIdx.x, (int)i,(int)j,  (int)nnzA, (int)nnzB);
            //      }
            //      __syncthreads();

            // convert global data pointer to the local pointer of this block
            GB_DECLAREA (aki) ;
            GB_DECLAREB (bkj) ;

            #if GB_A_IS_FULL && GB_B_IS_FULL
            {
                cij_exists = true ;
                for (int64_t k = threadIdx.x ; k < nnzA ; k += s)
                { 
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    GB_MULTADD ( cij, aki, bkj, i, k, j ) ; // cij += aki * bkj
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < nnzA ; k += s)
                { 
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    int8_t b = (Ab [pA+k] && Bb [pB+k]) ;
                    cij_exists |= b ;
                    if (b)
                    {
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;        // cij += aki * bkj
                    }
                }
            }
            #elif GB_A_IS_FULL && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < nnzA ; k += s)
                { 
                    if (Bb [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;        // cij += aki * bkj
                        cij_exists = true ;
                    }
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_FULL
            {
                for ( int64_t k = threadIdx.x ; k < nnzA ; k += s)
                { 
                    if (Ab [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;        // cij += aki * bkj
                        cij_exists = true ;
                    }
                }
            }
            #endif
        }

        //----------------------------------------------------------------------
        // reduce per-thread sums to a single scalar
        //----------------------------------------------------------------------

        // Do vote here for control.
        thread_block_tile<32> tile = tiled_partition<32>( this_thread_block() );
        cij_exists = tile.any( cij_exists);
        tile.sync();

        #if !GB_C_ISO
        // FIXME: the ANY monoid needs the cij_exists for each thread
        cij = warp_ReduceSum<T_Z, 32> ( tile, cij);
        #endif

        // write result for this block to global mem
        if (threadIdx.x == 0)
        {
            if (cij_exists)
            {
                GB_PUTC (cij, Cx, pair_id) ;        // Cx [pair_id] = (T_C) cij
                Ci [pair_id] = i ;
            }
            else
            {
                // cij is a zombie
                zc++;
                Ci [pair_id] = GB_FLIP (i) ;
            }
        }
        //__syncthreads ( ) ;

        if( threadIdx.x ==0 && zc > 0)
        {
            GB_cuda_atomic_add <int64_t>( &(C->nzombies), zc) ;
        }
    }
}

