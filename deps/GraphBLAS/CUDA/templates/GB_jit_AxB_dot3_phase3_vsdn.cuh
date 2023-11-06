//******************************************************************************
//  Sparse dot products in batch form, sparse - dense case. 
//  Each thread in this kernel is responsible for m vector-pairs(x,y), 
//  m = 256/sz, where sz is in {4, 16, 64, 256}
//  We know each non-zero on the sparse side will hit a dense value.
//  Template on <T_C, T_A, T_B, T_X, T_Y, T_Z >
//  Parameters:

//  matrix<T_C> *C         <- C result matrix 
//  matrix<T_C> *M         <- Mask matrix 
//  matrix<T_A> *A         <- A matrix to multiply, sparse 
//  matrix<T_B> *B         <- B matrix to multiply, dense in sparse format? 
//******************************************************************************

#pragma once
#include <limits>
#include <cstdint>
#include <stdio.h>
#include "GB_cuda_kernel.h"
#include "GB_mxm_shared_definitions.h"
#include "GB_hash.h"
#include "GB_hyper_hash_lookup.h"
#include <cooperative_groups.h>
#define tile_sz 32
//#include "local_cub/block/block_reduce.cuh"
#include <cub/block/block_reduce.cuh>
#include "GB_cuda_dot3_defn.h"

using namespace cooperative_groups;

//------------------------------------------------------------------------------
// reduce_sum_int64
//------------------------------------------------------------------------------

// for counting zombies only (always int64_t)
template< int warpSize >
__device__ int64_t reduce_sum_int64(thread_block_tile<warpSize> g, int64_t val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down(val,i) ;
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// AxB_dot3_phase3_vsdn
//------------------------------------------------------------------------------

template<
    typename T_C, typename T_A, typename T_B,
    typename T_Z, typename T_X, typename T_Y,
    uint64_t srcode>
__global__ void AxB_dot3_phase3_vsdn
( 
  int64_t start,
  int64_t end,
  int64_t *Bucket,  // do the work in Bucket [start:end-1]
  GrB_Matrix C, 
  GrB_Matrix M, 
  GrB_Matrix A, 
  GrB_Matrix B,
  int sz            // unused (FIXME: remove this)
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

    #if GB_A_IS_HYPER || GB_A_IS_SPARSE
    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Ap = A->p ;
    #endif

    #if GB_A_IS_BITMAP
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    #if GB_B_IS_HYPER || GB_B_IS_SPARSE
    const int64_t *__restrict__ Bi = B->i ;
    const int64_t *__restrict__ Bp = B->p ;
    #endif

    #if GB_B_IS_BITMAP
    const int8_t *__restrict__ Bb = B->b ;
    #endif

    #if GB_A_IS_HYPER
    const int64_t anvec = A->nvec ;
    const int64_t *__restrict__ Ah = A->h ;
    const int64_t *__restrict__ A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *__restrict__ A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *__restrict__ A_Yx = (int64_t *)
        ((A->Y == NULL) ? NULL : A->Y->x) ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
    #endif

    #if GB_B_IS_HYPER
    const int64_t bnvec = B->nvec ;
    const int64_t *__restrict__ Bh = B->h ;
    const int64_t *__restrict__ B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
    const int64_t *__restrict__ B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
    const int64_t *__restrict__ B_Yx = (int64_t *)
        ((B->Y == NULL) ? NULL : B->Y->x) ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;
    #endif

//   typedef cub::BlockReduce<int, 32> BlockReduce;
//   __shared__ typename BlockReduce::TempStorage temp_storage;

//   if( threadIdx.x ==0)
//      printf("thd:%d %d dots/thrd, nvec = %d blockDim=%d\n",threadIdx.x, sz, nvec, blockDim.x);
//   __syncthreads();

    int64_t pair_id; 

    int64_t zc = 0 ;

//       if (threadIdx.x ==0)
//         printf("thd%u pi=%lld\n",tid, start+threadIdx.x);
//       __syncthreads();

    int all_in_one = ( (end - start) == (M->p)[(M->nvec)] ) ;

    for (int64_t kk = start +threadIdx.x +blockIdx.x*blockDim.x; 
                 kk < end ;  
                 kk += gridDim.x*blockDim.x  )
    {

        int64_t pair_id = all_in_one ? kk : Bucket[ kk ];
        int64_t i = Mi[pair_id];  // cols from mask

        // FIXME: use another variable, not "k" here:
        int64_t k = Ci[pair_id] >> 4;  // vector of C encoded in phase1

        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBH_M (Mh, k) ;

        // Prep row offsets for both A and B

        // find A(:,i)
        int64_t pA, pA_end ;
        #if GB_A_IS_HYPER
        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
            i, &pA, &pA_end) ;
        #elif GB_A_IS_SPARSE
        pA     = Ap[i] ;
        pA_end = Ap[i+1] ;
        #else
        // A is bitmap or full
        pA     = (A->vlen)*i;
        pA_end = pA +(A->vlen);
        #endif

        // find B(:,j)
        int64_t pB, pB_end ;
        #if GB_B_IS_HYPER
        GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
            j, &pB, &pB_end) ;
        #elif GB_B_IS_SPARSE
        pB       = Bp[j];   // col of C
        pB_end   = Bp[j+1];
        #else
        // B is bitmap or full
        pB   = (B->vlen)*j;
        pB_end = pB +(B->vlen);
        #endif

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity
        bool cij_exists = false ;

        int64_t my_nzombies = 0;

        #if ( GB_A_IS_FULL )
        {
            int64_t nnzB = pB_end - pB ;
            if (nnzB > 0)
            {

                //--------------------------------------------------------------
                // A is full and B is sparse/hyper
                //--------------------------------------------------------------

                cij_exists = true ;
                for (int64_t p = pB ; p < pB_end ; ++p)
                {
                    int64_t k = Bi [p] ;        // next row index of B(:,j)
                    // cij += A(k,i) * B(k,j)
                    GB_GETA ( aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB ( bkj, Bx, p, ) ;              // bkj = B(k,j)
                    GB_MULTADD ( cij, aki, bkj, i, k, j) ; // cij += aki * bkj
                    GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
                }
            }
        }
        #elif ( GB_A_IS_BITMAP )
        {
            //------------------------------------------------------------------
            // A is bitmap and B is sparse/hyper
            //------------------------------------------------------------------

            for (int64_t p = pB ; p < pB_end ; ++p)
            {
                int64_t k = Bi [p] ;        // next row index of B(:,j)
                if (Ab [pA+k])              // check if A(k,i) exists
                {
                    // cij += A(k,i) * B(k,j)
                    GB_DOT_MERGE (pA+k, p) ;
                    GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
                }
            }
        }
        #elif ( GB_B_IS_FULL )
        {
            int64_t nnzA = pA_end - pA ;
            if (nnzA > 0)
            {

                //--------------------------------------------------------------
                // A is sparse/hyper and B is full
                //--------------------------------------------------------------

                cij_exists = true ;
                for (int64_t p = pA ; p < pA_end ; ++p)
                {
                    int64_t k = Ai [p] ;        // next row index of A(:,i)
                    // cij += A(k,i) * B(k,j)
                    GB_GETA ( aki, Ax, p, ) ;              // aki = A(i,k)
                    GB_GETB ( bkj, Bx, pB+k, ) ;           // bkj = B(j,k)
                    GB_MULTADD ( cij, aki, bkj, i, k, j) ; // cij += aik * bjk
                    GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
                }
            }
        }
        #elif ( GB_B_IS_BITMAP )
        {

            //------------------------------------------------------------------
            // A is sparse/hyper and B is bitmap
            //------------------------------------------------------------------

            for (int64_t p = pA ; p < pA_end ; ++p)
            {
                int64_t k = Ai [p] ;        // next row index of A(:,i)
                if (Bb [pB+k])              // check if B(k,j) exists
                {
                    // cij += A(k,i) * B(k,j)
                    GB_DOT_MERGE (p, pB+k) ;
                    GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
                }
            }
        }
        #endif

        GB_CIJ_EXIST_POSTCHECK
        if (cij_exists)
        {
            GB_PUTC (cij, Cx, pair_id) ;        // Cx [pair_id] = (T_C) cij
            Ci [pair_id] = i ;
        }
        else
        {
            my_nzombies++ ;
            Ci [pair_id] = GB_FLIP (i) ;
        }

        // FIXME: use the same method as vsvs for counting zombies
        // sum up the zombie count:
        thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());
        zc += reduce_sum_int64<tile_sz>(tile, my_nzombies);
    }

    if(threadIdx.x == 0 && zc > 0)
    {
        // this threadblock accumulates its zombie count into the global
        // zombie count
        GB_cuda_atomic_add <int64_t>( &(C->nzombies), zc) ;
    }
}

