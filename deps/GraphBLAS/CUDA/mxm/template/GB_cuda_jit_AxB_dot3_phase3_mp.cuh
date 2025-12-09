//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase3_mp.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This CUDA kernel produces the semi-ring product of two sparse matrices of
// types GB_A_TYPE and GB_B_TYPE and common index space size n, to a  output
// matrix of type GB_C_TYPE. The matrices are sparse, with different numbers of
// non-zeros and different sparsity patterns.  ie. we want to produce C = A'*B
// in the sense of the given semi-ring.

// This version uses a merge-path algorithm, when the sizes nnzA and nnzB are
// relatively close in size, neither is very sparse nor dense, for any size of
// N.  Handles arbitrary sparsity patterns with guaranteed load balance.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number of
// active threads = min( min(g_xnz, g_ynz), 32) 

// Thus, threadblock b owns a part of the index set spanned by g_xi and g_yi.
// Its job is to find the intersection of the index sets g_xi and g_yi, perform
// the semi-ring dot product on those items in the intersection, and finally
// reduce this data to a scalar, on exit write it to g_odata [b].

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  GrB_Matrix C           <- result matrix 
//  GrB_Matrix M           <- mask matrix
//  GrB_Matrix A           <- input matrix A
//  GrB_Matrix B           <- input matrix B

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_mp_kernel
//------------------------------------------------------------------------------

//#include <time.h>
  
__global__ void GB_cuda_AxB_dot3_phase3_mp_kernel
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,    // do the work in Bucket [start:end-1]
    GrB_Matrix C,
    GrB_Matrix M,
    GrB_Matrix A,
    GrB_Matrix B,
    const void *theta
)
{

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *)A->x  ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *)B->x  ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *)C->x  ;
          GB_Ci_SIGNED_TYPE *__restrict__ Ci = (GB_Ci_SIGNED_TYPE *) C->i ;

    const GB_Mp_TYPE *__restrict__ Mp = (GB_Mp_TYPE *) M->p ;
    const GB_Mi_TYPE *__restrict__ Mi = (GB_Mi_TYPE *) M->i ;
    #if GB_M_IS_HYPER
    const GB_Mj_TYPE *__restrict__ Mh = (GB_Mj_TYPE *) M->h ;
    #endif

    // A and B are either sparse or hypersparse
    const GB_Ai_TYPE *__restrict__ Ai = (GB_Ai_TYPE *) A->i ;
    const GB_Bi_TYPE *__restrict__ Bi = (GB_Bi_TYPE *) B->i ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Bp_TYPE *__restrict__ Bp = (GB_Bp_TYPE *) B->p ;
    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;
    ASSERT (GB_B_IS_HYPER || GB_B_IS_SPARSE) ;

    #if GB_A_IS_HYPER
    const int64_t anvec = A->nvec ;
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    const void *A_Yp = (void *) ((A->Y == NULL) ? NULL : A->Y->p) ;
    const void *A_Yi = (void *) ((A->Y == NULL) ? NULL : A->Y->i) ;
    const void *A_Yx = (void *) ((A->Y == NULL) ? NULL : A->Y->x) ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;
    #endif

    #if GB_B_IS_HYPER
    const int64_t bnvec = B->nvec ;
    const GB_Bj_TYPE *__restrict__ Bh = (GB_Bj_TYPE *) B->h ;
    const void *B_Yp = (void *) ((B->Y == NULL) ? NULL : B->Y->p) ;
    const void *B_Yi = (void *) ((B->Y == NULL) ? NULL : B->Y->i) ;
    const void *B_Yx = (void *) ((B->Y == NULL) ? NULL : B->Y->x) ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;
    #endif

    // zombie count
    uint64_t zc = 0;

    // set thread ID
//  int tid_global = threadIdx.x+ blockDim.x* blockIdx.x;
    int tid = threadIdx.x;


    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());
    int all_in_one = ( (end - start) == Mp [(M->nvec)] ) ;

    // Main loop over pairs 
    int64_t kk ;
    for (kk = start+ blockIdx.x; // warp per C(i,j)=A(:,i)'*B(:,j) dot product
         kk < end;  
         kk += gridDim.x )
    {

        //----------------------------------------------------------------------
        // get A(:,i) and B(:,j)
        //----------------------------------------------------------------------

        int64_t pair_id = all_in_one ? kk : Bucket [kk] ;
        int64_t i = Mi[pair_id];
        int64_t k = Ci[pair_id] >> 4;
        // assert: Ci [pair_id] & 0xF == GB_BUCKET_MERGEPATH

        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBh_M (Mh, k) ;

        // find A(:,i)
        int64_t pA_start, pA_end ;
        #if GB_A_IS_HYPER
        GB_hyper_hash_lookup (GB_Ap_IS_32, GB_Aj_IS_32,
            Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
            i, &pA_start, &pA_end) ;
        #else
        pA_start = Ap[i] ;
        pA_end   = Ap[i+1] ;
        #endif

        // find B(:,j)
        int64_t pB_start, pB_end ;
        #if GB_B_IS_HYPER
        GB_hyper_hash_lookup (GB_Bp_IS_32, GB_Bj_IS_32,
            Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
            j, &pB_start, &pB_end) ;
        #else
        pB_start = Bp[j] ;
        pB_end   = Bp[j+1] ;
        #endif

        //----------------------------------------------------------------------
        // compute cij
        //----------------------------------------------------------------------
    
        __shared__ int64_t Xi_s[shared_vector_size];
        __shared__ int64_t Yi_s[shared_vector_size];

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity
        int cij_exists = 0 ;

//      int64_t total_ainz = pA_start - pA_end ;
//      int64_t total_bjnz = pB_start - pB_end ;

//      if (total_ainz < total_bjnz)
        {
            // A(:,i) is sparser than B(:,j)
            #define MP_FLIP 0

            #define pX       pA
            #define pX_start pA_start
            #define pX_end   pA_end
            #define Xi       Ai

            #define pY       pB
            #define pY_start pB_start
            #define pY_end   pB_end
            #define Yi       Bi

            #include "template/GB_cuda_jit_AxB_dot3_phase3_mp_guts.cuh"
        }
#if 0
        else
        {
            // B(:,j) is sparser than A(:,i)
            // (this works but it has the same performance)
            #define MP_FLIP 1

            #define pX       pB
            #define pX_start pB_start
            #define pX_end   pB_end
            #define Xi       Bi

            #define pY       pA
            #define pY_start pA_start
            #define pY_end   pA_end
            #define Yi       Ai

            // flip the roles of A(:,i) and B(:,j)
            #include "template/GB_cuda_jit_AxB_dot3_phase3_mp_guts.cuh"
        }
#endif

        //----------------------------------------------------------------------
        // reduce sum per-thread values to a single scalar, get OR of flag
        //----------------------------------------------------------------------

        // Do vote here for control.
        cij_exists = tile.any (cij_exists) ;
        tile.sync ( ) ;

        #if !GB_C_ISO
        if (cij_exists)
        {
            // FIXME: the ANY monoid needs the cij_exists for each thread
            cij = GB_cuda_tile_reduce_ztype (tile, cij) ;
        }
        #endif

// HACK
//int64_t end_time = (int64_t) clock ( ) ;
//cij = end_time - start_time ;
//cij_exists = 1 ;

        // write result for this block to global mem
        if (tid == 0)
        {
            if (cij_exists)
            {
                // Cx [pair_id] = (GB_C_TYPE) cij
                GB_PUTC (cij, Cx, pair_id) ;
                Ci [pair_id] = i ;
            }
            else
            {
               // cij is a zombie
               zc++;
               Ci [pair_id] = GB_ZOMBIE (i) ;
            }
        }
        //__syncthreads(); 
    }

    //--------------------------------------------------------------------------
    // sum up the global zombie count
    //--------------------------------------------------------------------------

    if( tid ==0 && zc > 0)
    {
        GB_cuda_atomic_add <uint64_t>( &(C->nzombies), zc) ;
    }
}

