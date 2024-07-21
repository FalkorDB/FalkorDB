//------------------------------------------------------------------------------
// GraphBLAS/CUDA/JitKernels/GB_cuda_jit_AxB_dot3_phase3_vsvs.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//******************************************************************************
//  Sparse dot version of Matrix-Matrix multiply with mask 
//  Each thread in this kernel is responsible for m vector-pairs(x,y), 
//  finding intersections and producting the final dot product for each
//  using a serial merge algorithm on the sparse vectors. 
//  m = 256/sz, where sz is in {4, 16, 64, 256}
//  For a vector-pair, sz = xnz + ynz 
//  Parameters:

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  C         <- result matrix 
//  M         <- mask matrix
//  A         <- input matrix A
//  B         <- input matrix B

//  Blocksize is 1024, uses warp and block reductions to count zombies produced.
//******************************************************************************

//------------------------------------------------------------------------------
// GB_block_ReduceSum_uint64
//------------------------------------------------------------------------------

__inline__ __device__ uint64_t GB_block_ReduceSum_uint64
(
    thread_block g,
    uint64_t val
)
{
    // Shared mem for 32 partial sums
    static __shared__ uint64_t shared [tile_sz] ;

    // FIXME: assumes tile_sz is 32:  (use an #if .. #else ... #endif)
    int lane = threadIdx.x & 31 ; // % tile_sz;
    int wid  = threadIdx.x >> 5 ; // / tile_sz;
    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz> (g) ;

    // Each warp performs partial reduction
    val = GB_cuda_warp_sum_uint64 (tile, val) ;    

    // Wait for all partial reductions
    if (lane == 0)
    {
        shared [wid] = val ; // Write reduced value to shared memory
    }

    g.sync();                     // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x <  (blockDim.x / tile_sz ) ) ? shared[lane] : 0;

    // Final reduce within first warp
    if (wid == 0)
    {
        val = GB_cuda_warp_sum_uint64 (tile, val) ;
    }

    return (val) ;
}

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_vsvs_kernel
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase3_vsvs_kernel
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,  // do the work in Bucket [start:end-1]
    GrB_Matrix C,
    GrB_Matrix M,
    GrB_Matrix A,
    GrB_Matrix B
)
{

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *)A->x  ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *)B->x  ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *)C->x  ;
          int64_t *__restrict__ Ci = C->i ;
    const int64_t *__restrict__ Mi = M->i ;
    #if GB_M_IS_HYPER
    const int64_t *__restrict__ Mh = M->h ;
    #endif

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;
    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Ap = A->p ;

    ASSERT (GB_B_IS_HYPER || GB_B_IS_SPARSE) ;
    const int64_t *__restrict__ Bi = B->i ;
    const int64_t *__restrict__ Bp = B->p ;

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

    uint64_t my_nzombies = 0 ;

    GB_M_NVALS (mnz) ;
    int all_in_one = ( (end - start) == mnz ) ;

    for (int64_t kk = start + threadIdx.x + blockDim.x*blockIdx.x ;
                 kk < end ;
                 kk += blockDim.x*gridDim.x )
    {
        int64_t pair_id = all_in_one ? kk : Bucket[ kk ];

        int64_t i = Mi [pair_id] ;
        int64_t k = Ci [pair_id]>>4 ;

        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBH_M (Mh, k) ;

        // find A(:,i):  A is always sparse or hypersparse
        int64_t pA, pA_end ;
        #if GB_A_IS_HYPER
        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
           i, &pA, &pA_end) ;
        #else
        pA     = Ap [i] ;
        pA_end = Ap [i+1] ;
        #endif

        // find B(:,j):  B is always sparse or hypersparse
        int64_t pB, pB_end ;
        #if GB_B_IS_HYPER
        GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
           j, &pB, &pB_end) ;
        #else
        pB     = Bp [j] ;
        pB_end = Bp [j+1] ;
        #endif

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity

        bool cij_exists = false;

        while (pA < pA_end && pB < pB_end )
        {
            int64_t ia = Ai [pA] ;
            int64_t ib = Bi [pB] ;
            #if GB_IS_PLUS_PAIR_REAL_SEMIRING && GB_Z_IGNORE_OVERFLOW
                cij += (ia == ib) ;
            #else
                if (ia == ib)
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT_MERGE (pA, pB) ;
                    GB_DOT_TERMINAL (cij) ;   // break if cij == terminal
                }
            #endif
            pA += ( ia <= ib);  // incr pA if A(ia,i) at or before B(ib,j)
            pB += ( ib <= ia);  // incr pB if B(ib,j) at or before A(ia,i)
        }


        GB_CIJ_EXIST_POSTCHECK ;
        if (cij_exists)
        {
            GB_PUTC (cij, Cx, pair_id) ;    // Cx [pair_id] = (GB_C_TYPE) cij
            Ci [pair_id] = i ;
        }
        else
        {
            // cij is a zombie
            my_nzombies++;
            Ci [pair_id] = GB_FLIP (i) ;
        }
    }

    // FIXME: use this in spdn and vsdn:
    this_thread_block().sync(); 

    my_nzombies = GB_block_ReduceSum_uint64 (this_thread_block(), my_nzombies) ;
    this_thread_block().sync(); 

    if( threadIdx.x == 0 && my_nzombies > 0)
    {
        GB_cuda_atomic_add <uint64_t>( &(C->nzombies), my_nzombies) ;
    }
}

