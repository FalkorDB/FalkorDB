//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase3_spdn.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This CUDA kernel produces the semi-ring product of two sparse matrices of
// types GB_A_TYPE and GB_B_TYPE and common index space size n, to an output
// matrix of type GB_C_TYPE. The matrices are sparse, with different numbers of
// non-zeros and different sparsity patterns.  ie. we want to produce C = A'*B
// in the sense of the given semi-ring.

// This version uses an entire threadblock to compute each C(i,j) dot product.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_spdn_kernel
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase3_spdn_kernel
(
    int64_t start,      // start of vector pairs for this kernel
    int64_t end,        // end of vector pairs for this kernel
    int64_t *Bucket,    // do the work in Bucket [start:end-1]
    GrB_Matrix C,       // result matrix 
    GrB_Matrix M,       // mask matrix
    GrB_Matrix A,       // input matrix A
    GrB_Matrix B,       // input matrix B
    const void *theta
)
{

    // sparse-times-dense or dense-times-sparse
    #if !(((GB_A_IS_SPARSE || GB_A_IS_HYPER) &&         \
           (GB_B_IS_BITMAP || GB_B_IS_FULL))            \
            ||                                          \
          ((GB_B_IS_SPARSE || GB_B_IS_HYPER) &&         \
           (GB_A_IS_BITMAP || GB_A_IS_FULL)))
    #error "spdn: for sparse-dense or dense-sparse cases only"
    #endif

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *)A->x  ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *)B->x  ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *)C->x  ;
          GB_Ci_SIGNED_TYPE *__restrict__ Ci = (GB_Ci_SIGNED_TYPE *) C->i ;
    const GB_Mi_TYPE *__restrict__ Mi = (GB_Mi_TYPE *) M->i ;
    #if GB_M_IS_HYPER
    const GB_Mj_TYPE *__restrict__ Mh = (GB_Mj_TYPE *) M->h ;
    #endif

    #if GB_A_IS_HYPER || GB_A_IS_SPARSE
    const GB_Ai_TYPE *__restrict__ Ai = (GB_Ai_TYPE *) A->i ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    #else
    const int64_t avlen = A->vlen ;
    #endif

    #if GB_A_IS_BITMAP
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    #if GB_B_IS_HYPER || GB_B_IS_SPARSE
    const GB_Bi_TYPE *__restrict__ Bi = (GB_Bi_TYPE *) B->i ;
    const GB_Bp_TYPE *__restrict__ Bp = (GB_Bp_TYPE *) B->p ;
    #else
    const int64_t bvlen = B->vlen ;
    #endif

    #if GB_B_IS_BITMAP
    const int8_t *__restrict__ Bb = B->b ;
    #endif

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

    // zombie count for this threadblock
    uint64_t zc = 0 ;

    thread_block_tile<tile_sz> tile = 
        tiled_partition<tile_sz> (this_thread_block()) ;

    GB_M_NVALS (mnz) ;
    ASSERT (GB_M_IS_SPARSE || GB_M_IS_HYPER) ;
    int64_t cnz_in_bucket = end - start ;
    int all_in_one = (cnz_in_bucket == mnz) ;

    // Main loop over pairs 
    int64_t kk ;
    for (kk = start + blockIdx.x ; // warp per C(i,j)=A(:,i)'*B(:,j) dot product
         kk < end ;
         kk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get M(i,j) and C(i,j)
        //----------------------------------------------------------------------

        int64_t pair_id = all_in_one ? kk : Bucket [kk] ;
        int64_t i = Mi [pair_id] ;
        int64_t k = Ci [pair_id] >> 4 ;
        // assert: Ci [pair_id] & 0xF == GB_BUCKET_SPDN
        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBh_M (Mh, k) ;

        //----------------------------------------------------------------------
        // get A(:,i)
        //----------------------------------------------------------------------

        #if GB_A_IS_HYPER
        int64_t pA, pA_end ;
        GB_hyper_hash_lookup (GB_Ap_IS_32, GB_Aj_IS_32,
            Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits, i, &pA, &pA_end) ;
        #elif GB_A_IS_SPARSE
        int64_t pA     = Ap [i] ;
        int64_t pA_end = Ap [i+1] ;
        #else
        // A is bitmap or full: only pA is needed
        int64_t pA = avlen * i ;
        #endif

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        #if GB_B_IS_HYPER
        int64_t pB, pB_end ;
        GB_hyper_hash_lookup (GB_Bp_IS_32, GB_Bj_IS_32,
            Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits, j, &pB, &pB_end) ;
        #elif GB_B_IS_SPARSE
        int64_t pB     = Bp [j] ;
        int64_t pB_end = Bp [j+1] ;
        #else
        // B is bitmap or full: only pB is needed
        int64_t pB = bvlen * j ;
        #endif

        //----------------------------------------------------------------------
        // C(i,j) = A(:,i)'*B(:,j) using the entire threadblock
        //----------------------------------------------------------------------

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity
        int cij_exists = 0 ;

        #if ( GB_A_IS_FULL )
        {

            //------------------------------------------------------------------
            // A is full and B is sparse/hyper
            //------------------------------------------------------------------

            cij_exists = true ;
            for (int64_t p = pB + threadIdx.x ; p < pB_end ; p += blockDim.x)
            {
                int64_t k = Bi [p] ;        // next row index of B(:,j)
                // cij += A(k,i) * B(k,j)
                GB_GETA ( aki, Ax, pA+k, ) ;           // aki = A(k,i)
                GB_GETB ( bkj, Bx, p, ) ;              // bkj = B(k,j)
                // cij += aki * bkj
                GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
            }

        }
        #elif ( GB_A_IS_BITMAP )
        {
            //------------------------------------------------------------------
            // A is bitmap and B is sparse/hyper
            //------------------------------------------------------------------

            for (int64_t p = pB + threadIdx.x ; p < pB_end ; p += blockDim.x)
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

            //------------------------------------------------------------------
            // A is sparse/hyper and B is full
            //------------------------------------------------------------------

            cij_exists = true ;
            for (int64_t p = pA + threadIdx.x ; p < pA_end ; p += blockDim.x)
            {
                int64_t k = Ai [p] ;        // next row index of A(:,i)
                // cij += A(k,i) * B(k,j)
                GB_GETA ( aki, Ax, p, ) ;               // aki = A(i,k)
                GB_GETB ( bkj, Bx, pB+k, ) ;            // bkj = B(j,k)
                // cij += aik * bjk
                GB_MULTADD ( cij, aki, bkj, i, k, j) ;
                GB_DOT_TERMINAL (cij) ;     // break if cij == terminal
            }

        }
        #elif ( GB_B_IS_BITMAP )
        {

            //------------------------------------------------------------------
            // A is sparse/hyper and B is bitmap
            //------------------------------------------------------------------

            for (int64_t p = pA + threadIdx.x ; p < pA_end ; p += blockDim.x)
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

        //----------------------------------------------------------------------
        // save C(i,j) or declare it a zombie
        //----------------------------------------------------------------------

        GB_CIJ_EXIST_POSTCHECK

        //----------------------------------------------------------------------
        // reduce sum per-thread values to a single scalar, get OR of flag
        //----------------------------------------------------------------------

        // Do vote here for control
        cij_exists = tile.any (cij_exists) ;
        tile.sync ( ) ;

        #if !GB_C_ISO
        if (cij_exists)
        {
            // FIXME: the ANY monoid needs cij_exists for each thread
            cij = GB_cuda_tile_reduce_ztype (tile, cij) ;
        }
        #endif

        // write result for this block to global mem
        if (threadIdx.x == 0)
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
                zc++ ;
                Ci [pair_id] = GB_ZOMBIE (i) ;
            }
        }
        //__syncthreads(); 
    }

    //--------------------------------------------------------------------------
    // sum up the global zombie count
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && zc > 0)
    {
        GB_cuda_atomic_add <uint64_t> (&(C->nzombies), zc) ;
    }
}

