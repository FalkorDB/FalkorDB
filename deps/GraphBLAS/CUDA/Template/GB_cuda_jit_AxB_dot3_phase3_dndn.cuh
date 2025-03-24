//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase3_dndn.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This CUDA kernel produces the semiring product of two dense matrices of
// types GB_A_TYPE and GB_B_TYPE and common index space size n, to an output
// matrix of type GB_C_TYPE. The matrices are dense, with uniform non-zeros and
// sparsity patterns.  ie. we want to produce C = A'*B in the sense of the
// given semi-ring.

// This version uses a simple warp-based dense dot product algorithm, when the
// vectors coming from both A and B are dense, for any size of N.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x. s= 32 with a variable number
// of active threads = min( min(nzA, nzB), 32) 

// Thus, threadblock b owns a semi-ring dot product on a pair of vectors. 
// The work is to load the data, do the multiply and add work and finally 
// reduce this data to a scalar, and write it to Cx[pair].

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_dndn_kernel
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase3_dndn_kernel
(
    GrB_Matrix C,   // result matrix
    GrB_Matrix M,   // mask matrix
    GrB_Matrix A,   // input matrix A
    GrB_Matrix B,   // input matrix B
    const void *theta
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *)A->x  ;
    #endif
    #if !GB_A_IS_PATTERN
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *)B->x  ;
    #endif
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *)C->x  ;
          GB_Ci_SIGNED_TYPE *__restrict__ Ci = (GB_Ci_SIGNED_TYPE *) C->i ;
    const GB_Mi_TYPE *__restrict__ Mi = (GB_Mi_TYPE *) M->i ;
    #if GB_M_IS_HYPER
    const GB_Mj_TYPE *__restrict__ Mh = (GB_Mj_TYPE *) M->h ;
    #endif
    // A and B are either bitmap or full
    #if GB_A_IS_BITMAP
    const int8_t  *__restrict__ Ab = A->b ;
    #endif
    #if GB_B_IS_BITMAP
    const int8_t  *__restrict__ Bb = B->b ;
    #endif

    // zombie count
    uint64_t zc = 0 ;

    GB_M_NVALS (mnz) ;

    // total items to be inspected
    int64_t vlen = A->vlen ;
    ASSERT (vlen == B->vlen) ;
    ASSERT (vlen > 0) ;

    //--------------------------------------------------------------------------
    // compute C(i,j) = A(:,i)'*B(:,j) for each entry in M(i,j)
    //--------------------------------------------------------------------------

    for (int64_t pM = blockIdx.x ; pM < mnz ; pM += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get M(i,j) and C(i,j)
        //----------------------------------------------------------------------

        int64_t i = Mi [pM] ;
        int64_t kth = Ci [pM] ;             // C(i,j) is in the kth vector of C
        bool cij_exists = false ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity

        //----------------------------------------------------------------------
        // The threadblock cooperates to compute a single entry C(i,j)
        //----------------------------------------------------------------------

        #ifndef GB_MASK_STRUCT
        // skip if C(i,j) is a prezombie
        if (kth >= 0)
        #endif
        {

            // j = kth or j = Mh [kth] if C and M are hypersparse
            int64_t j = GBh_M (Mh, kth) ;
            int64_t pA = vlen * i ;
            int64_t pB = vlen * j ;

            GB_DECLAREA (aki) ;
            GB_DECLAREB (bkj) ;

            #if GB_A_IS_FULL && GB_B_IS_FULL
            {
                cij_exists = true ;
                for (int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    GB_MULTADD ( cij, aki, bkj, i, k, j ) ; // cij += aki * bkj
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                    int8_t b = (Ab [pA+k] && Bb [pB+k]) ;
                    cij_exists |= b ;
                    if (b)
                    {
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                    }
                }
            }
            #elif GB_A_IS_FULL && GB_B_IS_BITMAP
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    if (Bb [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                        cij_exists = true ;
                    }
                }
            }
            #elif GB_A_IS_BITMAP && GB_B_IS_FULL
            {
                for ( int64_t k = threadIdx.x ; k < vlen ; k += blockDim.x)
                { 
                    if (Ab [pB+k])
                    {
                        GB_GETA (aki, Ax, pA+k, ) ;           // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB+k, ) ;           // bkj = B(k,j)
                        // cij += aki * bkj
                        GB_MULTADD ( cij, aki, bkj, i, k, j ) ;
                        cij_exists = true ;
                    }
                }
            }
            #endif
        }

        //----------------------------------------------------------------------
        // reduce per-thread sums to a single scalar
        //----------------------------------------------------------------------

        // FIXME: no need to do this if C(i,j) is a zombie (cij_exists is
        // always false), or if A and B are both full and C(i,j) is not a
        // zombie (cij_exists is always true).

        // FIXME: this only works if the size of the thread block is 32,
        // right?

        // Do vote here for control.
        thread_block_tile<32> tile = tiled_partition<32> (this_thread_block()) ;

        // FIXME: tile.any takes an int predicate, not bool. How does this work?
        cij_exists = tile.any (cij_exists) ;
        tile.sync();

        #if !GB_C_ISO
        // FIXME: the ANY monoid needs the cij_exists for each thread
        cij = GB_cuda_tile_reduce_ztype (tile, cij) ;
        #endif

        // FIXME: if A and B are full, and GB_MASK_STRUCT is true, cij_exists
        // is always true because vlen > 0 always holds for this kernel.

        // FIXME: if kth < 0, C(i,j) is a prezombie, and Ci [pM] already holds
        // GB_ZOMBIE (i).

        // write result for this block to global mem
        if (threadIdx.x == 0)
        {
            if (cij_exists)
            {
                // Cx [pM] = (GB_C_TYPE) cij
                GB_PUTC (cij, Cx, pM) ;
                Ci [pM] = i ;
            }
            else
            {
                // cij is a zombie
                zc++ ;
                Ci [pM] = GB_ZOMBIE (i) ;
            }
        }

        // __syncthreads ( ) ;

        if( threadIdx.x ==0 && zc > 0)
        {
            GB_cuda_atomic_add <uint64_t>( &(C->nzombies), zc) ;
        }
    }
}

