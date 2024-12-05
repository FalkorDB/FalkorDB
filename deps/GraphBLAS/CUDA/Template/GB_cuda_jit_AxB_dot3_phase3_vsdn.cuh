//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase3_vsdn.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//******************************************************************************
//  Sparse dot products in batch form, sparse - dense case. 
//  Each thread in this kernel is responsible for m vector-pairs(x,y), 
//  m = 256/sz, where sz is in {4, 16, 64, 256}
//  We know each non-zero on the sparse side will hit a dense value.
//  Parameters:

//  C         <- C result matrix 
//  M         <- Mask matrix 
//  A         <- A matrix to multiply, sparse 
//  B         <- B matrix to multiply, dense in sparse format? 
//******************************************************************************

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase3_vsdn_kernel
//------------------------------------------------------------------------------

__global__ void GB_cuda_AxB_dot3_phase3_vsdn_kernel
( 
    int64_t start,
    int64_t end,
    int64_t *Bucket,  // do the work in Bucket [start:end-1]
    GrB_Matrix C, 
    GrB_Matrix M, 
    GrB_Matrix A, 
    GrB_Matrix B,
    const void *theta
)
{

    // sparse-times-dense or dense-times-sparse
    #if !(((GB_A_IS_SPARSE || GB_A_IS_HYPER) &&         \
           (GB_B_IS_BITMAP || GB_B_IS_FULL))            \
            ||                                          \
          ((GB_B_IS_SPARSE || GB_B_IS_HYPER) &&         \
           (GB_A_IS_BITMAP || GB_A_IS_FULL)))
    #error "vsdn: for sparse-dense or dense-sparse cases only"
    #endif

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

    #if GB_A_IS_HYPER || GB_A_IS_SPARSE
    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Ap = A->p ;
    #else
    const int64_t avlen = A->vlen ;
    #endif

    #if GB_A_IS_BITMAP
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    #if GB_B_IS_HYPER || GB_B_IS_SPARSE
    const int64_t *__restrict__ Bi = B->i ;
    const int64_t *__restrict__ Bp = B->p ;
    #else
    const int64_t bvlen = B->vlen ;
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

    uint64_t zc = 0 ;       // zombie count

    GB_M_NVALS (mnz) ;
    ASSERT (GB_M_IS_SPARSE || GB_M_IS_HYPER) ;
    int64_t cnz_in_bucket = end - start ;
    int all_in_one = (cnz_in_bucket == mnz) ;

    for (int64_t kk = start + threadIdx.x + blockIdx.x*blockDim.x ;
                 kk < end ;
                 kk += gridDim.x*blockDim.x)
    {

        //----------------------------------------------------------------------
        // get the entry C(i,j)
        //----------------------------------------------------------------------

        int64_t pair_id = all_in_one ? kk : Bucket[ kk ];
        int64_t i = Mi [pair_id] ;
        int64_t k = Ci [pair_id] >> 4;  // vector of C encoded in phase1
        // assert: Ci [pair_id] & 0xF == GB_BUCKET_VSDN
        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBH_M (Mh, k) ;

        //----------------------------------------------------------------------
        // get A(:,i)
        //----------------------------------------------------------------------

        #if GB_A_IS_HYPER
        int64_t pA, pA_end ;
        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
            i, &pA, &pA_end) ;
        #elif GB_A_IS_SPARSE
        int64_t pA     = Ap[i] ;
        int64_t pA_end = Ap[i+1] ;
        #else
        // A is bitmap or full: only pA is needed
        int64_t pA = avlen * i ;
        #endif

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        #if GB_B_IS_HYPER
        int64_t pB, pB_end ;
        GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
            j, &pB, &pB_end) ;
        #elif GB_B_IS_SPARSE
        int64_t pB       = Bp [j] ;
        int64_t pB_end   = Bp [j+1] ;
        #else
        // B is bitmap or full: only pB is needed
        int64_t pB = bvlen * j ;
        #endif

        //----------------------------------------------------------------------
        // C(i,j) = A(:,i)'*B(:,j)
        //----------------------------------------------------------------------

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity
        bool cij_exists = false ;

        uint64_t my_nzombies = 0 ;

        #if ( GB_A_IS_FULL )
        {
            int64_t nnzB = pB_end - pB ;
            if (nnzB > 0)
            {

                //--------------------------------------------------------------
                // A is full and B is sparse/hyper
                //--------------------------------------------------------------

                cij_exists = true ;
                for (int64_t p = pB ; p < pB_end ; p++)
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

            for (int64_t p = pB ; p < pB_end ; p++)
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
                for (int64_t p = pA ; p < pA_end ; p++)
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

            for (int64_t p = pA ; p < pA_end ; p++)
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
        if (cij_exists)
        {
            // Cx [pair_id] = (GB_C_TYPE) cij
            GB_PUTC (cij, Cx, pair_id) ;
            Ci [pair_id] = i ;
        }
        else
        {
            my_nzombies++ ;
            Ci [pair_id] = GB_ZOMBIE (i) ;
        }

        // sum up the zombie count:
        thread_block_tile<tile_sz> tile =
            tiled_partition<tile_sz> (this_thread_block ()) ;
        zc += GB_cuda_tile_sum_uint64 (tile, my_nzombies) ;
    }

    if (threadIdx.x == 0 && zc > 0)
    {
        // this threadblock accumulates its zombie count into the global
        // zombie count
        GB_cuda_atomic_add <uint64_t>( &(C->nzombies), zc) ;
    }
}

