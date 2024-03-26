//------------------------------------------------------------------------------
// GraphBLAS/CUDA/JitKernels/GB_cuda_jit_AxB_dot3_phase3_mp.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
  
__global__ void GB_cuda_AxB_dot3_phase3_mp_kernel
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,    // do the work in Bucket [start:end-1]
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

    // A and B are either sparse or hypersparse
    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Bi = B->i ;
    const int64_t *__restrict__ Ap = A->p ;
    const int64_t *__restrict__ Bp = B->p ;
    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;
    ASSERT (GB_B_IS_HYPER || GB_B_IS_SPARSE) ;

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

    // zombie count
    int64_t zc = 0;

    // set thread ID
//  int tid_global = threadIdx.x+ blockDim.x* blockIdx.x;
    int tid = threadIdx.x;

    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());
    int all_in_one = ( (end - start) == (M->p)[(M->nvec)] ) ;

    // Main loop over pairs 
    int64_t kk ;
    for (kk = start+ blockIdx.x; // warp per C(i,j)=A(:,i)'*B(:,j) dot product
         kk < end;  
         kk += gridDim.x )
    {

        int64_t pair_id = all_in_one ? kk : Bucket [kk] ;
        int64_t i = Mi[pair_id];
        int64_t k = Ci[pair_id] >> 4;

        // j = k or j = Mh [k] if C and M are hypersparse
        int64_t j = GBH_M (Mh, k) ;

        // find A(:,i)
        int64_t pA_start, pA_end ;
        #if GB_A_IS_HYPER
        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits,
            i, &pA_start, &pA_end) ;
        #else
        pA_start = Ap[i] ;
        pA_end   = Ap[i+1] ;
        #endif

        int64_t ainz = pA_end - pA_start ;

        GB_DECLAREA (aki) ;
        GB_DECLAREB (bkj) ;
        GB_DECLARE_IDENTITY (cij) ;         // GB_Z_TYPE cij = identity

        int cij_exists = 0 ;       // FIXME: make a bool

        __shared__ int64_t Ai_s[shared_vector_size];
        int shared_steps_A = (ainz + shared_vector_size -1)/shared_vector_size;

        int64_t step_end = (shared_steps_A <= 1? ainz : shared_vector_size);
        for ( int64_t i = tid; i< step_end; i+= blockDim.x)
        {
            Ai_s[i] = Ai[ i + pA_start];
        }   
        this_thread_block().sync();
         

        // find B(:,j)
        int64_t pB_start, pB_end ;
        #if GB_B_IS_HYPER
        GB_hyper_hash_lookup (Bh, bnvec, Bp, B_Yp, B_Yi, B_Yx, B_hash_bits,
           j, &pB_start, &pB_end) ;
        #else
        pB_start = Bp[j] ;
        pB_end   = Bp[j+1] ;
        #endif

        int64_t bjnz = pB_end - pB_start;          // bjnz
        int shared_steps_B = (bjnz + shared_vector_size -1)/shared_vector_size;
         
        __shared__ int64_t Bj_s[shared_vector_size];

        step_end = (shared_steps_B <= 1 ? bjnz : shared_vector_size);
        for ( int64_t i =tid; i< step_end; i+= blockDim.x)
        {
            Bj_s[i] = Bi[ i + pB_start];
        }   
        this_thread_block().sync();

        //we want more than one intersection per thread
        while ( (shared_steps_A > 0) && (shared_steps_B > 0) )
        {
            int64_t awork = pA_end - pA_start;
            int64_t bwork = pB_end - pB_start;
            if ( shared_steps_A > 1) awork = shared_vector_size;  
            if ( shared_steps_B > 1) bwork = shared_vector_size;  
            int64_t nxy = awork + bwork;

            // ceil Divide by 32 = blockDim.x :
            int work_per_thread = (nxy + blockDim.x -1)/blockDim.x;
            int diag     = GB_IMIN( work_per_thread*tid, nxy);
            int diag_end = GB_IMIN( diag + work_per_thread, nxy);

            // bwork takes place of bjnz:
            int x_min = GB_IMAX( (diag - bwork) , 0);

            //awork takes place of ainz:
            int x_max = GB_IMIN( diag, awork);

            while ( x_min < x_max)
            {
                //binary search for correct diag break
                int pivot = (x_min +x_max) >> 1;
                int64_t Apiv =  Ai_s[pivot] ;
                int64_t Bpiv = Bj_s[diag -pivot -1] ;

                x_min = (pivot + 1)* (Apiv < Bpiv)  + x_min * (1 - (Apiv < Bpiv));
                x_max = pivot * (1 - (Apiv < Bpiv)) + x_max * (Apiv < Bpiv);

            }

            int xcoord = x_min;
            int ycoord = diag -x_min -1;
            int64_t Atest = Ai_s[xcoord] ;
            int64_t Btest = Bj_s[ycoord] ;
            if ( (diag > 0) && (diag < nxy ) && (ycoord >= 0 ) && (Atest == Btest)) 
            { 
                diag--; //adjust for intersection incrementing both pointers 
            }
            // two start points are known now
            int tx_start = xcoord; // +pA_start;
            int ty_start = diag -xcoord; // +pB_start; 


            x_min = GB_IMAX( (diag_end - bwork), 0); //bwork replace bjnz
            x_max = GB_IMIN( diag_end, awork);      //awork replace ainz

            while ( x_min < x_max) 
            {
                int pivot = (x_min +x_max) >> 1;
                int64_t Apiv = Ai_s[pivot] ;
                int64_t Bpiv = Bj_s[diag_end -pivot -1] ;

                x_min = (pivot + 1)* (Apiv < Bpiv)   + x_min * (1 - (Apiv < Bpiv));
                x_max = pivot * (1 - (Apiv < Bpiv))  + x_max * (Apiv < Bpiv);
            }

            xcoord = x_min;
            ycoord = diag_end -x_min -1;

            // two end points are known now
            int tx_end = xcoord; // +pA_start; 
            int ty_end = diag_end - xcoord; // + pB_start; 

            //merge-path dot product
            int64_t pA = tx_start;       // pA
            int64_t pB = ty_start;       // pB

            while ( pA < tx_end && pB < ty_end ) 
            {
                int64_t Aind = Ai_s[pA] ;
                int64_t Bind = Bj_s[pB] ;
                #if GB_IS_PLUS_PAIR_REAL_SEMIRING && GB_Z_IGNORE_OVERFLOW
                    cij += (Aind == Bind) ;
                #else
                    if (Aind == Bind)
                    {
                        // cij += aki * bkj
                        GB_DOT_MERGE (pA + pA_start, pB + pB_start) ;
                        // TODO check terminal condition, using tile.any
                    }
                #endif
                pA += (Aind <= Bind) ;
                pB += (Aind >= Bind) ;
            }
            GB_CIJ_EXIST_POSTCHECK ;

            this_thread_block().sync();

            if  (  (shared_steps_A >= 1) 
            && (shared_steps_B >= 1) 
            && ( Ai_s[awork-1] == Bj_s[bwork-1]) )
            {

                pA_start += shared_vector_size;
                shared_steps_A -= 1;
                if (shared_steps_A == 0) break;
                pB_start += shared_vector_size;
                shared_steps_B -= 1;
                if (shared_steps_B == 0) break;

                step_end = ( (pA_end - pA_start) < shared_vector_size ? (pA_end - pA_start) : shared_vector_size);
                for ( int64_t i = tid; i< step_end; i+= blockDim.x)
                {
                    Ai_s[i] = Ai[ i + pA_start];
                }   
                this_thread_block().sync();

                step_end = ( (pB_end - pB_start) < shared_vector_size ? (pB_end - pB_start) : shared_vector_size);
                for ( int64_t i = tid; i< step_end; i+= blockDim.x)
                {
                    Bj_s[i] = Bi[ i + pB_start];
                }   
                this_thread_block().sync();

            } 
            else if ( (shared_steps_A >= 1) && (Ai_s[awork-1] < Bj_s[bwork-1] ))
            {
                pA_start += shared_vector_size;
                shared_steps_A -= 1;
                if (shared_steps_A == 0) break;

                step_end= ( (pA_end - pA_start) < shared_vector_size ? (pA_end - pA_start) : shared_vector_size);
                for ( int64_t i = tid; i< step_end; i+= blockDim.x)
                {
                    Ai_s[i] = Ai[ i + pA_start];
                }   
                this_thread_block().sync();

            }
            else if ( (shared_steps_B >= 1) && (Bj_s[bwork-1] < Ai_s[awork-1]) )
            {
                pB_start += shared_vector_size;
                shared_steps_B -= 1;
                if (shared_steps_B == 0) break;

                step_end = ( (pB_end - pB_start) < shared_vector_size ? (pB_end - pB_start) : shared_vector_size);
                for ( int64_t i = tid; i< step_end; i+= blockDim.x)
                {
                    Bj_s[i] = Bi[ i + pB_start];
                }   
                this_thread_block().sync();
            }
        } // end while shared_steps A > 0 && shared_steps_B >0

        //tile.sync( ) ;

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
            cij = GB_cuda_warp_reduce_ztype (tile, cij) ;
        }
        #endif

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
               Ci [pair_id] = GB_FLIP (i) ;
            }
        }
        //__syncthreads(); 
    }

    //--------------------------------------------------------------------------
    // sum up the global zombie count
    //--------------------------------------------------------------------------

    if( tid ==0 && zc > 0)
    {
        GB_cuda_atomic_add <uint64_t>( &(C->nzombies), (uint64_t) zc) ;
    }
}

