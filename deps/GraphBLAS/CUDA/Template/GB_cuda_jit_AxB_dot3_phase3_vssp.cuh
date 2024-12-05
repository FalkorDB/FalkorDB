//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_AxB_dot3_phase3_vssp.cuh
//------------------------------------------------------------------------------

// This version uses a binary-search algorithm, when the sizes nnzA and nnzB
// are far apart in size, neither is very spare nor dense, for any size of N.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

//  int64_t start          <- start of vector pairs for this kernel
//  int64_t end            <- end of vector pairs for this kernel
//  int64_t *Bucket        <- array of pair indices for all kernels 
//  GrB_Matrix C         <- result matrix 
//  GrB_Matrix M         <- mask matrix
//  GrB_Matrix A         <- input matrix A
//  GrB_Matrix B         <- input matrix B

__global__ void GB_cuda_AxB_dot3_phase3_vssp_kernel
(
    int64_t start,
    int64_t end,
    int64_t *Bucket,    // do the work defined by Bucket [start:end-1]
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

    // zombie count (only maintained by threadIdx.x == zero)
    uint64_t zc = 0 ;

    GB_M_NVALS (mnz) ;
    int all_in_one = ( (end - start) == mnz ) ;

    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( this_thread_block());

    // Main loop over pairs in Bucket [start:end-1]
    for (int64_t kk = start+ blockIdx.x; 
                 kk < end ;  
                 kk += gridDim.x)
    {

        int64_t pair_id = all_in_one ? kk : Bucket[ kk ];

        int64_t i = Mi[pair_id];
        int64_t k = Ci[pair_id] >> 4;
        // assert: Ci [pair_id] & 0xF == GB_BUCKET_VSSP

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

        int64_t nnzA = pA_end - pA;
        int64_t nnzB = pB_end - pB;

        //Search for each nonzero in the smaller vector to find intersection 

        if (nnzA <= nnzB)
        {
            //------------------------------------------------------------------
            // A(:,i) is very sparse compared to B(:,j)
            //------------------------------------------------------------------

            while (pA+ threadIdx.x< pA_end && pB< pB_end)
            {
                int64_t ia = Ai [pA+ threadIdx.x] ;
                int64_t ib = Bi [pB] ;
                 /*
                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    pA++ ;
                }
                */
                pA += ( ia < ib )*blockDim.x;
                if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    // discard all entries B(ib:ia-1,j)
                    int64_t pleft = pB + 1 ;
                    int64_t pright = pB_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                    //ASSERT (pleft > pB) ;
                    pB = pleft ;
                }
                else if (ia == ib) // ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT_MERGE (pA, pB);
                    //GB_DOT_TERMINAL (cij) ;   // break if cij == terminal
                    pA+= blockDim.x ;
                    pB++ ;
                }
            }
        }
        else
        {
            //------------------------------------------------------------------
            // B(:,j) is very sparse compared to A(:,i)
            //------------------------------------------------------------------

            while (pA < pA_end && pB+ threadIdx.x < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB + threadIdx.x] ;

                pB += ( ib < ia)*blockDim.x;

                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    // discard all entries A(ia:ib-1,i)
                    int64_t pleft = pA + 1 ;
                    int64_t pright = pA_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                    //ASSERT (pleft > pA) ;
                    pA = pleft ;
                }
                /*
                else if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    pB++ ;
                }
                */
                else if (ia == ib)// ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT_MERGE (pA, pB) ;
                    //GB_DOT_TERMINAL (cij) ;   // break if cij == terminal
                    pA++ ;
                    pB+=blockDim.x ;
                }
            }

        }
        GB_CIJ_EXIST_POSTCHECK ;
	this_thread_block().sync();

	cij_exists = tile.any( cij_exists) ;
	tile.sync ( ) ;
      
	#if  !GB_C_ISO
        if ( cij_exists)
        {
	    cij = GB_cuda_tile_reduce_ztype (tile, cij) ;
	}
        #endif

	if (threadIdx.x == 0) 
	{
            if (cij_exists)
	    {
	        Ci[pair_id] = i ;
	        GB_PUTC (cij, Cx, pair_id) ;
            }
	    else 
	    {
		zc++; 
		//printf(" %lld, %lld is zombie %d!\n",i,j,zc);
		Ci[pair_id] = GB_ZOMBIE( i ) ;
            }
	}
    }
    this_thread_block().sync();

    //--------------------------------------------------------------------------
    // update the zombie count
    //--------------------------------------------------------------------------

    if (threadIdx.x ==0 && zc > 0)
    {
        // this threadblock accumulates its zombie count into the global
        // zombie count
        //printf("vssp warp %d zombie count = %d\n", blockIdx.x, zc);
        GB_cuda_atomic_add<uint64_t>( &(C->nzombies), zc) ;
        //printf(" vssp Czombie = %lld\n",C->nzombies);
    }
}

