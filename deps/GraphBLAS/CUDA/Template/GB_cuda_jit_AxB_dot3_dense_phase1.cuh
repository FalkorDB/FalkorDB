//------------------------------------------------------------------------------
// GraphBLAS/CUDA/JitKernels/GB_cuda_jit_AxB_dot3_dense_phase1.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// phase1 for dot3, A and B are bitmap/full.
// dense phase1: symbolic load balancing and data partition.

// This kernel scans the non-zero pattern in A and B, takes into account the
// mask and computes total work required to form C. Then it computes the vector
// k that contains each entry C(i,j) that isn't a zombie, or sets C(i,j) to its
// zombie status.

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_dense_phase1_kernel: lookup i,k pairs and store in Ci 
//------------------------------------------------------------------------------

// GB_cuda_AxB_dot3_dense_phase1_kernel is a CUDA kernel that scans all entries
// in M and assigns i,j coordinates for each entries and stores in Mi and Ci. 
// A and B are both bitmap/full.  C and M are sparse/hypersparse.

__global__ void GB_cuda_AxB_dot3_dense_phase1_kernel
(
    // input/output:
    GrB_Matrix C,           // final output matrix
    const GrB_Matrix M      // mask matrix
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    const int64_t *__restrict__ Mp = M->p ;
    const int64_t *__restrict__ Mi = M->i ;
    #if !GB_MASK_STRUCT
    const GB_M_TYPE *__restrict__ Mx = (GB_M_TYPE *) M->x ;
    #endif
    const int64_t mnvec = M->nvec ;
    const GB_M_NVALS (mnz) ;

    int64_t *__restrict__ Ci = C->i ;   // for zombies, or vector k

    // Ci [p] for an entry C(i,j) contains either GB_FLIP(i) if C(i,j) is a
    // zombie, or k otherwise, where C(:,j) is the kth vector of C (j = Ch [k]
    // if hypersparse or j = k if standard sparse).

    //--------------------------------------------------------------------------
    // determine the vector k of all entries in C(i,j), one chunk at a time
    //--------------------------------------------------------------------------

#if 0
    __shared__ int64_t ks [chunk_size] ;
#endif

//  int64_t chunk_max = GB_ICEIL (mnz, chunk_size) ;
//  for (int64_t chunk = blockIdx.x ; chunk < chunk_max ; chunk += gridDim.x )

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < mnz ;
                 pfirst += gridDim.x << log2_chunk_size)
    {

        //----------------------------------------------------------------------
        // find the vector k that contains each entry C(i,j) in this chunk
        //----------------------------------------------------------------------

        // This threadblock works on Mi/Mx and Ci/Cx, in positions pfirst to
        // pfirst + my_chunk_size - 1.

#if 0
        int64_t my_chunk_size = GB_cuda_ek_slice (Mp, mnvec, mnz, pfirst,
            chunk_size, /* output: */ ks) ;
#else
        int64_t my_chunk_size, mnvec1 ;
        float slope ;
        int64_t kfirst = GB_cuda_ek_slice_setup (Mp, mnvec, mnz, pfirst,
            chunk_size, &my_chunk_size, &mnvec1, &slope) ;
#endif

        //----------------------------------------------------------------------
        // assign entries in C(i,j): either its vector k or its zombie status
        //----------------------------------------------------------------------

//      for (int64_t pM = pfirst + threadIdx.x ;
//                   pM < pfirst + my_chunk_size ;
//                   pM += blockDim.x)

        for (int64_t kk = threadIdx.x ; kk < my_chunk_size ; kk += blockDim.x)
        {

#if 0
            int64_t k = ks [kk] ;       // get the k value of Mi,Mx [pM].
#else
            int64_t k = GB_cuda_ek_slice_entry (kk, pfirst, Mp, mnvec1, kfirst,
                slope) ;
#endif

            int64_t pM = kk + pfirst ;

            #if GB_MASK_STRUCT
            {
                // no need to check the value of M(i,j); no prezombies
                Ci [pM] = k ;
            }
            #else
            {
                bool mij = (bool) GB_MCAST (Mx, pM, ) ;
                int64_t i = Mi [pM] ;
                Ci [pM] = (!mij) * (GB_FLIP (i))
                        +   mij  * (k) ;
            }
            #endif
        }
    }
}

