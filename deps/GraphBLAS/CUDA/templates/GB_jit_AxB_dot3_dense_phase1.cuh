//------------------------------------------------------------------------------
// GB_jit_AxB_dot3_dense_phase1: phase1 for dot3, A and B are bitmap/full
//------------------------------------------------------------------------------

// dense phase1: symbolic load balancing and data partition
// to assign work to different 'buckets' for later compute

//  This kernel scans the non-zero pattern in A and B, takes into account the
//  mask and computes total work required to form C. Then it classifies each
//  dot product into a set of buckets for efficient compute. 

#pragma once

#include <limits>
#include "GB_cuda_kernel.h"
#include "GB_mxm_shared_definitions.h"
#include "GB_cuda_buckets.h"
#include <cub/block/block_scan.cuh>
#include <cooperative_groups.h>

using namespace cooperative_groups;

//------------------------------------------------------------------------------
// GB_jit_AxB_dot3_dense_phase1: lookup i,j pairs and store in Mi, Ci 
//------------------------------------------------------------------------------

// GB_AxB_dense_phase1 is a CUDA kernel that scans all entries in M and
// assigns i,j coordinates for each entries and stores in Mi and Ci. 

template<typename T_M, uint64_t srcode, int chunk_size = 128>
__global__ void GB_jit_AxB_dot3_dense_phase1
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
    const int64_t mvlen = M->vlen ;
//  const int64_t mnz =  GB_nnz(M) ;
    const GB_M_NVALS (mnz) ;

    int64_t *__restrict__ Ci = C->i ;   // for zombies, or bucket assignment

    // Ci [p] for an entry C(i,j) contains either GB_FLIP(i) if C(i,j) is a
    // zombie, or (k << 4) + bucket otherwise, where C(:,j) is the kth vector
    // of C (j = Ch [k] if hypersparse or j = k if standard sparse), and
    // where bucket is the bucket assignment for C(i,j).
    // bucket can be recovered from Ci by bucket = Ci & 0xF

    // ASSERT (mnz > 0) ;
    // ASSERT (gridDim.x <= mnz) ;

    // shared cache used for coordinate search
    __shared__ int64_t ks [chunk_size] ;

    //--------------------------------------------------------------------------
    // assign all entries of C to the buckets
    //--------------------------------------------------------------------------

    // all threads in this block will compute the same values for these:
    int64_t pfirst, plast, kfirst, klast ;

    int64_t chunk_max = GB_ICEIL (mnz, chunk_size) ;
    //      (mnz + chunk_size -1)/chunk_size;
    for ( int64_t chunk = blockIdx.x;
                  chunk < chunk_max;
                  chunk += gridDim.x )
    {

        //----------------------------------------------------------------------
        // determine the work done by this iteration, "chunk"
        //----------------------------------------------------------------------

        // The slice for each task contains entries pfirst:plast-1 of M and C.
        // This iteration "chunk" computes Ci and Cx [pfirst...plast-1], using
        // Mi and Mx [pfirst:plast-1].  All threads in the thread block are
        // used for this "chunk".
        pfirst = chunk_size * chunk ;
        plast  = pfirst + chunk_size ;
        // plast = GB_IMIN (plast, mnz) ;
        if (plast > mnz) plast = mnz ;
        int64_t my_chunk_size = plast - pfirst ;

        // find the first vector of the slice for this chunk: the
        // vector that owns the entry Mi [pfirst] and Mx [pfirst].
        kfirst = GB_search_for_vector_device (pfirst, Mp, 0, mnvec, mvlen) ;

        // find the last vector of the slice for task blockIdx.x: the
        // vector that owns the entry Mi [plast-1] and Mx [plast-1].
        klast = GB_search_for_vector_device (plast-1, Mp, kfirst, mnvec, mvlen);

        // number of vectors in C and M for this "chunk" iteration, where
        // Mp [kfirst:klast] will be operated on.
        int64_t nk = klast - kfirst + 1 ;

        //----------------------------------------------------------------------
        // fill ks to find all indices
        //----------------------------------------------------------------------

        // search for k values for each entry pfirst:plast-1
        float slope = ((float) nk) / ((float) my_chunk_size) ;
        int64_t mnvec1 = mnvec - 1 ;
        for (int64_t kk = threadIdx.x ; kk < my_chunk_size ; kk += blockDim.x)
        {
            // get a rough estimate of k for the kkth entry in ks
            int64_t k = kfirst + (int64_t) (slope * ((float) kk)) ;
            // k cannot be smaller than kfirst, but might be bigger than
            // mnvec-1, so ensure it is in the valid range, kfirst to mnvec-1
            // k = GB_IMIN (k, mnvec-1) ;
            if (k > mnvec1) k = mnvec1 ; 
            // look for p in Mp, where p is in range pfirst:plast-1
            // where pfirst >= 0 and plast < mnz
            int64_t p = kk + pfirst ;
            // linear-time search for the k value of the pth entry
            while ( Mp [ k + 1 ] <= p ) k++ ;
            while ( Mp [ k     ] >  p ) k-- ;
            ks [kk] = k ;
        }
        this_thread_block().sync();

        //----------------------------------------------------------------------
        // assign entries in C(i,j) to the buckets
        //----------------------------------------------------------------------

        for ( int64_t pM = pfirst + threadIdx.x;
                      pM < pfirst + my_chunk_size;
                      pM += blockDim.x )
        {
            int64_t k = ks [pM - pfirst] ;  // get the k value of Mi,Mx [pM].
            // j = k or j = Mh [k] if C and M are hypersparse, but j is not
            // needed here.

            #if GB_MASK_STRUCT
            {
                // no need to check the value of M(i,j); no prezombies
                Ci[pM] = (k << 4) ;
            }
            #else
            {
                bool mij = (bool) GB_MCAST (Mx,pM,) ;
                int64_t i = Mi [ pM ] ;
                // FIXME: no need for k<<4, just place k or GB_FLIP(i) in Ci
                Ci[pM] = (!mij) * ( GB_FLIP(i) << 4)
                       +   mij  * ((k<<4) ) ;
            }
            #endif
        }
    }
}

