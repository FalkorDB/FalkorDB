//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_add_sparse.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A+B kernel on the GPU.   A and B cannot be jumbled on input, and C is
// returned as unjumbled.  No mask is exploited.

// phase1 (GPU): expand A and B into their COO form

// phase 2 (GPU):  use the merge-path method to find outer tasks for C, each
//      of which are fairly large (16K entries to handle in A and B).

// phase 3 (GPU): compute the size of the set union of A and B for each task t
//      of C.  Let S[t] = size of intersection of A_t and B_t, where A_t and
//      B_t are the set of entries in the t-th task found in phase 2.  Then #
//      of entries in C_t is |A_t| + |B_t| - S[t].  Use a set intersection
//      since it's faster to compute than set union; it can trim its inputs.
//      Each task C_t is completely independent of other tasks, because of the
//      merge-path construction in phase 2.

// phase 4 (CPU):  compute the cumulative sum of |C_t|.  This gives the total
//      size of C.  Allocate C (Ci, Cj, Cx each of size cnz).  Cj will be
//      compressed later into Cp and Ch, much like the CUDA builder kernel.

// phase 5 (GPU): same tasks as phase 3 but now compute the output matrix C in
//      coordinate form: (i,j,cij), using a set-union merge-path method.  The
//      output of this phase is the final matrix C in sorted coordinate form:
//      Cj, Ci, Cx, of size cnz.

// phases 6 and 7 (GPU):  convert (Cj,Ci,Cx) into the hypersparse form
//      Ch,Cp,Ci,Cx, using templated methods from CUDA/builder

//------------------------------------------------------------------------------

#include "template/GB_cuda_ek_slice.cuh"
#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_threadblock_sum_uint64.cuh"
#include "include/GB_add_shared_definitions.h"

// FIXME: move these to CUDA/include/GB_cuda_geometry.hpp, and tune them:

// for phase1 (create Aj, Bj)
#define CHUNKSIZE1 256
#define LOG2_CHUNKSIZE1 8

// for phase2, defines the size of the large outer tasks:
#define CHUNKSIZE2 16384
#define LOG2_CHUNKSIZE2 14

// for phase3 and phase5 (computing the set merge):
#define CHUNKSIZE3 1024
#define LOG2_CHUNKSIZE3 10

// for phase1, phase3, and phase5:
// #define BLOCKDIM1 256
#define BLOCKDIM1 32

// for phase2 only: using a single thread per threadblock
#define BLOCKDIM2 32

// for phase6 and phase7: (same geometry as the builder kernel)
#define BLOCKDIM6 128
#define CHUNKSIZE6 256
#define LOG2_CHUNKSIZE6 8

#define GB_FREE_WORKSPACE                               \
{                                                       \
    GB_FREE_MEMORY (&Task_Astart, Task_Astart_mem) ;    \
    GB_FREE_MEMORY (&Task_Bstart, Task_Bstart_mem) ;    \
    GB_FREE_MEMORY (&Task_Cstart, Task_Cstart_mem) ;    \
    GB_FREE_MEMORY (&Aj , Aj_mem ) ;                    \
    GB_FREE_MEMORY (&Bj , Bj_mem ) ;                    \
    GB_FREE_MEMORY (&W_0, W_0_mem) ;                    \
    GB_FREE_MEMORY (&W_6, W_6_mem) ;                    \
    GB_FREE_MEMORY (&W_7, W_7_mem) ;                    \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    /* GB_phbix_free (C) is not called; */      \
    /* it is done in the caller if needed */    \
    GB_FREE_WORKSPACE ;                         \
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase1: construct Aj and Bj
//------------------------------------------------------------------------------

// FIXME: instead of global Aj and Bj arrays, use GB_cuda_ek_slice_search to
// compute kfirst for each chunk (see shared_version), which is 1/256 smaller.
// Then use GB_cuda_ek_slice_entry to lookup Aj [p] and Bj [p] as needed.
// (see getk in shared_version).

#include "template/GB_cuda_extractTuples_template.cuh"

__global__ void GB_cuda_add_sparse_phase1
(
    // outputs:
    GB_Aj_TYPE *__restrict__ Aj, // j = Aj [p], col index of pth entry of A
    GB_Bj_TYPE *__restrict__ Bj, // j = Bj [p], col index of pth entry of B
    // inputs:
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t anz,
    const int64_t bnz
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    const int64_t anvec = A->nvec ;

    const GB_Bp_TYPE *__restrict__ Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bj_TYPE *__restrict__ Bh = (GB_Bj_TYPE *) B->h ;
    const int64_t bnvec = B->nvec ;

    //--------------------------------------------------------------------------
    // construct the column indices for A and B
    //--------------------------------------------------------------------------

    GB_cuda_extractTuples_template <GB_Ap_TYPE, GB_Aj_TYPE, GB_A_IS_HYPER,
        CHUNKSIZE1, LOG2_CHUNKSIZE1> (Aj, Ap, Ah, anvec, anz) ;

    GB_cuda_extractTuples_template <GB_Bp_TYPE, GB_Bj_TYPE, GB_B_IS_HYPER,
        CHUNKSIZE1, LOG2_CHUNKSIZE1> (Bj, Bp, Bh, bnvec, bnz) ;
}

//------------------------------------------------------------------------------
// mergepath: search the diagonal of A and B
//------------------------------------------------------------------------------

// The mergepath method searches the lists Ai,Aj [0:na-1] and Bi,Bj [0:nb-1]
// along the given diagonal (shorthand: A [0:na-1] and B [0:nb-1]).  It
// computes astart and bstart as the first positions in A [astart] and B
// [bstart] to start the merge.

// FUTURE: other uses of mergepath will need a single pair of arrays, Ai and
// Bi, not Aj and Bj.  This template could extend to those cases using a
// template parameter and "if constexpr (...)" to control access to Aj and Bj.
// Then place this method in its own template file, in CUDA/slice/template.

template
<
    typename T,     // integer type to use for scalars (int32_t or int64_t)
    typename T_Ai,  // type of Ai
    typename T_Aj,  // type of Aj
    typename T_Bi,  // type of Bi
    typename T_Bj   // type of Bj
>
__device__ void mergepath
(
    // outputs:
    T *astart,      // mergepath results: starting position in A
    T *bstart,      // mergepath results: starting position in B
    // inputs:
    const T na,                     // # of entries in A
    const T nb,                     // # of entries in B
    const T diag,                   // diagonal to search
    const T_Ai *__restrict__ Ai,    // row indices of A
    const T_Aj *__restrict__ Aj,    // col indices of A
    const T_Bi *__restrict__ Bi,    // row indices of B
    const T_Bj *__restrict__ Bj     // col indices of B
)
{

    //--------------------------------------------------------------------------
    // find the range of positions in Ai,Aj to search
    //--------------------------------------------------------------------------

    T amin = GB_IMAX (diag - nb, 0) ;
    T amax = GB_IMIN (diag, na) ;

    //--------------------------------------------------------------------------
    // binary search along the diagonal
    //--------------------------------------------------------------------------

    while (amin < amax)
    {

        //----------------------------------------------------------------------
        // cut the diagonal (amin:amax) in half
        //----------------------------------------------------------------------

        T pA = (amin + amax) >> 1 ;
        T pB = diag - pA - 1 ;

        //----------------------------------------------------------------------
        // compare the entries at A [pA] and B [pB]
        //----------------------------------------------------------------------

        // afirst is true if A [pA] comes before B [pB]
        T_Aj jA = Aj [pA] ;     // col index if A [pA]
        T_Bj jB = Bj [pB] ;     // col index of B [pB]
        T afirst = ((jA < jB) || (jA == jB && Ai [pA] < Bi [pB])) ;

        //----------------------------------------------------------------------
        // if (afirst) amin = pA+1 else amax = pA
        //----------------------------------------------------------------------

        amin = (pA + 1) * (afirst) + amin * (1-afirst) ;
        amax = pA * (1-afirst)     + amax * (afirst) ;
    }

    //--------------------------------------------------------------------------
    // finalize the search
    //--------------------------------------------------------------------------

    (*bstart) = diag - amin ;
    T bprior = (*bstart) - 1 ;
    if ((amin < na) && (bprior >= 0) &&
        (Ai [amin] == Bi [bprior]) && (Aj [amin] == Bj [bprior]))
    {
        // The last entry in B of the prior partition matches the first entry
        // in A of the current partition.  Adjust the partitions by moving the
        // last entry of B in the prior partition (at bstart-1) into this
        // current partition as the first entry in B for this partition
        // (revising bstart).
        (*bstart)-- ;
    }
    (*astart) = amin ;
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase2: construct tasks
//------------------------------------------------------------------------------

// This method divides the work to compute C=A+B into large tasks, each of
// which is found via the mergepath method on the coordinate forms of A and B.
// The tasks are later computed using a single threadblock each, in subsequent
// phases.  This kernel launch uses a single thread to compute the starting
// points of a single task.  As a result, there will be a lot of warp
// divergence in this kernel, but very little work is done since the tasks are
// very large.  The work should be done on the GPU because this kernel reads
// the Ai, Aj, Bi, and Bj arrays.  That data should remain on the GPU for
// subsequent phases.

// Each large task operates on CHUNKSIZE2 entries in A and B.

__global__ void GB_cuda_add_sparse_phase2
(
    // outputs:
    int64_t *__restrict__ Task_Astart,      // array of size # tasks+1
    int64_t *__restrict__ Task_Bstart,      // array of size # tasks+1
    // inputs:
    const int64_t ntasks,                   // # of outer tasks
    const GB_Ai_TYPE *__restrict__ Ai,      // row indices of A
    const GB_Aj_TYPE *__restrict__ Aj,      // col indices of A
    const GB_Bi_TYPE *__restrict__ Bi,      // row indices of B
    const GB_Bj_TYPE *__restrict__ Bj,      // col indices of B
    const int64_t anz,
    const int64_t bnz
)
{

    //--------------------------------------------------------------------------
    // iterate through all outer tasks: one thread per task
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0)
    {

        for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
        {

            //------------------------------------------------------------------
            // construct the task via mergepath
            //------------------------------------------------------------------

            // compute the starting point of each task using merge-path method

            int64_t diag = t << LOG2_CHUNKSIZE2 ;
            int64_t astart, bstart ;
            mergepath <int64_t, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&astart, &bstart, anz, bnz, diag, Ai, Aj, Bi, Bj) ;

            //------------------------------------------------------------------
            // save the results in global memory
            //------------------------------------------------------------------

            Task_Astart [t] = astart ;
            Task_Bstart [t] = bstart ;
        }

        //----------------------------------------------------------------------
        // sentinel value for the last task
        //----------------------------------------------------------------------

        if (blockIdx.x == 0)
        {
            Task_Astart [ntasks] = anz ;
            Task_Bstart [ntasks] = bnz ;
        }
    }
}

//------------------------------------------------------------------------------
// set_intersection_size: compute intersection of A and B for a single thread
//------------------------------------------------------------------------------

// compute the size of the set intersection of a chunk of Ai,Aj [pa:pa_end-1]
// and Bi,Bj [pb:pb_end-1]

template
<   
    typename T_Ai,      // type of Ai_chunk
    typename T_Aj,      // type of Aj_chunk
    typename T_Bi,      // type of Bi_chunk
    typename T_Bj       // type of Bj_chunk
>
__device__ int set_intersection_size
(
    // inputs:
    int pa,
    int pb,
    const int pa_end,
    const int pb_end,
    const T_Ai *__restrict__ Ai_chunk,
    const T_Aj *__restrict__ Aj_chunk,
    const T_Bi *__restrict__ Bi_chunk,
    const T_Bj *__restrict__ Bj_chunk
)
{
    int intersection = 0 ;
    while (pa < pa_end && pb < pb_end)
    {
        // get the two entries to compare: A [pa] and B [pb]
        const GB_Ai_TYPE iA = Ai_chunk [pa] ;
        const GB_Aj_TYPE jA = Aj_chunk [pa] ;
        const GB_Bi_TYPE iB = Bi_chunk [pb] ;
        const GB_Bj_TYPE jB = Bj_chunk [pb] ;
        // compare the two entries
        int afirst = ((jA < jB) || (jA == jB && iA < iB)) ;
        int amatch = (jA == jB && iA == iB) ;
        // count the size of the intersection
        intersection += amatch ;
        // advance pa and pb
        pa += ( afirst || amatch) ;
        pb += (!afirst || amatch) ;
    }
    return (intersection) ;
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase3: compute intersection of A and B for each task
//------------------------------------------------------------------------------

// C, A, and B have been split into tasks.  For task t, the entries in A are
// in Ax, Ai, Aj [Task_Astart [t] ... Task_Astart [t+1]-1], and in B 
// at Bx, Bi, Bj [Task_Bstart [t] ... Task_Bstart [t+1]-1].

// Each task is done by a single threadblock.  All threads take part in the
// work by using the mergepath method to split the work for each thread.

// FIXME: if A and B are disjoint, then Task_Cstart [0..ntasks] can be
// computed as:
//
//      for t = 0 to ntasks-1
//          pA     = Task_Astart [t] ;
//          pA_end = Task_Astart [t+1] ;
//          pB     = Task_Bstart [t] ;
//          pB_end = Task_Bstart [t+1] ;
//          Task_Cstart [t] = (pA_end - pA) + (pB_end - pB) ;
//
// GB_encodify_ewise / enumify_ewise needs to add the A_and_B_disjoint flag
// to exploit this.

__global__ void GB_cuda_add_sparse_phase3
(
    // outputs:
    int64_t *Task_Cstart,       // array of size # tasks+1; Task_Cstart [t]
                                // is the size of the set union
                                // of A and B for task t
    // inputs:
    const int64_t *__restrict__ Task_Astart,    // array of size # tasks+1
    const int64_t *__restrict__ Task_Bstart,    // array of size # tasks+1
    const int64_t ntasks,                       // # of outer tasks
    const GB_Ai_TYPE *__restrict__ Ai,          // row indices of A
    const GB_Aj_TYPE *__restrict__ Aj,          // col indices of A
    const GB_Bi_TYPE *__restrict__ Bi,          // row indices of B
    const GB_Bj_TYPE *__restrict__ Bj           // col indices of B
)
{

    //--------------------------------------------------------------------------
    // iterate through all outer tasks: one threadblock per task
    //--------------------------------------------------------------------------

    for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the details of this task
        //----------------------------------------------------------------------

        int64_t pA     = Task_Astart [t] ;
        int64_t pA_end = Task_Astart [t+1] ;
        int64_t pB     = Task_Bstart [t] ;
        int64_t pB_end = Task_Bstart [t+1] ;
        int64_t AB_size = (pA_end - pA) + (pB_end - pB) ;   // |A_t| + |B_t|
        uint64_t AB_intersection = 0 ;

        //----------------------------------------------------------------------
        // compute the size of the set intersection for this task
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {

            //------------------------------------------------------------------
            // get the current chunk
            //------------------------------------------------------------------

            int na = GB_IMIN (pA_end - pA, CHUNKSIZE3) ;
            int nb = GB_IMIN (pB_end - pB, CHUNKSIZE3) ;
            int nab ;
            if (na < CHUNKSIZE3 && nb < CHUNKSIZE3)
            {
                // this is the very last chunk of the entire task; do it all
                nab = na + nb ;
            }
            else
            {
                // this chunk is in the middle of the task; just do the first
                // half of the diagonals of the na,nb chunk
                nab = GB_IMIN (na, nb) ;
            }

            // work_per_thread = ceil (nab / blockdim)
            int work_per_thread = GB_ICEIL (nab, blockDim.x) ;
            int diag = GB_IMIN (work_per_thread * threadIdx.x, nab) ;
            int diag_end  = GB_IMIN (diag + work_per_thread, nab) ;
            int diag_last = GB_IMIN (work_per_thread * blockDim.x, nab) ;

            // get pointers to the current chunk
            const GB_Ai_TYPE *Ai_chunk = Ai + pA ;
            const GB_Aj_TYPE *Aj_chunk = Aj + pA ;
            const GB_Bi_TYPE *Bi_chunk = Bi + pB ;
            const GB_Bj_TYPE *Bj_chunk = Bj + pB ;

            //------------------------------------------------------------------
            // each thread searches for its starting point
            //------------------------------------------------------------------

            int pa, pb ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa, &pb, na, nb, diag,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            //------------------------------------------------------------------
            // each thread searches for its ending point
            //------------------------------------------------------------------

            int pa_end, pb_end ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_end,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            //------------------------------------------------------------------
            // compute the size of the set intersection
            //------------------------------------------------------------------

            int my_intersection = set_intersection_size
                <GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (pa, pb, pa_end, pb_end,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;
            AB_intersection += my_intersection ;

            //------------------------------------------------------------------
            // find the last diagonal of the last thread
            //------------------------------------------------------------------

            // All threads find the last diagonal of all threads.
            // Alternatively: the last thread could broadcast pa_end and pb_end
            // of the last thread in the threadblock to the entire threadblock.

            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_last,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            pA += pa_end ;
            pB += pb_end ;
        }

        //----------------------------------------------------------------------
        // compute the size of C for this task
        //----------------------------------------------------------------------

        // all threads cooperate to sum up their AB_intersection, and the
        // result is summed into thread 0
        AB_intersection = GB_cuda_threadblock_sum_uint64 (AB_intersection) ;

        if (threadIdx.x == 0)
        {
            // |C_t| = |A_t union B_t| = |A_t| + |B_t| - |A_t intersection B_t|
            // FIXME: for eWiseMult, |C_t| = AB_intersection
            Task_Cstart [t] = AB_size - AB_intersection ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase5: compute pattern and values of C (coordinate form)
//------------------------------------------------------------------------------

__global__ void GB_cuda_add_sparse_phase5
(
    // outputs:
    GrB_Matrix C,
    GB_Cj_TYPE *Cj,             // size cnz+1, Cj [-1..cnz-1]
    // inputs:
    const int64_t *__restrict__ Task_Cstart,       // array of size # tasks+1
    const int64_t *__restrict__ Task_Astart,       // array of size # tasks+1
    const int64_t *__restrict__ Task_Bstart,       // array of size # tasks+1
    const int64_t ntasks,               // # of outer tasks
    const GB_Ai_TYPE *__restrict__ Ai,  // row indices of A
    const GB_Aj_TYPE *__restrict__ Aj,  // col indices of A
    const GB_Bi_TYPE *__restrict__ Bi,  // row indices of B
    const GB_Bj_TYPE *__restrict__ Bj,  // col indices of B
    const GrB_Matrix A,
    const GrB_Matrix B,
    const void *theta                   // theta scalar for index binary ops
    #if GB_IS_EWISEUNION
    , const GB_X_TYPE alpha_scalar      // alpha scalar, for eWiseUnion
    , const GB_Y_TYPE beta_scalar       // beta scalar, for eWiseUnion
    #endif
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #if !GB_C_ISO
    const GB_A_TYPE  *__restrict__ Ax = (GB_A_TYPE  *) A->x ;
    const GB_B_TYPE  *__restrict__ Bx = (GB_B_TYPE  *) B->x ;
          GB_C_TYPE  *__restrict__ Cx = (GB_C_TYPE  *) C->x ;
    #endif
          GB_Ci_TYPE *__restrict__ Ci = (GB_Ci_TYPE *) C->i ;

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    // cub::Block* workspace for ExclusiveSum of pc of each thread
    using BlockScan = cub::BlockScan <uint16_t, BLOCKDIM1,
        cub::BLOCK_SCAN_WARP_SCANS> ;
    __shared__ typename BlockScan::TempStorage W ;

    //--------------------------------------------------------------------------
    // compute the pattern and values of C = A+B
    //--------------------------------------------------------------------------

    for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the details of this task
        //----------------------------------------------------------------------

        int64_t pA     = Task_Astart [t] ;
        int64_t pA_end = Task_Astart [t+1] ;
        int64_t pB     = Task_Bstart [t] ;
        int64_t pB_end = Task_Bstart [t+1] ;
        int64_t pC     = Task_Cstart [t] ;
//      int64_t pC_end = Task_Cstart [t+1] ;        // not needed

        //----------------------------------------------------------------------
        // compute C = A+B for this task, while entries in A and B appear
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {

            //------------------------------------------------------------------
            // get the current chunk
            //------------------------------------------------------------------

            // using the same chunksize as phase3
            int na = GB_IMIN (pA_end - pA, CHUNKSIZE3) ;
            int nb = GB_IMIN (pB_end - pB, CHUNKSIZE3) ;
            int nab ;
            if (na < CHUNKSIZE3 && nb < CHUNKSIZE3)
            {
                // this is the very last chunk of the entire task; do it all
                nab = na + nb ;
            }
            else
            {
                // this chunk is in the middle of the task; just do the first
                // half of the diagonals of the na,nb chunk
                nab = GB_IMIN (na, nb) ;
            }

            // work_per_thread = ceil (nab / blockdim)
            int work_per_thread = GB_ICEIL (nab, blockDim.x) ;
            int diag = GB_IMIN (work_per_thread * threadIdx.x, nab) ;
            int diag_end = GB_IMIN (diag + work_per_thread, nab) ;
            int diag_last = GB_IMIN (work_per_thread * blockDim.x, nab) ;

            // get pointers to the current chunk
            const GB_Ai_TYPE *Ai_chunk = Ai + pA ;
            const GB_Aj_TYPE *Aj_chunk = Aj + pA ;
            const GB_Bi_TYPE *Bi_chunk = Bi + pB ;
            const GB_Bj_TYPE *Bj_chunk = Bj + pB ;
                  GB_Ci_TYPE *Ci_chunk = Ci + pC ;
                  GB_Cj_TYPE *Cj_chunk = Cj + pC ;
            #if !GB_C_ISO
            const GB_A_TYPE  *Ax_chunk = Ax + pA ;
            const GB_B_TYPE  *Bx_chunk = Bx + pB ;
                  GB_C_TYPE  *Cx_chunk = Cx + pC ;
            #endif

            //------------------------------------------------------------------
            // each thread searches for its starting point
            //------------------------------------------------------------------

            int pa, pb ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa, &pb, na, nb, diag,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            //------------------------------------------------------------------
            // each thread searches for its ending point
            //------------------------------------------------------------------

            int pa_end, pb_end ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_end,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            //------------------------------------------------------------------
            // compute the size of the set intersection (repeat of phase 3)
            //------------------------------------------------------------------

            int my_intersection = set_intersection_size
                <GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (pa, pb, pa_end, pb_end,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            //------------------------------------------------------------------
            // cumulative sum across all threads of the set union
            //------------------------------------------------------------------

            // This thread's set union is pc = |A| + |B| - |intersection(A,B)|,
            // which is the size of the set union found by each thread.
            // FIXME: eWiseMult: pc = my_intersection
            uint16_t pc = (pa_end - pa) + (pb_end - pb) - my_intersection ;
            uint16_t pc_end ;   // the block aggregate for all threads
            this_thread_block ( ).sync ( ) ;

            // pc is then replaced with a cumulative sum across all threads
            BlockScan (W).ExclusiveSum (pc, pc, pc_end) ;
            this_thread_block ( ).sync ( ) ;

            //------------------------------------------------------------------
            // compute the pattern and values of C for this thread
            //------------------------------------------------------------------

            while (pa < pa_end && pb < pb_end)
            {
                // get the two entries to compare: A [pa] and B [pb]
                GB_Ai_TYPE iA = Ai_chunk [pa] ;
                GB_Aj_TYPE jA = Aj_chunk [pa] ;
                GB_Bi_TYPE iB = Bi_chunk [pb] ;
                GB_Bj_TYPE jB = Bj_chunk [pb] ;

                // compare the two entries
                int afirst = ((jA < jB) || (jA == jB && iA < iB)) ;
                int amatch = (jA == jB && iA == iB) ;

                // cij = aij + bij
                if (afirst)
                {
                    // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
                    // FIXME: for eWiseMult: do nothing except pa++
                    Ci_chunk [pc] = iA ;
                    Cj_chunk [pc] = jA ;
                    GB_ADD_AIJ_PLUS_BETA (Cx_chunk, pc,
                        Ax_chunk, pa, GB_A_ISO, beta_scalar, iA, jA) ;
                    pa++ ;
                }
                else if (!amatch)
                {
                    // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
                    // FIXME: for eWiseMult: do nothing except pb++
                    Ci_chunk [pc] = iB ;
                    Cj_chunk [pc] = jB ;
                    GB_ADD_ALPHA_PLUS_BIJ (Cx_chunk, pc,
                        alpha_scalar, Bx_chunk, pb, GB_B_ISO, iB, jB) ;
                    pb++ ;
                }
                else
                {
                    // cij = aij + bij
                    // FIXME: for eWiseMult: same
                    Ci_chunk [pc] = iA ;
                    Cj_chunk [pc] = jA ;
                    GB_ADD_AIJ_PLUS_BIJ (Cx_chunk, pc,
                        Ax_chunk, pa, GB_A_ISO,
                        Bx_chunk, pb, GB_B_ISO, iA, jA) ;
                    pa++ ;
                    pb++ ;
                }
                pc++ ;
            }

            // FIXME: for eWiseMult: skip
            for ( ; pa < pa_end ; pa++, pc++)
            {
                // get the indices of A (i,j)
                GB_Ai_TYPE iA = Ai_chunk [pa] ;
                GB_Aj_TYPE jA = Aj_chunk [pa] ;
                // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
                Ci_chunk [pc] = iA ;
                Cj_chunk [pc] = jA ;
                GB_ADD_AIJ_PLUS_BETA (Cx_chunk, pc,
                    Ax_chunk, pa, GB_A_ISO, beta_scalar, iA, jA) ;
            }

            // FIXME: for eWiseMult: skip
            for ( ; pb < pb_end ; pb++, pc++)
            {
                // get the indices of B (i,j)
                GB_Bi_TYPE iB = Bi_chunk [pb] ;
                GB_Bj_TYPE jB = Bj_chunk [pb] ;
                // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
                Ci_chunk [pc] = iB ;
                Cj_chunk [pc] = jB ;
                GB_ADD_ALPHA_PLUS_BIJ (Cx_chunk, pc,
                    alpha_scalar, Bx_chunk, pb, GB_B_ISO, iB, jB) ;
            }

            //------------------------------------------------------------------
            // find the last diagonal of the last thread
            //------------------------------------------------------------------

            // All threads find the last diagonal of all threads.
            // Alternatively: the last thread could broadcast pa_end and pb_end
            // of the last thread in the threadblock to the entire threadblock.

            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_last,
                Ai_chunk, Aj_chunk, Bi_chunk, Bj_chunk) ;

            // all threads compute the same values of pA, pB, and pC:
            pA += pa_end ;
            pB += pb_end ;
            pC += pc_end ;
        }

        //----------------------------------------------------------------------
        // C = A or C = A+beta for entries remaining in A
        //----------------------------------------------------------------------

        // FIXME: for eWiseMult: skip
        for (pA = pA + threadIdx.x ;
             pA < pA_end ;
             pA += blockDim.x, pC += blockDim.x)
        {
            // get the indices of A (i,j)
            GB_Ai_TYPE iA = Ai [pA] ;
            GB_Aj_TYPE jA = Aj [pA] ;
            // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
            Ci [pC] = iA ;
            Cj [pC] = jA ;
            GB_ADD_AIJ_PLUS_BETA (Cx, pC, Ax, pA, GB_A_ISO,
                beta_scalar, iA, jA) ;
        }

        //----------------------------------------------------------------------
        // C = B or C = alpha+B for entries remaining in B
        //----------------------------------------------------------------------

        // FIXME: for eWiseMult: skip
        for (pB = pB + threadIdx.x ;
             pB < pB_end ;
             pB += blockDim.x, pC += blockDim.x)
        {
            // get the indices of B (i,j)
            GB_Bi_TYPE iB = Bi [pB] ;
            GB_Bj_TYPE jB = Bj [pB] ;
            // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
            Ci [pC] = iB ;
            Cj [pC] = jB ;
            GB_ADD_ALPHA_PLUS_BIJ (Cx, pC, alpha_scalar,
                Bx, pB, GB_B_ISO, iB, jB) ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase6: find leading entries in Cj
//------------------------------------------------------------------------------

#include "template/GB_cuda_construct_JDelta.cuh"

#define ITEMS_PER_THREAD6 ( CHUNKSIZE6 / BLOCKDIM6 )

__global__ void GB_cuda_add_sparse_phase6
(
    // outputs:
    uint16_t *JDelta,           // size cnz+1, in JDelta [-1..cnz-1]
    GB_Cp_TYPE *JDeltaSum,      // size nchunks_in_C+2
    // inputs, not modified, except for Cj [-1] sentinel value:
    GB_Cj_TYPE *Cj,             // size cnz+1: Cj [-1..cnz-1]
    int64_t cnz,                // # of entries in Cj
    int64_t nchunks_in_C        // # of chunks in Cj
)
{

    auto unload_Cj = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        return (Cj [p]) ;
    } ;

    GB_cuda_construct_JDelta
    <
        GB_Cp_TYPE,         // type of JDeltasum
        GB_Cj_TYPE,         // type of Cj
        CHUNKSIZE6,         // size of each chunk
        LOG2_CHUNKSIZE6,    // log2 (chunksize)
        BLOCKDIM6,          // blockdim of kernel launch
        ITEMS_PER_THREAD6   // # of items per thread (chunksize/blockdim)
    >
        (JDelta, JDeltaSum, Cj, cnz, nchunks_in_C, unload_Cj) ;
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase7: construct Cp and Ch
//------------------------------------------------------------------------------

// This phase is skipped if C->vdim is 1.

#include "template/GB_cuda_construct_Cphix.cuh"

__global__ void GB_cuda_add_sparse_phase7
(
    // outputs
    GrB_Matrix C,
    // inputs, not modified:
    uint16_t *JDelta,       // size nvals+1, in JDelta [-1..nvals-1]
    GB_Cp_TYPE *JDeltaSum,  // size nchunks+1
    GB_Cj_TYPE *Cj,         // size nvals+1: Key_out [-1 ... nvals-1]
    int64_t nvals,          // # of entries in C
    int64_t nchunks         // # of chunks to build C
)
{

    auto unload_Ci = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        // unused
        return (0) ;
    } ;

    auto unload_Cj = [](GB_Cj_TYPE *Cj, int64_t p)
    {
        // j = Cj [p]
        return (Cj [p]) ;
    } ;

    auto unload_Cx = [](void *Sx, int64_t p)
    {
        // unused
        return (0) ;
    } ;

    GB_cuda_construct_Cphix
    <   
        GB_Cp_TYPE,             // type of C->p
        GB_Cj_TYPE,             // type of C->h
        GB_Ci_TYPE,             // type of C->i
        GB_Cj_TYPE,             // type of Cj workspace
        void,                   // type of Cx, not used
        void,                   // type of Sx, not used
        CHUNKSIZE6,             // chunksize for work done by a threadblock
        LOG2_CHUNKSIZE6,        // log2 (chunksize)
        true,                   // C is a matrix; construct C->h
        false,                  // C->i is not constructed
        false                   // C->x is not constructed
    >
        (C, JDelta, JDeltaSum, Cj, NULL, nvals, nchunks,
            unload_Ci, unload_Cj, unload_Cx) ;
}

//------------------------------------------------------------------------------
// add sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_ADD_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_ADD_SPARSE_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_bix_alloc) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int data_arena = GxB_NARENAS + device ;
    uint64_t mem = GB_mem (data_arena, 0) ;

    GB_Aj_TYPE *Aj = NULL ; uint64_t Aj_mem = mem ;
    GB_Bj_TYPE *Bj = NULL ; uint64_t Bj_mem = mem ;

    int64_t *Task_Astart = NULL ; uint64_t Task_Astart_mem = mem ;
    int64_t *Task_Bstart = NULL ; uint64_t Task_Bstart_mem = mem ;
    int64_t *Task_Cstart = NULL ; uint64_t Task_Cstart_mem = mem ;
    void *W_0 = NULL ; uint64_t W_0_mem = mem ;    // size cnz+1: Cj
    void *W_6 = NULL ; uint64_t W_6_mem = mem ;    // size nvals+1: JDelta
    void *W_7 = NULL ; uint64_t W_7_mem = mem ;    // size nchunks+2: JDeltaSum

    GB_A_NHELD (anz) ;          // # of entries in A
    GB_B_NHELD (bnz) ;          // # of entries in B

    // # of outer tasks to compute all of C=A+B
    // ntasks = ceil ((anz + bnz) / chunksize2
    int64_t ntasks = GB_ICEIL (anz + bnz, CHUNKSIZE2) ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block1 (BLOCKDIM1) ;
    dim3 block2 (BLOCKDIM2) ;
    dim3 block6 (BLOCKDIM6) ;

    #if GB_IS_EWISEUNION
    const GB_X_TYPE alpha_scalar = (*((GB_X_TYPE *) alpha_scalar_in)) ;
    const GB_Y_TYPE beta_scalar  = (*((GB_Y_TYPE *) beta_scalar_in )) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 1: construct Aj and Bj (tuple form of A and B)
    //--------------------------------------------------------------------------

    Aj = (GB_Aj_TYPE *) GB_MALLOC_MEMORY (anz, sizeof (GB_Aj_TYPE), &Aj_mem) ;
    Bj = (GB_Bj_TYPE *) GB_MALLOC_MEMORY (bnz, sizeof (GB_Bj_TYPE), &Bj_mem) ;
    if (Aj == NULL || Bj == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 1: phase1
    GB_cuda_add_sparse_phase1 <<<grid, block1, 0, stream>>>
        (/* outputs: */ Aj, Bj,
         /* inputs: */  A, B, anz, bnz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 2: construct outer tasks
    //--------------------------------------------------------------------------

    Task_Astart = (int64_t *) GB_MALLOC_MEMORY (ntasks+1, sizeof (int64_t),
        &Task_Astart_mem) ;
    Task_Bstart = (int64_t *) GB_MALLOC_MEMORY (ntasks+1, sizeof (int64_t),
        &Task_Bstart_mem) ;
    Task_Cstart = (int64_t *) GB_MALLOC_MEMORY (ntasks+1, sizeof (int64_t),
        &Task_Cstart_mem) ;

    if (Task_Astart == NULL || Task_Bstart == NULL || Task_Cstart == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    const GB_Ai_TYPE *__restrict__ Ai = (GB_Ai_TYPE *) A->i ;
    const GB_Bi_TYPE *__restrict__ Bi = (GB_Bi_TYPE *) B->i ;

    // KERNEL LAUNCH 2: phase2
    GB_cuda_add_sparse_phase2 <<<grid, block2, 0, stream>>>
        (   // outputs:
            Task_Astart, Task_Bstart,
            // inputs:
            ntasks, Ai, Aj, Bi, Bj, anz, bnz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase3: compute the size of C for each task
    //--------------------------------------------------------------------------

    // KERNEL LAUNCH 3: phase3
    GB_cuda_add_sparse_phase3 <<<grid, block1, 0, stream>>>
    (   // outputs:
        Task_Cstart,
        // inputs:
        Task_Astart, Task_Bstart, ntasks, Ai, Aj, Bi, Bj) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase4: exclusive cumulative sum of size C on CPU, and allocate C
    //--------------------------------------------------------------------------

    int64_t cnz = 0 ;
    for (int64_t t = 0 ; t < ntasks ; t++)
    {
        int64_t s = Task_Cstart [t] ;
        Task_Cstart [t] = cnz ;
        cnz += s ;
    }
    Task_Cstart [ntasks] = cnz ;

    // allocate C->[phix]
    GB_OK (GB_bix_alloc (C, cnz, (C->vdim == 1) ? GxB_SPARSE : GxB_HYPERSPARSE,
        /* bitmap_calloc: */ false, /* numeric: */ true, GB_C_ISO)) ;
    C->nvals = cnz ;
    C->jumbled = false ;

    //--------------------------------------------------------------------------
    // phase5: numerical phase, compute C = A+B in coordinate form
    //--------------------------------------------------------------------------

    // allocate Cj workspace, with the Cj [-1] sentinel value
    W_0 = GB_MALLOC_MEMORY (cnz + 1, sizeof (GB_Cj_TYPE), &W_0_mem) ;
    GB_Cj_TYPE *Cj = ((GB_Cj_TYPE *) W_0) + 1 ;
    if (Cj == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 4: phase5
    GB_cuda_add_sparse_phase5 <<<grid, block1, 0, stream>>>
    (
        /* outputs: */ C, Cj,
        /* inputs:  */ Task_Cstart, Task_Astart, Task_Bstart, ntasks,
            Ai, Aj, Bi, Bj, A, B, theta
            #if GB_IS_EWISEUNION
            , alpha_scalar, beta_scalar
            #endif
            ) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase6: find leading entries of each column of C
    //--------------------------------------------------------------------------

    uint16_t *JDelta = NULL ;
    GB_Cp_TYPE *JDeltaSum = NULL ;
    int64_t nchunks_in_C = GB_ICEIL (cnz, CHUNKSIZE6) ;

    if (C->vdim != 1)
    {
        W_6 = GB_MALLOC_MEMORY (cnz+1+CHUNKSIZE6, sizeof (uint16_t), &W_6_mem) ;
        W_7 = GB_MALLOC_MEMORY (nchunks_in_C+2, sizeof (GB_Cp_TYPE), &W_7_mem) ;
        if (W_6 == NULL || W_7 == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        // shift by one so the [-1] entry can be used:
        JDelta = ((uint16_t *) W_6) + 1 ;
        JDeltaSum = ((GB_Cp_TYPE *) W_7) + 1 ;

        // KERNEL LAUNCH 5: phase6
        GB_cuda_add_sparse_phase6 <<<grid, block6, 0, stream>>>
            (/* outputs: */ JDelta, JDeltaSum,
             /* inputs:  */ Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ( )) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // phase7: construct Cp and Ch
    //--------------------------------------------------------------------------

    int64_t cnvec = 0 ;
    if (C->vdim == 1)
    {
        // C is a sparse vector
        cnvec = 1 ;
    }
    else
    {
        // FIXME: make this a template function too:
        // FIXME: do in on the GPU?
        // overwrite JDeltaSum with its exclusive cumulative sum
        for (int64_t chunk = 0 ; chunk < nchunks_in_C ; chunk++)
        {
            int64_t s = JDeltaSum [chunk] ;
            JDeltaSum [chunk] = cnvec ;
            cnvec += s ;
        }
        JDeltaSum [nchunks_in_C] = cnvec ;
    }

    // allocate C->p and C->h
    C->p_mem = mem ;
    C->p = GB_MALLOC_MEMORY (cnvec+1, sizeof (GB_Cp_TYPE), &(C->p_mem)) ;
    if (C->vdim == 1)
    {
        // C is a sparse vector
        C->h = NULL ;
        C->plen = 1 ;
    }
    else
    {
        // C is a hypersparse matrix
        C->h_mem = mem ;
        C->h = GB_MALLOC_MEMORY (cnvec, sizeof (GB_Cj_TYPE), &(C->h_mem)) ;
        C->plen = cnvec ;
    }
    if (C->p == NULL || (C->h == NULL && C->vdim != 1))
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;

    if (C->vdim == 1)
    {
        // C is a sparse vector
        GB_Cp_TYPE *Cp = (GB_Cp_TYPE *) C->p ;
        Cp [0] = 0 ;
        Cp [1] = cnz ;
    }
    else
    {
        // C is a hypersparse matrix
        // KERNEL LAUNCH 6: phase7
        GB_cuda_add_sparse_phase7 <<<grid, block6, 0, stream>>>
            (C, JDelta, JDeltaSum, Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ( )) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

