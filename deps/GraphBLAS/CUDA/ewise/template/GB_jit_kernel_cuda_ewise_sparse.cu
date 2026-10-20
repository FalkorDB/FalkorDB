//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_ewise_sparse.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = ewise(A,B) kernel on the GPU.   A and B cannot be jumbled on input, and
// C is returned as unjumbled.  No mask is exploited.  A and B are sparse or
// hypersparse.  C is constructed as hypersparse.

// This kernel can handle multiple kinds of methods:

//  (1) eWiseAdd, with GB_SET_UNION 1 and GB_IS_EWISEUNION 0, to compute C=A+B

//  (2) eWiseUnion, with GB_SET_UNION 1 and GB_IS_EWISEUNION 1, to compute
//      C=A+B using the eWiseUnion method.

//  (3) eWiseMult, with GB_SET_INTERSECTION 1, to compute C=eWiseMult(A,op,B).
//      This works for all cases but is efficient only if A and B have about
//      the same number of entries (within a wide factor, say 64).  If nnz(A)
//      << nnz(B) or visa versa, then another kernel will be used to scan the
//      sparser matrix and use binary search on the other.

//  (4) C<M>=A where C is initially empty, and M is structural.  This uses
//      GB_SET_INTERSECTION 1 and GB_IS_MASK_ASSIGN 1, and M is passed in as
//      the B matrix (values are not used).  The operator is always 1st_ctype.
//      As in eWiseMult, this method will be not be used when nnz(A) and nnz(B)
//      differ greatly.  Another kernel will be used instead.

//  (5) C<!M>=A, where C is initially empty, and M is structural.  This uses
//      GB_SET_DIFFERENCE 1 and GB_IS_MASK_ASSIGN 1 (computing the set A minus
//      M), and M is passed in as the B matrix (values are not used).  The
//      operator is always 1st_ctype.  If nnz(A) << nnz(M), this method is not
//      used; instead, another kernel will scan all of A and use binary search
//      to look up entries in M.

//------------------------------------------------------------------------------

// The column index j of each entry of A and B is not stored.  It is computed on
// demand by the getk method, which uses the small per-slice arrays built in
// phase 1.  This avoids building large Aj and Bj column-index arrays in global
// memory.  (The old phase 1 built the full Aj and Bj arrays; phase 1 now builds
// the small slice arrays instead, and getk does the lookup.)

// phase 1 (GPU): build the small per-slice arrays A_slice_kfirst and
//      B_slice_kfirst.  getk uses these to find the column of any entry.  Each
//      array holds one value per CHUNKSIZE1 entries, so it is a tiny fraction of
//      the size of a full column-index array.

// phase 2 (GPU):  use the merge-path method to find outer tasks for C, each
//      of which are fairly large (16K entries to handle in A and B).  getk
//      supplies the column of each entry during the search.

// phase 3 (GPU): compute the size of the set operator on A and B for each task
//      t of C, where the set operator is UNION for eWiseAdd and eWiseUnion,
//      INTERSECTION for eWiseMult or C<M>=A, or DIFFERENCE for C<!M>=A.  Each
//      task fills shared memory with the (i,j) of its current chunk, using getk,
//      and then merges from shared memory.  Each task C_t is completely
//      independent of other tasks, because of the merge-path construction in
//      phase 2.

// phase 4 (CPU):  compute the cumulative sum of |C_t|.  This gives the total
//      size of C.  Allocate C (Ci, Cj, Cx, each of size cnz).  Cj will be
//      compressed later into Cp and Ch, much like the CUDA builder kernel.

// phase 5 (GPU): same tasks as phase 3, with the same shared-memory fill, but
//      now compute the output matrix C in coordinate form: (i,j,cij), using the
//      merge-path method.  The output of this phase is the final matrix C in
//      sorted coordinate form: Cj, Ci, Cx, of size cnz.

// phases 6 and 7 (GPU):  convert (Cj,Ci,Cx) into the hypersparse form
//      Cp,Ch,Ci,Cx, using templated methods from CUDA/builder

//------------------------------------------------------------------------------

#include "template/GB_cuda_ek_slice.cuh"
#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_threadblock_sum_uint64.cuh"

// FIXME: move these to CUDA/include/GB_cuda_geometry.hpp, and tune them:

// for phase1 (build the per-slice arrays) and getk: size of one ek_slice slice
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
// wider blocks for phase3 (occupancy)
#define BLOCKDIM3 256
// phase5 uses the same width as phase3, but kept separate so it can be tuned
// independently.  NOTE: phase5's cub::BlockScan MUST use this same width.
#define BLOCKDIM5 BLOCKDIM3

// for phase6 and phase7: (same geometry as the builder kernel)
#define BLOCKDIM6 128
#define CHUNKSIZE6 256
#define LOG2_CHUNKSIZE6 8

#define CHUNK_DELTA_MASK1 (CHUNKSIZE1 - 1)

#define GB_FREE_WORKSPACE                                     \
{                                                             \
    GB_FREE_MEMORY (&A_slice_kfirst, A_slice_kfirst_mem) ;    \
    GB_FREE_MEMORY (&B_slice_kfirst, B_slice_kfirst_mem) ;    \
    GB_FREE_MEMORY (&Task_Astart, Task_Astart_mem) ;          \
    GB_FREE_MEMORY (&Task_Bstart, Task_Bstart_mem) ;          \
    GB_FREE_MEMORY (&Task_Cstart, Task_Cstart_mem) ;          \
    GB_FREE_MEMORY (&W_0, W_0_mem) ;                          \
    GB_FREE_MEMORY (&W_6, W_6_mem) ;                          \
    GB_FREE_MEMORY (&W_7, W_7_mem) ;                          \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    /* GB_phbix_free (C) is not called; */      \
    /* it is done in the caller if needed */    \
    GB_FREE_WORKSPACE ;                         \
}

//------------------------------------------------------------------------------
// getk: find the vector k that owns entry p (replaces the global Aj/Bj)
//------------------------------------------------------------------------------

// Given an entry at position p, return the vector k that contains it (so that
// Ap [k] <= p < Ap [k+1]).  The caller gets the column index j from k:
// j = k if the matrix is sparse, or j = Ah [k] if it is hypersparse.

// getk reads kfirst and klast from the small per-slice array (built in phase 1),
// computes the slope of the slice, and calls GB_cuda_ek_slice_entry.  It is used
// in place of the old global Aj/Bj arrays: phase 2 calls getk directly, and
// phases 3 and 5 call getk to fill shared memory.

template <typename T> __device__ int64_t getk
(
    const int64_t p,              // position of the entry
    const int64_t *Slice_kfirst,  // the small array, size nchunks+1
    const T *Ap,                  // column pointers, size anvec+1
    const int64_t anvec1          // anvec-1
)
{
    int64_t slice  = p >> LOG2_CHUNKSIZE1   ;  // which slice
    int64_t pdelta = p & CHUNK_DELTA_MASK1  ;  // offset in the slice
    int64_t kfirst = Slice_kfirst [slice]   ;
    int64_t klast  = Slice_kfirst [slice+1] ;
    float   slope  = ((float) (klast - kfirst + 1)) / ((float) CHUNKSIZE1) ;

    return (GB_cuda_ek_slice_entry<T> (p, pdelta, Ap, anvec1, kfirst, slope)) ;
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_sparse_phase1: build the per-slice arrays for getk
//------------------------------------------------------------------------------

// Phase 1 builds A_slice_kfirst and B_slice_kfirst.  For each slice of
// CHUNKSIZE1 entries, it stores kfirst: the vector k that owns the first entry
// of the slice.  getk later uses these values to look up the column of any
// entry.  Each array holds one value per CHUNKSIZE1 entries (plus a sentinel),
// so it is 1/CHUNKSIZE1 of the size of a full column-index array.

__global__ void GB_cuda_ewise_sparse_phase1
(
    // outputs:
    int64_t *A_slice_kfirst,  // small array for A, size A_slice_nchunks+1
    int64_t *B_slice_kfirst,  // small array for B, size B_slice_nchunks+1
    // inputs:
    int64_t A_slice_nchunks,
    int64_t B_slice_nchunks,
    const GrB_Matrix A,
    const GrB_Matrix B
)
{
    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    const int64_t bnvec = B->nvec ;
    const GB_Bp_TYPE *__restrict__ Bp = (GB_Bp_TYPE *) B->p ;

    // one thread per slice of A.  k is reused as the lower bound of each binary
    // search.  A thread visits its slices in increasing order, so the owning
    // vector k never decreases.  Reusing k makes each search shorter.
    int64_t k = 0 ;
    for (int64_t slice = blockIdx.x * blockDim.x + threadIdx.x ;
            slice < A_slice_nchunks ;
            slice += blockDim.x * gridDim.x)
    {
        int64_t pfirst = slice << LOG2_CHUNKSIZE1 ;  // first entry of the slice
        GB_cuda_ek_slice_search<GB_Ap_TYPE> (&k, Ap, anvec, pfirst) ;
        A_slice_kfirst [slice] = k ;
    }

    // one thread per slice of B
    k = 0 ;
    for (int64_t slice = blockIdx.x * blockDim.x + threadIdx.x ;
            slice < B_slice_nchunks ;
            slice += blockDim.x * gridDim.x)
    {
        int64_t pfirst = slice << LOG2_CHUNKSIZE1 ;  // first entry of the slice
        GB_cuda_ek_slice_search<GB_Bp_TYPE> (&k, Bp, bnvec, pfirst) ;
        B_slice_kfirst [slice] = k ;
    }

    // the sentinel for the last slice
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        A_slice_kfirst [A_slice_nchunks] = anvec ;
        B_slice_kfirst [B_slice_nchunks] = bnvec ;
    }
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
// GB_cuda_ewise_sparse_phase2: construct tasks
//------------------------------------------------------------------------------

// This method divides the work to compute C=A+B into large tasks, each of
// which is found via the mergepath method on the coordinate forms of A and B.
// The column of each entry is computed on demand with getk; there are no Aj/Bj
// arrays.  The tasks are later computed using a single threadblock each, in
// subsequent phases.  This kernel launch uses a single thread to compute the
// starting points of a single task.  As a result, there will be a lot of warp
// divergence in this kernel, but very little work is done since the tasks are
// very large.

// This kernel inlines its own copy of the mergepath search so it can call getk
// in place of reading Aj/Bj.  See the FIXME on the mergepath template: the two
// copies can later be merged with an "if constexpr" flag.

// Each large task operates on CHUNKSIZE2 entries in A and B.

__global__ void GB_cuda_ewise_sparse_phase2
(
    // outputs:
    int64_t *__restrict__ Task_Astart,      // array of size # tasks+1
    int64_t *__restrict__ Task_Bstart,      // array of size # tasks+1
    // inputs:
    const int64_t ntasks,                   // # of outer tasks
    const GrB_Matrix A,                          // gives Ap, Ah, Ai, anvec
    const GrB_Matrix B,                          // gives Bp, Bh, Bi, bnvec
    const int64_t *__restrict__ A_slice_kfirst,  // small array for A
    const int64_t *__restrict__ B_slice_kfirst,  // small array for B
    const int64_t anz,
    const int64_t bnz
)
{

    // inputs derived from A and B
    const GB_Ap_TYPE *Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *Ah = (GB_Aj_TYPE *) A->h ;  // NULL if sparse
    const GB_Ai_TYPE *Ai = (GB_Ai_TYPE *) A->i ;
    const int64_t anvec1 = A->nvec - 1 ;

    const GB_Bp_TYPE *Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bj_TYPE *Bh = (GB_Bj_TYPE *) B->h ;
    const GB_Bi_TYPE *Bi = (GB_Bi_TYPE *) B->i ;
    const int64_t bnvec1 = B->nvec - 1 ;

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
            int64_t amin = GB_IMAX (diag - bnz, 0) ;
            int64_t amax = GB_IMIN (diag, anz) ;

            while (amin < amax)
            {
                int64_t pivot = (amin + amax) >> 1 ;
                int64_t pB = diag - pivot - 1 ;

                GB_Ai_TYPE iA = Ai [pivot] ;
                int64_t kA = getk<GB_Ap_TYPE> (pivot, A_slice_kfirst, Ap, anvec1) ;
                GB_Aj_TYPE jA = GB_A_IS_HYPER ? Ah [kA] : (GB_Aj_TYPE) kA ;

                GB_Bi_TYPE iB = Bi [pB] ;
                int64_t kB = getk<GB_Bp_TYPE> (pB, B_slice_kfirst, Bp, bnvec1) ;
                GB_Bj_TYPE jB = GB_B_IS_HYPER ? Bh [kB] : (GB_Bj_TYPE) kB ;

                int64_t afirst = ((jA < jB) || (jA == jB && iA < iB)) ;
                amin = (pivot + 1) * afirst + amin * (1 - afirst) ;
                amax = pivot * (1 - afirst) + amax * afirst ;
            }

            // tie-break: keep a matched pair together across the task boundary
            int64_t astart = amin ;
            int64_t bstart = diag - amin ;
            int64_t bprior = bstart - 1 ;
            if (astart < anz && bprior >= 0)
            {
                GB_Ai_TYPE iA = Ai [astart] ;
                int64_t kA = getk<GB_Ap_TYPE> (astart, A_slice_kfirst, Ap, anvec1) ;
                GB_Aj_TYPE jA = GB_A_IS_HYPER ? Ah [kA] : (GB_Aj_TYPE) kA ;

                GB_Bi_TYPE iB = Bi [bprior] ;
                int64_t kB = getk<GB_Bp_TYPE> (bprior, B_slice_kfirst, Bp, bnvec1) ;
                GB_Bj_TYPE jB = GB_B_IS_HYPER ? Bh [kB] : (GB_Bj_TYPE) kB ;

                if (iA == iB && jA == jB) bstart-- ;
            }

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
// GB_cuda_ewise_sparse_phase3: compute intersection of A and B for each task
//------------------------------------------------------------------------------

// C, A, and B have been split into tasks.  For task t, the entries in A are
// at Ax, Ai [Task_Astart [t] ... Task_Astart [t+1]-1], and in B at
// Bx, Bi [Task_Bstart [t] ... Task_Bstart [t+1]-1].  The column of each entry
// is computed with getk; there are no Aj/Bj arrays.

// Each task is done by a single threadblock.  The threads first fill shared
// memory with the (i,j) of the current chunk (rows copied from global, columns
// from getk), then use the mergepath method to split the work for each thread.

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

__global__ void GB_cuda_ewise_sparse_phase3
(
    // outputs:
    int64_t *Task_Cstart,       // array of size # tasks+1; Task_Cstart [t]
                                // is the size of the set union
                                // of A and B for task t
    // inputs:
    const int64_t *__restrict__ Task_Astart,    // array of size # tasks+1
    const int64_t *__restrict__ Task_Bstart,    // array of size # tasks+1
    const int64_t ntasks,                       // # of outer tasks
    const GrB_Matrix A,                         // gives Ap, Ah, Ai, anvec
    const GrB_Matrix B,                         // gives Bp, Bh, Bi, bnvec
    const int64_t *__restrict__ A_slice_kfirst, // small array for A
    const int64_t *__restrict__ B_slice_kfirst  // small array for B
)
{

    // four shared arrays for one chunk
    __shared__ GB_Ai_TYPE Ai_s [CHUNKSIZE3] ;
    __shared__ GB_Aj_TYPE Aj_s [CHUNKSIZE3] ;
    __shared__ GB_Bi_TYPE Bi_s [CHUNKSIZE3] ;
    __shared__ GB_Bj_TYPE Bj_s [CHUNKSIZE3] ;

    // inputs derived from A and B
    const GB_Ap_TYPE *Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *Ah = (GB_Aj_TYPE *) A->h ;  // NULL if sparse
    const GB_Ai_TYPE *Ai = (GB_Ai_TYPE *) A->i ;
    const int64_t anvec1 = A->nvec - 1 ;

    const GB_Bp_TYPE *Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bj_TYPE *Bh = (GB_Bj_TYPE *) B->h ;
    const GB_Bi_TYPE *Bi = (GB_Bi_TYPE *) B->i ;
    const int64_t bnvec1 = B->nvec - 1 ;

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
        uint64_t AB_intersection = 0 ;

        #if GB_SET_UNION
        int64_t AB_size = (pA_end - pA) + (pB_end - pB) ;   // |A_t| + |B_t|
        #elif GB_SET_DIFFERENCE
        int64_t A_size = (pA_end - pA) ;    // |A_t|
        #endif

        //----------------------------------------------------------------------
        // compute the size of the set intersection for this task
        //----------------------------------------------------------------------

#if 0
        if (threadIdx.x == 0 && t == 0)
        {
            printf ("task %ld: pA %ld %ld, pB: %ld %ld\n",
                t, pA, pA_end, pB, pB_end) ;
        }
#endif

        bool ok = true ;

        while (pA < pA_end && pB < pB_end && ok)
        {

#if 0
        if (threadIdx.x == 0 && t == 0)
        {
            printf ("\n==== continue task %ld: pA %ld %ld, pB: %ld %ld\n",
                t, pA, pA_end, pB, pB_end) ;
        }
#endif

            //------------------------------------------------------------------
            // get the current chunk
            //------------------------------------------------------------------

            int na = GB_IMIN (pA_end - pA, CHUNKSIZE3) ;
            int nb = GB_IMIN (pB_end - pB, CHUNKSIZE3) ;

            //------------------------------------------------------------------
            // fill shared memory
            //------------------------------------------------------------------

            // wait: the last chunk's readers must finish
            __syncthreads () ;

            // A: rows and columns
            for (int kk = threadIdx.x ; kk < na ; kk += blockDim.x)
            {
                int64_t p = pA + kk ;
                Ai_s [kk] = Ai [p] ;

                int64_t k = getk<GB_Ap_TYPE> (p, A_slice_kfirst, Ap, anvec1) ;
                Aj_s [kk] = GB_A_IS_HYPER ? Ah [k] : (GB_Aj_TYPE) k ;
            }

            // B: rows and columns
            for (int kk = threadIdx.x ; kk < nb ; kk += blockDim.x)
            {
                int64_t p = pB + kk ;
                Bi_s [kk] = Bi [p] ;

                int64_t k = getk<GB_Bp_TYPE> (p, B_slice_kfirst, Bp, bnvec1) ;
                Bj_s [kk] = GB_B_IS_HYPER ? Bh [k] : (GB_Bj_TYPE) k ;
            }

            // wait: all writers must finish before the merge reads
            __syncthreads () ;

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

#if 0
            if (t == 0)
            {
                printf ("task 0, thread %d: na %d, nb: %d, nab: %d, diag %d diag_end %d diag_last %d\n",
                    threadIdx.x, na, nb, nab, diag, diag_end, diag_last) ;
            }
#endif

            //------------------------------------------------------------------
            // each thread searches for its starting point
            //------------------------------------------------------------------

            int pa, pb ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa, &pb, na, nb, diag, Ai_s, Aj_s, Bi_s, Bj_s) ;

            //------------------------------------------------------------------
            // each thread searches for its ending point
            //------------------------------------------------------------------

            int pa_end, pb_end ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_end, Ai_s, Aj_s, Bi_s, Bj_s) ;

            //------------------------------------------------------------------
            // compute the size of the set intersection
            //------------------------------------------------------------------

            int my_intersection = set_intersection_size
                <GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (pa, pb, pa_end, pb_end, Ai_s, Aj_s, Bi_s, Bj_s) ;
            AB_intersection += my_intersection ;

            //------------------------------------------------------------------
            // find the last diagonal of the last thread
            //------------------------------------------------------------------

            // All threads find the last diagonal of all threads.
            // Alternatively: the last thread could broadcast pa_end and pb_end
            // of the last thread in the threadblock to the entire threadblock.

            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_last, Ai_s, Aj_s, Bi_s, Bj_s) ;

            pA += pa_end ;
            pB += pb_end ;

#if 1
            if (t == 0 && pa_end == 0 && pb_end == 0)
            {
                ok = false ;
                if (threadIdx.x == 0)
                {
                printf ("task %ld: infinite loop! pA %ld %ld, pB: %ld %ld\n",
                    t, pA, pA_end, pB, pB_end) ;
                }
            }
#endif

        }

        //----------------------------------------------------------------------
        // compute the size of C for this task
        //----------------------------------------------------------------------

        // all threads cooperate to sum up their AB_intersection, and the
        // result is summed into thread 0
        AB_intersection = GB_cuda_threadblock_sum_uint64 (AB_intersection) ;

        if (threadIdx.x == 0)
        {
            #if GB_SET_UNION
            {
                // |C_t| = |A_t union B_t| = |A_t| + |B_t| - |A_t intersect B_t|
                Task_Cstart [t] = AB_size - AB_intersection ;
            }
            #elif GB_SET_DIFFERENCE
            {
                // |C_t| = |A_t - B_t| = |A_t| - |A_t intersection B_t|
                Task_Cstart [t] = A_size - AB_intersection ;
            }
            #else
            {
                // |C_t| = |A_t intersection B_t|
                Task_Cstart [t] = AB_intersection ;
            }
            #endif
//          printf ("task %ld, size %ld\n", t, Task_Cstart [t]) ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_sparse_phase5: compute pattern and values of C (coordinate form)
//------------------------------------------------------------------------------

__global__ void GB_cuda_ewise_sparse_phase5
(
    // outputs:
    GrB_Matrix C,
    GB_Cj_TYPE *Cj,                              // size cnz+1, Cj [-1..cnz-1]
    // inputs:
    const int64_t *__restrict__ Task_Cstart,     // array of size # tasks+1
    const int64_t *__restrict__ Task_Astart,     // array of size # tasks+1
    const int64_t *__restrict__ Task_Bstart,     // array of size # tasks+1
    const int64_t ntasks,                        // # of outer tasks
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *__restrict__ A_slice_kfirst,  // small array for A
    const int64_t *__restrict__ B_slice_kfirst,  // small array for B
    const void *theta                   // theta scalar for index binary ops
    #if GB_IS_EWISEUNION
    , const GB_X_TYPE alpha_scalar      // alpha scalar, for eWiseUnion
    , const GB_Y_TYPE beta_scalar       // beta scalar, for eWiseUnion
    #endif
)
{

    // four shared arrays for one chunk
    __shared__ GB_Ai_TYPE Ai_s [CHUNKSIZE3] ;
    __shared__ GB_Aj_TYPE Aj_s [CHUNKSIZE3] ;
    __shared__ GB_Bi_TYPE Bi_s [CHUNKSIZE3] ;
    __shared__ GB_Bj_TYPE Bj_s [CHUNKSIZE3] ;

    // inputs derived from A and B
    const GB_Ap_TYPE *Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Aj_TYPE *Ah = (GB_Aj_TYPE *) A->h ;  // NULL if sparse
    const GB_Ai_TYPE *Ai = (GB_Ai_TYPE *) A->i ;
    const int64_t anvec1 = A->nvec - 1 ;

    const GB_Bp_TYPE *Bp = (GB_Bp_TYPE *) B->p ;
    const GB_Bj_TYPE *Bh = (GB_Bj_TYPE *) B->h ;
    const GB_Bi_TYPE *Bi = (GB_Bi_TYPE *) B->i ;
    const int64_t bnvec1 = B->nvec - 1 ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #if !GB_C_ISO
    const GB_A_TYPE  *__restrict__ Ax = (GB_A_TYPE  *) A->x ;
    #if !GB_IS_MASK_ASSIGN
    const GB_B_TYPE  *__restrict__ Bx = (GB_B_TYPE  *) B->x ;
    #endif
          GB_C_TYPE  *__restrict__ Cx = (GB_C_TYPE  *) C->x ;
    #endif
          GB_Ci_TYPE *__restrict__ Ci = (GB_Ci_TYPE *) C->i ;

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    // cub::Block* workspace for ExclusiveSum of pc of each thread
    using BlockScan = cub::BlockScan <uint16_t, BLOCKDIM5,
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

            //------------------------------------------------------------------
            // fill shared memory
            //------------------------------------------------------------------

            // wait: the last chunk's readers must finish
            __syncthreads () ;

            // A: rows and columns
            for (int kk = threadIdx.x ; kk < na ; kk += blockDim.x)
            {
                int64_t p = pA + kk ;
                Ai_s [kk] = Ai [p] ;

                int64_t k = getk<GB_Ap_TYPE> (p, A_slice_kfirst, Ap, anvec1) ;
                Aj_s [kk] = GB_A_IS_HYPER ? Ah [k] : (GB_Aj_TYPE) k ;
            }

            // B: rows and columns
            for (int kk = threadIdx.x ; kk < nb ; kk += blockDim.x)
            {
                int64_t p = pB + kk ;
                Bi_s [kk] = Bi [p] ;

                int64_t k = getk<GB_Bp_TYPE> (p, B_slice_kfirst, Bp, bnvec1) ;
                Bj_s [kk] = GB_B_IS_HYPER ? Bh [k] : (GB_Bj_TYPE) k ;
            }

            // wait: all writers must finish before the merge reads
            __syncthreads () ;

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
            GB_Ci_TYPE *Ci_chunk = Ci + pC ;
            GB_Cj_TYPE *Cj_chunk = Cj + pC ;

            #if !GB_C_ISO
            const GB_A_TYPE  *Ax_chunk = Ax + pA ;
            #if !GB_IS_MASK_ASSIGN
            const GB_B_TYPE  *Bx_chunk = Bx + pB ;
            #endif
                  GB_C_TYPE  *Cx_chunk = Cx + pC ;
            #endif

            //------------------------------------------------------------------
            // each thread searches for its starting point
            //------------------------------------------------------------------

            int pa, pb ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa, &pb, na, nb, diag, Ai_s, Aj_s, Bi_s, Bj_s) ;

            //------------------------------------------------------------------
            // each thread searches for its ending point
            //------------------------------------------------------------------

            int pa_end, pb_end ;
            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_end, Ai_s, Aj_s, Bi_s, Bj_s) ;

            //------------------------------------------------------------------
            // compute the size of the set intersection (repeat of phase 3)
            //------------------------------------------------------------------

            int my_intersection = set_intersection_size
                <GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (pa, pb, pa_end, pb_end, Ai_s, Aj_s, Bi_s, Bj_s) ;

            //------------------------------------------------------------------
            // cumulative sum across all threads of the set union
            //------------------------------------------------------------------

            #if GB_SET_UNION
            // This thread's set union is pc = |A| + |B| - |intersection(A,B)|.
            uint16_t pc = (pa_end - pa) + (pb_end - pb) - my_intersection ;
            #elif GB_SET_DIFFERENCE
            // This thread's set difference is pc = |A-B| =
            // |A|- |intersection(A,B)|.
            uint16_t pc = (pa_end - pa) - my_intersection ;
            #else
            // This thread's set intersection is pc = |intersection(A,B)|.
            uint16_t pc = my_intersection ;
            #endif
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
                GB_Ai_TYPE iA = Ai_s [pa] ;
                GB_Aj_TYPE jA = Aj_s [pa] ;
                GB_Bi_TYPE iB = Bi_s [pb] ;
                GB_Bj_TYPE jB = Bj_s [pb] ;

                // compare the two entries
                int afirst = ((jA < jB) || (jA == jB && iA < iB)) ;
                int amatch = (jA == jB && iA == iB) ;

                if (afirst)
                {
                    // The (i,j) entry is in A but not B
                    // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
                    // or C<!M>=A using the 1st_ctype operator (where B is M)
                    #if GB_SET_UNION || GB_SET_DIFFERENCE
                    Ci_chunk [pc] = iA ;
                    Cj_chunk [pc] = jA ;
                    GB_EWISE_AIJ_OP_BETA (Cx_chunk, pc,
                        Ax_chunk, pa, GB_A_ISO, beta_scalar, iA, jA) ;
                    pc++ ;
                    #endif
                    pa++ ;
                }
                else if (!amatch)
                {
                    // The (i,j) entry is in B but not A
                    // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
                    #if GB_SET_UNION
                    Ci_chunk [pc] = iB ;
                    Cj_chunk [pc] = jB ;
                    GB_EWISE_ALPHA_OP_BIJ (Cx_chunk, pc,
                        alpha_scalar, Bx_chunk, pb, GB_B_ISO, iB, jB) ;
                    pc++ ;
                    #endif
                    pb++ ;
                }
                else
                {
                    // The (i,j) entry is in both A and B
                    // cij = aij + bij for eWiseAdd, eWiseUnion, eWiseMult,
                    // and C<M>=A using the 1st_ctype operator (where B is M)
                    #if GB_SET_UNION || GB_SET_INTERSECTION
                    Ci_chunk [pc] = iA ;
                    Cj_chunk [pc] = jA ;
                    GB_EWISE_AIJ_OP_BIJ (Cx_chunk, pc,
                        Ax_chunk, pa, GB_A_ISO,
                        Bx_chunk, pb, GB_B_ISO, iA, jA) ;
                    pc++ ;
                    #endif
                    pa++ ;
                    pb++ ;
                }
            }

            //------------------------------------------------------------------
            // copy any entries that remain in A, which are not in B
            //------------------------------------------------------------------

            #if GB_SET_UNION || GB_SET_DIFFERENCE
            for ( ; pa < pa_end ; pa++, pc++)
            {
                // get the indices of A (i,j)
                GB_Ai_TYPE iA = Ai_s [pa] ;
                GB_Aj_TYPE jA = Aj_s [pa] ;
                // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
                // or C<!M>=A using the 1st_ctype operator
                Ci_chunk [pc] = iA ;
                Cj_chunk [pc] = jA ;
                GB_EWISE_AIJ_OP_BETA (Cx_chunk, pc,
                    Ax_chunk, pa, GB_A_ISO, beta_scalar, iA, jA) ;
            }
            #endif

            //------------------------------------------------------------------
            // copy any entries that remain in B, which are not in A
            //------------------------------------------------------------------

            #if GB_SET_UNION
            for ( ; pb < pb_end ; pb++, pc++)
            {
                // get the indices of B (i,j)
                GB_Bi_TYPE iB = Bi_s [pb] ;
                GB_Bj_TYPE jB = Bj_s [pb] ;
                // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
                Ci_chunk [pc] = iB ;
                Cj_chunk [pc] = jB ;
                GB_EWISE_ALPHA_OP_BIJ (Cx_chunk, pc,
                    alpha_scalar, Bx_chunk, pb, GB_B_ISO, iB, jB) ;
            }
            #endif

            //------------------------------------------------------------------
            // find the last diagonal of the last thread
            //------------------------------------------------------------------

            // All threads find the last diagonal of all threads.
            // Alternatively: the last thread could broadcast pa_end and pb_end
            // of the last thread in the threadblock to the entire threadblock.

            mergepath <int, GB_Ai_TYPE, GB_Aj_TYPE, GB_Bi_TYPE, GB_Bj_TYPE>
                (&pa_end, &pb_end, na, nb, diag_last, Ai_s, Aj_s, Bi_s, Bj_s) ;

            // all threads compute the same values of pA, pB, and pC:
            pA += pa_end ;
            pB += pb_end ;
            pC += pc_end ;
        }

        //----------------------------------------------------------------------
        // C = A or C = A+beta for entries remaining in A
        //----------------------------------------------------------------------

        #if GB_SET_UNION || GB_SET_DIFFERENCE
        int64_t pc ;
        for (pA = pA + threadIdx.x, pc = pC + threadIdx.x ;
             pA < pA_end ;
             pA += blockDim.x, pc += blockDim.x)
        {
            // get the indices of A (i,j)
            GB_Ai_TYPE iA = Ai [pA] ;
            int64_t k = getk<GB_Ap_TYPE> (pA, A_slice_kfirst, Ap, anvec1) ;
            GB_Aj_TYPE jA = GB_A_IS_HYPER ? Ah [k] : (GB_Aj_TYPE) k ;
            // cij = aij + beta for eWiseUnion; cij = aij for eWiseAdd
            // or C<!M>=A using the 1st_ctype operator
            Ci [pc] = iA ;
            Cj [pc] = jA ;
            GB_EWISE_AIJ_OP_BETA (Cx, pc, Ax, pA, GB_A_ISO,
                beta_scalar, iA, jA) ;
        }
        #endif

        //----------------------------------------------------------------------
        // C = B or C = alpha+B for entries remaining in B
        //----------------------------------------------------------------------

        #if GB_SET_UNION
        for (pB = pB + threadIdx.x, pc = pC + threadIdx.x ;
             pB < pB_end ;
             pB += blockDim.x, pc += blockDim.x)
        {
            // get the indices of B (i,j)
            GB_Bi_TYPE iB = Bi [pB] ;
            int64_t k = getk<GB_Bp_TYPE> (pB, B_slice_kfirst, Bp, bnvec1) ;
            GB_Bj_TYPE jB = GB_B_IS_HYPER ? Bh [k] : (GB_Bj_TYPE) k ;
            // cij = alpha + bij for eWiseUnion; cij = bij for eWiseAdd
            Ci [pc] = iB ;
            Cj [pc] = jB ;
            GB_EWISE_ALPHA_OP_BIJ (Cx, pc, alpha_scalar,
                Bx, pB, GB_B_ISO, iB, jB) ;
        }
        #endif
    }
}

//------------------------------------------------------------------------------
// GB_cuda_ewise_sparse_phase6: find leading entries in Cj
//------------------------------------------------------------------------------

#include "template/GB_cuda_construct_JDelta.cuh"

#define ITEMS_PER_THREAD6 ( CHUNKSIZE6 / BLOCKDIM6 )

__global__ void GB_cuda_ewise_sparse_phase6
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
// GB_cuda_ewise_sparse_phase7: construct Cp and Ch
//------------------------------------------------------------------------------

// This phase is skipped if C->vdim is 1.

#include "template/GB_cuda_construct_Cphix.cuh"

__global__ void GB_cuda_ewise_sparse_phase7
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
// cuda_ewise_sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO (GB_jit_kernel)
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

    int64_t *Task_Astart = NULL ; uint64_t Task_Astart_mem = mem ;
    int64_t *Task_Bstart = NULL ; uint64_t Task_Bstart_mem = mem ;
    int64_t *Task_Cstart = NULL ; uint64_t Task_Cstart_mem = mem ;

    int64_t *A_slice_kfirst = NULL ; uint64_t A_slice_kfirst_mem = mem ;
    int64_t *B_slice_kfirst = NULL ; uint64_t B_slice_kfirst_mem = mem ;

    void *W_0 = NULL ; uint64_t W_0_mem = mem ;    // size cnz+1: Cj
    void *W_6 = NULL ; uint64_t W_6_mem = mem ;    // size nvals+1: JDelta
    void *W_7 = NULL ; uint64_t W_7_mem = mem ;    // size nchunks+2: JDeltaSum

    GB_A_NHELD (anz) ;          // # of entries in A
    GB_B_NHELD (bnz) ;          // # of entries in B
//  printf ("anz: %ld, bnz: %ld\n", anz, bnz) ;

    int64_t A_slice_nchunks = GB_ICEIL (anz, CHUNKSIZE1) ;
    int64_t B_slice_nchunks = GB_ICEIL (bnz, CHUNKSIZE1) ;

    // # of outer tasks to compute all of C=A+B
    // ntasks = ceil ((anz + bnz) / chunksize2
    int64_t ntasks = GB_ICEIL (anz + bnz, CHUNKSIZE2) ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block1 (BLOCKDIM1) ;
    dim3 block2 (BLOCKDIM2) ;
    dim3 block6 (BLOCKDIM6) ;
    dim3 block3 (BLOCKDIM3) ;
    dim3 block5 (BLOCKDIM5) ;

    #if GB_IS_EWISEUNION
    const GB_X_TYPE alpha_scalar = (*((GB_X_TYPE *) alpha_scalar_in)) ;
    const GB_Y_TYPE beta_scalar  = (*((GB_Y_TYPE *) beta_scalar_in )) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 1: build the small slice arrays for getk
    //--------------------------------------------------------------------------

    A_slice_kfirst = (int64_t*) GB_MALLOC_MEMORY (A_slice_nchunks + 1,
            sizeof (int64_t), &A_slice_kfirst_mem) ;
    B_slice_kfirst = (int64_t*) GB_MALLOC_MEMORY (B_slice_nchunks + 1,
            sizeof (int64_t), &B_slice_kfirst_mem) ;
    if (A_slice_kfirst == NULL || B_slice_kfirst == NULL)
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 1: phase1
    // printf ("\ncuda ewise start phase1:\n") ;
    GB_cuda_ewise_sparse_phase1 <<<grid, block1, 0, stream>>>
        (/* outputs: */ A_slice_kfirst, B_slice_kfirst,
         /* inputs:  */ A_slice_nchunks, B_slice_nchunks, A, B) ;
    CUDA_OK (cudaGetLastError ()) ;
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

    // KERNEL LAUNCH 2: phase2
    // printf ("\ncuda ewise start phase2:\n") ;
    GB_cuda_ewise_sparse_phase2 <<<grid, block2, 0, stream>>>
        (   // outputs:
            Task_Astart, Task_Bstart,
            // inputs:
            ntasks, A, B, A_slice_kfirst, B_slice_kfirst, anz, bnz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase3: compute the size of C for each task
    //--------------------------------------------------------------------------

    // KERNEL LAUNCH 3: phase3
    // printf ("\ncuda ewise start phase3:\n") ;
    GB_cuda_ewise_sparse_phase3 <<<grid, block3, 0, stream>>>
    (   // outputs:
        Task_Cstart,
        // inputs:
        Task_Astart, Task_Bstart, ntasks, A, B,
        A_slice_kfirst, B_slice_kfirst) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase4: exclusive cumulative sum of size C on CPU, and allocate C
    //--------------------------------------------------------------------------

    // printf ("\ncuda ewise start phase4:\n") ;
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
    // printf ("\ncuda ewise start phase5:\n") ;
    GB_cuda_ewise_sparse_phase5 <<<grid, block5, 0, stream>>>
    (
        /* outputs: */ C, Cj,
        /* inputs:  */ Task_Cstart, Task_Astart, Task_Bstart, ntasks,
            A, B, A_slice_kfirst, B_slice_kfirst, theta
            #if GB_IS_EWISEUNION
            , alpha_scalar, beta_scalar
            #endif
            ) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase6: find leading entries of each column of C
    //--------------------------------------------------------------------------

    // printf ("\ncuda ewise start phase6:\n") ;
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
        GB_cuda_ewise_sparse_phase6 <<<grid, block6, 0, stream>>>
            (/* outputs: */ JDelta, JDeltaSum,
             /* inputs:  */ Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ( )) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // phase7: construct Cp and Ch
    //--------------------------------------------------------------------------

    // printf ("\ncuda ewise start phase7:\n") ;
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
        GB_cuda_ewise_sparse_phase7 <<<grid, block6, 0, stream>>>
            (C, JDelta, JDeltaSum, Cj, cnz, nchunks_in_C) ;
        CUDA_OK (cudaGetLastError ( )) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // printf ("\ncuda ewise done\n") ;
    C->magic = GB_MAGIC ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

