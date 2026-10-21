//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_add_sparse.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A+B kernel on the GPU.   A and B cannot be jumbled on input, and C is
// returned as unjumbled.  No mask is exploited.

#include "template/GB_cuda_ek_slice.cuh"

#define CHUNKSIZE1 256
#define CHUNKMASK
#define LOG2_CHUNKSIZE1 8

#define CHUNK_FIRST_MASK1 0xFF
#define CHUNK_DELTA_MASK1 0xFFFFFFFFFFFFFF00

#define CHUNKSIZE2 16384
#define LOG2_CHUNKSIZE2 14

#define CHUNKSIZE3 256
#define LOG2_CHUNKSIZE3 8

// phase1 (GPU):  slices A and B into small chunks and sets up the ek_slice
//      methods with GB_cuda_ek_slice_search for each chunk.  The results are
//      saved in 2 workspace arrays of size equal to the # of chunks of A and
//      B: A_slice_kfirst and B_slice_kfirst.  This setup will be used in
//      phases 2 and 5 to compute the column index for each entry in A and B.
//      Note that the slope is not computed; that is computed as needed from
//      A_slice_kfirst and B_slice_kfirst, later on.

// phase 2 (GPU):  use the merge-path method to find outer tasks for C, each
//      of which are fairly large (16K entries to handle in A and B).

/*
// phase 3 (GPU): compute the size of the set intersection of A and B for
        each outer task t of C.  Let S[t] = size of intersection of A_t and
        B_t, where A_t and B_t are the set of entries in the t-th outer task
        found in phase 2.  Then # of entries in C_t is |A_t| + |B_t| - S[t].
        Do each outer task with a merge-path method like the dot3_mp method.
        Use a set intersection since it's faster to compute than set union; it
        can trim its inputs.  Each outer task C_t is completely independent of
        other outer tasks, because of the merge-path construction in phase 2.

// phase 4 (CPU):  compute the cumulative sum of |C_t|.  This gives the total
        size of C.  Allocate C (Ci, Cj, Cx each of size cnz).  Cj will be
        compressed later into Cp and Ch, much like the CUDA builder kernel.

// phase 5 (GPU): same outer tasks as phase 3 but now compute the output
        matrix C in coordinate form: (i,j,cij), using a set-union, merge-path
        method like dot3, but it cannot trim its inputs A and B.  During this
        phase, use shared memory of size 256 to load Ai_s and Bi_s from Ai and
        Bi, then use GB_cuda_ek_slice_entry to compute all column indices in
        Ak_s and Bk_s of all entries, before starting the merge-path searches.
        This makes the binary search (per thread) of merge-path very efficient.

        The output of this phase is the final matrix C in sorted coordinate
        form: Cj, Ci, Cx, of size cnz.

// phase 6 (GPU):  convert (Cj,Ci,Cx) into the hypersparse form Ch,Cp,Ci,Cx,
        use a method like GB_cuda_builder_phase3_no_dupl, phase4, and
        phase5_transplant of that method,except j is in Cj, not Key_out.  Ci
        and Cx are not modified in this phase.
*/

#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&A_slice_kfirst, A_slice_kfirst_mem) ;  \
    GB_FREE_MEMORY (&B_slice_kfirst, B_slice_kfirst_mem) ;  \
    GB_FREE_MEMORY (&Task_Astart   , Task_Astart_mem) ;     \
    GB_FREE_MEMORY (&Task_Bstart   , Task_Bstart_mem) ;     \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE ;

//------------------------------------------------------------------------------
// getk: helper method to query a single entry with GB_cuda_ek_slice_entry 
//------------------------------------------------------------------------------

// given an entry at position p, determine the vector k that contains it
template <typename T> __device__ int64_t getk
(
    // inputs, not modified:
    const int64_t p,                // position of entry to query
    const int64_t *Slice_kfirst,    // array of size nchunks+1
    const T *Ap,                    // array of size anvec+1
    const int64_t anvec1            // anvec-1
)
{
    int64_t chunk  = p >> LOG2_CHUNKSIZE1 ;
    int64_t pdelta = p && CHUNK_DELTA_MASK1 ;
    int64_t kfirst = Slice_kfirst [chunk] ;
    int64_t klast  = Slice_kfirst [chunk+1] ;
    float slope = ((float) (klast - kfirst + 1)) / ((float) CHUNKSIZE1) ;
    return (GB_cuda_ek_slice_entry<T> (pA, pdelta, Ap, anvec1, kfirst, slope)) ;
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase1: construct ek_slice info for each chunk of A and B
//------------------------------------------------------------------------------

// Unlike the typical usage of GB_cuda_ek_slice_setup where all threads in a
// threadblock call the setup for the same chunk and then use the results for
// their own part of this chunk, this method uses a single thread to setup each
// chunk.  The results of GB_cuda_ek_slice_search are not used here, but
// constructed for later phases.  The computation will be irregular since each
// thread does its own binary search.  However, threads t and t+1 in a single
// warp will operate on nearby entries of the matrix, so there will be some
// reuse of the Ap and Bp arrays within that warp, and some change of avoiding
// warp divergence.

__global__ void GB_cuda_add_sparse_phase1
(
    // outputs:
    int64_t *A_slice_kfirst,    // array of size A_slice_nchunks+1
    int64_t *B_slice_kfirst,    // array of size B_slice_nchunks+1
    // inputs:
    int64_t A_slice_nchunks,    // # of chunks of A for ek_slice
    int64_t B_slice_nchunks,    // # of chunks of B for ek_slice
    GrB_Matrix A,
    GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;

    const int64_t bnvec = B->nvec ;
    const GB_Bp_TYPE *__restrict__ Bp = (GB_Bp_TYPE *) B->p ;

    //--------------------------------------------------------------------------
    // setup ek_slice for each chunk of A
    //--------------------------------------------------------------------------

    int64_t k = 0 ;
    for (int64_t chunk = blockIdx.x * blockDim.x + threadIdx.x ;
                 chunk < A_slice_nchunks ;
                 chunk += blockDim.x & gridDim.x)        // grid-stride loop
    {
        int64_t pfirst = chunk << LOG2_CHUNKSIZE1 ;
        GB_cuda_ek_slice_search<GB_Ap_TYPE> (&k, Ap, anvec, pfirst) ;
        A_slice_kfirst [chunk] = k ;
    }

    //--------------------------------------------------------------------------
    // setup ek_slice for each chunk of B
    //--------------------------------------------------------------------------

    k = 0 ;
    for (int64_t chunk = blockIdx.x * blockDim.x + threadIdx.x ;
                 chunk < B_slice_nchunks ;
                 chunk += blockDim.x & gridDim.x)        // grid-stride loop
    {
        int64_t pfirst = chunk << LOG2_CHUNKSIZE1 ;
        GB_cuda_ek_slice_search<GB_Bp_TYPE> (&k, Bp, bnvec, pfirst) ;
        B_slice_kfirst [chunk] = k ;
    }

    //--------------------------------------------------------------------------
    // sentinal value for the last chunks of A and B
    //--------------------------------------------------------------------------

    if (blockDim.x == 0 && threadIdx.x == 0)
    {
        A_slice_kfirst [A_slice_nchunks] = anvec ;
        B_slice_kfirst [B_slice_nchunks] = anvec ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase2: construct outer tasks
//------------------------------------------------------------------------------

// This method divides the work to compute C=A+B into outer tasks, each of
// which is found via the mergepath method on the implicit, sorted, coordinate
// forms of A and B.  This kernel launch uses a single thread to compute the
// starting points of a single outer task.  As a result, there will be a lot
// of warp divergence in this kernel, but very little work is done since the
// outer tasks are very large (16K).

__global__ void GB_cuda_add_sparse_phase2
(
    // outputs:
    int64_t *Task_Astart,       // array of size # tasks+1
    int64_t *Task_Bstart,       // array of size # tasks+1
    // inputs:
    int64_t ntasks,             // # of outer tasks
    int64_t *A_slice_kfirst,    // array of size A_slice_nchunks+1
    int64_t *B_slice_kfirst,    // array of size B_slice_nchunks+1
    GrB_Matrix A,
    GrB_Matrix B,
    int64_t anz,
    int64_t bnz
)
{

    for (int64_t task = blockIdx.x * blockDim.x + threadIdx.x ;
                 task < ntasks ;
                 task += blockDim.x & gridDim.x)        // grid-stride loop
    {

        //---------------------------------------------------------------------
        // construct the task via mergepath
        //---------------------------------------------------------------------

        // a single thread computes the starting point of each outer task
        // using the merge-path method. 

        int64_t diag = task << LOG2_CHUNKSIZE2 ;
        int64_t amin = GB_IMAX (diag - bnz, 0) ;
        int64_t amax = GB_IMIN (diag, anz) ;

        while (amin < amax)
        {

            //------------------------------------------------------------------
            // cut the diagonal (amin:amax) in half
            //------------------------------------------------------------------

            int64_t pivot = (amin + amax) >> 1 ;

            //------------------------------------------------------------------
            // find the row/col indices (iA,jA) of the entry of A, at the pivot
            //------------------------------------------------------------------

            int64_t pA = pivot ;
            GB_Ai_TYPE iA = Ai [pA] ;
            int64_t k = getk<GB_Ap_TYPE> (pA, A_slice_kfirst, Ap, anvec1) ;
            GB_Aj_TYPE jA = GBh_A (Ah, k) ;

            //------------------------------------------------------------------
            // find the row/col indices (iB,jB) of the entry of B at position pB
            //------------------------------------------------------------------

            int64_t pB = diag - pivot - 1 ;
            GB_Bi_TYPE iB = Bi [pB] ;
            k = getk<GB_Bp_TYPE> (pB, B_slice_kfirst, Bp, bnvec1) ;
            GB_Bj_TYPE jB = GBh_B (Bh, k) ;

            //------------------------------------------------------------------
            // determine which entry comes first
            //------------------------------------------------------------------

            int64_t afirst = (int64_t) ((jA < jB) || (jA == jB && iA < iB)) ;

            //------------------------------------------------------------------
            // if (afirst) amin = pivot+1 else amax = pivot
            //------------------------------------------------------------------

            amin = (pivot + 1) * (afirst) + amin * (1-afirst) ;
            amax = pivot * (1-afirst)     + amax * (afirst) ;
        }

        //----------------------------------------------------------------------
        // adjust the start point of this task
        //----------------------------------------------------------------------

        int64_t astart = amin ;
        int64_t bstart = diag - astart ;

        if ((astart < anz) && (bstart >= 1))
        {
            // get iA for the entry (iA,jA) at position astart
            int64_t pA = astart ;
            GB_Ai_TYPE iA = Ai [pA] ;
            // get iB for the entry (iB,jB) at position bstart-1
            int64_t pB = bstart - 1 ;
            GB_Bi_TYPE iB = Bi [pB] ;
            if (iA == iB)
            {
                // row indices match; check the column indices.
                // get jA for the entry (iA,jA) at position astart
                int64_t k = getk<GB_Ap_TYPE> (pA, A_slice_kfirst, Ap, anvec1) ;
                GB_Aj_TYPE jA = GBh_A (Ah, k) ;
                // get jB for the entry (iB,jB) at position bstart-1
                k = getk<GB_Bp_TYPE> (pB, B_slice_kfirst, Bp, bnvec1) ;
                GB_Bj_TYPE jB = GBh_B (Bh, k) ;
                if (jA == jB)
                {
                    // Column indices match, so these entries in A and B are in
                    // the same place in the matrix and must appear in the same
                    // task.  Adjust this task by moving the last entry of B in
                    // the prior task (at bstart-1) into this task as the first
                    // entry in B for this task (revising bstart).
                    bstart-- ;
                }
            }
        }

        //----------------------------------------------------------------------
        // save the results in global memory
        //----------------------------------------------------------------------

        Task_Astart [task] = astart ;
        Task_Bstart [task] = bstart ;
    }

    //--------------------------------------------------------------------------
    // sentinal value for the last task
    //--------------------------------------------------------------------------

    if (blockDim.x == 0 && threadIdx.x == 0)
    {
        Task_Astart [ntasks] = anz ;
        Task_Bstart [ntasks] = bnz ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_add_sparse_phase3: compute intersection of A and B for each task
//------------------------------------------------------------------------------

// C, A, and B have been split into tasks.  For task t, the entries in A are
// in Ax, Ai, Ak [Task_Astart [t] ... Task_Astart [t+1]-1], and in B 
// at Bx, Bi, Bk [Task_Bstart [t] ... Task_Bstart [t+1]-1].

// Each task is done by a single threadblock.  All threads take part in the
// work by using the mergepath method to split the work for each thread.

__global__ void GB_cuda_add_sparse_phase3
(
    // outputs:
    int64_t *Task_Csize,        // array of size # tasks+1; Task_Csize [t]
                                // is the size of the set intersection (union?)
                                // of A and B for task t
    // inputs:
    int64_t *Task_Astart,       // array of size # tasks+1
    int64_t *Task_Bstart,       // array of size # tasks+1
    int64_t ntasks,             // # of outer tasks
    int64_t *A_slice_kfirst,    // array of size A_slice_nchunks+1
    int64_t *B_slice_kfirst,    // array of size B_slice_nchunks+1
    GrB_Matrix A,
    GrB_Matrix B,
    int64_t anz,
    int64_t bnz
)
{

    for (int64_t t = blockIdx.x ; t < ntasks ; t += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the details of this task
        //----------------------------------------------------------------------

        int64_t pA_start = Task_Astart [t] ;
        int64_t pA_end   = Task_Astart [t+1] ;
        int64_t pB_start = Task_Bstart [t] ;
        int64_t pB_end   = Task_Bstart [t+1] ;

        int64_t na_task = pA_end - pA_start ;   // # entries in A for this task
        int64_t nb_task = pB_end - pB_start ;   // # entries in B for this task

        //----------------------------------------------------------------------
        // compute the size of the set intersection for this task
        //----------------------------------------------------------------------

        while (pA_start < pA_end && pB_start < pB_end)
        {

            //------------------------------------------------------------------
            // load the next chunks
            //------------------------------------------------------------------

            int64_t na = GB_IMIN (pA_start - pA_end, CHUNKSIZE3) ;
            int64_t nb = GB_IMIN (pB_start - pB_end, CHUNKSIZE3) ;

            __shared__ GB_Ai_TYPE Ai_s [CHUNKSIZE3] ;
            __shared__ GB_Aj_TYPE Aj_s [CHUNKSIZE3] ;
            __shared__ GB_Bi_TYPE Bi_s [CHUNKSIZE3] ;
            __shared__ GB_Bj_TYPE Bj_s [CHUNKSIZE3] ;

            for (int kk = threadIdx.x ; kk < na ; kk += blockDim.x)
            {
                Ai_s [kk] = Ai [kk + pA_start] ;
            }

            for (int kk = threadIdx.x ; kk < nb ; kk += blockDim.x)
            {
                Bi_s [kk] = Bi [kk + pB_start] ;
            }

            for (int kk = threadIdx.x ; kk < na ; kk += blockDim.x)
            {
                int64_t pA = kk + pA_start ;
                int64_t k = getk<GB_Ap_TYPE> (pA, A_slice_kfirst, Ap, anvec1) ;
                GB_Aj_TYPE jA = GBh_A (Ah, k) ;
                Aj_s [kk] = jA ;
            }

            for (int kk = threadIdx.x ; kk < nb ; kk += blockDim.x)
            {
                int64_t pB = kk + pB_start ;
                int64_t k = getk<GB_Bp_TYPE> (pB, B_slice_kfirst, Bp, anvec1) ;
                GB_Bj_TYPE jB = GBh_B (Bh, k) ;
                Bj_s [kk] = jB ;
            }

            //------------------------------------------------------------------
            // each thread finds the start of its own mini chunk
            //------------------------------------------------------------------

            int nab = na + nb ;
            // work_per_thread = ceil (nab / blockDim.x)
            // FIXME: use >> LOG2_BLOCKDIM instead
            int work_per_thread = (nab + blockDim.x - 1) / blockDim.x ;
            int diag = GB_IMIN (work_per_thread * threadIdx.x, nab) ;
            int amin = GB_IMAX (diag - nb, 0) ;
            int amax = GB_IMIN (diag, na) ;

            while (amin < amax)
            {

                //--------------------------------------------------------------
                // cut the diagonal (amin:amax) in half
                //--------------------------------------------------------------

                int pivot = (amin + amax) >> 1 ;

                //--------------------------------------------------------------
                // find row/col indices (iA,jA) of the entry of A, at the pivot
                //--------------------------------------------------------------

                int64_t iA = Ai_s [pivot] ;
                int64_t jA = Aj_s [pivot] ;

                //--------------------------------------------------------------
                // find row/col indices (iB,jB) of the entry of B, at b
                //--------------------------------------------------------------

                int b = diag - pivot - 1 ;
                int64_t iB = Bi_s [b] ;
                int64_t jB = Bj_s [b] ;

                int64_t afirst = (int64_t) ((jA < jB) || (jA == jB && iA < iB));

                //--------------------------------------------------------------
                // if (afirst) amin = pivot+1 else amax = pivot
                //--------------------------------------------------------------

                amin = (pivot + 1) * (afirst) + amin * (1-afirst) ;
                amax = pivot * (1-afirst)     + amax * (afirst) ;
            }

            int bstart = diag - amin ;
            if ((astart < na) && (bstart >= 1))
            { 
                int64_t iA = Ai_s [astart] ;
                int64_t jA = Aj_s [astart] ;
                int64_t iB = Bi_s [bstart-1] ;
                int64_t jB = Bj_s [bstart-1] ;

                ...

            }



        }
    }
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

    int64_t *A_slice_kfirst = NULL ; uint64_t A_slice_kfirst_mem = mem ;
    int64_t *B_slice_kfirst = NULL ; uint64_t B_slice_kfirst_mem = mem ;

    int64_t *Task_Astart = NULL ; uint64_t Task_Astart_mem = mem ;
    int64_t *Task_Bstart = NULL ; uint64_t Task_Bstart_mem = mem ;

    GB_A_NHELD (anz) ;          // # of entries in A
    GB_B_NHELD (bnz) ;          // # of entries in B

    // chunks of A and B for ek_slice:
    int64_t A_slice_nchunks = (anz + CHUNKSIZE1 - 1) >> LOG2_CHUNKSIZE1 ;
    int64_t B_slice_nchunks = (bnz + CHUNKSIZE1 - 1) >> LOG2_CHUNKSIZE1 ;

    // # of outer tasks to compute all of C=A+B
    int64_t ntasks = (anz + bnz + CHUNKSIZE2 - 1) >> LOG2_CHUNKSIZE2 ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block1 (BLOCKDIM1) ;
    dim3 block2 (BLOCKDIM2) ;

    // ...

    //--------------------------------------------------------------------------
    // phase 1: setup ek_slice for each chunk of A and B
    //--------------------------------------------------------------------------

    A_slice_kfirst = GB_MALLOC_MEMORY (A_slice_nchunks+1, sizeof (int64_t),
        &A_slice_kfirst_mem) ;
    B_slice_kfirst = GB_MALLOC_MEMORY (B_slice_nchunks+1, sizeof (int64_t),
        &B_slice_kfirst_mem) ;

    if (A_slice_kfirst == NULL || B_slice_kfirst == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 1: phase1
    GB_cuda_add_sparse_phase1 <<<grid, block1, 0, stream>>>
        (/* outputs: */ A_slice_kfirst, B_slice_kfirst,
         /* inputs: */  A_slice_nchunks, B_slice_nchunks, A, B) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 2: construct outer tasks
    //--------------------------------------------------------------------------

    Task_Astart = GB_MALLOC_MEMORY (ntasks+1, sizeof (int64_t),
        &Task_Astart_mem) ;
    Task_Bstart = GB_MALLOC_MEMORY (ntasks+1, sizeof (int64_t),
        &Task_Bstart_mem) ;

    if (Task_Astart == NULL || Task_Bstart == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // KERNEL LAUNCH 2: phase2
    GB_cuda_add_sparse_phase2 <<<grid, block1, 0, stream>>>
        (   // outputs:
            Task_Astart, Task_Bstart,
            // inputs:
            ntasks, A_slice_kfirst, B_slice_kfirst, A, B, anz, bnz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase3:
    //--------------------------------------------------------------------------

    // ...

    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
}

