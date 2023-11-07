//------------------------------------------------------------------------------
// GB_msort_1: sort a 1-by-n list of integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// A parallel mergesort of an array of 1-by-n integers.

#include "GB_msort_1.h"

//------------------------------------------------------------------------------
// GB_msort_1_binary_search: binary search for the pivot
//------------------------------------------------------------------------------

// The Pivot value is Z [pivot], and a binary search for the Pivot is made in
// the array X [p_pstart...p_end-1], which is sorted in non-decreasing order on
// input.  The return value is pleft, where
//
//    X [p_start ... pleft-1] <= Pivot and
//    X [pleft ... p_end-1] >= Pivot holds.
//
// pleft is returned in the range p_start to p_end.  If pleft is p_start, then
// the Pivot is smaller than all entries in X [p_start...p_end-1], and the left
// list X [p_start...pleft-1] is empty.  If pleft is p_end, then the Pivot is
// larger than all entries in X [p_start...p_end-1], and the right list X
// [pleft...p_end-1] is empty.

static int64_t GB_msort_1_binary_search    // return pleft
(
    const int64_t *restrict Z_0,         // Pivot is Z [pivot]
    const int64_t pivot,
    const int64_t *restrict X_0,         // search in X [p_start..p_end_-1]
    const int64_t p_start,
    const int64_t p_end
)
{

    //--------------------------------------------------------------------------
    // find where the Pivot appears in X
    //--------------------------------------------------------------------------

    // binary search of X [p_start...p_end-1] for the Pivot
    int64_t pleft = p_start ;
    int64_t pright = p_end - 1 ;
    while (pleft < pright)
    { 
        int64_t pmiddle = (pleft + pright) >> 1 ;
        // less = (X [pmiddle] < Pivot)
        bool less = GB_lt_1 (X_0, pmiddle,
                             Z_0, pivot) ;
        pleft  = less ? (pmiddle+1) : pleft ;
        pright = less ? pright : pmiddle ;
    }

    // binary search is narrowed down to a single item
    // or it has found the list is empty:
    ASSERT (pleft == pright || pleft == pright + 1) ;

    // If found is true then X [pleft == pright] == Pivot.  If duplicates
    // appear then X [pleft] is any one of the entries equal to the Pivot
    // in the list.  If found is false then
    //    X [p_start ... pleft-1] < Pivot and
    //    X [pleft+1 ... p_end-1] > Pivot holds.
    //    The value X [pleft] may be either < or > Pivot.
    bool found = (pleft == pright) && GB_eq_1 (X_0, pleft,
                                               Z_0, pivot) ;

    // Modify pleft and pright:
    if (!found && (pleft == pright))
    { 
        if (GB_lt_1 (X_0, pleft,
                     Z_0, pivot))
        {
            pleft++ ;
        }
        else
        {
//          pright++ ;  // (not needed)
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // If found is false then
    //    X [p_start ... pleft-1] < Pivot and
    //    X [pleft ... p_end-1] > Pivot holds,
    //    and pleft-1 == pright

    // If X has no duplicates, then whether or not Pivot is found,
    //    X [p_start ... pleft-1] < Pivot and
    //    X [pleft ... p_end-1] >= Pivot holds.

    // If X has duplicates, then whether or not Pivot is found,
    //    X [p_start ... pleft-1] <= Pivot and
    //    X [pleft ... p_end-1] >= Pivot holds.

    return (pleft) ;
}

//------------------------------------------------------------------------------
// GB_msort_1_create_merge_tasks
//------------------------------------------------------------------------------

// Recursively constructs ntasks tasks to merge two arrays, Left and Right,
// into Sresult, where Left is L [pL_start...pL_end-1], Right is R
// [pR_start...pR_end-1], and Sresult is S [pS_start...pS_start+total_work-1],
// and where total_work is the total size of Left and Right.
//
// Task tid will merge L [L_task [tid] ... L_task [tid] + L_len [tid] - 1] and
// R [R_task [tid] ... R_task [tid] + R_len [tid] -1] into the merged output
// array S [S_task [tid] ... ].  The task tids created are t0 to
// t0+ntasks-1.

void GB_msort_1_create_merge_tasks
(
    // output:
    int64_t *restrict L_task,        // L_task [t0...t0+ntasks-1] computed
    int64_t *restrict L_len,         // L_len  [t0...t0+ntasks-1] computed
    int64_t *restrict R_task,        // R_task [t0...t0+ntasks-1] computed
    int64_t *restrict R_len,         // R_len  [t0...t0+ntasks-1] computed
    int64_t *restrict S_task,        // S_task [t0...t0+ntasks-1] computed
    // input:
    const int t0,                       // first task tid to create
    const int ntasks,                   // # of tasks to create
    const int64_t pS_start,             // merge into S [pS_start...]
    const int64_t *restrict L_0,     // Left = L [pL_start...pL_end-1]
    const int64_t pL_start,
    const int64_t pL_end,
    const int64_t *restrict R_0,     // Right = R [pR_start...pR_end-1]
    const int64_t pR_start,
    const int64_t pR_end
)
{

    //--------------------------------------------------------------------------
    // get problem size
    //--------------------------------------------------------------------------

    int64_t nleft  = pL_end - pL_start ;        // size of Left array
    int64_t nright = pR_end - pR_start ;        // size of Right array
    int64_t total_work = nleft + nright ;       // total work to do
    ASSERT (ntasks >= 1) ;
    ASSERT (total_work > 0) ;

    //--------------------------------------------------------------------------
    // create the tasks
    //--------------------------------------------------------------------------

    if (ntasks == 1)
    { 

        //----------------------------------------------------------------------
        // a single task will merge all of Left and Right into Sresult
        //----------------------------------------------------------------------

        L_task [t0] = pL_start ; L_len [t0] = nleft ;
        R_task [t0] = pR_start ; R_len [t0] = nright ;
        S_task [t0] = pS_start ;

    }
    else
    {

        //----------------------------------------------------------------------
        // partition the Left and Right arrays for multiple merge tasks
        //----------------------------------------------------------------------

        int64_t pleft, pright ;
        if (nleft >= nright)
        { 
            // split Left in half, and search for its pivot in Right
            pleft = (pL_end + pL_start) >> 1 ;
            pright = GB_msort_1_binary_search (
                        L_0, pleft,
                        R_0, pR_start, pR_end) ;
        }
        else
        { 
            // split Right in half, and search for its pivot in Left
            pright = (pR_end + pR_start) >> 1 ;
            pleft = GB_msort_1_binary_search (
                        R_0, pright,
                        L_0, pL_start, pL_end) ;
        }

        //----------------------------------------------------------------------
        // partition the tasks according to the work of each partition
        //----------------------------------------------------------------------

        // work0 is the total work in the first partition
        int64_t work0 = (pleft - pL_start) + (pright - pR_start) ;
        int ntasks0 = (int) round ((double) ntasks *
            (((double) work0) / ((double) total_work))) ;

        // ensure at least one task is assigned to each partition
        ntasks0 = GB_IMAX (ntasks0, 1) ;
        ntasks0 = GB_IMIN (ntasks0, ntasks-1) ;
        int ntasks1 = ntasks - ntasks0 ;

        //----------------------------------------------------------------------
        // assign ntasks0 to the first half
        //----------------------------------------------------------------------

        // ntasks0 tasks merge L [pL_start...pleft-1] and R [pR_start..pright-1]
        // into the result S [pS_start...work0-1].

        GB_msort_1_create_merge_tasks (
            L_task, L_len, R_task, R_len, S_task, t0, ntasks0, pS_start,
            L_0, pL_start, pleft,
            R_0, pR_start, pright) ;

        //----------------------------------------------------------------------
        // assign ntasks1 to the second half
        //----------------------------------------------------------------------

        // ntasks1 tasks merge L [pleft...pL_end-1] and R [pright...pR_end-1]
        // into the result S [pS_start+work0...pS_start+total_work].

        int t1 = t0 + ntasks0 ;     // first task id of the second set of tasks
        int64_t pS_start1 = pS_start + work0 ;  // 2nd set starts here in S
        GB_msort_1_create_merge_tasks (
            L_task, L_len, R_task, R_len, S_task, t1, ntasks1, pS_start1,
            L_0, pleft,  pL_end,
            R_0, pright, pR_end) ;
    }
}

//------------------------------------------------------------------------------
// GB_msort_1_merge: merge two sorted lists via a single thread
//------------------------------------------------------------------------------

// merge Left [0..nleft-1] and Right [0..nright-1] into S [0..nleft+nright-1] */

static void GB_msort_1_merge
(
    int64_t *restrict S_0,              // output of length nleft + nright
    const int64_t *restrict Left_0,     // left input of length nleft
    const int64_t nleft,
    const int64_t *restrict Right_0,    // right input of length nright
    const int64_t nright
)
{
    int64_t p, pleft, pright ;

    // merge the two inputs, Left and Right, while both inputs exist
    for (p = 0, pleft = 0, pright = 0 ; pleft < nleft && pright < nright ; p++)
    {
        if (GB_lt_1 (Left_0,  pleft,
                     Right_0, pright))
        { 
            // S [p] = Left [pleft++]
            S_0 [p] = Left_0 [pleft] ;
            pleft++ ;
        }
        else
        { 
            // S [p] = Right [pright++]
            S_0 [p] = Right_0 [pright] ;
            pright++ ;
        }
    }

    // either input is exhausted; copy the remaining list into S
    if (pleft < nleft)
    { 
        int64_t nremaining = (nleft - pleft) ;
        memcpy (S_0 + p, Left_0 + pleft, nremaining * sizeof (int64_t)) ;
    }
    else if (pright < nright)
    { 
        int64_t nremaining = (nright - pright) ;
        memcpy (S_0 + p, Right_0 + pright, nremaining * sizeof (int64_t)) ;
    }
}

//------------------------------------------------------------------------------
// GB_msort_1: parallel mergesort
//------------------------------------------------------------------------------

GrB_Info GB_msort_1     // sort array A of size 1-by-n
(
    int64_t *restrict A_0,   // size n array
    const int64_t n,
    int nthreads                // # of threads to use
)
{

    //--------------------------------------------------------------------------
    // handle small problems with a single thread
    //--------------------------------------------------------------------------

    if (nthreads <= 1 || n <= GB_BASECASE)
    { 
        // sequential quicksort
        GB_qsort_1 (A_0, n) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // determine # of tasks
    //--------------------------------------------------------------------------

    // determine the number of levels to create, which must always be an
    // even number.  The # of levels is chosen to ensure that the # of leaves
    // of the task tree is between 4*nthreads and 16*nthreads.

    //  2 to 4 threads:     4 levels, 16 qsort leaves
    //  5 to 16 threads:    6 levels, 64 qsort leaves
    // 17 to 64 threads:    8 levels, 256 qsort leaves
    // 65 to 256 threads:   10 levels, 1024 qsort leaves
    // 256 to 1024 threads: 12 levels, 4096 qsort leaves
    // ...

    int k = (int) (2 + 2 * ceil (log2 ((double) nthreads) / 2)) ;
    int ntasks = 1 << k ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t *restrict W = NULL ; size_t W_size = 0 ;
    W = GB_MALLOC_WORK (n + 6*ntasks + 1, int64_t, &W_size) ;
    if (W == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    int64_t *T = W ;
    int64_t *restrict W_0    = T ; T += n ;
    int64_t *restrict L_task = T ; T += ntasks ;
    int64_t *restrict L_len  = T ; T += ntasks ;
    int64_t *restrict R_task = T ; T += ntasks ;
    int64_t *restrict R_len  = T ; T += ntasks ;
    int64_t *restrict S_task = T ; T += ntasks ;
    int64_t *restrict Slice  = T ; T += (ntasks+1) ;  

    //--------------------------------------------------------------------------
    // partition and sort the leaves
    //--------------------------------------------------------------------------

    GB_eslice (Slice, n, ntasks) ;
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    { 
        int64_t leaf = Slice [tid] ;
        int64_t leafsize = Slice [tid+1] - leaf ;
        GB_qsort_1 (A_0 + leaf, leafsize) ;
    }

    //--------------------------------------------------------------------------
    // merge each level
    //--------------------------------------------------------------------------

    int nt = 1 ;
    for ( ; k >= 2 ; k -= 2)
    {

        //----------------------------------------------------------------------
        // merge level k into level k-1, from A into W
        //----------------------------------------------------------------------

        // TODO: skip k and k-1 for each group of 4 sublists of A if they are
        // already sorted with respect to each other.

        // this could be done in parallel if ntasks was large
        for (int tid = 0 ; tid < ntasks ; tid += 2*nt)
        { 
            // create 2*nt tasks to merge two A sublists into one W sublist
            GB_msort_1_create_merge_tasks (
                L_task, L_len, R_task, R_len, S_task, tid, 2*nt, Slice [tid],
                A_0, Slice [tid],    Slice [tid+nt],
                A_0, Slice [tid+nt], Slice [tid+2*nt]) ;
        }

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        { 
            // merge A [pL...pL+nL-1] and A [pR...pR+nR-1] into W [pS..]
            int64_t pL = L_task [tid], nL = L_len [tid] ;
            int64_t pR = R_task [tid], nR = R_len [tid] ;
            int64_t pS = S_task [tid] ;

            GB_msort_1_merge (
                W_0 + pS,
                A_0 + pL, nL,
                A_0 + pR, nR) ;
        }
        nt = 2*nt ;

        //----------------------------------------------------------------------
        // merge level k-1 into level k-2, from W into A
        //----------------------------------------------------------------------

        // this could be done in parallel if ntasks was large
        for (int tid = 0 ; tid < ntasks ; tid += 2*nt)
        { 
            // create 2*nt tasks to merge two W sublists into one A sublist
            GB_msort_1_create_merge_tasks (
                L_task, L_len, R_task, R_len, S_task, tid, 2*nt, Slice [tid],
                W_0, Slice [tid],    Slice [tid+nt],
                W_0, Slice [tid+nt], Slice [tid+2*nt]) ;
        }

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        { 
            // merge A [pL...pL+nL-1] and A [pR...pR+nR-1] into W [pS..]
            int64_t pL = L_task [tid], nL = L_len [tid] ;
            int64_t pR = R_task [tid], nR = R_len [tid] ;
            int64_t pS = S_task [tid] ;
            GB_msort_1_merge (
                A_0 + pS,
                W_0 + pL, nL,
                W_0 + pR, nR) ;
        }
        nt = 2*nt ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK (&W, W_size) ;
    return (GrB_SUCCESS) ;
}

