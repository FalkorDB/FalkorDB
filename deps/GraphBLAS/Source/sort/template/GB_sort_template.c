//------------------------------------------------------------------------------
// GB_sort_template: sort all vectors in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//  macros:
//  GB_SORT (func)      defined as GB_sort_func_TYPE_ascend or _descend,
//                      GB_sort_ISO_ascend or _descend, or GB_sort_func_UDT
//  GB_C_TYPE           bool, int8_, ... or GB_void for UDT or ISO
//  GB_Ci_TYPE          the type of C->i (uint32_t or uint64_t)
//  GB_ADDR(A,p)        A+p for builtin, A + p * GB_SIZE otherwise
//  GB_SIZE             size of each entry: sizeof (GB_C_TYPE) for built-in
//  GB_GETX(x,X,i)      x = X [i] for built-in, memcpy for UDT
//  GB_COPY(A,i,C,k)    A[i] = C [k]
//  GB_SWAP(A,i,k)      swap A[i] and A[k]
//  GB_LT               compare two entries, x < y, or x > y for descending sort

#ifndef GB_SORT_BASECASE
#define GB_SORT_BASECASE (64 * 1024)
#endif

//------------------------------------------------------------------------------
// GB_SORT (partition): use a pivot to partition an array
//------------------------------------------------------------------------------

// C.A.R Hoare partition method, partitions an array in-place via a pivot.
// k = partition (A, n) partitions A [0:n-1] such that all entries in
// A [0:k] are <= all entries in A [k+1:n-1].

static inline int64_t GB_SORT (partition)
(
    GB_C_TYPE  *restrict A_0,   // size n arrays to partition
    GB_Ci_TYPE *restrict A_1,   // size n array (uint32_t or uint64_t)
    const int64_t n,            // size of the array(s) to partition
    uint64_t *seed              // random number seed, modified on output
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
)
{

    // select a pivot at random
    uint64_t pivot = GB_rand (seed) % ((uint64_t) n) ;

    // Pivot = A [pivot]
    GB_GETX (Pivot0, A_0, pivot) ;       // Pivot0 = A_0 [pivot]
    int64_t Pivot1 = A_1 [pivot] ;

    // At the top of the while loop, A [left+1...right-1] is considered, and
    // entries outside this range are in their proper place and not touched.
    // Since the input specification of this function is to partition A
    // [0..n-1], left must start at -1 and right must start at n.
    int64_t left = -1 ;
    int64_t right = n ;

    // keep partitioning until the left and right sides meet
    while (true)
    {
        // loop invariant:  A [0..left] < pivot and A [right..n-1] > Pivot,
        // so the region to be considered is A [left+1 ... right-1].

        // increment left until finding an entry A [left] >= Pivot
        bool less ;
        do
        { 
            left++ ;
            // a0 = A_0 [left]
            GB_GETX (a0, A_0, left) ;
            // less =   (a0, A_1 [left]) < (Pivot0, Pivot1)
            GB_LT (less, a0, A_1 [left],    Pivot0, Pivot1) ;
        }
        while (less) ;

        // decrement right until finding an entry A [right] <= Pivot
        do
        { 
            right-- ;
            // a0 = A_0 [right]
            GB_GETX (a1, A_0, right) ;
            // less =   (Pivot0, Pivot1) < (a1, A_1 [right])
            GB_LT (less, Pivot0, Pivot1,    a1, A_1 [right]) ;
        }
        while (less) ;

        // now A [0..left-1] < pivot and A [right+1..n-1] > pivot, but
        // A [left] > pivot and A [right] < pivot, so these two entries
        // are out of place and must be swapped.

        // However, if the two sides have met, the partition is finished.
        if (left >= right)
        { 
            // A has been partitioned into A [0:right] and A [right+1:n-1].
            // k = right+1, so A is split into A [0:k-1] and A [k:n-1].
            return (right + 1) ;
        }

        // since A [left] > pivot and A [right] < pivot, swap them
        GB_SWAP (A_0, left, right) ;
        int64_t t1 = A_1 [left] ; A_1 [left] = A_1 [right] ; A_1 [right] = t1 ;

        // after the swap this condition holds:
        // A [0..left] < pivot and A [right..n-1] > pivot
    }
}

//------------------------------------------------------------------------------
// GB_SORT (quicksort): recursive single-threaded quicksort
//------------------------------------------------------------------------------

static void GB_SORT (quicksort)    // sort A [0:n-1]
(
    GB_C_TYPE  *restrict A_0,   // size n arrays to sort
    GB_Ci_TYPE *restrict A_1,   // size n array
    const int64_t n,            // size of the array(s) to sort
    uint64_t *seed              // random number seed
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
)
{

    if (n < 8)
    {
        // in-place insertion sort on A [0:n-1], where n is small
        for (int64_t k = 1 ; k < n ; k++)
        {
            for (int64_t j = k ; j > 0 ; j--)
            { 
                // a0 = A_0 [j]
                GB_GETX (a0, A_0, j) ;
                // a1 = A_0 [j-1]
                GB_GETX (a1, A_0, j-1) ;
                // break if A [j] >= A [j-1]
                bool less ;
                // less =   (a0, A_1 [j]) < (a1, A_1 [j-1])
                GB_LT (less, a0, A_1 [j],    a1, A_1 [j-1]) ;
                if (!less) break ;
                // swap A [j-1] and A [j]
                GB_SWAP (A_0, j-1, j) ;
                int64_t t1 = A_1 [j-1] ; A_1 [j-1] = A_1 [j] ; A_1 [j] = t1 ;
            }
        }
    }
    else
    { 
        // partition A [0:n-1] into A [0:k-1] and A [k:n-1]
        int64_t k = GB_SORT (partition) (A_0, A_1, n, seed
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;

        // sort each partition

        // sort A [0:k-1]
        GB_SORT (quicksort) (A_0, A_1, k, seed
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;

        // sort A [k:n-1]
        GB_SORT (quicksort) (GB_ADDR (A_0, k), A_1 + k, n-k, seed
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;
    }
}

//------------------------------------------------------------------------------
// GB_SORT (binary_search): binary search for the pivot
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

static int64_t GB_SORT (binary_search)  // return pleft
(
    const GB_C_TYPE  *restrict Z_0,     // Pivot is Z [pivot]
    const GB_Ci_TYPE *restrict Z_1,
    const int64_t pivot,
    const GB_C_TYPE  *restrict X_0,     // search in X [p_start..p_end_-1]
    const GB_Ci_TYPE *restrict X_1,
    const int64_t p_start,
    const int64_t p_end
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
)
{

    //--------------------------------------------------------------------------
    // find where the Pivot appears in X
    //--------------------------------------------------------------------------

    // binary search of X [p_start...p_end-1] for the Pivot
    int64_t pleft = p_start ;
    int64_t pright = p_end - 1 ;
    GB_GETX (Pivot0, Z_0, pivot) ;       // Pivot0 = Z_0 [pivot]
    int64_t Pivot1 = Z_1 [pivot] ;
    bool less ;
    while (pleft < pright)
    { 
        int64_t pmiddle = (pleft + pright) >> 1 ;
        // x0 = X_0 [pmiddle]
        GB_GETX (x0, X_0, pmiddle) ;
        // less =   (x0, X_1 [pmiddle]) < (Pivot0, Pivot1)
        GB_LT (less, x0, X_1 [pmiddle],    Pivot0, Pivot1) ;
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
    bool found = (pleft == pright) && (X_1 [pleft] == Pivot1) ;

    // Modify pleft and pright:
    if (!found && (pleft == pright))
    { 
        // x0 = X_0 [pleft]
        GB_GETX (x0, X_0, pleft) ;
        // less =   (x0, X_1 [pleft]) < (Pivot0, Pivot1)
        GB_LT (less, x0, X_1 [pleft],    Pivot0, Pivot1) ;
        if (less)
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
// GB_SORT (create_merge_tasks)
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

static void GB_SORT (create_merge_tasks)
(
    // output:
    int64_t *restrict L_task,        // L_task [t0...t0+ntasks-1] computed
    int64_t *restrict L_len,         // L_len  [t0...t0+ntasks-1] computed
    int64_t *restrict R_task,        // R_task [t0...t0+ntasks-1] computed
    int64_t *restrict R_len,         // R_len  [t0...t0+ntasks-1] computed
    int64_t *restrict S_task,        // S_task [t0...t0+ntasks-1] computed
    // input:
    const int t0,                    // first task tid to create
    const int ntasks,                // # of tasks to create
    const int64_t pS_start,          // merge into S [pS_start...]
    const GB_C_TYPE  *restrict L_0,  // Left = L [pL_start...pL_end-1]
    const GB_Ci_TYPE *restrict L_1,
    const int64_t pL_start,
    const int64_t pL_end,
    const GB_C_TYPE  *restrict R_0,  // Right = R [pR_start...pR_end-1]
    const GB_Ci_TYPE *restrict R_1,
    const int64_t pR_start,
    const int64_t pR_end
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
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
            pright = GB_SORT (binary_search) (
                        L_0, L_1, pleft,
                        R_0, R_1, pR_start, pR_end
                        #if GB_SORT_UDT
                        , csize, xsize, flt, fcast
                        #endif
                        ) ;
        }
        else
        { 
            // split Right in half, and search for its pivot in Left
            pright = (pR_end + pR_start) >> 1 ;
            pleft = GB_SORT (binary_search) (
                        R_0, R_1, pright,
                        L_0, L_1, pL_start, pL_end
                        #if GB_SORT_UDT
                        , csize, xsize, flt, fcast
                        #endif
                        ) ;
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

        GB_SORT (create_merge_tasks) (
            L_task, L_len, R_task, R_len, S_task, t0, ntasks0, pS_start,
            L_0, L_1, pL_start, pleft,
            R_0, R_1, pR_start, pright
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;

        //----------------------------------------------------------------------
        // assign ntasks1 to the second half
        //----------------------------------------------------------------------

        // ntasks1 tasks merge L [pleft...pL_end-1] and R [pright...pR_end-1]
        // into the result S [pS_start+work0...pS_start+total_work].

        int t1 = t0 + ntasks0 ;     // first task id of the second set of tasks
        int64_t pS_start1 = pS_start + work0 ;  // 2nd set starts here in S
        GB_SORT (create_merge_tasks) (
            L_task, L_len, R_task, R_len, S_task, t1, ntasks1, pS_start1,
            L_0, L_1, pleft,  pL_end,
            R_0, R_1, pright, pR_end
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;
    }
}

//------------------------------------------------------------------------------
// GB_SORT (merge): merge two sorted lists via a single thread
//------------------------------------------------------------------------------

// merge Left [0..nleft-1] and Right [0..nright-1] into S [0..nleft+nright-1] */

static void GB_SORT (merge)
(
    GB_C_TYPE  *restrict S_0,           // output of length nleft + nright
    GB_Ci_TYPE *restrict S_1,
    const GB_C_TYPE  *restrict L_0,     // left input of length nleft
    const GB_Ci_TYPE *restrict L_1,
    const int64_t nleft,
    const GB_C_TYPE  *restrict R_0,     // right input of length nright
    const GB_Ci_TYPE *restrict R_1,
    const int64_t nright
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
)
{
    int64_t p, pleft, pright ;

    // merge the two inputs, Left and Right, while both inputs exist
    for (p = 0, pleft = 0, pright = 0 ; pleft < nleft && pright < nright ; p++)
    {
        // left0 = L_0 [pleft]
        GB_GETX (left0, L_0, pleft) ;
        // right0 = R_0 [pright]
        GB_GETX (right0, R_0, pright) ;
        bool less ;
        // less =   (left0, L_1 [pleft]) < (right0, R_1 [pright])
        GB_LT (less, left0, L_1 [pleft],    right0, R_1 [pright]) ;
        if (less)
        { 
            // S [p] = Left [pleft++]
            GB_COPY (S_0, p, L_0, pleft) ;
            S_1 [p] = L_1 [pleft] ;
            pleft++ ;
        }
        else
        { 
            // S [p] = Right [pright++]
            GB_COPY (S_0, p, R_0, pright) ;
            S_1 [p] = R_1 [pright] ;
            pright++ ;
        }
    }

    // either input is exhausted; copy the remaining list into S
    if (pleft < nleft)
    { 
        int64_t nremaining = (nleft - pleft) ;
        memcpy (GB_ADDR (S_0, p), GB_ADDR (L_0, pleft), nremaining * GB_SIZE) ;
        memcpy (S_1 + p, L_1 + pleft, nremaining * sizeof (GB_Ci_TYPE)) ;
    }
    else if (pright < nright)
    { 
        int64_t nremaining = (nright - pright) ;
        memcpy (GB_ADDR (S_0, p), GB_ADDR (R_0, pright), nremaining * GB_SIZE) ;
        memcpy (S_1 + p, R_1 + pright, nremaining * sizeof (GB_Ci_TYPE)) ;
    }
}

//------------------------------------------------------------------------------
// GB_SORT (vector) parallel mergesort of a single vector
//------------------------------------------------------------------------------

static void GB_SORT (vector)    // sort the pair of arrays A_0, A_1
(
    GB_C_TYPE  *restrict A_0,   // size n array
    GB_Ci_TYPE *restrict A_1,   // size n array
    GB_C_TYPE  *restrict W_0,   // workspace of size n * GB_SIZE bytes
    GB_Ci_TYPE *restrict W_1,   // workspace of size n (uint32_t or uint64_t)
    int64_t    *restrict W,     // int64_t workspace of size 6*ntasks+1
    const int64_t n,
    const int kk,
    const int ntasks,
    const int nthreads          // # of threads to use
    #if GB_SORT_UDT
    , size_t csize              // size of GB_C_TYPE
    , size_t xsize              // size of op->xtype
    , GxB_binary_function flt   // function to test for < (ascend), > (descend)
    , GB_cast_function fcast    // cast entry to inputs of flt
    #endif
)
{

    //--------------------------------------------------------------------------
    // split up workspace
    //--------------------------------------------------------------------------

    ASSERT (nthreads > 2 && n >= GB_SORT_BASECASE) ;
    int64_t *T = W ;
    int64_t *restrict L_task = T ; T += ntasks ;
    int64_t *restrict L_len  = T ; T += ntasks ;
    int64_t *restrict R_task = T ; T += ntasks ;
    int64_t *restrict R_len  = T ; T += ntasks ;
    int64_t *restrict S_task = T ; T += ntasks ;
    int64_t *restrict Slice  = T ; T += (ntasks+1) ;

    //--------------------------------------------------------------------------
    // partition and sort the leaves
    //--------------------------------------------------------------------------

    GB_e_slice (Slice, n, ntasks) ;
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    { 
        int64_t leaf = Slice [tid] ;
        int64_t leafsize = Slice [tid+1] - leaf ;
        uint64_t seed = tid ;
        GB_SORT (quicksort) (GB_ADDR (A_0, leaf), A_1 + leaf, leafsize, &seed
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;
    }

    //--------------------------------------------------------------------------
    // merge each level
    //--------------------------------------------------------------------------

    int nt = 1 ;
    for (int k = kk ; k >= 2 ; k -= 2)
    {

        //----------------------------------------------------------------------
        // merge level k into level k-1, from A into W
        //----------------------------------------------------------------------

        // this could be done in parallel if ntasks was large
        for (tid = 0 ; tid < ntasks ; tid += 2*nt)
        { 
            // create 2*nt tasks to merge two A sublists into one W sublist
            GB_SORT (create_merge_tasks) (
                L_task, L_len, R_task, R_len, S_task, tid, 2*nt, Slice [tid],
                A_0, A_1, Slice [tid],    Slice [tid+nt],
                A_0, A_1, Slice [tid+nt], Slice [tid+2*nt]
                #if GB_SORT_UDT
                , csize, xsize, flt, fcast
                #endif
                ) ;
        }

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        { 
            // merge A [pL...pL+nL-1] and A [pR...pR+nR-1] into W [pS..]
            int64_t pL = L_task [tid], nL = L_len [tid] ;
            int64_t pR = R_task [tid], nR = R_len [tid] ;
            int64_t pS = S_task [tid] ;

            GB_SORT (merge) (
                GB_ADDR (W_0, pS), W_1 + pS,
                GB_ADDR (A_0, pL), A_1 + pL, nL,
                GB_ADDR (A_0, pR), A_1 + pR, nR
                #if GB_SORT_UDT
                , csize, xsize, flt, fcast
                #endif
                ) ;
        }
        nt = 2*nt ;

        //----------------------------------------------------------------------
        // merge level k-1 into level k-2, from W into A
        //----------------------------------------------------------------------

        // this could be done in parallel if ntasks was large
        for (tid = 0 ; tid < ntasks ; tid += 2*nt)
        { 
            // create 2*nt tasks to merge two W sublists into one A sublist
            GB_SORT (create_merge_tasks) (
                L_task, L_len, R_task, R_len, S_task, tid, 2*nt, Slice [tid],
                W_0, W_1, Slice [tid],    Slice [tid+nt],
                W_0, W_1, Slice [tid+nt], Slice [tid+2*nt]
                #if GB_SORT_UDT
                , csize, xsize, flt, fcast
                #endif
                ) ;
        }

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        { 
            // merge A [pL...pL+nL-1] and A [pR...pR+nR-1] into W [pS..]
            int64_t pL = L_task [tid], nL = L_len [tid] ;
            int64_t pR = R_task [tid], nR = R_len [tid] ;
            int64_t pS = S_task [tid] ;
            GB_SORT (merge) (
                GB_ADDR (A_0, pS), A_1 + pS,
                GB_ADDR (W_0, pL), W_1 + pL, nL,
                GB_ADDR (W_0, pR), W_1 + pR, nR
                #if GB_SORT_UDT
                , csize, xsize, flt, fcast
                #endif
                ) ;
        }
        nt = 2*nt ;
    }
}

//------------------------------------------------------------------------------
// sort all vectors in a matrix
//------------------------------------------------------------------------------

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_WERK_POP (SortTasks, int64_t) ;          \
    GB_FREE_MEMORY (&C_skipped, C_skipped_size) ; \
    GB_FREE_MEMORY (&W_0, W_0_size) ;             \
    GB_FREE_MEMORY (&W_1, W_1_size) ;             \
    GB_FREE_MEMORY (&W, W_size) ;                 \
}

#ifdef GB_JIT_KERNEL
GB_JIT_GLOBAL GB_JIT_KERNEL_SORT_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SORT_PROTO (GB_jit_kernel)
#else
static GrB_Info GB_SORT (mtx)
(
    GrB_Matrix C,               // matrix sorted in-place
    #if GB_SORT_UDT
    GrB_BinaryOp op,            // comparator for user-defined types only
    #endif
    int nthreads,               // # of threads to use
    GB_Werk Werk
)
#endif
{

    //--------------------------------------------------------------------------
    // get callback functions (JIT kernels only)
    //--------------------------------------------------------------------------

    GB_GET_CALLBACKS ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    #if GB_SORT_UDT
    ASSERT (op->ztype == GrB_BOOL) ;
    ASSERT (op->xtype == op->ytype) ;
    #endif

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_werk_pop) ;
    GB_GET_CALLBACK (GB_werk_push) ;
    GB_GET_CALLBACK (GB_p_slice) ;
    #endif

    //--------------------------------------------------------------------------
    // get input
    //--------------------------------------------------------------------------

    int64_t cnvec = C->nvec ;
    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ci_TYPE *restrict Ci = (GB_Ci_TYPE *) C->i ;
    GB_C_TYPE  *restrict Cx = (GB_C_TYPE  *) C->x ;

    // workspace
    GB_C_TYPE  *restrict W_0 = NULL ; size_t W_0_size = 0 ;
    GB_Ci_TYPE *restrict W_1 = NULL ; size_t W_1_size = 0 ;
    int64_t    *restrict W   = NULL ; size_t W_size   = 0 ;
    int64_t *restrict C_skipped = NULL ;
    size_t C_skipped_size = 0 ;
    GB_WERK_DECLARE (SortTasks, int64_t) ;

    #if GB_SORT_UDT
    // get typesize, and function pointers for operators and typecasting
    GrB_Type ctype = C->type ;
    size_t csize = ctype->size ;
    size_t xsize = op->xtype->size ;
    GxB_binary_function flt = op->binop_function ;
    ASSERT (flt != NULL) ;
    GB_cast_function fcast = GB_cast_factory (op->xtype->code, ctype->code) ;
    #endif

    //==========================================================================
    // phase1: sort all short vectors
    //==========================================================================

    // slice the C matrix into tasks for phase 1

    int ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;
    ntasks = GB_IMIN (ntasks, cnvec) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    GB_WERK_PUSH (SortTasks, 3*ntasks + 2, int64_t) ;
    if (SortTasks == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    int64_t *restrict C_max   = SortTasks ;                  // size ntasks
    int64_t *restrict C_skip  = SortTasks + ntasks ;         // size ntasks+1
    int64_t *restrict C_slice = SortTasks + 2*ntasks + 1;    // size ntasks+1

    GB_p_slice (C_slice, Cp, C->p_is_32, cnvec, ntasks, false) ;

    // sort all short vectors in parallel, one thread per vector
    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        const int64_t kfirst = C_slice [tid] ;
        const int64_t klast  = C_slice [tid+1] ;
        int64_t task_max_length = 0 ;
        int64_t n_skipped = 0 ;
        for (int64_t k = kfirst ; k < klast ; k++)
        {
            // sort the vector C(:,k), unless it is too long
            const int64_t pC_start = GB_IGET (Cp, k) ;
            const int64_t pC_end   = GB_IGET (Cp, k+1) ;
            const int64_t cknz = pC_end - pC_start ;
            if (cknz <= GB_SORT_BASECASE || nthreads == 1)
            { 
                uint64_t seed = k ;
                GB_SORT (quicksort) (GB_ADDR (Cx, pC_start), Ci + pC_start,
                    cknz, &seed
                    #if GB_SORT_UDT
                    , csize, xsize, flt, fcast
                    #endif
                    ) ;
            }
            else
            { 
                n_skipped++ ;
            }
            task_max_length = GB_IMAX (task_max_length, cknz) ;
        }
        C_max  [tid] = task_max_length ;
        C_skip [tid] = n_skipped ;
    }

    // find max vector length and return if all vectors are now sorted
    int64_t max_length = 0 ;
    for (tid = 0 ; tid < ntasks ; tid++)
    { 
        max_length = GB_IMAX (max_length, C_max [tid]) ;
    }

    if (max_length <= GB_SORT_BASECASE || nthreads == 1)
    { 
        // all vectors are sorted
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //==========================================================================
    // phase2: sort all long vectors in parallel
    //==========================================================================

    //--------------------------------------------------------------------------
    // construct a list of vectors that must still be sorted
    //--------------------------------------------------------------------------

    GB_cumsum1_64 ((uint64_t *) C_skip, ntasks) ;
    int64_t total_skipped = C_skip [ntasks] ;

    C_skipped = GB_MALLOC_MEMORY (total_skipped, sizeof (int64_t),
        &C_skipped_size) ;
    if (C_skipped == NULL)
    { 
        // out of memory
        GB_FREE_WORKSPACE ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        const int64_t kfirst = C_slice [tid] ;
        const int64_t klast  = C_slice [tid+1] ;
        int64_t n_skipped = C_skip [tid] ;
        for (int64_t k = kfirst ; k < klast ; k++)
        {
            const int64_t pC_start = GB_IGET (Cp, k) ;
            const int64_t pC_end   = GB_IGET (Cp, k+1) ;
            const int64_t cknz = pC_end - pC_start ;
            if (cknz > GB_SORT_BASECASE)
            { 
                // C(:,k) was not sorted
                C_skipped [n_skipped++] = k ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // determine # of tasks for each vector in phase 2
    //--------------------------------------------------------------------------

    // determine the number of levels to create, which must always be an
    // even number.  The # of levels is chosen to ensure that the # of leaves
    // of the task tree is between 4*nthreads and 16*nthreads.

    //  2 to 4 threads:     4 levels, 16 quicksort leaves
    //  5 to 16 threads:    6 levels, 64 quicksort leaves
    // 17 to 64 threads:    8 levels, 256 quicksort leaves
    // 65 to 256 threads:   10 levels, 1024 quicksort leaves
    // 256 to 1024 threads: 12 levels, 4096 quicksort leaves
    // ...

    int kk = (int) (2 + 2 * ceil (log2 ((double) nthreads) / 2)) ;
    int ntasks2 = 1 << kk ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    W   = GB_MALLOC_MEMORY (max_length + 6*ntasks2 + 1, sizeof (int64_t),
        &W_size) ;
    W_0 = (GB_C_TYPE *) GB_MALLOC_MEMORY (max_length, GB_SIZE,
        &W_0_size) ;
    W_1 = (GB_Ci_TYPE *) GB_MALLOC_MEMORY (max_length, sizeof (GB_Ci_TYPE),
        &W_1_size) ;

    if (W == NULL || W_0 == NULL || W_1 == NULL)
    { 
        // out of memory
        GB_FREE_WORKSPACE ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // sort each long vector using all available threads
    //--------------------------------------------------------------------------

    for (int64_t t = 0 ; t < total_skipped ; t++)
    { 
        const int64_t k = C_skipped [t] ;
        const int64_t pC_start = GB_IGET (Cp, k) ;
        const int64_t pC_end   = GB_IGET (Cp, k+1) ;
        const int64_t cknz = pC_end - pC_start ;
        ASSERT (cknz > GB_SORT_BASECASE) ;
        GB_SORT (vector) (GB_ADDR (Cx, pC_start), Ci + pC_start,
            W_0, W_1, W, cknz, kk, ntasks2, nthreads
            #if GB_SORT_UDT
            , csize, xsize, flt, fcast
            #endif
            ) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    C->jumbled = true ;
    return (GrB_SUCCESS) ;
}

#undef GB_SORT
#undef GB_C_TYPE

