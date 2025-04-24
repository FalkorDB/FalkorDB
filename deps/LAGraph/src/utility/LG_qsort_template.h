//------------------------------------------------------------------------------
// LG_qsort_template: quicksort of a K-by-n array
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// This file is #include'd in LG_qsort*.c to create specific versions for
// different kinds of sort keys and auxiliary arrays.  Requires an inline or
// macro definition of the LG_lt function.  The LG_lt function has the form
// LG_lt (A,i,B,j) and returns true if A[i] < B[j].

// All of these functions are static; there will be versions of them in each
// variant of LG_qsort*, and given unique names via #define's in the
// #include'ing file.

//------------------------------------------------------------------------------
// LG_partition: use a pivot to partition an array
//------------------------------------------------------------------------------

// C.A.R Hoare partition method, partitions an array in-place via a pivot.
// k = partition (A, n) partitions A [0:n-1] such that all entries in
// A [0:k] are <= all entries in A [k+1:n-1].

static inline int64_t LG_partition
(
    LG_args (A),            // array(s) to partition
    const int64_t n,        // size of the array(s) to partition
    uint64_t *seed,         // random number seed, modified on output
    LG_void *tx
)
{

    // select a pivot at random
    int64_t pivot = ((n < LG_RANDOM15_MAX) ?
        LG_Random15 (seed) : LG_Random60 (seed)) % n ;

    // get the Pivot
    int64_t Pivot_0 [1] ; Pivot_0 [0] = A_0 [pivot] ;
    #if LG_K > 1
    int64_t Pivot_1 [1] ; Pivot_1 [0] = A_1 [pivot] ;
    #endif
    #if LG_K > 2
    int64_t Pivot_2 [1] ; Pivot_2 [0] = A_2 [pivot] ;
    #endif

    // At the top of the while loop, A [left+1...right-1] is considered, and
    // entries outside this range are in their proper place and not touched.
    // Since the input specification of this function is to partition A
    // [0..n-1], left must start at -1 and right must start at n.
    int64_t left = -1 ;
    int64_t right = n ;

    // keep partitioning until the left and right sides meet
    while (true)
    {
        // loop invariant:  A [0..left] < pivot and A [right..n-1] > pivot,
        // so the region to be considered is A [left+1 ... right-1].

        // increment left until finding an entry A [left] >= pivot
        do { left++ ; } while (LG_lt (A, left, Pivot, 0)) ;

        // decrement right until finding an entry A [right] <= pivot
        do { right-- ; } while (LG_lt (Pivot, 0, A, right)) ;

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
        LG_swap (A, left, right) ;

        // after the swap this condition holds:
        // A [0..left] < pivot and A [right..n-1] > pivot
    }
}

//------------------------------------------------------------------------------
// LG_quicksort: recursive single-threaded quicksort
//------------------------------------------------------------------------------

static void LG_quicksort    // sort A [0:n-1]
(
    LG_args (A),            // array(s) to sort
    const int64_t n,        // size of the array(s) to sort
    uint64_t *seed,         // random number seed
    LG_void *tx
)
{

    if (n < 20)
    {
        // in-place insertion sort on A [0:n-1], where n is small
        for (int64_t k = 1 ; k < n ; k++)
        {
            for (int64_t j = k ; j > 0 && LG_lt (A, j, A, j-1) ; j--)
            {
                // swap A [j-1] and A [j]
                LG_swap (A, j-1, j) ;
            }
        }
    }
    else
    {
        // partition A [0:n-1] into A [0:k-1] and A [k:n-1]
        int64_t k = LG_partition (LG_arg (A), n, seed, tx) ;

        // sort each partition
        LG_quicksort (LG_arg (A), k, seed, tx) ;             // sort A [0:k-1]
        LG_quicksort (LG_arg_offset (A, k), n-k, seed, tx) ; // sort A [k+1:n-1]
    }
}
