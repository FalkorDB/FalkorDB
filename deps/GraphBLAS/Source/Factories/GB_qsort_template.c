//------------------------------------------------------------------------------
// GB_qsort_template: quicksort of a K-by-n array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd in GB_qsort*.c to create specific versions for
// different kinds of sort keys and auxiliary arrays.  Requires an inline or
// macro definition of the GB_lt function.  The GB_lt function has the form
// GB_lt (A,i,B,j) and returns true if A[i] < B[j].

// All of these functions are static; there will be versions of them in each
// variant of GB_qsort*, and given unique names via #define's in the
// #include'ing file.

//------------------------------------------------------------------------------
// GB_partition: use a pivot to partition an array
//------------------------------------------------------------------------------

// C.A.R Hoare partition method, partitions an array in-place via a pivot.
// k = partition (A, n) partitions A [0:n-1] such that all entries in
// A [0:k] are <= all entries in A [k+1:n-1].

static inline int64_t GB_partition
(
    GB_args (A),            // array(s) to partition
    const int64_t n,        // size of the array(s) to partition
    uint64_t *seed          // random number seed, modified on output
)
{

    // select a pivot at random
    int64_t pivot = ((n < GB_RAND_MAX) ? GB_rand15 (seed) : GB_rand (seed)) % n;

    // get the Pivot
    int64_t Pivot_0 [1] ; Pivot_0 [0] = A_0 [pivot] ;
    #if GB_K > 1
    int64_t Pivot_1 [1] ; Pivot_1 [0] = A_1 [pivot] ;
    #endif
    #if GB_K > 2
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
        do { left++ ; } while (GB_lt (A, left, Pivot, 0)) ;

        // decrement right until finding an entry A [right] <= pivot
        do { right-- ; } while (GB_lt (Pivot, 0, A, right)) ;

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
        GB_swap (A, left, right) ;

        // after the swap this condition holds:
        // A [0..left] < pivot and A [right..n-1] > pivot
    }
}

//------------------------------------------------------------------------------
// GB_quicksort: recursive single-threaded quicksort
//------------------------------------------------------------------------------

static void GB_quicksort    // sort A [0:n-1]
(
    GB_args (A),            // array(s) to sort
    const int64_t n,        // size of the array(s) to sort
    uint64_t *seed          // random number seed
)
{

    if (n < 20)
    {
        // in-place insertion sort on A [0:n-1], where n is small
        for (int64_t k = 1 ; k < n ; k++)
        {
            for (int64_t j = k ; j > 0 && GB_lt (A, j, A, j-1) ; j--)
            { 
                // swap A [j-1] and A [j]
                GB_swap (A, j-1, j) ;
            }
        }
    }
    else
    { 
        // partition A [0:n-1] into A [0:k-1] and A [k:n-1]
        int64_t k = GB_partition (GB_arg (A), n, seed) ;

        // sort each partition
        GB_quicksort (GB_arg (A), k, seed) ;                // sort A [0:k-1]
        GB_quicksort (GB_arg_offset (A, k), n-k, seed) ;    // sort A [k+1:n-1]
    }
}

