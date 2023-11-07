//------------------------------------------------------------------------------
// GB_transpose_method: select method for GB_transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_transpose.h"

// GB_transpose can use choose between a merge-sort-based method that takes
// O(anz*log(anz)) time, or a bucket-sort method that takes O(anz+m+n) time.
// The bucket sort has 3 methods: sequential, atomic, and non-atomic.

bool GB_transpose_method        // if true: use GB_builder, false: use bucket
(
    const GrB_Matrix A,         // matrix to transpose
    int *nworkspaces_bucket,    // # of slices of A for the bucket method
    int *nthreads_bucket        // # of threads to use for the bucket method
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // if available, A->nvec_nonempty is used to select the method
    int64_t anvec = (A->nvec_nonempty < 0) ? A->nvec : A->nvec_nonempty ;
    int64_t anz = GB_nnz (A) ;
    int64_t avlen = A->vlen ;
//  int64_t avdim = A->vdim ;
    int anzlog = (anz   == 0) ? 1 : (int) GB_CEIL_LOG2 (anz) ;
    int mlog   = (avlen == 0) ? 1 : (int) GB_CEIL_LOG2 (avlen) ;
    double bucket_factor ;

    // determine # of threads for bucket method
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz + avlen, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // select between the atomic and non-atomic bucket method
    //--------------------------------------------------------------------------

    bool atomics ;
    if (nthreads <= 2)
    { 
        // sequential bucket method: no atomics needed
        // 2 threads: always use non-atomic method
        atomics = false ;
    }
    else if ((double) nthreads * (double) avlen > (double) anz)
    { 
        // non-atomic workspace is too high; use atomic method
        atomics = true ;
    }
    else
    {
        // select between atomic and non-atomic methods.  This rule is based on
        // performance on a 4-core system with 4 threads with gcc 7.5.  The icc
        // compiler has much slower atomics than gcc and so atol should likely
        // be smaller when using icc.

        int atol ;
        if (anzlog < 14)
        { 
            atol = -4 ;     // fewer than 16K entries in A
        }
        else
        { 
            switch (anzlog)
            {
                case 14: atol = -4 ; break ;        // 16K entries in A
                case 15: atol = -3 ; break ;        // 32K
                case 16: atol = -2 ; break ;        // 64K
                case 17: atol = -1 ; break ;        // 128K
                case 18: atol =  0 ; break ;        // 256K
                case 19: atol =  1 ; break ;        // 512K
                case 20: atol =  2 ; break ;        // 1M
                case 21: atol =  3 ; break ;        // 2M
                case 22: atol =  4 ; break ;        // 4M
                case 23: atol =  5 ; break ;        // 8M
                case 24: atol =  6 ; break ;        // 16M
                case 25: atol =  8 ; break ;        // 32M
                case 26: atol =  9 ; break ;        // 64M
                case 27: atol =  9 ; break ;        // 128M
                case 28: atol = 10 ; break ;        // 256M
                default: atol = 10 ; break ;        // > 256M
            }
        }

        if (anzlog - mlog <= atol)
        { 
            // use atomic method
            // anzlog - mlog is the log2 of the average row degree, rounded.
            // If the average row degree is <= 2^atol, use the atomic method.
            // That is, the atomic method works better for sparser matrices,
            // and the non-atomic works better or denser matrices.  However,
            // the threshold changes as the problem gets larger, in terms of #
            // of entries in A, when the atomic method becomes more attractive
            // relative to the non-atomic method.  The atomic has the
            // advantange of needing much less workspace, which becomes more
            // important for larger problems.
            atomics = true ;
        }
        else
        { 
            // use non-atomic method
            atomics = false ;
        }
    }

    (*nworkspaces_bucket) = (atomics) ? 1 : nthreads ;
    (*nthreads_bucket) = nthreads ;

    //--------------------------------------------------------------------------
    // select between GB_builder method and bucket method
    //--------------------------------------------------------------------------

    // As the problem gets larger, the GB_builder method gets faster relative
    // to the bucket method, in terms of the "constants" in the O(a log a) work
    // for GB_builder, or O (a+m+n) for the bucket method.  Clearly, O (a log
    // a) and O (a+m+n) do not fully model the performance of these two
    // methods.  Perhaps this is because of cache effects.  The bucket method
    // has more irregular memory accesses.  The GB_builder method uses
    // mergesort, which has good memory locality.

    if (anzlog < 14)
    { 
        bucket_factor = 0.5 ;       // fewer than 2^14 = 16K entries
    }
    else
    { 
        switch (anzlog)
        {
            case 14: bucket_factor = 0.6 ; break ;      // 16K entries in A
            case 15: bucket_factor = 0.7 ; break ;      // 32K
            case 16: bucket_factor = 1.0 ; break ;      // 64K
            case 17: bucket_factor = 1.7 ; break ;      // 128K
            case 18: bucket_factor = 3.0 ; break ;      // 256K
            case 19: bucket_factor = 4.0 ; break ;      // 512K
            case 20: bucket_factor = 6.0 ; break ;      // 1M
            case 21: bucket_factor = 7.0 ; break ;      // 2M
            case 22: bucket_factor = 8.0 ; break ;      // 4M
            case 23: bucket_factor = 5.0 ; break ;      // 8M
            case 24: bucket_factor = 5.0 ; break ;      // 16M
            case 25: bucket_factor = 5.0 ; break ;      // 32M
            case 26: bucket_factor = 5.0 ; break ;      // 64M
            case 27: bucket_factor = 5.0 ; break ;      // 128M
            case 28: bucket_factor = 5.0 ; break ;      // 256M
            default: bucket_factor = 5.0 ; break ;      // > 256M
        }
    }

    double bucket_work  = (double) (anz + avlen + anvec) * bucket_factor ;
    double builder_work = (log2 ((double) anz + 1) * (anz)) ;

    //--------------------------------------------------------------------------
    // select the method with the least amount of work
    //--------------------------------------------------------------------------

    bool use_builder = (builder_work < bucket_work) ;
    return (use_builder) ;
}

