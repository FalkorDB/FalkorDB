//------------------------------------------------------------------------------
// GB_p_slice_template: partition Work for a set of tasks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a templatized method where _TYPE is 32 or 64 for uint32_t and
// uint64_t, or float.

//------------------------------------------------------------------------------
// GB_p_slice_worker_TYPE: partition Work for a set of tasks
//------------------------------------------------------------------------------

static void GB_p_slice_worker_TYPE
(
    int64_t *restrict Slice,            // size ntasks+1
    const GB_Work_TYPE *restrict Work,  // array size n+1
    int tlo,                            // assign to Slice [(tlo+1):(thi-1)]
    int thi                     
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    ASSERT (Work != NULL) ;
    ASSERT (Slice != NULL) ;
    ASSERT (0 <= tlo && tlo < thi - 1) ;
    for (int t = tlo+1 ; t <= thi-1 ; t++)
    {
        ASSERT (Slice [t] == -1) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // assign work to Slice [(tlo+1):(thi-1)]
    //--------------------------------------------------------------------------

    // klo = Slice [tlo] and khi = Slice [thi] are defined on input, where
    // tlo < thi - 1.  This determines the task boundaries for tasks
    // tlo+1 to thi-1, which defines Slice [(tlo+1):(thi-1)].

    int64_t klo = Slice [tlo] ;
    int64_t khi = Slice [thi] ;         ASSERT (0 <= klo && klo <= khi) ;
    GB_Work_TYPE p1 = Work [klo] ;
    GB_Work_TYPE p2 = Work [khi] ;      ASSERT (p1 <= p2) ;

    if (p1 == p2 || klo == khi)
    {

        //----------------------------------------------------------------------
        // no work is left so simply fill in with empty tasks
        //----------------------------------------------------------------------

        int64_t k = klo ;
        for (int64_t t = tlo+1 ; t <= thi-1 ; t++)
        { 
            Slice [t] = k ;
        }

    }
    else // p1 < p2 && klo < khi
    {

        //----------------------------------------------------------------------
        // find task t that evenly partitions the work p1:p2 to tasks tlo:thi
        //----------------------------------------------------------------------

        ASSERT (p1 < p2) ;
        int64_t k = (klo + khi) / 2 ;       ASSERT (klo <= k && k <= khi) ;
        GB_Work_TYPE p = Work [k] ;         ASSERT (p1 <= p && p <= p2) ;
        double ntasks = thi - tlo ;
        double ratio = (((double) (p - p1)) / ((double) (p2 - p1))) ;
        int t = tlo + (int) floor (ratio * ntasks) ;
        t = GB_IMAX (t, tlo+1) ;
        t = GB_IMIN (t, thi-1) ;            ASSERT (tlo < t && t < thi) ;

        //----------------------------------------------------------------------
        // assign work to task t
        //----------------------------------------------------------------------

        ASSERT (Slice [t] == -1) ;
        Slice [t] = k ;

        //----------------------------------------------------------------------
        // recursively partition for tasks (tlo+1):(t-1) and (t+1):(thi-1)
        //----------------------------------------------------------------------

        if (tlo < t-1)
        { 
            GB_p_slice_worker_TYPE (Slice, Work, tlo, t) ;
        }
        if (t < thi-1)
        { 
            GB_p_slice_worker_TYPE (Slice, Work, t, thi) ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_p_slice_TYPE: partition Work for a set of tasks
//------------------------------------------------------------------------------

void GB_p_slice_TYPE     // slice Work, uint32_t, uint64_t, or float
(
    // output:
    int64_t *restrict Slice,    // size ntasks+1
    // input:
    const GB_Work_TYPE *Work,   // array size n+1
    const int64_t n,
    const int ntasks            // # of tasks
    #ifdef GB_ENABLE_PERFECT_BALANCE
    , const bool perfectly_balanced
    #endif
)
{

    ASSERT (Work != NULL) ;

    #ifdef GB_DEBUG
    for (int taskid = 0 ; taskid <= ntasks ; taskid++)
    {
        Slice [taskid] = -1 ;
    }
    #endif

    if (n == 0 || ntasks <= 1 || Work [n] == 0)
    { 
        // matrix is empty, or a single thread is used
        memset ((void *) Slice, 0, ntasks * sizeof (int64_t)) ;
        Slice [ntasks] = n ;
    }
    else
    {
        // slice Work by # of entries
        Slice [0] = 0 ;
        Slice [ntasks] = n ;
        #ifdef GB_ENABLE_PERFECT_BALANCE
        if (perfectly_balanced)
        {
            // this method is costly, and should only be used if the
            // work is to be perfectly balanced (in particular, when there
            // is just one task per thread, with static scheduling).  The Work
            // array must be uint32_t or uint64_t.
            const double work = (double) (Work [n]) ;
            int64_t k = 0 ;
            for (int taskid = 1 ; taskid < ntasks ; taskid++)
            { 
                // binary search to find k so that Work [k] == (taskid*work) /
                // ntasks.  The exact value will not typically not be found;
                // just pick what the binary search comes up with.
                int64_t wtask = (int64_t) GB_PART (taskid, work, ntasks) ;
                int64_t pright = n ;
                GB_trim_binary_search_TYPE (wtask, Work, &k, &pright) ;
                Slice [taskid] = k ;
            }
        }
        else
        #endif
        { 
            // this is much faster, and results in good load balancing if
            // there is more than one task per thread, and dynamic
            // scheduling is used.
            GB_p_slice_worker_TYPE (Slice, Work, 0, ntasks) ;
        }
    }

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    ASSERT (Slice [0] == 0) ;
    ASSERT (Slice [ntasks] == n) ;
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        ASSERT (Slice [taskid] <= Slice [taskid+1]) ;
    }
    #endif
}

#undef GB_Work_TYPE
#undef GB_p_slice_TYPE
#undef GB_p_slice_worker_TYPE
#undef GB_trim_binary_search_TYPE

