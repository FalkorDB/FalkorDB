//------------------------------------------------------------------------------
// GB_task_cumsum: cumulative sum of Cp and fine tasks in TaskList
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Cp is never NULL.  C is created as sparse or hypersparse.

#include "GB.h"

void GB_task_cumsum
(
    int64_t *Cp,                        // size Cnvec+1
    const int64_t Cnvec,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp != NULL) ;
    ASSERT (Cnvec >= 0) ;
    ASSERT (Cnvec_nonempty != NULL) ;
    ASSERT (TaskList != NULL) ;
    ASSERT (ntasks >= 0) ;
    ASSERT (nthreads >= 1) ;

    //--------------------------------------------------------------------------
    // local cumulative sum of the fine tasks
    //--------------------------------------------------------------------------

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t k = TaskList [taskid].kfirst ;
        if (TaskList [taskid].klast < 0)
        { 
            // Compute the sum of all fine tasks for vector k, in Cp [k].  Also
            // compute the cumulative sum of TaskList [taskid].pC, for the
            // tasks that work on vector k.  The first fine task for vector k
            // starts with TaskList [taskid].pC = 0, which is an offset from
            // the final Cp [k].  A subsequent fine task t for a vector k
            // starts on offset of TaskList [t].pC from the start of C(:,k).
            // Cp [k] has not been cumsum'd across all of Cp.  It is just the
            // count of the entries in C(:,k).  The final Cp [k] is added to
            // each fine task below, after the GB_cumsum of Cp.
            int64_t pC = Cp [k] ;
            Cp [k] += TaskList [taskid].pC ;
            TaskList [taskid].pC = pC ;
        }
    }

    //--------------------------------------------------------------------------
    // replace Cp with its cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, Cnvec, Cnvec_nonempty, nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // shift the cumulative sum of the fine tasks
    //--------------------------------------------------------------------------

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t k = TaskList [taskid].kfirst ;
        if (TaskList [taskid].klast < 0)
        { 
            // TaskList [taskid].pC is currently an offset for this task into
            // C(:,k).  The first fine task for vector k has an offset of zero,
            // the 2nd fine task has an offset equal to the # of entries
            // computed by the first task, and so on.  Cp [k] needs to be added
            // to all offsets to get the final starting position for each fine
            // task in C.
            TaskList [taskid].pC += Cp [k] ;
        }
        else
        { 
            // The last fine task to operate on vector k needs know its own
            // pC_end, which is Cp [k+1].  Suppose that task is taskid-1.  If
            // this taskid is the first fine task for vector k, then TaskList
            // [taskid].pC is set to Cp [k] above.  If all coarse tasks are
            // also given TaskList [taskid].pC = Cp [k], then taskid-1 will
            // always know its pC_end, which is TaskList [taskid].pC,
            // regardless of whether taskid is a fine or coarse task.
            TaskList [taskid].pC = Cp [k] ;
        }
    }

    // The last task is ntasks-1.  It may be a fine task, in which case it
    // computes the entries in C from TaskList [ntasks-1].pC to
    // TaskList [ntasks].pC-1, inclusive.
    TaskList [ntasks].pC = Cp [Cnvec] ;

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // nthreads, ntasks, Cnvec) ;
    for (int t = 0 ; t < ntasks ; t++)
    {
        int64_t k = TaskList [t].kfirst ;
        int64_t klast = TaskList [t].klast ;
        if (klast < 0)
        {
            // this is a fine task for vector k
            int64_t pA     = TaskList [t].pA ;
            int64_t pA_end = TaskList [t].pA_end ;
            int64_t pB     = TaskList [t].pB ;
            int64_t pB_end = TaskList [t].pB_end ;
            int64_t pC     = TaskList [t].pC ;
            int64_t pC_end = TaskList [t+1].pC ;
            int64_t pM     = TaskList [t].pM ;
            int64_t pM_end = TaskList [t].pM_end ;
            ASSERT (k >= 0 && k < Cnvec) ;
            // pA:(pA_end-1) must reside inside A(:,j), and pB:(pB_end-1) must
            // reside inside B(:,j), but these cannot be checked here since A
            // and B are not available.  These basic checks can be done:
            ASSERT (pA == -1 || (0 <= pA && pA <= pA_end)) ;
            ASSERT (pB == -1 || (0 <= pB && pB <= pB_end)) ;
            ASSERT (pM == -1 || (0 <= pM && pM <= pM_end)) ;
            // pC and pC_end can be checked exactly.  This task t computes
            // entries pC:(pC_end-1) of C, inclusive.
            ASSERT (Cp [k] <= pC && pC <= pC_end && pC_end <= Cp [k+1]) ;
        }
        else
        {
            // this is a coarse task for vectors k:klast, inclusive
            ASSERT (k >= 0 && k < Cnvec) ;
            ASSERT (klast >= 0 && klast <= Cnvec) ;
            ASSERT (k <= klast) ;
        }
    }
    #endif
}

