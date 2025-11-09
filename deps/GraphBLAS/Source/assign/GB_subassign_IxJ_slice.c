//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice: slice IxJ for subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Slice IxJ for a scalar assignment method and for bitmap assignments.

// Construct a set of tasks to compute C(I,J)<...> = x or += x, for a subassign
// method that performs scalar assignment, based on slicing the Cartesian
// product IxJ.  If enough tasks can be constructed by just slicing J, then all
// tasks are coarse.  Each coarse tasks computes all of C(I,J(kfirst:klast-1)),
// for its range of indices kfirst:klast-1, inclusive.

// Otherwise, if not enough coarse tasks can be constructed, then all tasks are
// fine.  Each fine task computes a slice of C(I(iA_start:iA_end-1), jC) for a
// single index jC = J(kfirst).

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   +   -   S       15:  C(I,J)<!M> += x, with S
        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   +   -   S       19:  C(I,J)<!M,repl> += x, with S

// There are 10 methods that perform scalar assignment: the 6 listed above, and
// Methods 05, 07, 09, and 11.  The latter 4 methods do not need to iterate
// over the entire IxJ space, because of the mask M:

        //  M   -   -   -   -   -       05:  C(I,J)<M> = x
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x
        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   +   -   S       11:  C(I,J)<M,repl> += x, with S

// As a result, they do not use GB_subassign_IxJ_slice to define their tasks.
// Instead, Methods 05 and 07 slice the matrix M, and Methods 09 and 11 slice
// the matrix addition M+S.

#include "assign/GB_subassign_methods.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_MEMORY (&TaskList, TaskList_size) ;   \
}

//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice
//------------------------------------------------------------------------------

#if 0
GrB_Info GB_subassign_IxJ_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    // input:
    const int64_t nI,
    const int64_t nJ,
    GB_Werk Werk
)
#endif

GB_CALLBACK_SUBASSIGN_IXJ_SLICE_PROTO (GB_subassign_IxJ_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_TaskList_size != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;

    (*p_TaskList  ) = NULL ;
    (*p_TaskList_size) = 0 ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;
    int ntasks, max_ntasks = 0, nthreads ;
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // allocate the initial TaskList
    //--------------------------------------------------------------------------

    double work = ((double) nI) * ((double) nJ) ;
    nthreads = GB_nthreads (work, chunk, nthreads_max) ;
    int ntasks0 = (nthreads == 1) ? 1 : (32 * nthreads) ;
    GB_REALLOC_TASK_WORK (TaskList, ntasks0, max_ntasks) ;

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (nJ == 0 || ntasks0 == 1)
    { 
        // construct a single coarse task that does all the work
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = nJ-1 ;
        (*p_TaskList  ) = TaskList ;
        (*p_TaskList_size) = TaskList_size ;
        (*p_ntasks    ) = (nJ == 0) ? 0 : 1 ;
        (*p_nthreads  ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // construct the tasks: all fine or all coarse
    //--------------------------------------------------------------------------

    // The desired number of tasks is ntasks0.  If this is less than or equal
    // to |J|, then all tasks can be coarse, and each coarse task handles one
    // or more indices in J.  Otherise, multiple fine tasks are constructed for
    // each index in J.

    if (ntasks0 <= nJ)
    {

        //----------------------------------------------------------------------
        // all coarse tasks: slice just J
        //----------------------------------------------------------------------

        ntasks = ntasks0 ;
        for (int taskid = 0 ; taskid < ntasks ; taskid++)
        { 
            // the coarse task computes C (I, J (j:jlast-1))
            int64_t j, jlast ;
            GB_PARTITION (j, jlast, nJ, taskid, ntasks) ;
            ASSERT (j <= jlast) ;
            ASSERT (jlast <= nJ) ;
            TaskList [taskid].kfirst = j ;
            TaskList [taskid].klast  = jlast - 1 ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // all fine tasks: slice both I and J
        //----------------------------------------------------------------------

        // create at least 2 fine tasks per index in J
        int nI_fine_tasks = ntasks0 / nJ ;
        nI_fine_tasks = GB_IMAX (nI_fine_tasks, 2) ;
        ntasks = 0 ;

        GB_REALLOC_TASK_WORK (TaskList, nJ * nI_fine_tasks, max_ntasks) ;

        //----------------------------------------------------------------------
        // construct fine tasks for index j
        //----------------------------------------------------------------------

        for (int64_t j = 0 ; j < nJ ; j++)
        {
            // create nI_fine_tasks for each index in J
            for (int t = 0 ; t < nI_fine_tasks ; t++)
            { 
                // this fine task computes C (I (iA_start:iA_end-1), jC)
                int64_t iA_start, iA_end ;
                GB_PARTITION (iA_start, iA_end, nI, t, nI_fine_tasks) ;
                TaskList [ntasks].kfirst = j ;
                TaskList [ntasks].klast  = -1 ;
                TaskList [ntasks].pA     = iA_start ;
                TaskList [ntasks].pA_end = iA_end ;
                ntasks++ ;
            }
        }
    }

    ASSERT (ntasks <= max_ntasks) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_TaskList  ) = TaskList ;
    (*p_TaskList_size) = TaskList_size ;
    (*p_ntasks    ) = ntasks ;
    (*p_nthreads  ) = nthreads ;
    return (GrB_SUCCESS) ;
}

