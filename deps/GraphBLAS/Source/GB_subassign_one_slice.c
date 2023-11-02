//------------------------------------------------------------------------------
// GB_subassign_one_slice: slice the entries and vectors for subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs a set of tasks to compute C for a subassign method, based on
// slicing a single input matrix (M or A).  Fine tasks must also find their
// location in their vector C(:,jC).  Currently this method is only used to
// slice M, not A.

// This method is used by GB_subassign_05, 06n, and 07.  Each of those methods
// apply this function to M, but they use TaskList[...].pA and pA_end to
// partition the matrix.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x       for M
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x      for M
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A       for M

// C: not bitmap

#include "GB_subassign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE           \
{                                   \
    GB_WERK_POP (Coarse, int64_t) ; \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_FREE_WORK (&TaskList, TaskList_size) ;   \
}

//------------------------------------------------------------------------------
// GB_subassign_one_slice
//------------------------------------------------------------------------------

GrB_Info GB_subassign_one_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    // input:
    const GrB_Matrix C,             // output matrix C
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,             // matrix to slice
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;
    ASSERT_MATRIX_OK (C, "C for 1_slice", GB0) ;
    ASSERT_MATRIX_OK (M, "M for 1_slice", GB0) ;

    ASSERT (!GB_IS_BITMAP (C)) ;

    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_JUMBLED (M)) ;

    (*p_TaskList  ) = NULL ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // get M and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Mp = M->p ;
    const int64_t *restrict Mh = M->h ;
//  const int8_t  *restrict Mb = M->b ;
    const int64_t *restrict Mi = M->i ;
    const int64_t mnz = GB_nnz_held (M) ;
    const int64_t mnvec = M->nvec ;
    const int64_t mvlen = M->vlen ;

    const int64_t *restrict Cp = C->p ;
    const int64_t *restrict Ch = C->h ;
    const int64_t *restrict Ci = C->i ;
    const bool C_is_hyper = (Ch != NULL) ;
    const int64_t nzombies = C->nzombies ;
    const int64_t Cnvec = C->nvec ;
    const int64_t Cvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // allocate the initial TaskList
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (Coarse, int64_t) ;     // size ntasks1+1
    int ntasks1 = 0 ;
    int nthreads = GB_nthreads (mnz, chunk, nthreads_max) ;
    GB_task_struct *restrict TaskList = NULL ; size_t TaskList_size = 0 ;
    int max_ntasks = 0 ;
    int ntasks = 0 ;
    int ntasks0 = (nthreads == 1) ? 1 : (32 * nthreads) ;
    GB_REALLOC_TASK_WORK (TaskList, ntasks0, max_ntasks) ;

    GB_GET_C_HYPER_HASH ;

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (mnvec == 0 || ntasks0 == 1)
    { 
        // construct a single coarse task that does all the work
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = mnvec-1 ;
        (*p_TaskList  ) = TaskList ;
        (*p_TaskList_size) = TaskList_size ;
        (*p_ntasks    ) = (mnvec == 0) ? 0 : 1 ;
        (*p_nthreads  ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // determine # of threads and tasks for the subassign operation
    //--------------------------------------------------------------------------

    double target_task_size = ((double) mnz) / (double) (ntasks0) ;
    target_task_size = GB_IMAX (target_task_size, chunk) ;
    ntasks1 = ((double) mnz) / target_task_size ;
    ntasks1 = GB_IMAX (ntasks1, 1) ;

    //--------------------------------------------------------------------------
    // slice the work into coarse tasks
    //--------------------------------------------------------------------------

    // M may be hypersparse, sparse, bitmap, or full
    GB_WERK_PUSH (Coarse, ntasks1 + 1, int64_t) ;
    if (Coarse == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_pslice (Coarse, Mp, mnvec, ntasks1, false) ;

    //--------------------------------------------------------------------------
    // construct all tasks, both coarse and fine
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < ntasks1 ; t++)
    {

        //----------------------------------------------------------------------
        // coarse task computes C (I, J(k:klast)) = M (I, k:klast)
        //----------------------------------------------------------------------

        int64_t k = Coarse [t] ;
        int64_t klast = Coarse [t+1] - 1 ;

        if (k >= mnvec)
        { 

            //------------------------------------------------------------------
            // all tasks have been constructed
            //------------------------------------------------------------------

            break ;

        }
        else if (k < klast)
        { 

            //------------------------------------------------------------------
            // coarse task has 2 or more vectors
            //------------------------------------------------------------------

            // This is a non-empty coarse-grain task that does two or more
            // entire vectors of M, vectors k:klast, inclusive.
            GB_REALLOC_TASK_WORK (TaskList, ntasks + 1, max_ntasks) ;
            TaskList [ntasks].kfirst = k ;
            TaskList [ntasks].klast  = klast ;
            ntasks++ ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task has 0 or 1 vectors
            //------------------------------------------------------------------

            // As a coarse-grain task, this task is empty or does a single
            // vector, k.  Vector k must be removed from the work done by this
            // and any other coarse-grain task, and split into one or more
            // fine-grain tasks.

            for (int tt = t ; tt < ntasks1 ; tt++)
            {
                // remove k from the initial slice tt
                if (Coarse [tt] == k)
                { 
                    // remove k from task tt
                    Coarse [tt] = k+1 ;
                }
                else
                { 
                    // break, k not in task tt
                    break ;
                }
            }

            //------------------------------------------------------------------
            // get the vector of C
            //------------------------------------------------------------------

            ASSERT (k >= 0 && k < mnvec) ;
            int64_t j = GBH (Mh, k) ;
            ASSERT (j >= 0 && j < nJ) ;
            GB_LOOKUP_VECTOR_jC (false, 0) ;

            bool jC_dense = (pC_end - pC_start == Cvlen) ;

            //------------------------------------------------------------------
            // determine the # of fine-grain tasks to create for vector k
            //------------------------------------------------------------------

            int64_t mknz = (Mp == NULL) ? mvlen : (Mp [k+1] - Mp [k]) ;
            int nfine = ((double) mknz) / target_task_size ;
            nfine = GB_IMAX (nfine, 1) ;

            // make the TaskList bigger, if needed
            GB_REALLOC_TASK_WORK (TaskList, ntasks + nfine, max_ntasks) ;

            //------------------------------------------------------------------
            // create the fine-grain tasks
            //------------------------------------------------------------------

            if (nfine == 1)
            { 

                //--------------------------------------------------------------
                // this is a single coarse task for all of vector k
                //--------------------------------------------------------------

                TaskList [ntasks].kfirst = k ;
                TaskList [ntasks].klast  = k ;
                ntasks++ ;

            }
            else
            {

                //--------------------------------------------------------------
                // slice vector M(:,k) into nfine fine tasks
                //--------------------------------------------------------------

                ASSERT (ntasks < max_ntasks) ;

                for (int tfine = 0 ; tfine < nfine ; tfine++)
                {

                    // this fine task operates on vector M(:,k)
                    TaskList [ntasks].kfirst = k ;
                    TaskList [ntasks].klast  = -1 ;

                    // slice M(:,k) for this task
                    int64_t p1, p2 ;
                    GB_PARTITION (p1, p2, mknz, tfine, nfine) ;
                    int64_t pM_start = GBP (Mp, k, mvlen) ;
                    int64_t pM     = pM_start + p1 ;
                    int64_t pM_end = pM_start + p2 ;
                    TaskList [ntasks].pA     = pM ;
                    TaskList [ntasks].pA_end = pM_end ;

                    if (jC_dense)
                    { 
                        // do not slice C(:,jC) if it is dense
                        TaskList [ntasks].pC     = pC_start ;
                        TaskList [ntasks].pC_end = pC_end ;
                    }
                    else
                    { 
                        // find where this task starts and ends in C(:,jC)
                        int64_t iM_start = GBI (Mi, pM, mvlen) ;
                        int64_t iC1 = GB_ijlist (I, iM_start, Ikind, Icolon) ;
                        int64_t iM_end = GBI (Mi, pM_end-1, mvlen) ;
                        int64_t iC2 = GB_ijlist (I, iM_end, Ikind, Icolon) ;

                        // If I is an explicit list, it must be already sorted
                        // in ascending order, and thus iC1 <= iC2.  If I is
                        // GB_ALL or GB_STRIDE with inc >= 0, then iC1 < iC2.
                        // But if inc < 0, then iC1 > iC2.  iC_start and iC_end
                        // are used for a binary search bracket, so iC_start <=
                        // iC_end must hold.
                        int64_t iC_start = GB_IMIN (iC1, iC2) ;
                        int64_t iC_end   = GB_IMAX (iC1, iC2) ;

                        // this task works on Ci,Cx [pC:pC_end-1]
                        int64_t pleft = pC_start ;
                        int64_t pright = pC_end - 1 ;
                        bool found, is_zombie ;
                        GB_SPLIT_BINARY_SEARCH_ZOMBIE (iC_start, Ci,
                            pleft, pright, found, nzombies, is_zombie) ;
                        TaskList [ntasks].pC = pleft ;

                        pleft = pC_start ;
                        pright = pC_end - 1 ;
                        GB_SPLIT_BINARY_SEARCH_ZOMBIE (iC_end, Ci,
                            pleft, pright, found, nzombies, is_zombie) ;
                        TaskList [ntasks].pC_end = (found) ? (pleft+1) : pleft ;
                    }

                    ASSERT (TaskList [ntasks].pA <= TaskList [ntasks].pA_end) ;
                    ASSERT (TaskList [ntasks].pC <= TaskList [ntasks].pC_end) ;
                    ntasks++ ;
                }
            }
        }
    }

    ASSERT (ntasks <= max_ntasks) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    (*p_TaskList  ) = TaskList ;
    (*p_TaskList_size) = TaskList_size ;
    (*p_ntasks    ) = ntasks ;
    (*p_nthreads  ) = nthreads ;
    return (GrB_SUCCESS) ;
}

