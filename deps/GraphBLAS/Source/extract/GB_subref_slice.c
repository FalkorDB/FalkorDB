//------------------------------------------------------------------------------
// GB_subref_slice: construct coarse/fine tasks for C = A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determine the tasks for computing C=A(I,J).  The matrix C has Cnvec vectors,
// and these are divided into coarse and fine tasks.  A coarse task will
// compute one or more whole vectors of C.  A fine task operates on a slice of
// a single vector of C.  The slice can be done by the # of entries in the
// corresponding vector of A, or by the list of indices I, depending on how the
// work is done for that method.

// The (kC)th vector will access A(imin:imax,kA) in Ai,Ax [pA:pA_end-1], where
// pA = Ap_start [kC] and pA_end = Ap_end [kC].

// The computation of each vector C(:,kC) = A(I,kA) is by done using one of 12
// different cases, depending on the vector, as determined by GB_subref_method.
// Not all vectors in C are computed using the same method.

// Note that J can have duplicates.  kC is unique (0:Cnvec-1) but the
// corresponding vector kA in A may repeat, if J has duplicates.  Duplicates in
// J are not exploited, since the coarse/fine tasks are constructed by slicing
// the list of vectors Ch of size Cnvec, not the vectors of A.

// Compare this function with GB_ewise_slice, which constructs coarse/fine
// tasks for the eWise operations (C=A+B, C=A.*B, and C<M>=Z).

// The matrices C and A are sparse or hypersparse, but the matrices themselves
// do not appear in this method.

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_WERK_POP (Coarse, int64_t) ;             \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_FREE_MEMORY (&Cwork, Cwork_size) ;       \
    GB_FREE_MEMORY (&TaskList, TaskList_size) ; \
    GB_Matrix_free (&R) ;                       \
}

#define GB_RETURN_RESULTS                   \
{                                           \
    (*p_TaskList     ) = TaskList ;         \
    (*p_TaskList_size) = TaskList_size ;    \
    (*p_ntasks       ) = ntasks ;           \
    (*p_nthreads     ) = nthreads ;         \
    (*p_post_sort    ) = post_sort ;        \
    (*R_handle       ) = R ;                \
    (*p_Cwork        ) = Cwork ;            \
    (*p_Cwork_size   ) = Cwork_size ;       \
}

#include "extract/GB_subref.h"

GrB_Info GB_subref_slice    // phase 1 of GB_subref
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,              // # of tasks constructed
    int *p_nthreads,            // # of threads for subref operation
    bool *p_post_sort,          // true if a final post-sort is needed
    GrB_Matrix *R_handle,       // R = inverse (I), if needed
    uint64_t **p_Cwork,         // workspace of size max(2,C->nvec+1)
    size_t *p_Cwork_size,
    // from phase0:
    const void *Ap_start,       // location of A(imin:imax,kA)
    const void *Ap_end,
    const int64_t Cnvec,        // # of vectors of C
    const bool need_qsort,      // true if C must be sorted
    const int Ikind,            // GB_ALL, GB_RANGE, GB_STRIDE or GB_LIST
    const int64_t nI,           // length of I
    const int64_t Icolon [3],   // for GB_RANGE and GB_STRIDE
    // original input:
    const int64_t avlen,        // A->vlen
    const int64_t anz,          // nnz (A)
    const bool Ap_is_32,        // if true, Ap_start/end are 32-bit; else 64
    const void *I,
    const bool I_is_32,         // if true, I is 32-bit; else 64 bit
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_TaskList_size != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;
    ASSERT (p_post_sort != NULL) ;
    ASSERT (p_Cwork != NULL) ;
    ASSERT (p_Cwork_size != NULL) ;
    ASSERT (R_handle != NULL) ;

    ASSERT ((Cnvec > 0) == (Ap_start != NULL)) ;
    ASSERT ((Cnvec > 0) == (Ap_end != NULL)) ;

    (*p_TaskList) = NULL ;
    (*p_TaskList_size) = 0 ;
    (*p_Cwork) = NULL ;
    (*p_Cwork_size) = 0 ;

    uint64_t *restrict Cwork = NULL ; size_t Cwork_size = 0 ;
    GB_WERK_DECLARE (Coarse, int64_t) ;     // size ntasks1+1
    int ntasks1 = 0 ;
    GrB_Matrix R = NULL ;

    GrB_Info info ;

    GB_IDECL (Ap_start, const, u) ; GB_IPTR (Ap_start, Ap_is_32) ;
    GB_IDECL (Ap_end  , const, u) ; GB_IPTR (Ap_end  , Ap_is_32) ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // allocate the initial TaskList
    //--------------------------------------------------------------------------

    // Allocate the TaskList to hold at least 2*ntask0 tasks.  It will grow
    // later, if needed.  Usually, 64*nthreads_max is enough, but in a few cases
    // fine tasks can cause this number to be exceeded.  If that occurs,
    // TaskList is reallocated.

    // When the mask is present, it is often fastest to break the work up
    // into tasks, even when nthreads_max is 1.

    GB_task_struct *restrict TaskList = NULL ; size_t TaskList_size = 0 ;
    int max_ntasks = 0 ;
    int ntasks0 = (nthreads_max == 1) ? 1 : (32 * nthreads_max) ;
    GB_REALLOC_TASK_WORK (TaskList, ntasks0, max_ntasks) ;

    //--------------------------------------------------------------------------
    // to determine if R needs to be constructed
    //--------------------------------------------------------------------------

    bool need_I_inverse = false ;
    bool post_sort = false ;
    int64_t iinc = Icolon [GxB_INC] ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    Cwork = GB_MALLOC_MEMORY (GB_IMAX (2, Cnvec+1), sizeof (uint64_t),
        &Cwork_size) ;
    if (Cwork == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // estimate the work required for each vector of C
    //--------------------------------------------------------------------------

    int nthreads_for_Cwork = GB_nthreads (Cnvec, chunk, nthreads_max) ;

    int64_t kC ;
    #pragma omp parallel for num_threads(nthreads_for_Cwork) schedule(static) \
        reduction(||:need_I_inverse)
    for (kC = 0 ; kC < Cnvec ; kC++)
    { 
        // jC is the (kC)th vector of C = A(I,J)
        // int64_t jC = GBh_C (Ch, kC) ; // but this is not needed
        // C(:,kC) = A(I,kA) will be constructed
        int64_t pA      = GB_IGET (Ap_start, kC) ;
        int64_t pA_end  = GB_IGET (Ap_end  , kC) ;
        int64_t alen = pA_end - pA ;      // nnz (A (imin:imax,j))

        bool this_needs_I_inverse ; // true if this vector needs I inverse
        // amount of work for C(:,kC) = A (I,kA):
        int64_t work = GB_subref_work (&this_needs_I_inverse, alen, avlen,
            Ikind, nI, need_qsort, iinc) ;

        // log the result
        need_I_inverse = need_I_inverse || this_needs_I_inverse ;
        Cwork [kC] = work ;
    }

    Cwork [Cnvec] = 0 ;

    //--------------------------------------------------------------------------
    // replace Cwork with its cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cwork, false, Cnvec, NULL, nthreads_for_Cwork, Werk) ;
    double cwork = (double) Cwork [Cnvec] ;

    //--------------------------------------------------------------------------
    // determine # of threads and tasks to use for C=A(I,J)
    //--------------------------------------------------------------------------

    int nthreads = GB_nthreads (cwork, chunk, nthreads_max) ;
    int ntasks = 0 ;
    ntasks1 = (nthreads == 1) ? 1 : (32 * nthreads) ;
    double target_task_size = cwork / (double) (ntasks1) ;
    target_task_size = GB_IMAX (target_task_size, chunk) ;

    //--------------------------------------------------------------------------
    // invert I if required
    //--------------------------------------------------------------------------

    if (need_I_inverse)
    { 
        GB_OK (GB_I_inverse (I, I_is_32, nI, avlen, &R, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (Cnvec == 0 || ntasks1 == 1)
    { 
        // construct a single coarse task that computes all of C
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = Cnvec-1 ;
        ntasks = (Cnvec == 0) ? 0 : 1 ;
        nthreads = 1 ;

        // free workspace and return result
        GB_FREE_WORKSPACE ;
        GB_RETURN_RESULTS ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // slice the work into coarse tasks
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Coarse, ntasks1 + 1, int64_t) ;
    if (Coarse == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_p_slice (Coarse, Cwork, false, Cnvec, ntasks1, false) ;

    //--------------------------------------------------------------------------
    // construct all tasks, both coarse and fine
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < ntasks1 ; t++)
    {

        //----------------------------------------------------------------------
        // coarse task computes C (:,k:klast)
        //----------------------------------------------------------------------

        int64_t k = Coarse [t] ;
        int64_t klast = Coarse [t+1] - 1 ;

        if (k >= Cnvec)
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
            // entire vectors of C, vectors k:klast, inclusive.
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
            // determine the # of fine-grain tasks to create for vector k
            //------------------------------------------------------------------

            double ckwork = Cwork [k+1] - Cwork [k] ;
            int nfine = ckwork / target_task_size ;
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
                // slice vector k into nfine fine tasks
                //--------------------------------------------------------------

                // There are two kinds of fine tasks, depending on the method
                // used to compute C(:,kC) = A(I,kA).  If the method iterates
                // across all entries in A(imin:imax,kA), then those entries
                // are sliced (of size alen).  Three methods (1, 2, and 6)
                // iterate across all entries in I instead (of size nI).

                int64_t pA     = GB_IGET (Ap_start, k) ;
                int64_t pA_end = GB_IGET (Ap_end  , k) ;
                int64_t alen = pA_end - pA ;      // nnz (A (imin:imax,j))

                int method = GB_subref_method (alen, avlen, Ikind, nI,
                    need_qsort, iinc) ;

                if (method == 10)
                { 
                    // multiple fine tasks operate on a single vector C(:,kC)
                    // using method 10, and so a post-sort is needed.
                    post_sort = true ;
                }

                if (method == 1 || method == 2 || method == 6)
                {

                    // slice I for this task
                    nfine = GB_IMIN (nfine, nI) ;
                    nfine = GB_IMAX (nfine, 1) ;

                    for (int tfine = 0 ; tfine < nfine ; tfine++)
                    { 
                        // flag this as a fine task, and record the method.
                        // Methods 1, 2, and 6 slice I, not A(:,kA)
                        TaskList [ntasks].kfirst = k ;
                        TaskList [ntasks].klast = -method ;
                        // do not partition A(:,kA)
                        TaskList [ntasks].pA = pA ;
                        TaskList [ntasks].pA_end = pA_end ;
                        // partition I for this task
                        GB_PARTITION (TaskList [ntasks].pB,
                            TaskList [ntasks].pB_end, nI, tfine, nfine) ;
                        // unused
                        TaskList [ntasks].pM = -1 ;
                        TaskList [ntasks].pM_end = -1 ;
                        // no post sort
                        TaskList [ntasks].len = 0 ;
                        ntasks++ ;
                    }

                }
                else
                {

                    // slice A(:,kA) for this task
                    nfine = GB_IMIN (nfine, alen) ;
                    nfine = GB_IMAX (nfine, 1) ;

                    bool reverse = (method == 8 || method == 9) ;

                    for (int tfine = 0 ; tfine < nfine ; tfine++)
                    { 
                        // flag this as a fine task, and record the method.
                        // These methods slice A(:,kA).  Methods 8 and 9
                        // must do so in reverse order.
                        TaskList [ntasks].kfirst = k ;
                        TaskList [ntasks].klast = -method ;
                        // partition the items for this task
                        GB_PARTITION (TaskList [ntasks].pA,
                            TaskList [ntasks].pA_end, alen,
                            (reverse) ? (nfine-tfine-1) : tfine, nfine) ;
                        TaskList [ntasks].pA += pA ;
                        TaskList [ntasks].pA_end += pA  ;
                        // do not partition I
                        TaskList [ntasks].pB = 0 ;
                        TaskList [ntasks].pB_end = nI ;
                        // unused
                        TaskList [ntasks].pM = -1 ;
                        TaskList [ntasks].pM_end = -1 ;

                        // flag the task that does the post sort
                        bool do_post_sort = (tfine == 0 && method == 10) ;
                        TaskList [ntasks].len = do_post_sort ;
                        ntasks++ ;
                    }
                }
            }
        }
    }

    ASSERT (ntasks > 0) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    GB_RETURN_RESULTS ;
    return (GrB_SUCCESS) ;
}

