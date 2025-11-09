//------------------------------------------------------------------------------
// GB_ewise_slice: slice the entries and vectors for an ewise operation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: possible: could use variants for sparsity formats.

// Constructs a set of tasks to compute C, for an element-wise operation that
// operates on two input matrices, C=op(A,B).  These include:
// GB_add, GB_emult, and GB_masker, and many GB_subassign_* methods
// (02, 04, 06s_and_14, 08n, 08s_and_16, 09, 10_and_18, 11, 12_and_20).

// The mask is ignored for computing where to slice the work, but it is sliced
// once the location has been found.

// M, A, B: any sparsity structure (hypersparse, sparse, bitmap, or full).
// C: constructed as sparse or hypersparse in the caller.

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_WERK_POP (Coarse, int64_t) ;             \
    GB_FREE_MEMORY (&Cwork, Cwork_size) ;       \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_FREE_MEMORY (&TaskList, TaskList_size) ; \
}

#include "GB.h"

//------------------------------------------------------------------------------
// GB_ewise_slice
//------------------------------------------------------------------------------

//  GrB_Info GB_ewise_slice
//  (
//      // output:
//      GB_task_struct **p_TaskList,    // array of structs
//      size_t *p_TaskList_size,        // size of TaskList
//      int *p_ntasks,                  // # of tasks constructed
//      int *p_nthreads,                // # of threads for eWise operation
//      // input:
//      const int64_t Cnvec,            // # of vectors of C
//      const void *Ch,                 // vectors of C, if hypersparse
//      const bool Cj_is_32,            // if true, Ch is 32-bit, else 64-bit
//      const int64_t *restrict C_to_M, // mapping of C to M
//      const int64_t *restrict C_to_A, // mapping of C to A
//      const int64_t *restrict C_to_B, // mapping of C to B
//      bool Ch_is_Mh,                  // if true, then Ch == Mh; GB_add only
//      const GrB_Matrix M,             // mask matrix to slice (optional)
//      const GrB_Matrix A,             // matrix to slice
//      const GrB_Matrix B,             // matrix to slice
//      GB_Werk Werk
//  )

GB_CALLBACK_EWISE_SLICE_PROTO (GB_ewise_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_TaskList_size != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;

    ASSERT_MATRIX_OK (A, "A for ewise_slice", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ; 
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ; 

    ASSERT_MATRIX_OK (B, "B for ewise_slice", GB0) ;
    ASSERT (!GB_ZOMBIES (B)) ; 
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ; 

    ASSERT_MATRIX_OK_OR_NULL (M, "M for ewise_slice", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    (*p_TaskList  ) = NULL ;
    (*p_TaskList_size) = 0 ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;

    GB_MDECL (Cwork, , u) ; size_t Cwork_size = 0 ;
    GB_WERK_DECLARE (Coarse, int64_t) ;     // size ntasks1+1
    int ntasks1 = 0 ;

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
    int ntasks0 = (M == NULL && nthreads_max == 1) ? 1 : (32 * nthreads_max) ;
    GB_REALLOC_TASK_WORK (TaskList, ntasks0, max_ntasks) ;

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (Cnvec == 0 || ntasks0 == 1)
    { 
        // construct a single coarse task that computes all of C
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = Cnvec-1 ;
        (*p_TaskList  ) = TaskList ;
        (*p_TaskList_size) = TaskList_size ;
        (*p_ntasks    ) = (Cnvec == 0) ? 0 : 1 ;
        (*p_nthreads  ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // get A, B, and M
    //--------------------------------------------------------------------------

    const int64_t vlen = A->vlen ;
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    #ifdef GB_DEBUG
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    #endif
    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    bool Ch_is_Ah = (Ch != NULL && A->h != NULL && Ch == A->h) ;
    bool Ch_is_Bh = (Ch != NULL && B->h != NULL && Ch == B->h) ;
    ASSERT (GB_IMPLIES (Ch_is_Ah, Cj_is_32 == A->j_is_32)) ;
    ASSERT (GB_IMPLIES (Ch_is_Bh, Cj_is_32 == B->j_is_32)) ;

    GB_Mp_DECLARE (Mp, const) ; GB_Mp_PTR (Mp, M) ;
    void *Mi = NULL ;
    bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    bool Mi_is_32 = false ;
    if (M != NULL)
    { 
        // Ch_is_Mh is true if either true on input (for GB_add, which denotes
        // that Ch is a deep copy of M->h), or if Ch is a shallow copy of M->h.
        // If Ch is a deep copy, the integer sizes of Ch and Mh can differ.
        Ch_is_Mh = Ch_is_Mh || (Ch != NULL && M_is_hyper && Ch == M->h) ;
        Mi = M->i ;
        Mi_is_32 = M->i_is_32 ;
    }

    GB_IDECL (Ch, const, u) ; GB_IPTR (Ch, Cj_is_32) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz (A) ;
    int64_t bnz = GB_nnz (A) ;
    int64_t cnzmax = anz + bnz + Cnvec ;
    bool Cwork_is_32 = (cnzmax < UINT32_MAX) ;
    size_t cwsize = (Cwork_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Cwork = GB_MALLOC_MEMORY (Cnvec+1, cwsize, &Cwork_size) ;
    if (Cwork == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_IPTR (Cwork, Cwork_is_32) ;

    //--------------------------------------------------------------------------
    // compute an estimate of the work for each vector of C
    //--------------------------------------------------------------------------

    int nthreads_for_Cwork = GB_nthreads (Cnvec, chunk, nthreads_max) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads_for_Cwork) schedule(static)
    for (k = 0 ; k < Cnvec ; k++)
    {

        //----------------------------------------------------------------------
        // get the C(:,j) vector
        //----------------------------------------------------------------------

        int64_t j = GBh_C (Ch, k) ;

        //----------------------------------------------------------------------
        // get the corresponding vector of A
        //----------------------------------------------------------------------

        int64_t kA ;
        if (C_to_A != NULL)
        { 
            // A is hypersparse and the C_to_A mapping has been created
            ASSERT (GB_IS_HYPERSPARSE (A)) ;
            kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                ASSERT (j == GBh_A (Ah, kA)) ;
            }
        }
        else if (Ch_is_Ah)
        { 
            // A is hypersparse, but Ch is a shallow copy of A->h
            ASSERT (GB_IS_HYPERSPARSE (A)) ;
            kA = k ;
            ASSERT (j == GB_IGET (Ah, kA)) ;
        }
        else
        { 
            // A is sparse, bitmap, or full
            ASSERT (!GB_IS_HYPERSPARSE (A)) ;
            kA = j ;
        }

        //----------------------------------------------------------------------
        // get the corresponding vector of B
        //----------------------------------------------------------------------

        int64_t kB ;
        if (C_to_B != NULL)
        { 
            // B is hypersparse and the C_to_B mapping has been created
            ASSERT (GB_IS_HYPERSPARSE (B)) ;
            kB = C_to_B [k] ;
            ASSERT (kB >= -1 && kB < B->nvec) ;
            if (kB >= 0)
            {
                ASSERT (j == GBh_B (Bh, kB)) ;
            }
        }
        else if (Ch_is_Bh)
        { 
            // B is hypersparse, but Ch is a shallow copy of B->h
            ASSERT (GB_IS_HYPERSPARSE (B)) ;
            kB = k ;
            ASSERT (j == GB_IGET (Bh, kB)) ;
        }
        else
        { 
            // B is sparse, bitmap, or full
            ASSERT (!GB_IS_HYPERSPARSE (B)) ;
            kB = j ;
        }

        //----------------------------------------------------------------------
        // estimate the work for C(:,j)
        //----------------------------------------------------------------------

        ASSERT (kA >= -1 && kA < A->nvec) ;
        ASSERT (kB >= -1 && kB < B->nvec) ;
        const int64_t aknz = (kA < 0) ? 0 :
            ((Ap == NULL) ? vlen : (GB_IGET (Ap, kA+1) - GB_IGET (Ap, kA))) ;
        const int64_t bknz = (kB < 0) ? 0 :
            ((Bp == NULL) ? vlen : (GB_IGET (Bp, kB+1) - GB_IGET (Bp, kB))) ;

        int64_t ckwork = aknz + bknz + 1 ;
        GB_ISET (Cwork, k, ckwork) ;            // Cwork [k] = ckwork ;
    }
    GB_ISET (Cwork, Cnvec, 0) ;                 // Cwork [Cnvec] = 0 ;

    //--------------------------------------------------------------------------
    // replace Cwork with its cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cwork, Cwork_is_32, Cnvec, NULL, nthreads_for_Cwork, Werk) ;
    double cwork = (double) GB_IGET (Cwork, Cnvec) ;

    //--------------------------------------------------------------------------
    // determine # of threads and tasks for the eWise operation
    //--------------------------------------------------------------------------

    int nthreads = GB_nthreads (cwork, chunk, nthreads_max) ;

    ntasks0 = (M == NULL && nthreads == 1) ? 1 : (32 * nthreads) ;
    double target_task_size = cwork / (double) (ntasks0) ;
    target_task_size = GB_IMAX (target_task_size, chunk) ;
    ntasks1 = cwork / target_task_size ;
    ntasks1 = GB_IMAX (ntasks1, 1) ;

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
    GB_p_slice (Coarse, Cwork, Cwork_is_32, Cnvec, ntasks1, false) ;

    //--------------------------------------------------------------------------
    // construct all tasks, both coarse and fine
    //--------------------------------------------------------------------------

    int ntasks = 0 ;

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
            // get the vector of C
            //------------------------------------------------------------------

            int64_t j = GBh_C (Ch, k) ;

            //------------------------------------------------------------------
            // get the corresponding vector of A
            //------------------------------------------------------------------

            int64_t kA ;
            if (C_to_A != NULL)
            { 
                // A is hypersparse and the C_to_A mapping has been created
                ASSERT (GB_IS_HYPERSPARSE (A)) ;
                kA = C_to_A [k] ;
            }
            else if (Ch_is_Ah)
            { 
                // A is hypersparse, but Ch is a shallow copy of A->h
                ASSERT (GB_IS_HYPERSPARSE (A)) ;
                kA = k ;
            }
            else
            { 
                // A is sparse, bitmap, or full
                ASSERT (!GB_IS_HYPERSPARSE (A)) ;
                kA = j ;
            }
            int64_t pA_start = (kA < 0) ? (-1) : GBp_A (Ap, kA, vlen) ;
            int64_t pA_end   = (kA < 0) ? (-1) : GBp_A (Ap, kA+1, vlen) ;
            bool a_empty = (pA_end == pA_start) ;

            //------------------------------------------------------------------
            // get the corresponding vector of B
            //------------------------------------------------------------------

            int64_t kB ;
            if (C_to_B != NULL)
            { 
                // B is hypersparse and the C_to_B mapping has been created
                ASSERT (GB_IS_HYPERSPARSE (B)) ;
                kB = C_to_B [k] ;
            }
            else if (Ch_is_Bh)
            { 
                // B is hypersparse, but Ch is a shallow copy of B->h
                ASSERT (GB_IS_HYPERSPARSE (B)) ;
                kB = k ;
            }
            else
            { 
                // B is sparse, bitmap, or full
                ASSERT (!GB_IS_HYPERSPARSE (B)) ;
                kB = j ;
            }
            int64_t pB_start = (kB < 0) ? (-1) : GBp_B (Bp, kB, vlen) ;
            int64_t pB_end   = (kB < 0) ? (-1) : GBp_B (Bp, kB+1, vlen) ;
            bool b_empty = (pB_end == pB_start) ;

            //------------------------------------------------------------------
            // get the corresponding vector of M, if present
            //------------------------------------------------------------------

            // M can have any sparsity structure (hyper, sparse, bitmap, full)

            int64_t pM_start = -1 ;
            int64_t pM_end   = -1 ;
            if (M != NULL)
            {
                int64_t kM ;
                if (C_to_M != NULL)
                { 
                    // M is hypersparse and the C_to_M mapping has been created
                    ASSERT (GB_IS_HYPERSPARSE (M)) ;
                    kM = C_to_M [k] ;
                }
                else if (Ch_is_Mh)
                { 
                    // M is hypersparse, but Ch is a copy of Mh
                    ASSERT (GB_IS_HYPERSPARSE (M)) ;
                    // Ch is a deep or shallow copy of Mh
                    kM = k ;
                }
                else
                { 
                    // M is sparse, bitmap, or full
                    ASSERT (!GB_IS_HYPERSPARSE (M)) ;
                    kM = j ;
                }
                pM_start = (kM < 0) ? -1 : GBp_M (Mp, kM, vlen) ;
                pM_end   = (kM < 0) ? -1 : GBp_M (Mp, kM+1, vlen) ;
            }
            bool m_empty = (pM_end == pM_start) ;

            //------------------------------------------------------------------
            // determine the # of fine-grain tasks to create for vector k
            //------------------------------------------------------------------

            double ckwork = GB_IGET (Cwork, k+1) - GB_IGET (Cwork, k) ;
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

                // first fine task starts at the top of vector k
                ASSERT (ntasks < max_ntasks) ;
                TaskList [ntasks].kfirst = k ;
                TaskList [ntasks].klast  = -1 ; // this is a fine task
                TaskList [ntasks].pM = (m_empty) ? -1 : pM_start ;
                TaskList [ntasks].pA = (a_empty) ? -1 : pA_start ;
                TaskList [ntasks].pB = (b_empty) ? -1 : pB_start ;
                TaskList [ntasks].len = 0 ;     // to be determined below
                ntasks++ ;
                int64_t ilast = 0, i = 0 ;

                for (int tfine = 1 ; tfine < nfine ; tfine++)
                { 
                    double target_work = ((nfine-tfine) * ckwork) / nfine ;
                    int64_t pM, pA, pB ;
                    GB_slice_vector (&i, &pM, &pA, &pB,
                        pM_start, pM_end, Mi,   Mi_is_32,
                        pA_start, pA_end, A->i, A->i_is_32,
                        pB_start, pB_end, B->i, B->i_is_32,
                        vlen, target_work) ;

                    // prior task ends at pM-1, pA-1, and pB-1
                    TaskList [ntasks-1].pM_end = pM ;
                    TaskList [ntasks-1].pA_end = pA ;
                    TaskList [ntasks-1].pB_end = pB ;

                    // prior task handles indices ilast:i-1
                    TaskList [ntasks-1].len = i - ilast ;

                    // this task starts at pM, pA, and pB 
                    ASSERT (ntasks < max_ntasks) ;
                    TaskList [ntasks].kfirst = k ;
                    TaskList [ntasks].klast  = -1 ; // this is a fine task
                    TaskList [ntasks].pM = pM ;
                    TaskList [ntasks].pA = pA ;
                    TaskList [ntasks].pB = pB ;

                    // advance to the next task
                    ntasks++ ;
                    ilast = i ;
                }

                // Terminate the last fine task.
                ASSERT (ntasks <= max_ntasks) ;
                TaskList [ntasks-1].pM_end = (m_empty) ? -1 : pM_end ;
                TaskList [ntasks-1].pA_end = (a_empty) ? -1 : pA_end ;
                TaskList [ntasks-1].pB_end = (b_empty) ? -1 : pB_end ;
                TaskList [ntasks-1].len = vlen - i ;
            }
        }
    }

    ASSERT (ntasks <= max_ntasks) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    (*p_TaskList     ) = TaskList ;
    (*p_TaskList_size) = TaskList_size ;
    (*p_ntasks       ) = ntasks ;
    (*p_nthreads     ) = nthreads ;
    return (GrB_SUCCESS) ;
}

