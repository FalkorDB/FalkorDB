//------------------------------------------------------------------------------
// GB_AxB_saxpy3_cumsum: finalize nnz(C(:,j)) and find cumulative sum of Cp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// phase3: fine tasks finalize their computation nnz(C(:,j))
// phase4: cumulative sum of C->p

#include "GB.h"

GB_CALLBACK_SAXPY3_CUMSUM_PROTO (GB_AxB_saxpy3_cumsum)
{

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (!GB_IS_FULL (C)) ;

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;
    const bool Cp_is_32 = C->p_is_32 ;
    ASSERT (Cp != NULL) ;

    //==========================================================================
    // phase3: count nnz(C(:,j)) for fine tasks
    //==========================================================================

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        // int64_t kk = SaxpyTasks [taskid].vector ;
        uint64_t hash_size = SaxpyTasks [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int team_size = SaxpyTasks [taskid].team_size ;
        int leader    = SaxpyTasks [taskid].leader ;
        int my_teamid = taskid - leader ;
        int64_t my_cjnz = 0 ;

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase3: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
            //------------------------------------------------------------------

            // Hf [i] == 2 if C(i,j) is an entry in C(:,j)

            int8_t *restrict Hf ;
            Hf = (int8_t *restrict) SaxpyTasks [taskid].Hf ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;
            for (int64_t i = istart ; i < iend ; i++)
            {
                if (Hf [i] == 2)
                { 
                    my_cjnz++ ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // phase3: fine hash task, C=A*B, C<M>=A*B, or C<!M>=A*B
            //------------------------------------------------------------------

            // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
            // and the index i of the entry is (Hf [hash] >> 2) - 1.

            uint64_t *restrict Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf;
            uint64_t mystart, myend ;
            GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ;
            for (uint64_t hash = mystart ; hash < myend ; hash++)
            {
                if ((Hf [hash] & 3) == 2)
                { 
                    my_cjnz++ ;
                }
            }
        }

        SaxpyTasks [taskid].my_cjnz = my_cjnz ; // count this task's nnz(C(:,j))
    }

    //==========================================================================
    // phase4: compute Cp with cumulative sum
    //==========================================================================

    //--------------------------------------------------------------------------
    // sum nnz (C (:,j)) for fine tasks
    //--------------------------------------------------------------------------

    // SaxpyTasks [taskid].my_cjnz is the # of unique entries found in C(:,j) by
    // that task.  Sum these terms to compute total # of entries in C(:,j).

    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = SaxpyTasks [taskid].vector ;
        GB_ISET (Cp, kk, 0) ; // Cp [kk] = 0 ;
    }

    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = SaxpyTasks [taskid].vector ;
        int64_t my_cjnz = SaxpyTasks [taskid].my_cjnz ;
        GB_IINC (Cp, kk, my_cjnz) ; // Cp [kk] += my_cjnz ;
        ASSERT (my_cjnz <= cvlen) ;
    }

    //--------------------------------------------------------------------------
    // cumulative sum for Cp (fine and coarse tasks)
    //--------------------------------------------------------------------------

    // Cp [kk] is now nnz (C (:,j)), for all vectors j, whether computed by
    // fine tasks or coarse tasks, and where j == GBh_B (Bh, kk) 

    #ifdef GBCOVER
    // tell GB_cumsum to fake a failure and return ok as false:
    if (GB_Global_hack_get (4)) GB_Global_hack_set (5, 1) ;
    #endif

    int nth = GB_nthreads (cnvec, chunk, nthreads) ;
    int64_t nvec_nonempty ;
    bool ok = GB_cumsum (Cp, Cp_is_32, cnvec, &nvec_nonempty, nth, Werk) ;
    if (ok)
    { 
        GB_nvec_nonempty_set (C, nvec_nonempty) ;
    }

    #ifdef GBCOVER
    // restore the hack (for test coverage only)
    if (GB_Global_hack_get (4)) GB_Global_hack_set (5, 0) ;
    #endif

    #ifdef GB_DEBUG
    int64_t cnz1 = 0, cnz2 = 0 ;
    if (Cp_is_32)
    {
        uint32_t *Cp_debug = C->p ;
        if (ok) cnz1 = Cp_debug [cnvec] ;
        for (int k = 0 ; k <= cnvec ; k++)
        {
            if (!ok && k < cnvec) cnz1 += Cp_debug [k] ;
        }
    }
    else
    {
        uint64_t *Cp_debug = C->p ;
        if (ok) cnz1 = Cp_debug [cnvec] ;
        for (int k = 0 ; k <= cnvec ; k++)
        {
            if (!ok && k < cnvec) cnz1 += Cp_debug [k] ;
        }
    }
    #endif

    if (!ok)
    { 
        // convert Cp to uint64_t and redo the cumulative sum
        ASSERT (Cp_is_32) ;
        ASSERT (!C->p_shallow) ;
        void *Cp_new = NULL ;
        size_t Cp_new_size = 0 ;
        Cp_new = GB_MALLOC_MEMORY (cnvec+1, sizeof (uint64_t), &Cp_new_size) ;
        if (Cp_new == NULL)
        { 
            return (GrB_OUT_OF_MEMORY) ;
        }
        // Cp_new = (uint64_t) Cp, casting from 32-bit to 64-bit
        GB_cast_int (Cp_new, GB_UINT64_code, Cp, GB_UINT32_code, cnvec+1, nth) ;
        GB_FREE_MEMORY (&Cp, C->p_size) ;
        C->p = Cp_new ;
        C->p_size = Cp_new_size ;
        C->p_is_32 = false ;
        // redo the cumsum (this will always succeed)
        GB_cumsum (C->p, false, cnvec, &nvec_nonempty, nth, Werk) ;
        GB_nvec_nonempty_set (C, nvec_nonempty) ;
    }

    #ifdef GB_DEBUG
    if (C->p_is_32)
    {
        uint32_t *Cp_debug = C->p ;
        cnz2 = Cp_debug [cnvec] ;
    }
    else
    {
        uint64_t *Cp_debug = C->p ;
        cnz2 = Cp_debug [cnvec] ;
    }
    ASSERT (cnz1 == cnz2) ;
    #endif

    //--------------------------------------------------------------------------
    // cumulative sum of nnz (C (:,j)) for each team of fine tasks
    //--------------------------------------------------------------------------

    int64_t cjnz_sum = 0 ;
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {
        if (taskid == SaxpyTasks [taskid].leader)
        { 
            cjnz_sum = 0 ;
        }
        int64_t my_cjnz = SaxpyTasks [taskid].my_cjnz ;
        SaxpyTasks [taskid].my_cjnz = cjnz_sum ;
        cjnz_sum += my_cjnz ;
    }

    return (GrB_SUCCESS) ;
}

