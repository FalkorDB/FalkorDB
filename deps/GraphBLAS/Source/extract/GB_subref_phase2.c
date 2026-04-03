//------------------------------------------------------------------------------
// GB_subref_phase2: find # of entries in C=A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_subref_phase2 counts the number of entries in each vector of C, for
// C=A(I,J) and then does a cumulative sum to find Cp.  A is sparse or
// hypersparse.

// Cp is either freed by phase2, or transplanted into C.

#include "extract/GB_subref.h"

GrB_Info GB_subref_phase2               // count nnz in each C(:,j)
(
    // computed by phase2:
    void **Cp_handle,                   // output of size Cnvec+1
    bool *p_Cp_is_32,                   // if true, Cp is 32-bit; else 64 bit
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    // tasks from phase1:
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads to use
    const GrB_Matrix R,                 // R = inverse (I), if needed
    uint64_t **p_Cwork,                 // workspace of size max(2,C->nvec+1)
    size_t Cwork_size,
    // analysis from phase0:
    const void *Ap_start,
    const void *Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // original input:
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const bool symbolic,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Cp_size_handle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for subref phase2", GB0) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    GB_IDECL (I       , const, u) ; GB_IPTR (I       , I_is_32) ;
    GB_IDECL (Ap_start, const, u) ; GB_IPTR (Ap_start, A->p_is_32) ;
    GB_IDECL (Ap_end  , const, u) ; GB_IPTR (Ap_end  , A->p_is_32) ;

    bool R_is_hyper = false ;
    int64_t rnvec = 0, R_hash_bits = 0 ;
    void *Rp = NULL, *Rh = NULL, *Ri = NULL ;
    void *R_Yp = NULL, *R_Yi = NULL, *R_Yx = NULL ;
    bool Rp_is_32 = false ;
    bool Rj_is_32 = false ;
    bool Ri_is_32 = false ;
    GB_IDECL (Rp, const, u) ;
    GB_IDECL (Rh, const, u) ;
    GB_IDECL (Ri, const, u) ;
    if (R != NULL)
    {
        R_is_hyper = GB_IS_HYPERSPARSE (R) ;
        rnvec = R->nvec ;
        Rp = R->p ;
        Rh = R->h ;
        Ri = R->i ;
        GB_IPTR (Rp, R->p_is_32) ;
        GB_IPTR (Rh, R->j_is_32) ;
        GB_IPTR (Ri, R->i_is_32) ;
        Rp_is_32 = R->p_is_32 ;
        Rj_is_32 = R->j_is_32 ;
        Ri_is_32 = R->i_is_32 ;
        GrB_Matrix R_Y = R->Y ;
        if (R_Y != NULL)
        {
            R_Yp = R_Y->p ;
            R_Yi = R_Y->i ;
            R_Yx = R_Y->x ;
            R_hash_bits = (R_Y->vdim - 1) ;
        }
    }

    (*Cp_handle) = NULL ;
    (*Cp_size_handle) = 0 ;
    uint64_t *restrict Cwork = (*p_Cwork) ;
    const bool Ai_is_32 = A->i_is_32 ;
    ASSERT (Cwork != NULL) ;

    // clear Cwork [k] for fine tasks that compute vector k
    #ifdef GB_DEBUG
    GB_memset (Cwork, 0xFF, (Cnvec+1) * sizeof (uint64_t), nthreads) ;
    #endif
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast < 0) ;
        if (fine_task)
        {
            // The set of fine tasks that compute C(:,kC) do not compute Cwork
            // [kC] directly.  Instead, they compute their partial results in
            // TaskList [taskid].pC, which is then summed by GB_task_cumsum.
            // That method sums up the work of each fine task and adds it to
            // Cwork [kC], which must be initialized here to zero.
            Cwork [kfirst] = 0 ;
        }
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of C
    //--------------------------------------------------------------------------

    #define GB_I_KIND Ikind
    #define GB_NEED_QSORT need_qsort

    #define GB_ANALYSIS_PHASE
    if (symbolic)
    { 
        #define GB_SYMBOLIC
        // symbolic extraction must handle zombies
        const bool may_see_zombies = (A->nzombies > 0) ;
        #include "extract/template/GB_subref_template.c"
    }
    else
    { 
        // iso and non-iso numeric extraction do not see zombies
        ASSERT (!GB_ZOMBIES (A)) ;
        #include "extract/template/GB_subref_template.c"
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Cwork and fine tasks in TaskList
    //--------------------------------------------------------------------------

    Cwork [Cnvec] = 0 ;
    GB_task_cumsum (Cwork, false, Cnvec, Cnvec_nonempty, TaskList, ntasks,
        nthreads, Werk) ;
    int64_t cnz = Cwork [Cnvec] ;

    //--------------------------------------------------------------------------
    // allocate the final result Cp
    //--------------------------------------------------------------------------

    // determine the final p_is_32 setting for the new matrix;
    // j_is_32 and i_is_32 have already been determined by GB_subref_phase0

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    ASSERT (p_Cp_is_32 != NULL) ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_AUTO_SPARSITY, cnz, nI, nJ, Werk) ;

    void *Cp = NULL ; size_t Cp_size = 0 ;

    if (Cp_is_32)
    { 
        // Cp is 32-bit; allocate and typecast from Cwork
        Cp = GB_MALLOC_MEMORY (GB_IMAX (2, Cnvec+1), sizeof (uint32_t),
            &Cp_size) ;
        if (Cp == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        GB_cast_int (Cp, GB_UINT32_code, Cwork, GB_UINT64_code, Cnvec+1,
            nthreads_max) ;
    }
    else
    { 
        // Cp is 64-bit; transplant Cwork as Cp
        Cp = Cwork ;
        Cp_size = Cwork_size ;
        (*p_Cwork) = NULL ;
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Cp_handle     ) = Cp ;
    (*Cp_size_handle) = Cp_size ;
    (*p_Cp_is_32    ) = Cp_is_32 ;
    return (GrB_SUCCESS) ;
}

