//------------------------------------------------------------------------------
// GB_subassign_26: C(:,j1:j2) = A ; append columns, no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 26: C(:,j1:j2) = A ; append columns, no S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           constructed

// C: hypersparse
// A: sparse

#include "assign/GB_subassign_methods.h"
#define GB_GENERIC
#define GB_SCALAR_ASSIGN 0
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;
#define GB_MEM_CHUNK (1024*1024)

GrB_Info GB_subassign_26
(
    GrB_Matrix C,
    // input:
    const int64_t Jcolon [3],       // j1:j2, with an increment of 1
    const GrB_Matrix A,
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (GB_IS_HYPERSPARSE (C)) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    ASSERT (!GB_PENDING (A)) ;          // FUTURE: could tolerate pending tuples
    ASSERT (!GB_ZOMBIES (A)) ;          // FUTURE: could tolerate zombies
    ASSERT (A->type == C->type) ;       // no typecasting
    ASSERT (!A->iso) ;                  // FUTURE: handle iso case
    ASSERT (!C->iso) ;                  // FUTURE: handle iso case

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const size_t csize = C->type->size ;
    int64_t Cnvec = C->nvec ;
    int64_t cnz = C->nvals ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    void *Ai = A->i ;
    GB_void *restrict Ax = (GB_void *) A->x ;
    int64_t anz = A->nvals ;
    bool Ai_is_32 = A->i_is_32 ;
    size_t aisize = (Ai_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    GB_Type_code aicode = (Ai_is_32) ? GB_UINT32_code : GB_UINT64_code ;

    int64_t j1 = Jcolon [GxB_BEGIN] ;
    int64_t j2 = Jcolon [GxB_END  ] ;
    ASSERT (Jcolon [GxB_INC] == 1) ;
    int64_t nJ = j2 - j1 + 1 ;
    ASSERT (nJ == A->vdim) ;

    //--------------------------------------------------------------------------
    // Method 26: C(:,j1:j2) = A ; append column(s), no S.
    //--------------------------------------------------------------------------

    // Time: Optimal.  Work is O(nnz(A)).

    //--------------------------------------------------------------------------
    // resize C if necessary
    //--------------------------------------------------------------------------

    int64_t cnz_new = cnz + anz ;

    if (Cnvec + nJ > C->plen)
    { 
        // double the size of C->h and C->p if needed
        int64_t plen_new = GB_IMIN (C->vdim, 2*(C->plen + nJ)) ;
        GB_OK (GB_hyper_realloc (C, plen_new, Werk)) ;
    }

    if (cnz_new > GB_nnz_max (C))
    { 
        // double the size of C->i and C->x if needed
        GB_OK (GB_ix_realloc (C, 2*cnz_new + 1)) ;
    }

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    GB_Ci_DECLARE (Ci, ) ; GB_Ci_PTR (Ci, C) ;
    GB_void *restrict Cx = (GB_void *) C->x ;
    bool Ci_is_32 = C->i_is_32 ;
    GB_Type_code cicode = (Ci_is_32) ? GB_UINT32_code : GB_UINT64_code ;

    //--------------------------------------------------------------------------
    // determine any parallelism to use
    //--------------------------------------------------------------------------

    ASSERT (Cnvec == 0 || GB_IGET (Ch, Cnvec-1) == j1-1) ;

    bool phase1_parallel = (nJ > GB_CHUNK_DEFAULT) ;
    bool phase2_parallel = (anz * (aisize + csize) > GB_MEM_CHUNK) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // phase1: append to Cp and Ch; find # new nonempty vectors, and properties
    //--------------------------------------------------------------------------

    int64_t Anvec_nonempty = 0 ;
    int nthreads = (phase1_parallel) ?
        GB_nthreads (nJ, chunk, nthreads_max) : 1 ;
    int64_t k ;

    // compute Cp, Ch, and Anvec_nonempty in parallel
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:Anvec_nonempty)
    for (k = 0 ; k < nJ ; k++)
    { 
        int64_t apk = GB_IGET (Ap, k) ;
        int64_t anzk = GB_IGET (Ap, k+1) - apk ;
        GB_ISET (Ch, Cnvec + k, j1 + k) ;
        GB_ISET (Cp, Cnvec + k, cnz + apk) ;
        Anvec_nonempty += (anzk > 0) ;
    }

    int64_t C_nvec_nonempty = GB_nvec_nonempty_get (C) ;
    if (C_nvec_nonempty >= 0)
    { 
//      C->nvec_nonempty += Anvec_nonempty ;
        GB_nvec_nonempty_set (C, C_nvec_nonempty + Anvec_nonempty) ;
    }
    C->nvec += nJ ;
    GB_ISET (Cp, C->nvec, cnz_new) ;
    C->nvals = cnz_new ;
    C->jumbled = C->jumbled || A->jumbled ;

    //--------------------------------------------------------------------------
    // phase2: append the indices and values to the end of Ci and Cx
    //--------------------------------------------------------------------------

    nthreads = (phase2_parallel) ? GB_nthreads (anz, chunk, nthreads_max) : 1 ;

    // copy Ci and Cx
    GB_cast_int (GB_IADDR (Ci, cnz), cicode, Ai, aicode, anz, nthreads) ;
    GB_memcpy (Cx + cnz * csize, Ax, anz * csize, nthreads) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

