//------------------------------------------------------------------------------
// GB_subref_phase3: C=A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// This function either frees Cp and Ch, or transplants then into C, as C->p
// and C->h.  Either way, the caller must not free them.

#include "GB_subref.h"
#include "GB_sort.h"
#include "GB_unused.h"

GrB_Info GB_subref_phase3   // C=A(I,J)
(
    GrB_Matrix C,               // output matrix, static header
    // from phase2:
    int64_t **Cp_handle,        // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // from phase1:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int ntasks,                           // # of tasks
    const int nthreads,                         // # of threads to use
    const bool post_sort,               // true if post-sort needed
    const int64_t *Mark,                // for I inverse buckets, size A->vlen
    const int64_t *Inext,               // for I inverse buckets, size nI
    const int64_t nduplicates,          // # of duplicates, if I inverted
    // from phase0:
    int64_t **Ch_handle,
    size_t Ch_size,
    const int64_t *restrict Ap_start,
    const int64_t *restrict Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // from GB_subref:
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // iso value of C
    // original input:
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,
    const GrB_Index *I,
    const bool symbolic,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT (Cp_handle != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    const int64_t *restrict Ch = (*Ch_handle) ;
    const int64_t *restrict Cp = (*Cp_handle) ;
    ASSERT (Cp != NULL) ;
    ASSERT_MATRIX_OK (A, "A for subref phase3", GB0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;    // GB_bitmap_subref is used instead

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int64_t cnz = Cp [Cnvec] ;
    bool C_is_hyper = (Ch != NULL) ;
    GrB_Type ctype = (symbolic) ? GrB_INT64 : A->type ;

    // allocate the result C (but do not allocate C->p or C->h)
    int sparsity = C_is_hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;
    // set C->iso = C_iso       OK
    GrB_Info info = GB_new_bix (&C, // sparse or hyper, existing header
        ctype, nI, nJ, GB_Ap_null, C_is_csc,
        sparsity, true, A->hyper_switch, Cnvec, cnz, true, C_iso) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE (Cp_handle, Cp_size) ;
        GB_FREE (Ch_handle, Ch_size) ;
        return (info) ;
    }

    // add Cp as the vector pointers for C, from GB_subref_phase1
    C->p = (int64_t *) Cp ; C->p_size = Cp_size ;
    (*Cp_handle) = NULL ;

    // add Ch as the hypersparse list for C, from GB_subref_phase0
    if (C_is_hyper)
    { 
        // transplant Ch into C
        C->h = (int64_t *) Ch ; C->h_size = Ch_size ;
        (*Ch_handle) = NULL ;
        C->nvec = Cnvec ;
    }

    // now Cp and Ch have been transplanted into C, so they must not be freed.
    ASSERT ((*Cp_handle) == NULL) ;
    ASSERT ((*Ch_handle) == NULL) ;
    C->nvec_nonempty = Cnvec_nonempty ;
    C->nvals = cnz ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // phase3: C = A(I,J)
    //--------------------------------------------------------------------------

    #define GB_PHASE_2_OF_2
    int64_t *restrict Ci = C->i ;
    int64_t *restrict Cx = (int64_t *) C->x ;

    if (symbolic)
    { 

        //----------------------------------------------------------------------
        // symbolic subref
        //----------------------------------------------------------------------

        ASSERT (!C_iso) ;

        // symbolic subref must handle zombies
        const int64_t nzombies = A->nzombies ;

        // symbolic copy: Cx is int64_t; the values of A ignored
        #define GB_COPY_RANGE(pC,pA,len)            \
            for (int64_t k = 0 ; k < (len) ; k++)   \
            {                                       \
                Cx [(pC) + k] = (pA) + k ;          \
            }
        #define GB_COPY_ENTRY(pC,pA) Cx [pC] = (pA) ;
        #define GB_CSIZE1 1
        #define GB_CSIZE2 (sizeof (int64_t))
        #define GB_SYMBOLIC
        #include "GB_subref_template.c"

    }
    else if (C_iso)
    { 

        //----------------------------------------------------------------------
        // iso numeric subref
        //----------------------------------------------------------------------

        // C is iso; no numeric values to extract; just set the iso value
        memcpy (Cx, cscalar, A->type->size) ;
        #define GB_COPY_RANGE(pC,pA,len) ;
        #define GB_COPY_ENTRY(pC,pA) ;
        #define GB_ISO_SUBREF
        #include "GB_subref_template.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // non-iso numeric subref
        //----------------------------------------------------------------------

        // TODO: create versions for asize = 1, 2, 4, 8, 16, and other

        ASSERT (C->type = A->type) ;
        const int64_t asize = A->type->size ;
        const GB_void *restrict Ax = (GB_void *) A->x ;
              GB_void *restrict Cx = (GB_void *) C->x ;

        // C and A have the same type
        #define GB_COPY_RANGE(pC,pA,len)                                \
            memcpy (Cx + (pC)*asize, Ax + (pA)*asize, (len) * asize) ;
        #define GB_COPY_ENTRY(pC,pA)                                    \
            memcpy (Cx + (pC)*asize, Ax + (pA)*asize, asize) ;
        #define GB_CSIZE1 asize
        #define GB_CSIZE2 asize
        #include "GB_subref_template.c"
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    info = GB_hypermatrix_prune (C, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_phybix_free (C) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // caller must not free Cp or Ch
    ASSERT_MATRIX_OK (C, "C output for subref phase3", GB0) ;
    return (GrB_SUCCESS) ;
}

