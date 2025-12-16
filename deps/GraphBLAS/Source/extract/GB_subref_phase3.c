//------------------------------------------------------------------------------
// GB_subref_phase3: C=A(I,J) where C and A are sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function either frees Cp and Ch, or transplants then into C, as C->p
// and C->h.  Either way, the caller must not free them.

#include "extract/GB_subref.h"
#include "sort/GB_sort.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_subref_phase3   // C=A(I,J)
(
    GrB_Matrix C,               // output matrix, static header
    // from phase2:
    void **Cp_handle,           // vector pointers for C
    const bool Cp_is_32,        // if true, Cp is 32-bit; else 64-bit
    size_t Cp_size,
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C
    // from phase1:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int ntasks,                           // # of tasks
    const int nthreads,                         // # of threads to use
    const bool post_sort,               // true if post-sort needed
    const GrB_Matrix R,                 // R = inverse (I), if needed
    // from phase0:
    void **Ch_handle,
    const bool Cj_is_32,        // if true, C->h is 32-bit; else 64-bit
    const bool Ci_is_32,        // if true, C->i is 32-bit; else 64-bit
    size_t Ch_size,
    const void *Ap_start,
    const void *Ap_end,
    const int64_t Cnvec,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    const int64_t nJ,
    // from GB_subref:
    const GrB_Type ctype,       // type of C to create
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // iso value of C
    // original input:
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,
    const void *I,
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const bool symbolic,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    ASSERT (Cp_handle != NULL) ;
    ASSERT (Ch_handle != NULL) ;

    GB_MDECL (Cp, const, u) ;
    Cp = (*Cp_handle) ;
    GB_IPTR (Cp, Cp_is_32) ;

    void *Ch = (*Ch_handle) ;

    bool Ap_is_32 = A->p_is_32 ;
    bool Ai_is_32 = A->i_is_32 ;

    GB_IDECL (I       , const, u) ; GB_IPTR (I       , I_is_32) ;
    GB_IDECL (Ap_start, const, u) ; GB_IPTR (Ap_start, Ap_is_32) ;
    GB_IDECL (Ap_end  , const, u) ; GB_IPTR (Ap_end  , Ap_is_32) ;

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

    ASSERT (Cp != NULL) ;
    ASSERT_MATRIX_OK (A, "A for subref phase3", GB0) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int64_t cnz = GB_IGET (Cp, Cnvec) ;
    bool C_is_hyper = (Ch != NULL) ;

    // allocate the result C (but do not allocate C->p or C->h)
    int sparsity = C_is_hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;
    GrB_Info info = GB_new_bix (&C, // sparse or hyper, existing header
        ctype, nI, nJ, GB_ph_null, C_is_csc,
        sparsity, true, A->hyper_switch, Cnvec, cnz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_MEMORY (Cp_handle, Cp_size) ;
        GB_FREE_MEMORY (Ch_handle, Ch_size) ;
        return (info) ;
    }

    // add Cp as the vector pointers for C, from GB_subref_phase2
    C->p = (*Cp_handle) ; C->p_size = Cp_size ;
    (*Cp_handle) = NULL ;

    // add Ch as the hypersparse list for C, from GB_subref_phase0
    if (C_is_hyper)
    { 
        // transplant Ch into C
        C->h = Ch ; C->h_size = Ch_size ;
        (*Ch_handle) = NULL ;
        C->nvec = Cnvec ;
    }

    // now Cp and Ch have been transplanted into C, so they must not be freed.
    ASSERT ((*Cp_handle) == NULL) ;
    ASSERT ((*Ch_handle) == NULL) ;
//  C->nvec_nonempty = Cnvec_nonempty ;
    GB_nvec_nonempty_set (C, Cnvec_nonempty) ;
    C->nvals = cnz ;
    C->magic = GB_MAGIC ;
    ASSERT (C->p_is_32 == Cp_is_32) ;
    ASSERT (C->j_is_32 == Cj_is_32) ;
    ASSERT (C->i_is_32 == Ci_is_32) ;

    //--------------------------------------------------------------------------
    // phase3: C = A(I,J)
    //--------------------------------------------------------------------------

    GB_Ci_DECLARE_U (Ci, ) ; GB_Ci_PTR (Ci, C) ;

    #define GB_PHASE_2_OF_2
    #define GB_I_KIND Ikind
    #define GB_NEED_QSORT need_qsort

    if (symbolic)
    { 

        //----------------------------------------------------------------------
        // symbolic subref: Cx is uint32_t or uint64_t; the values of A ignored
        //----------------------------------------------------------------------

        ASSERT (!C_iso) ;
        ASSERT (ctype == GrB_UINT32 || ctype == GrB_UINT64) ;

        // symbolic subref must handle zombies
        const bool may_see_zombies = (A->nzombies > 0) ;

        if (ctype == GrB_UINT32)
        {
            uint32_t *restrict Cx = (uint32_t *) C->x ;

            #define GB_COPY_RANGE(pC,pA,len)            \
                for (int64_t k = 0 ; k < (len) ; k++)   \
                {                                       \
                    Cx [(pC) + k] = (pA) + k ;          \
                }
            #define GB_COPY_ENTRY(pC,pA) Cx [pC] = (pA) ;
            #define GB_QSORT_1B(Ci,Cx,pC,clen)                          \
            {                                                           \
                if (Ci_is_32)                                           \
                {                                                       \
                    GB_qsort_1b_32_size4 (Ci32 + pC, Cx + pC, clen) ;   \
                }                                                       \
                else                                                    \
                {                                                       \
                    GB_qsort_1b_64_size4 (Ci64 + pC, Cx + pC, clen) ;   \
                }                                                       \
            }
            #define GB_SYMBOLIC
            #include "extract/template/GB_subref_template.c"

        }
        else
        {
            uint64_t *restrict Cx = (uint64_t *) C->x ;

            #define GB_COPY_RANGE(pC,pA,len)            \
                for (int64_t k = 0 ; k < (len) ; k++)   \
                {                                       \
                    Cx [(pC) + k] = (pA) + k ;          \
                }
            #define GB_COPY_ENTRY(pC,pA) Cx [pC] = (pA) ;
            #define GB_QSORT_1B(Ci,Cx,pC,clen)                          \
            {                                                           \
                if (Ci_is_32)                                           \
                {                                                       \
                    GB_qsort_1b_32_size8 (Ci32 + pC, Cx + pC, clen) ;   \
                }                                                       \
                else                                                    \
                {                                                       \
                    GB_qsort_1b_64_size8 (Ci64 + pC, Cx + pC, clen) ;   \
                }                                                       \
            }
            #define GB_SYMBOLIC
            #include "extract/template/GB_subref_template.c"
        }

    }
    else if (C_iso)
    { 

        //----------------------------------------------------------------------
        // iso numeric subref
        //----------------------------------------------------------------------

        // C is iso; no numeric values to extract; just set the iso value
        memcpy (C->x, cscalar, A->type->size) ;
        #define GB_COPY_RANGE(pC,pA,len) ;
        #define GB_COPY_ENTRY(pC,pA) ;
        #define GB_ISO_SUBREF
        #define GB_QSORT_1B(Ci,Cx,pC,clen)                          \
        {                                                           \
            if (Ci_is_32)                                           \
            {                                                       \
                GB_qsort_1_32 (Ci32 + pC, clen) ;                   \
            }                                                       \
            else                                                    \
            {                                                       \
                GB_qsort_1_64 (Ci64 + pC, clen) ;                   \
            }                                                       \
        }
        #include "extract/template/GB_subref_template.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // non-iso numeric subref
        //----------------------------------------------------------------------

        ASSERT (ctype == A->type) ;

        // using the JIT kernel
        info = GB_subref_sparse_jit (C, TaskList, ntasks, nthreads, post_sort,
            R, Ap_start, Ap_end, need_qsort, Ikind, nI, Icolon, A, I, I_is_32) ;

        if (info == GrB_NO_VALUE)
        { 
            // using the generic kernel
            GBURBLE ("(generic subref) ") ;
            ASSERT (C->type = A->type) ;
            const int64_t csize = C->type->size ;
            const GB_void *restrict Ax = (GB_void *) A->x ;
                  GB_void *restrict Cx = (GB_void *) C->x ;

            // C and A have the same type
            #define GB_COPY_RANGE(pC,pA,len)                                \
                memcpy (Cx + (pC)*csize, Ax + (pA)*csize, (len) * csize) ;
            #define GB_COPY_ENTRY(pC,pA)                                    \
                memcpy (Cx + (pC)*csize, Ax + (pA)*csize, csize) ;
            #define GB_QSORT_1B(Ci,Cx,pC,clen)                          \
            {                                                           \
                if (Ci_is_32)                                           \
                {                                                       \
                    GB_qsort_1b_32_generic (Ci32 + pC,                  \
                        (GB_void *) (Cx+(pC)*csize), csize, clen) ;     \
                }                                                       \
                else                                                    \
                {                                                       \
                    GB_qsort_1b_64_generic (Ci64 + pC,                  \
                        (GB_void *) (Cx+(pC)*csize), csize, clen) ;     \
                }                                                       \
            }
            #include "extract/template/GB_subref_template.c"
            info = GrB_SUCCESS ;
        }
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        info = GB_hyper_prune (C, Werk) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info != GrB_SUCCESS)
    { 
        // out of memory or JIT kernel failed
        GB_phybix_free (C) ;
        return (info) ;
    }

    // caller must not free Cp or Ch
    ASSERT_MATRIX_OK (C, "C output for subref phase3", GB0) ;
    return (GrB_SUCCESS) ;
}

