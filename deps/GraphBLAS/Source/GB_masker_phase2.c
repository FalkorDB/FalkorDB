//------------------------------------------------------------------------------
// GB_masker_phase2: phase2 for R = masker (C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// GB_masker_phase2 computes R = masker (C,M,Z).  It is preceded first by
// GB_add_phase0, which computes the list of vectors of R to compute (Rh) and
// their location in C and Z (R_to_[CZ]).  Next, GB_masker_phase1 counts the
// entries in each vector R(:,j) and computes Rp.

// GB_masker_phase2 computes the pattern and values of each vector of R(:,j),
// entirely in parallel.

// R, M, C, and Z can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  All cases of the mask M are handled: present and not
// complemented, and present and complemented.  The mask is always present.

// This function either frees Rp and Rh, or transplants then into R, as R->p
// and R->h.  Either way, the caller must not free them.

// R is iso if both C and Z are iso and zij == cij.

#include "GB_mask.h"
#include "GB_ek_slice.h"
#include "GB_unused.h"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (R) ;                    \
}

GrB_Info GB_masker_phase2           // phase2 for R = masker (C,M,Z)
(
    GrB_Matrix R,                   // output matrix, static header
    const bool R_is_csc,            // format of output matrix R
    // from phase1:
    int64_t **Rp_handle,            // vector pointers for R
    size_t Rp_size,
    const int64_t Rnvec_nonempty,   // # of non-empty vectors in R
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,     // array of structs
    const int R_ntasks,               // # of tasks
    const int R_nthreads,             // # of threads to use
    // analysis from phase0:
    const int64_t Rnvec,
    int64_t **Rh_handle,
    size_t Rh_size,
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    const int R_sparsity,
    // original input:
    const GrB_Matrix M,             // required mask
    const bool Mask_comp,           // if true, then M is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix C,
    const GrB_Matrix Z,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (M, "M for mask phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    ASSERT_MATRIX_OK (C, "C for mask phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ; 
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ; 

    ASSERT_MATRIX_OK (Z, "Z for mask phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (Z)) ; 
    ASSERT (!GB_JUMBLED (Z)) ;
    ASSERT (!GB_PENDING (Z)) ; 

    ASSERT (!GB_IS_BITMAP (C)) ;        // not used if C is bitmap

    ASSERT (C->vdim == Z->vdim && C->vlen == Z->vlen) ;
    ASSERT (C->vdim == M->vdim && C->vlen == M->vlen) ;
    ASSERT (C->type == Z->type) ;

    ASSERT (R != NULL && (R->static_header || GBNSTATIC)) ;

    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;

    ASSERT (Rp_handle != NULL) ;
    ASSERT (Rh_handle != NULL) ;
    int64_t *Rp = (*Rp_handle) ;
    int64_t *Rh = (*Rh_handle) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix R
    //--------------------------------------------------------------------------

    bool R_is_hyper = (R_sparsity == GxB_HYPERSPARSE) ;
    bool R_is_sparse_or_hyper = (R_sparsity == GxB_SPARSE) || R_is_hyper ;
    ASSERT (R_is_sparse_or_hyper == (Rp != NULL)) ;
    ASSERT (R_is_hyper == (Rh != NULL)) ;

    int64_t rnz = (R_is_sparse_or_hyper) ? Rp [Rnvec] : C->vlen*C->vdim ;

    size_t czsize = Z->type->size ;
    bool R_iso ;
    int64_t cnz = GB_nnz (C) ;
    int64_t znz = GB_nnz (Z) ;
    if (cnz == 0)
    { 
        // C is empty: R is iso if Z is iso
        R_iso = Z->iso ;
    }
    else if (znz == 0)
    { 
        // Z is empty: R is iso if C is iso
        R_iso = C->iso ;
    }
    else
    { 
        // C and Z are both non-empty:  R is iso if both C and Z are
        // iso, and have the same iso value.
        R_iso = (C->iso && Z->iso && (memcmp (C->x, Z->x, czsize) == 0)) ;
    }

    // allocate the result R (but do not allocate R->p or R->h)
    // set R->iso = R_iso   OK
    GrB_Info info = GB_new_bix (&R, // any sparsity, existing header
        C->type, C->vlen, C->vdim, GB_Ap_null, R_is_csc,
        R_sparsity, true, C->hyper_switch, Rnvec, rnz, true, R_iso) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory; caller must free R_to_M, R_to_C, R_to_Z
        GB_FREE (Rp_handle, Rp_size) ;
        GB_FREE (Rh_handle, Rh_size) ;
        return (info) ;
    }

    // add Rp as the vector pointers for R, from GB_masker_phase1
    if (R_is_sparse_or_hyper)
    { 
        R->nvec_nonempty = Rnvec_nonempty ;
        R->p = (int64_t *) Rp ; R->p_size = Rp_size ;
        R->nvals = rnz ;
        (*Rp_handle) = NULL ;
    }

    // add Rh as the hypersparse list for R, from GB_add_phase0
    if (R_is_hyper)
    { 
        R->h = (int64_t *) Rh ; R->h_size = Rh_size ;
        R->nvec = Rnvec ;
        (*Rh_handle) = NULL ;
    }

    // now Rp and Rh have been transplanted into R, so they must not be freed.
    ASSERT ((*Rp_handle) == NULL) ;
    ASSERT ((*Rh_handle) == NULL) ;
    R->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // generic worker
    //--------------------------------------------------------------------------

    #define GB_PHASE_2_OF_2
    if (R_iso)
    { 
        // R can be iso only if C and/or Z are iso
        GBURBLE ("(iso mask) ") ;
        #define GB_ISO_MASKER
        if (cnz == 0)
        { 
            // Z must be iso; copy its iso value into R
            memcpy (R->x, Z->x, czsize) ;
        }
        else
        { 
            // C must be iso; copy its iso value into R
            memcpy (R->x, C->x, czsize) ;
        }
        #include "GB_masker_template.c"
    }
    else
    { 
        #include "GB_masker_template.c"
    }

    //--------------------------------------------------------------------------
    // prune empty vectors from Rh
    //--------------------------------------------------------------------------

    GB_OK (GB_hypermatrix_prune (R, Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // caller must free R_to_M, R_to_C, and R_to_Z, but not Rp or Rh
    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (R, "R output for mask phase2", GB0) ;
    ASSERT (!GB_ZOMBIES (R)) ; 
    ASSERT (!GB_JUMBLED (R)) ;
    ASSERT (!GB_PENDING (R)) ; 
    return (GrB_SUCCESS) ;
}

