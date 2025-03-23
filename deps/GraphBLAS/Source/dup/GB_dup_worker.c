//------------------------------------------------------------------------------
// GB_dup_worker: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy.  The header for C may already exist.

// if numeric is false, C->x is allocated but not initialized.

// If *Chandle is not NULL on input, the header is reused.  It may be a static
// or dynamic header, depending on C->header_size.

#include "GB.h"
#include "get_set/GB_get_set.h"
#define GB_FREE_ALL \
    GB_FREE_MEMORY (&C_user_name, C_user_name_size) ;

GrB_Info GB_dup_worker      // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // output matrix, NULL or existing static/dynamic
    const bool C_iso,       // if true, construct C as iso
    const GrB_Matrix A,     // input matrix to copy
    const bool numeric,     // if true, duplicate the numeric values; if A is
                            // iso, only the first entry is copied, regardless
                            // of C_iso on input
    const GrB_Type ctype    // type of C, if numeric is false
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to duplicate", GB0) ;
    ASSERT (Chandle != NULL) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    int64_t anvec = A->nvec ;
    int64_t anvals = A->nvals ;
    int64_t anvec_nonempty = GB_nvec_nonempty_update (A) ;
    int64_t A_nzombies = A->nzombies ;
    bool A_jumbled = A->jumbled ;
    int sparsity_control = A->sparsity_control ;
    GrB_Type atype = A->type ;

    //--------------------------------------------------------------------------
    // copy the user_name of A, if present
    //--------------------------------------------------------------------------

    char *C_user_name = NULL ;
    size_t C_user_name_size = 0 ;
    if (A->user_name != NULL)
    { 
        info = GB_user_name_set (&C_user_name, &C_user_name_size,
            A->user_name, false) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory
            return (info) ;
        }
    }

    //--------------------------------------------------------------------------
    // create C
    //--------------------------------------------------------------------------

    // create C; allocate C->p and do not initialize it.
    // C has the exact same sparsity structure and integer sizes as A.

    // allocate a new user header for C if (*Chandle) is NULL, or reuse the
    // existing static or dynamic header if (*Chandle) is not NULL.
    GrB_Matrix C = (*Chandle) ;
    GB_OK (GB_new_bix (Chandle, // can be new or existing header
        numeric ? atype : ctype, A->vlen, A->vdim, GB_ph_malloc, A->is_csc,
        GB_sparsity (A), false, A->hyper_switch, A->plen, anz, true, C_iso,
        A->p_is_32, A->j_is_32, A->i_is_32)) ;
    C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // copy the contents of A into C
    //--------------------------------------------------------------------------

    C->nvec = anvec ;
//  C->nvec_nonempty = anvec_nonempty ;
    GB_nvec_nonempty_set (C, anvec_nonempty) ;
    C->nvals = anvals ;
    C->jumbled = A_jumbled ;        // C is jumbled if A is jumbled
    C->nzombies = A_nzombies ;      // zombies can be duplicated
    C->sparsity_control = sparsity_control ;

    size_t psize = A->p_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = A->j_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = A->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (A->p != NULL)
    { 
        GB_memcpy (C->p, A->p, (anvec+1) * psize, nthreads_max) ;
    }
    if (A->h != NULL)
    { 
        GB_memcpy (C->h, A->h, anvec * jsize, nthreads_max) ;
    }
    if (A->b != NULL)
    { 
        GB_memcpy (C->b, A->b, anz * sizeof (int8_t), nthreads_max) ;
    }
    if (A->i != NULL)
    { 
        GB_memcpy (C->i, A->i, anz * isize, nthreads_max) ;
    }
    if (numeric)
    { 
        ASSERT (C_iso == A->iso) ;
        ASSERT (C->type == A->type) ;
        GB_memcpy (C->x, A->x, (A->iso ? 1:anz) * atype->size, nthreads_max) ;
    }

    C->magic = GB_MAGIC ;      // C->p and C->h are now initialized

    //--------------------------------------------------------------------------
    // copy the user_name of A into C, if present
    //--------------------------------------------------------------------------

    C->user_name = C_user_name ;
    C->user_name_size = C_user_name_size ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    if (numeric) ASSERT_MATRIX_OK (C, "C duplicate of A", GB0) ;
    #endif
    return (GrB_SUCCESS) ;
}

