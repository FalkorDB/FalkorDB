//------------------------------------------------------------------------------
// GB_dup_worker: make a deep copy of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy.  The header for C may already exist.

// if numeric is false, C->x is allocated but not initialized.

// If *Chandle is not NULL on input, the header is reused.

// The input matrix A can include any pending work (pending tuples, zombies,
// or jumbled).  The pending work is copied into the output matrix C.  It is
// not finished.  This case is only supported if numeric is true.

// The p/j/i integers can differ from A.

#define GB_FREE_ALL                                     \
    GB_FREE_MEMORY (&C_user_name, C_user_name_mem) ;    \
    GB_phybix_free (C) ;                                \
    if (!preexisting_header)                            \
    {                                                   \
        GB_Matrix_free (Chandle) ;                      \
    }

#include "GB.h"
#include "get_set/GB_get_set.h"
#include "pending/GB_Pending.h"

GrB_Info GB_dup_worker      // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // output matrix, NULL or existing
    const bool C_iso,       // if true, construct C as iso
    const GrB_Matrix A,     // input matrix to copy
    const bool numeric,     // if true, duplicate the numeric values; if A is
                            // iso, only the first entry is copied, regardless
                            // of C_iso on input
    const GrB_Type ctype,   // type of C, if numeric is false
    const bool Cp_is_32,    // type of C->p
    const bool Cj_is_32,    // type of C->h and C->Y
    const bool Ci_is_32,    // type of C->i
    const int header_arena,
    const int data_arena,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    char *C_user_name = NULL ;
    uint64_t C_user_name_mem = 0 ;
    ASSERT_MATRIX_OK (A, "A to duplicate", GB0) ;
    ASSERT (Chandle != NULL) ;
    GrB_Matrix C = (*Chandle) ;
    bool preexisting_header = (C != NULL) ;
    ASSERT (GB_PENDING_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    GB_OK (GB_check_arena (header_arena)) ;
    GB_OK (GB_check_arena (data_arena)) ;

    uint64_t mem = GB_mem (data_arena, 0) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    int64_t anvec = A->nvec ;
    int64_t anvals = A->nvals ;
    int64_t anvec_nonempty = GB_nvec_nonempty_update (A) ;
    int64_t A_nzombies = A->nzombies ;
    bool A_jumbled = A->jumbled ;
    int sparsity_control = A->sparsity_control ;
    GrB_Type atype = A->type ;
    GB_Pending A_Pending = A->Pending ;

    //--------------------------------------------------------------------------
    // copy the user_name of A, if present
    //--------------------------------------------------------------------------

    C_user_name_mem = mem ;
    if (A->user_name != NULL)
    { 
        GB_OK (GB_user_name_set (&C_user_name, &C_user_name_mem,
            A->user_name, false, header_arena)) ;
    }

    //--------------------------------------------------------------------------
    // create C
    //--------------------------------------------------------------------------

    // C has the exact same sparsity structure and as A, but can have different
    // pji integer sizes.  A new header for C is allocated if (*Chandle) is
    // NULL on input, or the existing header is used if (*Chandle) is not NULL
    // on input.
    GB_OK (GB_new_bix (Chandle, // can be new or existing header
        numeric ? atype : ctype, A->vlen, A->vdim, GB_ph_malloc, A->is_csc,
        GB_sparsity (A), false, A->hyper_switch, A->plen, anz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32, header_arena, data_arena)) ;
    C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // allocate the pending tuples, if present
    //--------------------------------------------------------------------------

    if (A_Pending != NULL && numeric)
    { 
        // A has pending tuples; allocate space for them in C.  This case is
        // only supported if numeric is true.
        ASSERT (C_iso == A->iso) ;
        if (!GB_Pending_alloc (C, A->iso, A_Pending->type, A_Pending->op,
            A_Pending->nmax))
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // copy the A->Y hyper hash into C, if present
    //--------------------------------------------------------------------------

    ASSERT (C->Y == NULL) ;
    if (A->Y != NULL)
    { 
        GB_MATRIX_WAIT (A->Y) ;
        GrB_Type cytype = (C->j_is_32) ? GrB_UINT32 : GrB_UINT64 ;
        // create C->Y but just allocate C->Y->x
        GB_OK (GB_dup_worker (&(C->Y), /* Y is not iso: */ false, A->Y,
            /* numeric: */ false, cytype, C->j_is_32, C->j_is_32, C->j_is_32,
            data_arena, data_arena, Werk)) ;
        // typecast A->Y->x into C->Y->x
        GB_OK (GB_cast_matrix (C->Y, A->Y)) ;
    }

    //--------------------------------------------------------------------------
    // copy the A->[phbix] contents of A into C
    //--------------------------------------------------------------------------

    C->nvec = anvec ;
    GB_nvec_nonempty_set (C, anvec_nonempty) ;
    C->nvals = anvals ;
    C->jumbled = A_jumbled ;        // C is jumbled if A is jumbled
    C->nzombies = A_nzombies ;      // zombies can be duplicated
    C->sparsity_control = sparsity_control ;

    size_t psize = A->p_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = A->j_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = A->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    GB_Type_code cpcode = (C->p_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code apcode = (A->p_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code cjcode = (C->j_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code ajcode = (A->j_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code cicode = (C->i_is_32) ? GB_INT32_code  : GB_INT64_code ;
    GB_Type_code aicode = (A->i_is_32) ? GB_INT32_code  : GB_INT64_code ;

    if (A->p != NULL)
    { 
        GB_cast_int (C->p, cpcode, A->p, apcode, anvec+1, nthreads_max) ;
    }
    if (A->h != NULL)
    { 
        GB_cast_int (C->h, cjcode, A->h, ajcode, anvec, nthreads_max) ;
    }
    if (A->b != NULL)
    { 
        GB_memcpy (C->b, A->b, anz * sizeof (int8_t), nthreads_max) ;
    }
    if (A->i != NULL)
    { 
        GB_cast_int (C->i, cicode, A->i, aicode, anz, nthreads_max) ;
    }
    if (numeric)
    { 
        ASSERT (C_iso == A->iso) ;
        ASSERT (C->type == A->type) ;
        GB_memcpy (C->x, A->x, (A->iso ? 1:anz) * atype->size, nthreads_max) ;
    }

    //--------------------------------------------------------------------------
    // copy the pending tuples
    //--------------------------------------------------------------------------

    if (A_Pending != NULL && numeric)
    { 
        GB_Pending C_Pending = C->Pending ;
        int64_t n = A_Pending->n ;
        bool is_matrix = (A->vdim > 1) ;
        size_t jsize = (A->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t isize = (A->i_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t xsize = A_Pending->size ;
        GB_cast_int (C_Pending->i, cicode, A_Pending->i, aicode, n,
            nthreads_max) ;
        if (is_matrix)
        { 
            GB_cast_int (C_Pending->j, cjcode, A_Pending->j, ajcode, n,
                nthreads_max) ;
        }
        if (!A->iso)
        { 
            GB_memcpy (C_Pending->x, A_Pending->x, n * xsize, nthreads_max) ;
        }
        C_Pending->n = n ;
        C_Pending->sorted = A_Pending->sorted ;
    }

    //--------------------------------------------------------------------------
    // C->p and C->h are now initialized
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // copy the user_name of A into C, if present
    //--------------------------------------------------------------------------

    C->user_name = C_user_name ;
    C->user_name_mem = C_user_name_mem ;
    C_user_name = NULL ; C_user_name_mem = 0 ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    if (numeric) ASSERT_MATRIX_OK (C, "C duplicate of A", GB0) ;
    #endif
    return (GrB_SUCCESS) ;
}

