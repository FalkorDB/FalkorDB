//------------------------------------------------------------------------------
// GB_new_bix: create a matrix and allocate space
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Creates a matrix (with GB_new), then allocates a given space for indices and
// values.

// Ahandle must be non-NULL on input.

// If *Ahandle is NULL on input:

//      A new, dynamically allocated header for the matrix A is allocated.  If
//      successful, *Ahandle points to the new handle, and its contents, on
//      output.  If an out-of-memory condition occurs, the header is freed and
//      *Ahandle is NULL on output.  If successful, (*Ahandle)->static_header
//      will always be false on output.

// If *Ahandle is not NULL on input:

//      The existing header for A is used.  The pointer *Ahandle itself is not
//      modified on output, either on success or failure.  If successful, the
//      content of A has been created.  If an out-of-memory condition occurs,
//      the preexisting header is not freed and *Ahandle is unmodified on
//      output. (*Ahandle)->static_header is unchanged.

#include "GB.h"

GrB_Info GB_new_bix             // create a new matrix, incl. A->b, A->i, A->x
(
    GrB_Matrix *Ahandle,        // output matrix to create
    const GrB_Type type,        // type of output matrix
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_Ap_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity,         // hyper, sparse, bitmap, full, or auto
    const bool bitmap_calloc,   // if true, calloc A->b, otherwise use malloc
    const float hyper_switch,   // A->hyper_switch, unless auto
    const int64_t plen,         // size of A->p and A->h, if hypersparse
    const int64_t nzmax,        // number of nonzeros the matrix must hold;
                                // ignored if A is iso and full
    const bool numeric,         // if true, allocate A->x, else A->x is NULL
    const bool A_iso            // if true, allocate A as iso
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ahandle != NULL) ;

    //--------------------------------------------------------------------------
    // allocate the header and the vector pointers
    //--------------------------------------------------------------------------

    bool preexisting_header = (*Ahandle != NULL) ;
    GrB_Info info = GB_new (Ahandle, type, vlen, vdim,
        Ap_option, is_csc, sparsity, hyper_switch, plen) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory.
        // If *Ahandle was non-NULL on input, it has not been freed.
        ASSERT (preexisting_header == (*Ahandle != NULL)) ;
        return (info) ;
    }

    GrB_Matrix A = (*Ahandle) ;

    //--------------------------------------------------------------------------
    // allocate the bitmap (A->b), indices (A->i), and values (A->x)
    //--------------------------------------------------------------------------

    // set A->iso = A_iso   OK: burble in the caller
    info = GB_bix_alloc (A, nzmax, sparsity, bitmap_calloc, numeric, A_iso) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        // free all content of A
        GB_phybix_free (A) ;
        if (!preexisting_header)
        { 
            // free the header *Ahandle itself unless it existed on input
            GB_Matrix_free (Ahandle) ;
            ASSERT (*Ahandle == NULL) ;
        }
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

