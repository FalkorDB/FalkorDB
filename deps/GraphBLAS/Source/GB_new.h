//------------------------------------------------------------------------------
// GB_new.h: definitions for GB_new and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_NEW_H
#define GB_NEW_H

typedef enum                    // input parameter to GB_new and GB_new_bix
{
    GB_Ap_calloc,               // 0: calloc A->p, malloc A->h if hypersparse
    GB_Ap_malloc,               // 1: malloc A->p, malloc A->h if hypersparse
    GB_Ap_null                  // 2: do not allocate A->p or A->h
}
GB_Ap_code ;

GrB_Info GB_Matrix_new          // create a new matrix with no entries
(
    GrB_Matrix *A,              // handle of matrix to create
    GrB_Type type,              // type of matrix to create
    GrB_Index nrows,            // matrix dimension is nrows-by-ncols
    GrB_Index ncols
) ;

GrB_Info GB_new                 // create matrix, except for indices & values
(
    GrB_Matrix *Ahandle,        // handle of matrix to create
    const GrB_Type type,        // matrix type
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_Ap_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity,         // hyper, sparse, bitmap, full, or
                                // auto (hyper + sparse)
    const float hyper_switch,   // A->hyper_switch, ignored if auto
    const int64_t plen          // size of A->p and A->h, if A hypersparse.
                                // Ignored if A is not hypersparse.
) ;

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
    const bool iso              // if true, allocate A as iso
) ;

// See GB_callbacks.h:
// GB_CALLBACK_BIX_ALLOC_PROTO (GB_bix_alloc) ;

GrB_Info GB_ix_realloc      // reallocate space in a matrix
(
    GrB_Matrix A,               // matrix to allocate space for
    const int64_t nzmax_new     // new number of entries the matrix can hold
) ;

void GB_bix_free                // free A->b, A->i, and A->x of a matrix
(
    GrB_Matrix A                // matrix with content to free
) ;

void GB_phy_free                // free A->p, A->h, and A->Y of a matrix
(
    GrB_Matrix A                // matrix with content to free
) ;

void GB_hyper_hash_free         // free the A->Y hyper_hash of a matrix
(
    GrB_Matrix A                // matrix with content to free
) ;

void GB_phybix_free             // free all content of a matrix
(
    GrB_Matrix A                // matrix with content to free
) ;

void GB_Matrix_free             // free a matrix
(
    GrB_Matrix *Ahandle         // handle of matrix to free
) ;

#endif

