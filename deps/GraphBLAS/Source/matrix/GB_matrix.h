//------------------------------------------------------------------------------
// GB_matrix.h: definitions for basic methods for the GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MATRIX_H
#define GB_MATRIX_H

GrB_Info GB_new                 // create matrix, except for indices & values
(
    // output:
    GrB_Matrix *Ahandle,        // handle of matrix to create
    // inputs:
    const GrB_Type type,        // matrix type
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_ph_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity,         // hyper, sparse, bitmap, full, or auto
    const float hyper_switch,   // A->hyper_switch
    const int64_t plen,         // size of A->p and A->h, if A hypersparse.
                                // Ignored if A is not hypersparse.
    bool p_is_32,               // if true, A->p is 32 bit; 64 bit otherwise
    bool j_is_32,               // if true, A->h and A->Y are 32 bit; else 64
    bool i_is_32,               // if true, A->i is 32 bit; 64 bit otherwise
    const int header_arena,     // arena for header, if allocated
    const int data_arena        // arena for matrix data
) ;

/*
GrB_Info GB_new_bix             // create a new matrix, incl. A->b, A->i, A->x
(
    GrB_Matrix *Ahandle,        // output matrix to create
    const GrB_Type type,        // type of output matrix
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_ph_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity,         // hyper, sparse, bitmap, full, or auto
    const bool bitmap_calloc,   // if true, calloc A->b, otherwise use malloc
    const float hyper_switch,   // A->hyper_switch, unless auto
    const int64_t plen,         // size of A->p and A->h, if hypersparse
    const int64_t nzmax,        // number of nonzeros the matrix must hold;
                                // ignored if A is iso and full
    const bool numeric,         // if true, allocate A->x, else A->x is NULL
    const bool A_iso,           // if true, allocate A as iso
    bool p_is_32,               // if true, A->p is 32 bit; 64 bit otherwise
    bool j_is_32,               // if true, A->h and A->Y are 32 bit; else 64
    bool i_is_32,               // if true, A->i is 32 bit; 64 bit otherwise
    const int header_arena,     // arena for header, if allocated
    const int data_arena        // arena for matrix data
) ;
*/

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

void GB_hy_free                 // free A-h and A->Y of a matrix
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

/*
void GB_Matrix_free             // free a matrix
(
    GrB_Matrix *Ahandle         // handle of matrix to free
) ;
*/

GrB_Info GB_shallow_copy    // create a purely shallow matrix
(
    GrB_Matrix C,           // output matrix C, with a existing header
    const bool C_is_csc,    // desired CSR/CSC format of C
    const GrB_Matrix A,     // input matrix
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_matrix_header_new
//------------------------------------------------------------------------------

// Allocate an empty matrix header.

static inline GrB_Info GB_matrix_header_new
(
    GrB_Matrix *Ahandle,
    const int header_arena,
    const int data_arena
)
{
    ASSERT (Ahandle != NULL) ;
    uint64_t header_mem = GB_mem (header_arena, 0) ;
    (*Ahandle) = (GrB_Matrix) GB_CALLOC_MEMORY (1, sizeof (struct GB_Matrix_opaque), &header_mem) ;
    if (*Ahandle == NULL)
    {
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*Ahandle)->header_mem = header_mem ;
    (*Ahandle)->data_arena = data_arena ;
    (*Ahandle)->magic = GB_MAGIC2 ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_VECTOR_OK, GB_SCALAR_OK: check if typecast from GrB_Matrix is OK
//------------------------------------------------------------------------------

// The internal content of a GrB_Matrix and GrB_Vector are identical, and
// inside SuiteSparse:GraphBLAS, they can be typecasted between each other.
// This typecasting feature should not be done in user code, however, since it
// is not supported in the API.  All GrB_Vector objects can be safely
// typecasted into a GrB_Matrix, but not the other way around.  The GrB_Vector
// object is more restrictive.  The GB_VECTOR_OK(v) macro defines the content
// that all GrB_Vector objects must have.

// GB_VECTOR_OK(v) is used mainly for assertions, but also to determine when it
// is safe to typecast an n-by-1 GrB_Matrix (in standard CSC format) into a
// GrB_Vector.  The macro is also used in GB_Vector_check, to ensure the
// content of a GrB_Vector is valid.

#define GB_VECTOR_OK(v)                     \
(                                           \
    ((v) != NULL) &&                        \
    ((v)->is_csc == true) &&                \
    ((v)->plen == 1 || (v)->plen == -1) &&  \
    ((v)->vdim == 1) &&                     \
    ((v)->nvec == 1) &&                     \
    ((v)->h == NULL)                        \
)

// A GxB_Vector is a GrB_Vector of length 1
#define GB_SCALAR_OK(v) (GB_VECTOR_OK(v) && ((v)->vlen == 1))

#endif

