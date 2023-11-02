//------------------------------------------------------------------------------
// GB_shallow_copy: create a shallow copy of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a purely shallow copy C of a matrix A.  No typecasting is done.  If A
// has zombies or pending tuples, those are finished first.

// The CSR/CSC format of C and A can differ, but they have they same vlen and
// vdim.  This function is CSR/CSC agnostic, except that C_is_csc is used to
// set the C->is_csc state in C.  The matrix C has the same iso property as
// the matrix A.

// Shallow matrices are never passed back to the user.

// A has any sparsity structure (hypersparse, sparse, bitmap, or full).

// Compare this function with GB_shallow_op.c.

#include "GB_transpose.h"
#include "GB_unused.h"

#define GB_FREE_ALL ;

GrB_Info GB_shallow_copy    // create a purely shallow matrix
(
    GrB_Matrix C,           // output matrix C, with a static header
    const bool C_is_csc,    // desired CSR/CSC format of C
    const GrB_Matrix A,     // input matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_MATRIX_OK (A, "A for shallow copy", GB0) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    //--------------------------------------------------------------------------
    // construct a shallow copy of A for the pattern of C
    //--------------------------------------------------------------------------

    // allocate the struct for C, but do not allocate C->[p,h,b,i,x].
    // C has the exact same sparsity structure as A.
    GrB_Info info ;
    info = GB_new (&C, // sparse or hyper, existing header
        A->type, A->vlen, A->vdim, GB_Ap_null, C_is_csc,
        GB_sparsity (A), A->hyper_switch, 0) ;
    ASSERT (info == GrB_SUCCESS) ;

    //--------------------------------------------------------------------------
    // make a shallow copy of the vector pointers
    //--------------------------------------------------------------------------

    ASSERT (C->magic == GB_MAGIC2) ;    // C not yet initialized
    C->p_shallow = (A->p != NULL) ;     // C->p not freed when freeing C
    C->h_shallow = (A->h != NULL) ;     // C->h not freed when freeing C
    C->p = A->p ;                       // C->p is of size A->plen + 1
    C->h = A->h ;                       // C->h is of size A->plen
    C->p_size = A->p_size ;
    C->h_size = A->h_size ;
    C->plen = A->plen ;                 // C and A have the same hyperlist size
    C->nvec = A->nvec ;
    C->nvec_nonempty = A->nvec_nonempty ;
    C->jumbled = A->jumbled ;           // C is jumbled if A is jumbled
    C->nvals = A->nvals ;
    C->magic = GB_MAGIC ;
    C->iso = A->iso ;                   // OK: C has the same iso property as A
    if (A->iso)
    { 
        GB_BURBLE_MATRIX (A, "(iso copy) ") ;
    }

    //--------------------------------------------------------------------------
    // make a shallow copy of the A->Y hyper_hash
    //--------------------------------------------------------------------------

    C->Y = A->Y ;
    C->Y_shallow = (A->Y != NULL) ;

    //--------------------------------------------------------------------------
    // check for empty matrix
    //--------------------------------------------------------------------------

    if (GB_nnz_max (A) == 0)
    { 
        // C->p and C->h are shallow but the rest is empty
        C->b = NULL ;
        C->i = NULL ;
        C->x = NULL ;
        C->b_shallow = false ;
        C->i_shallow = false ;
        C->x_shallow = false ;
        C->jumbled = false ;
        C->iso = false ;
        ASSERT_MATRIX_OK (C, "C = quick copy of empty A", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // make a shallow copy of the pattern and values
    //--------------------------------------------------------------------------

    C->b = A->b ; C->b_shallow = (A->b != NULL) ; C->b_size = A->b_size ;
    C->i = A->i ; C->i_shallow = (A->i != NULL) ; C->i_size = A->i_size ;
    C->x = A->x ; C->x_shallow = (A->x != NULL) ; C->x_size = A->x_size ;

    ASSERT_MATRIX_OK (C, "C = pure shallow (A)", GB0) ;
    return (GrB_SUCCESS) ;
}

