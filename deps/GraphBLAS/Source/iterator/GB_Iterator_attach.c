//------------------------------------------------------------------------------
// GB_Iterator_attach: attach an iterator to matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB(Iterator_attach)
(
    // input/output:
    GxB_Iterator iterator,  // iterator to attach to the matrix A
    // input
    GrB_Matrix A,           // matrix to attach
    int format,             // by row, by col, or by entry (GxB_NO_FORMAT)
    GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL (iterator) ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;

    if ((format == GxB_BY_ROW &&  A->is_csc) ||
        (format == GxB_BY_COL && !A->is_csc))
    { 
        return (GrB_NOT_IMPLEMENTED) ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work on the matrix
    //--------------------------------------------------------------------------

    if (GB_ANY_PENDING_WORK (A))
    { 
        GrB_Info info ;
        GB_WERK ("GxB_Iterator_attach") ;
        GB_OK (GB_wait (A, "A", Werk)) ;
    }

    //--------------------------------------------------------------------------
    // clear the current position
    //--------------------------------------------------------------------------

    iterator->pstart = 0 ;
    iterator->pend = 0 ;
    iterator->p = 0 ;
    iterator->k = 0 ;

    //--------------------------------------------------------------------------
    // get the matrix and save its contents in the iterator
    //--------------------------------------------------------------------------

    iterator->pmax = (GB_nnz (A) == 0) ? 0 : GB_nnz_held (A) ;
    iterator->avlen = A->vlen ;
    iterator->avdim = A->vdim ;
    iterator->anvec = A->nvec ;
    iterator->Ap32 = (A->p_is_32) ? A->p : NULL ;
    iterator->Ap64 = (A->p_is_32) ? NULL : A->p ;
    iterator->Ah32 = (A->j_is_32) ? A->h : NULL ;
    iterator->Ah64 = (A->j_is_32) ? NULL : A->h ;
    iterator->Ai32 = (A->i_is_32) ? A->i : NULL ;
    iterator->Ai64 = (A->i_is_32) ? NULL : A->i ;
    iterator->Ab = A->b ;
    iterator->Ax = A->x ;
    iterator->type_size = A->type->size ;
    iterator->A_sparsity = GB_sparsity (A) ;
    iterator->iso = A->iso ;
    iterator->by_col = A->is_csc ;

    return (GrB_SUCCESS) ;
}

