//------------------------------------------------------------------------------
// GxB_pack_HyperHash: set the A->Y hyper_hash of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_pack_HyperHash assigns the input Y matrix as the A->Y hyper_hash of the
// hypersparse matrix A.  Normally, this method is called immediately after
// calling one of the four methods GxB_Matrix_(import/pack)_Hyper(CSR/CSC).
// For example, to unpack then pack a hypersparse CSC matrix:

//      GrB_Matrix Y = NULL ;
//
//      // to unpack all of A:
//      GxB_unpack_HyperHash (A, &Y, desc) ;    // first unpack A->Y into Y
//      GxB_Matrix_unpack_HyperCSC (A,          // then unpack the rest of A
//          &Ap, &Ah, &Ai, &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size,
//          &iso, &nvec, &jumbled, descriptor) ;
//
//      // use the unpacked contents of A here, but do not change Ah or nvec.
//      ...
//      
//      // to pack the data back into A:
//      GxB_Matrix_pack_HyperCSC (A, ...) ;     // pack most of A, except A->Y 
//      GxB_pack_HyperHash (A, &Y, desc) ;      // then pack A->Y

// The same process is used with GxB_Matrix_unpack_HyperCSR,
// and the GxB_Matrix_export_Hyper* and GxB_Matrix_import_Hyper* methods.

// If A is not hypersparse on input to GxB_pack_HyperHash, or if A already has
// a hyper_hash matrix, or if Y is NULL on input, then nothing happens and Y is
// unchanged.  This is not an error condition and this method returns
// GrB_SUCCESS.  In this case, if Y is non-NULL after calling this method, it
// owned by the user application and freeing it is the responsibility of the
// user application.

// Basic checks are perfomed on Y: Y must have the right dimensions:  if A is
// HyperCSR and m-by-n with nvec vectors present in Ah, then Y must be n-by-v
// where v is a power of 2; if A is HyperCSR and m-by-n, then Y must be m-by-v.
// nvals(Y) must equal nvec.  Y must be sparse, held by column, and have type
// int64.  It cannot have any pending work.  It cannot have a hyper_hash
// of its own.  If any of these conditions hold, GrB_INVALID is returned and
// A and Y are unchanged.

// If Y is moved into A as its hyper_hash, then the caller's Y is set to NULL
// to indicate that it has been moved into A.  It is no longer owned by the
// caller, but is instead an opaque component of the A matrix.  It will be
// freed by SuiteSparse:GraphBLAS if A is modified or freed.

// Results are undefined if the input Y was not created by GxB_unpack_HyperHash
// (see the example above) or if the Ah contents or nvec of the matrix A are
// modified after they were exported/unpacked by
// GxB_Matrix_(export/unpack)_Hyper(CSR/CSC).

#include "import_export/GB_export.h"
#define GB_FREE_ALL ;

GrB_Info GxB_pack_HyperHash         // move Y into A->Y
(
    GrB_Matrix A,                   // matrix to modify
    GrB_Matrix *Y,                  // hyper_hash matrix to pack into A
    const GrB_Descriptor desc       // unused
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;
    GB_RETURN_IF_NULL (Y) ;
    GB_RETURN_IF_INVALID (*Y) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (*Y) ;

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if ((*Y) == NULL || !GB_IS_HYPERSPARSE (A) || A->Y != NULL)
    { 
        // nothing to do.  A and Y are unchanged
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // basic error checks
    //--------------------------------------------------------------------------

    if ((*Y)->vlen != A->vdim || !GB_IS_POWER_OF_TWO ((*Y)->vdim) ||
        (*Y)->nvals != A->nvec || !GB_IS_SPARSE (*Y) || (*Y)->Y != NULL ||
        (!((*Y)->type == GrB_UINT64 || (*Y)->type == GrB_UINT32)) ||
        !(*Y)->is_csc || GB_ANY_PENDING_WORK (*Y))
    { 
        // Y is invalid
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // ensure Y has the same integers as A->h
    //--------------------------------------------------------------------------

    bool Aj_is_32 = A->j_is_32 ;
    GB_OK (GB_convert_int (*Y, Aj_is_32, Aj_is_32, Aj_is_32, false)) ;  // OK

    //--------------------------------------------------------------------------
    // pack the hyper_hash matrix Y into A
    //--------------------------------------------------------------------------

    A->Y = (*Y) ;
    (*Y) = NULL ;
    A->Y_shallow = false ;
    A->no_hyper_hash = false ;  // A now has a hyper_hash matrix A->Y
    ASSERT_MATRIX_OK (A, "A with new hyper_hash", GB0) ;
    return (GrB_SUCCESS) ;
}

