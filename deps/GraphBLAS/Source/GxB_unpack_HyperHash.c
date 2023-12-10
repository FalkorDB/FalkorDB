//------------------------------------------------------------------------------
// GxB_unpack_HyperHash: unpack the A->Y hyper_hash from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_unpack_HyperHash unpacks the hyper_hash from the hypersparse matrix A.
// Normally, this method is called immediately before calling one of the four
// methods GxB_Matrix_(export/unpack)_Hyper(CSR/CSC).  For example, to unpack
// then pack a hypersparse CSC matrix:

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

// If A is not hypersparse on input to GxB_unpack_HyperHash, or if A is
// hypersparse but does yet not have a hyper_hash, then Y is returned as NULL.
// This is not an error condition, and GrB_SUCCESS is returned.  The hyper_hash
// of a hypersparse matrix A is a matrix that provides quick access to the
// inverse of Ah.  It is not always needed and may not be present.  It is left
// as pending work to be computed when needed.

// GrB_Matrix_wait (A, GrB_MATERIALIZE) will ensure that the hyper_hash is
// constructed, if A is hypersparse.

// If Y is moved from A and returned as non-NULL to the caller, then it is
// the responsibility of the user application to free it, or to re-pack it back
// into A via GxB_pack_HyperHash, as shown in the example above.

// If this method is called to remove the hyper_hash Y from the hypersparse
// matrix A, and then GrB_Matrix_wait (A, GrB_MATERIALIZE) is called, a new
// hyper_hash matrix is constructed for A.

#include "GB_export.h"
#define GB_FREE_ALL ;

GrB_Info GxB_unpack_HyperHash       // move A->Y into Y
(
    GrB_Matrix A,                   // matrix to modify
    GrB_Matrix *Y,                  // hyper_hash matrix to move from A
    const GrB_Descriptor desc       // unused
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_unpack_HyperHash (A, &Y, desc)") ;
    GB_BURBLE_START ("GxB_unpack_HyperHash") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (Y) ;
    GB_RETURN_IF_FAULTY (*Y) ;

    //--------------------------------------------------------------------------
    // unpack the hyper_hash matrix Y from A
    //--------------------------------------------------------------------------

    (*Y) = A->Y ;
    A->Y = NULL ;
    A->Y_shallow = false ;

    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

