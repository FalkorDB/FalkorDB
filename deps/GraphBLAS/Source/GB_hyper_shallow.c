//------------------------------------------------------------------------------
// GB_hyper_shallow: create a sparse shallow version of a hypersparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// On input C must exist but the content of the C header is uninitialized
// except for C->static_header and C->header_size.  No memory is allocated to
// construct C as the hyper_shallow version of A.  C is purely shallow.  If A
// is iso then so is C.

#include "GB.h"
#include "GB_convert.h"

GrB_Matrix GB_hyper_shallow         // return C
(
    GrB_Matrix C,                   // output sparse matrix
    const GrB_Matrix A              // input hypersparse matrix, not modified.
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "hyper_shallow input", GB0) ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT (GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // construct the hyper_shallow version
    //--------------------------------------------------------------------------

    // save the C header status
    bool C_static_header = C->static_header ;
    bool C_header_size = C->header_size ;

    // copy the header
    memcpy (C, A, sizeof (struct GB_Matrix_opaque)) ;

    // restore the C header status
    C->static_header = C_static_header  ;
    C->header_size = C_header_size ;

    // remove the user_name
    C->user_name = NULL ;
    C->user_name_size = 0 ;

    // remove the hyperlist and the hyper_hash
    C->h = NULL ;
    C->h_shallow = false ;
    C->Y = NULL ;
    C->Y_shallow = false ;

    // flag all content of C as shallow
    C->p_shallow = true ;
    C->i_shallow = true ;
    C->x_shallow = true ;

    // C reduces in dimension to the # of vectors in A
    C->vdim = C->nvec ;
    C->plen = C->nvec ;
    C->nvec_nonempty = C->nvec ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "hyper_shallow output", GB0) ;
    ASSERT (GB_IS_SPARSE (C)) ;
    return (C) ;
}

