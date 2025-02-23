//------------------------------------------------------------------------------
// GB_Scalar_wrap: wrap a C scalar inside a GraphBLAS scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method construct a shallow statically-defined scalar, with no memory
// allocations.  The scalar is iso full, with a single entry.

// Note that since the header is statically allocated, it cannot be transfered
// automatically to the GPU when using CUDA.

#include "GB.h"
#include "scalar/GB_Scalar_wrap.h"

GrB_Scalar GB_Scalar_wrap   // create a new GrB_Scalar with one entry
(
    GrB_Scalar s,           // GrB_Scalar to create
    GrB_Type type,          // type of GrB_Scalar to create
    void *Sx                // becomes S->x, an array of size 1 * type->size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (s != NULL) ;

    //--------------------------------------------------------------------------
    // create the GrB_Scalar
    //--------------------------------------------------------------------------

    s->magic = GB_MAGIC ;
    s->header_size = 0 ;
    s->user_name = NULL ;
    s->user_name_size = 0 ;
    s->type = (type == NULL) ? GrB_BOOL : type ;
    s->logger = NULL ;
    s->logger_size = 0 ;

    s->plen = -1 ;
    s->vlen = 1 ;
    s->vdim = 1 ;
    s->nvec = 1 ;

//  s->nvec_nonempty = 1 ;
    GB_nvec_nonempty_set ((GrB_Matrix) s, 1) ;

    s->p = NULL ; s->p_size = 0 ; s->p_shallow = false ;
    s->h = NULL ; s->h_size = 0 ; s->h_shallow = false ;
    s->b = NULL ; s->b_size = 0 ; s->b_shallow = false ;
    s->i = NULL ; s->i_size = 0 ; s->i_shallow = false ;
    s->x = Sx   ; s->x_size = type->size ; s->x_shallow = true ;

    s->Y = NULL ;
    s->Y_shallow = false ;
    s->no_hyper_hash = true ;

    s->nvals = 0 ;

    s->Pending = NULL ;
    s->nzombies = 0 ;

    s->hyper_switch  = GxB_NEVER_HYPER ;
    s->bitmap_switch = 0.5 ;
    s->sparsity_control = GxB_FULL ;

    s->is_csc = true ;
    s->jumbled = false ;
    s->iso = true ;

    s->p_is_32 = false ;
    s->j_is_32 = false ;
    s->i_is_32 = false ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (s) ;
}

