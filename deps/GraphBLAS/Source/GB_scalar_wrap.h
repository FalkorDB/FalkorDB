//------------------------------------------------------------------------------
// GB_scalar_wrap.h: definitions for GB_scalar_wrap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SCALAR_WRAP_H
#define GB_SCALAR_WRAP_H

GrB_Scalar GB_Scalar_wrap   // create a new GrB_Scalar with one entry
(
    GrB_Scalar s,           // GrB_Scalar to create
    GrB_Type type,          // type of GrB_Scalar to create
    void *Sx                // becomes S->x, an array of size 1 * type->size
) ;

// stype can be NULL if op is positional

// wrap a bare scalar inside a statically-allocated GrB_Scalar
#define GB_SCALAR_WRAP(scalar,bare,stype)                                   \
    struct GB_Scalar_opaque scalar ## _header ;                             \
    size_t ssize = (stype == NULL) ? 1 : (stype->size) ;                    \
    GB_void Sx [GB_VLA(ssize)] ;                                            \
    GrB_Scalar scalar = GB_Scalar_wrap (& scalar ## _header, stype, Sx) ;   \
    memcpy (Sx, &bare, ssize) ;

// wrap a void *bare scalar inside a statically-allocated GrB_Scalar
#define GB_SCALAR_WRAP_UDT(scalar,bare,scalar_type)                         \
    GrB_Type stype = scalar_type ;                                          \
    struct GB_Scalar_opaque scalar ## _header ;                             \
    size_t ssize = (stype == NULL) ? 1 : (stype->size) ;                    \
    GB_void Sx [GB_VLA(ssize)] ;                                            \
    GrB_Scalar scalar = NULL ;                                              \
    if (bare != NULL && stype != NULL)                                      \
    {                                                                       \
        scalar = GB_Scalar_wrap (& scalar ## _header, stype, Sx) ;          \
        memcpy (Sx, bare, ssize) ;                                          \
    }

#endif

